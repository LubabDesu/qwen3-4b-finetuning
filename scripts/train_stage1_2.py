#!/usr/bin/env python3
"""Run Stage 1-2 training from a terminal/nohup process.

This intentionally reuses the notebook definitions so the terminal run stays in
lockstep with `cse151b_post_training_curriculum.ipynb`.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "cse151b_post_training_curriculum.ipynb"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CORE_CELL_IDS = (4, 5, 6, 7, 8, 9, 16)
EVAL_CELL_IDS = (10, 11)
PREFLIGHT_CELL_ID = 18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Stage 1_2 model outside the notebook.")
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the Stage 1_2 pre-train inspection cell.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run the notebook's held-out eval after training. This requires vLLM.",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Save only the LoRA adapter and skip merged 16-bit checkpoint creation.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="Save a resumable Trainer checkpoint every N optimizer steps.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=3,
        help="Keep only the most recent N local Trainer checkpoints.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to a Trainer checkpoint directory, for example checkpoints/trainer_stage1_2/checkpoint-300.",
    )
    parser.add_argument(
        "--drive-target",
        type=str,
        default=os.environ.get("STAGE1_2_DRIVE_TARGET"),
        help=(
            "Optional backup target. Use an rclone remote like "
            "'gdrive:qwen3-4b-finetuning/stage1_2' or a mounted local path."
        ),
    )
    parser.add_argument(
        "--dry-run-setup",
        action="store_true",
        help="Load notebook definitions and Stage 1_2 records, then exit before model loading/training.",
    )
    return parser.parse_args()


def load_notebook_cells() -> list[dict[str, Any]]:
    with NOTEBOOK_PATH.open() as f:
        return json.load(f)["cells"]


def exec_cell(cells: list[dict[str, Any]], cell_id: int, namespace: dict[str, Any]) -> None:
    source = "".join(cells[cell_id].get("source", []))
    filename = f"{NOTEBOOK_PATH.name}#cell-{cell_id}"
    exec(compile(source, filename, "exec"), namespace)


def load_notebook_namespace(args: argparse.Namespace) -> SimpleNamespace:
    os.chdir(REPO_ROOT)
    cells = load_notebook_cells()
    module_name = "__stage1_2_train__"
    module = ModuleType(module_name)
    module.__file__ = str(NOTEBOOK_PATH)
    sys.modules[module_name] = module
    namespace = module.__dict__

    for cell_id in CORE_CELL_IDS:
        print(f"[stage1_2] loading notebook cell {cell_id}", flush=True)
        exec_cell(cells, cell_id, namespace)

    if not args.skip_preflight:
        print("[stage1_2] running Stage 1_2 preflight", flush=True)
        exec_cell(cells, PREFLIGHT_CELL_ID, namespace)

    if args.eval:
        for cell_id in EVAL_CELL_IDS:
            print(f"[stage1_2] loading eval notebook cell {cell_id}", flush=True)
            exec_cell(cells, cell_id, namespace)

    return SimpleNamespace(**namespace)


def is_rclone_target(target: str) -> bool:
    return ":" in target and not target.startswith("/") and not target.startswith("./") and not target.startswith("../")


def backup_path(src: Path, target: str | None, *, label: str) -> None:
    if not target:
        return
    if not src.exists():
        print(f"[stage1_2] backup skipped, missing {label}: {src}", flush=True)
        return

    if is_rclone_target(target):
        dest = f"{target.rstrip('/')}/{src.name}"
        cmd = [
            "rclone",
            "copy",
            str(src),
            dest,
            "--transfers=4",
            "--checkers=8",
            "--drive-chunk-size=128M",
        ]
        print(f"[stage1_2] backing up {label} to {dest}", flush=True)
        try:
            result = subprocess.run(cmd, check=False)
        except FileNotFoundError:
            print("[stage1_2] WARNING: rclone is not installed; Drive backup skipped.", flush=True)
            return
        if result.returncode != 0:
            print(f"[stage1_2] WARNING: rclone backup failed with exit code {result.returncode}", flush=True)
        return

    dest_root = Path(os.path.expanduser(target)).resolve()
    dest = dest_root / src.name
    print(f"[stage1_2] backing up {label} to {dest}", flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dest, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dest)


def train_stage_with_checkpoints(ns: SimpleNamespace, args: argparse.Namespace) -> tuple[Path, Path | None]:
    from transformers import Trainer, TrainerCallback, TrainingArguments

    class BackupOnSaveCallback(TrainerCallback):
        def __init__(self, target: str | None) -> None:
            self.target = target
            self.synced: set[int] = set()

        def on_save(self, trainer_args, state, control, **kwargs):  # type: ignore[no-untyped-def]
            step = int(state.global_step or 0)
            if not self.target or step <= 0 or step in self.synced:
                return control
            checkpoint_dir = Path(trainer_args.output_dir) / f"checkpoint-{step}"
            backup_path(checkpoint_dir, self.target, label=f"checkpoint step {step}")
            self.synced.add(step)
            return control

    cfg = ns.STAGE1_2
    print(f"[stage1_2] Loading model for {cfg.name}: {ns.STAGE1_2_BASE_MODEL}", flush=True)
    model, tokenizer = ns.load_model_for_training(ns.STAGE1_2_BASE_MODEL, cfg)
    print(f"[stage1_2] Tokenizer EOS/PAD: {tokenizer.eos_token!r} / {tokenizer.pad_token!r}", flush=True)
    train_dataset = ns.format_records_with_tokenizer(ns.stage1_2_records, tokenizer)

    lora_dir = ns.CHECKPOINT_DIR / f"lora_{cfg.name}"
    merged_dir = ns.CHECKPOINT_DIR / f"merged_{cfg.name}"
    output_dir = ns.CHECKPOINT_DIR / f"trainer_{cfg.name}"

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        num_train_epochs=cfg.epochs if cfg.max_steps == -1 else 1,
        max_steps=cfg.max_steps,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.scheduler,
        optim="adamw_8bit",
        bf16=True,
        fp16=False,
        logging_steps=5,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="none",
        seed=ns.SEED,
        gradient_checkpointing=False,
        remove_unused_columns=False,
    )

    callbacks = [BackupOnSaveCallback(args.drive_target)] if args.drive_target else None
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        data_collator=ns.make_causal_lm_collator(tokenizer),
        args=training_args,
        callbacks=callbacks,
    )

    print(
        "[stage1_2] checkpointing enabled: "
        f"save_steps={args.save_steps}, save_total_limit={args.save_total_limit}, output_dir={output_dir}",
        flush=True,
    )
    if args.drive_target:
        print(f"[stage1_2] checkpoint backup target: {args.drive_target}", flush=True)

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    lora_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))
    print(f"[stage1_2] Saved LoRA adapter: {lora_dir}", flush=True)
    backup_path(lora_dir, args.drive_target, label="final LoRA adapter")

    merged_path = None
    if not args.no_merge:
        merged_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
        merged_path = merged_dir
        print(f"[stage1_2] Saved merged 16-bit model: {merged_dir}", flush=True)
        backup_path(merged_dir, args.drive_target, label="final merged model")

    del trainer, model
    gc.collect()
    if ns.torch.cuda.is_available():
        ns.torch.cuda.empty_cache()
    return lora_dir, merged_path


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    args = parse_args()
    ns = load_notebook_namespace(args)
    if args.dry_run_setup:
        print(f"[stage1_2] dry run OK; loaded {len(ns.stage1_2_records)} Stage 1_2 records", flush=True)
        return

    print("[stage1_2] starting full Stage 1_2 training", flush=True)
    lora_stage1_2, merged_stage1_2 = train_stage_with_checkpoints(ns, args)
    print(f"[stage1_2] LoRA adapter: {lora_stage1_2}", flush=True)
    print(f"[stage1_2] merged checkpoint: {merged_stage1_2}", flush=True)

    if args.eval:
        print("[stage1_2] running eval", flush=True)
        metrics = ns.evaluate_model(str(merged_stage1_2), "stage1_2", ns.eval_set)
        print(f"[stage1_2] eval metrics: {metrics}", flush=True)

    print("[stage1_2] done", flush=True)


if __name__ == "__main__":
    main()
