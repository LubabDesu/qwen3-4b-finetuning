#!/usr/bin/env python3
"""Merge and evaluate Stage 1-2 Trainer checkpoints."""

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
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

SETUP_CELL_IDS = (4, 5, 6, 7, 8, 9, 10, 11)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved Stage 1_2 checkpoints.")
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=[100, 200, 300, 400, 500, 530],
        help="Checkpoint step numbers to evaluate.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("checkpoints/trainer_stage1_2"),
        help="Local directory containing checkpoint-N folders.",
    )
    parser.add_argument(
        "--drive-source",
        default=None,
        help=(
            "Optional rclone source containing checkpoint-N folders, for example "
            "'gdrive:151B_SP26_Competition/checkpoints/stage1_2/trainer_stage1_2'."
        ),
    )
    parser.add_argument(
        "--merged-root",
        type=Path,
        default=Path("checkpoints/eval_merged_stage1_2"),
        help="Where temporary merged models are written.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Eval only the first N eval rows.")
    parser.add_argument("--batch-size", type=int, default=20, help="vLLM eval batch size.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override notebook EVAL_MAX_NEW_TOKENS for generation during eval.",
    )
    parser.add_argument("--keep-merged", action="store_true", help="Keep merged checkpoint models after eval.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip evals whose summary JSON already exists.")
    parser.add_argument(
        "--public-only",
        action="store_true",
        help="Evaluate on public.jsonl instead of the held-out mixed eval set.",
    )
    parser.add_argument(
        "--public-path",
        type=Path,
        default=Path("artifacts/post_training_curriculum/eval/public.jsonl"),
        help="Path to competition-style public.jsonl.",
    )
    parser.add_argument(
        "--public-sample-size",
        type=int,
        default=None,
        help="Deterministically sample N public rows before applying --limit.",
    )
    parser.add_argument("--public-seed", type=int, default=42, help="Seed for deterministic public sampling.")
    return parser.parse_args()


def load_notebook_cells() -> list[dict[str, Any]]:
    with NOTEBOOK_PATH.open() as f:
        return json.load(f)["cells"]


def exec_cell(cells: list[dict[str, Any]], cell_id: int, namespace: dict[str, Any]) -> None:
    source = "".join(cells[cell_id].get("source", []))
    exec(compile(source, f"{NOTEBOOK_PATH.name}#cell-{cell_id}", "exec"), namespace)


def load_eval_namespace() -> SimpleNamespace:
    os.chdir(REPO_ROOT)
    module_name = "__stage1_2_checkpoint_eval__"
    module = ModuleType(module_name)
    module.__file__ = str(NOTEBOOK_PATH)
    sys.modules[module_name] = module
    namespace = module.__dict__
    cells = load_notebook_cells()
    for cell_id in SETUP_CELL_IDS:
        print(f"[checkpoint_eval] loading notebook cell {cell_id}", flush=True)
        exec_cell(cells, cell_id, namespace)
    try:
        from judger import Judger

        namespace["judger"] = Judger(strict_extract=False)
        print("[checkpoint_eval] using scripts/judger.py Judger(strict_extract=False)", flush=True)
    except Exception as exc:
        raise RuntimeError(
            "Could not import scripts/judger.py. Public eval should use the competition Judger, "
            "so fix this before running public-only eval. If the error mentions `utils`, copy the "
            "competition utils.py into scripts/utils.py."
        ) from exc
    return SimpleNamespace(**namespace)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_public_eval_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    import random

    rows = load_jsonl(args.public_path)
    rows = [{"eval_source": "public", **row} for row in rows]
    if args.public_sample_size is not None and args.public_sample_size < len(rows):
        rng = random.Random(args.public_seed)
        rows = rng.sample(rows, args.public_sample_size)
    if args.limit:
        rows = rows[: args.limit]
    print(
        "[checkpoint_eval] public eval rows "
        f"path={args.public_path} n={len(rows)} sample_size={args.public_sample_size} seed={args.public_seed}",
        flush=True,
    )
    return rows


def ensure_local_checkpoint(checkpoint_dir: Path, drive_source: str | None) -> None:
    if checkpoint_dir.exists():
        return
    if not drive_source:
        raise FileNotFoundError(f"Missing local checkpoint and no --drive-source was provided: {checkpoint_dir}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    remote = f"{drive_source.rstrip('/')}/{checkpoint_dir.name}"
    print(f"[checkpoint_eval] downloading {remote} -> {checkpoint_dir}", flush=True)
    result = subprocess.run(
        [
            "rclone",
            "copy",
            remote,
            str(checkpoint_dir),
            "--transfers=4",
            "--checkers=4",
            "--drive-chunk-size=128M",
            "--log-level",
            "INFO",
        ],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"rclone copy failed for {remote} with exit code {result.returncode}")


def merge_checkpoint(ns: SimpleNamespace, checkpoint_dir: Path, merged_dir: Path) -> Path:
    if (merged_dir / "model.safetensors.index.json").exists() or any(merged_dir.glob("model-*.safetensors")):
        print(f"[checkpoint_eval] using existing merged model: {merged_dir}", flush=True)
        return merged_dir

    from unsloth import FastLanguageModel

    print(f"[checkpoint_eval] merging {checkpoint_dir} -> {merged_dir}", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(checkpoint_dir),
        max_seq_length=ns.MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
        trust_remote_code=True,
    )
    tokenizer = ns.normalize_tokenizer_special_tokens(tokenizer)
    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

    del model
    gc.collect()
    if ns.torch.cuda.is_available():
        ns.torch.cuda.empty_cache()
        ns.torch.cuda.ipc_collect()
    return merged_dir


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    args = parse_args()
    ns = load_eval_namespace()
    if args.max_new_tokens is not None:
        ns.EVAL_MAX_NEW_TOKENS = args.max_new_tokens
        print(f"[checkpoint_eval] EVAL_MAX_NEW_TOKENS={ns.EVAL_MAX_NEW_TOKENS}", flush=True)
    eval_rows = load_public_eval_rows(args) if args.public_only else ns.eval_set
    args.merged_root.mkdir(parents=True, exist_ok=True)

    for step in args.steps:
        stage_name = f"stage1_2_public_ckpt_{step}" if args.public_only else f"stage1_2_ckpt_{step}"
        summary_path = ns.EVAL_DIR / f"{stage_name}_eval_summary.json"
        if args.skip_existing and summary_path.exists():
            print(f"[checkpoint_eval] skipping existing eval summary: {summary_path}", flush=True)
            continue

        checkpoint_dir = args.checkpoint_root / f"checkpoint-{step}"
        ensure_local_checkpoint(checkpoint_dir, args.drive_source)
        if not (checkpoint_dir / "adapter_model.safetensors").exists():
            raise FileNotFoundError(f"Checkpoint is missing adapter_model.safetensors: {checkpoint_dir}")

        merged_dir = args.merged_root / f"merged_checkpoint-{step}"
        try:
            merge_checkpoint(ns, checkpoint_dir, merged_dir)
            print(f"[checkpoint_eval] evaluating {stage_name}", flush=True)
            metrics = ns.evaluate_model(
                str(merged_dir),
                stage_name,
                eval_rows,
                limit=args.limit,
                batch_size=args.batch_size,
            )
            print(f"[checkpoint_eval] {stage_name} metrics: {metrics}", flush=True)
        finally:
            if not args.keep_merged and merged_dir.exists():
                print(f"[checkpoint_eval] removing merged temp model: {merged_dir}", flush=True)
                shutil.rmtree(merged_dir)
                gc.collect()


if __name__ == "__main__":
    main()
