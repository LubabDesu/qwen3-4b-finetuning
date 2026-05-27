#!/usr/bin/env python3
"""Train a rejection-sampling SFT adapter from reviewed RS-SFT rows."""

from __future__ import annotations

import argparse
import gc
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import train_stage1_2 as stage_trainer  # noqa: E402


DEFAULT_DATASET_PATH = (
    REPO_ROOT
    / "artifacts/grpo_rs_sft/sft/ckpt500_combined_review_cleaned_plus_mcq_longcap_plus_recovered16_waitle20.jsonl"
)
DEFAULT_BASE_MODEL = REPO_ROOT / "checkpoints/merged_grpo_ramp1p2_unified_ckpt500"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RS-SFT from rejection-sampled accepted completions.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="RS-SFT JSONL with question/target rows.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=str(DEFAULT_BASE_MODEL),
        help="Merged model/checkpoint to start from.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="rs_sft_ckpt500",
        help="Output suffix for checkpoints/lora_*, checkpoints/merged_*, and checkpoints/trainer_*.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--drive-target", type=str, default=os.environ.get("RS_SFT_DRIVE_TARGET"))
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--dry-run-setup", action="store_true")
    parser.add_argument("--lazy-tokenize", action="store_true")
    parser.add_argument("--no-merge", action="store_true")
    return parser.parse_args()


def load_rs_sft_namespace(args: argparse.Namespace) -> SimpleNamespace:
    os.chdir(REPO_ROOT)
    cells = stage_trainer.load_notebook_cells()
    module = ModuleType("__rs_sft_train__")
    module.__file__ = str(stage_trainer.NOTEBOOK_PATH)
    sys.modules[module.__name__] = module
    namespace: dict[str, Any] = module.__dict__

    for cell_id in stage_trainer.CORE_CELL_IDS:
        print(f"[rs_sft] loading notebook cell {cell_id}", flush=True)
        stage_trainer.exec_cell(cells, cell_id, namespace)

    stage_trainer.apply_dataset_override(namespace, args.dataset_path)
    namespace["STAGE1_2_BASE_MODEL"] = args.base_model
    namespace["STAGE1_2"] = replace(
        namespace["STAGE1_2"],
        name=args.run_name,
        lr=args.learning_rate,
        epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
    )

    if not args.skip_preflight:
        print("[rs_sft] running RS-SFT preflight", flush=True)
        stage_trainer.exec_cell(cells, stage_trainer.PREFLIGHT_CELL_ID, namespace)

    return SimpleNamespace(**namespace)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    args = parse_args()
    ns = load_rs_sft_namespace(args)
    if args.dry_run_setup:
        print(f"[rs_sft] dry run OK; loaded {len(ns.stage1_2_records)} RS-SFT records", flush=True)
        print(f"[rs_sft] base model: {ns.STAGE1_2_BASE_MODEL}", flush=True)
        print(f"[rs_sft] run config: {ns.STAGE1_2}", flush=True)
        return

    print("[rs_sft] starting training", flush=True)
    lora_dir, merged_dir = stage_trainer.train_stage_with_checkpoints(ns, args)
    print(f"[rs_sft] LoRA adapter: {lora_dir}", flush=True)
    print(f"[rs_sft] merged checkpoint: {merged_dir}", flush=True)

    gc.collect()
    if ns.torch.cuda.is_available():
        ns.torch.cuda.empty_cache()
    print("[rs_sft] done", flush=True)


if __name__ == "__main__":
    main()
