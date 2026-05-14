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
        "--eval-base-model",
        action="store_true",
        help="Evaluate the base model directly instead of merging/evaluating Trainer checkpoints.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model name/path for --eval-base-model. Defaults to the notebook BASE_MODEL.",
    )
    parser.add_argument(
        "--base-stage-name",
        default=None,
        help="Stage name prefix for --eval-base-model outputs. Defaults to base_public or base_eval.",
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
        "--drive-results-target",
        default=None,
        help=(
            "Optional rclone destination for eval result files, for example "
            "'gdrive:151B_SP26_Competition/eval/stage1_2_public'."
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
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Override vLLM max_model_len. Needed when --max-new-tokens is close to or above 8192.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Override vLLM gpu_memory_utilization.",
    )
    parser.add_argument(
        "--vllm-dtype",
        default=None,
        help="Override vLLM dtype, for example bfloat16.",
    )
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override vLLM enforce_eager. Use --no-enforce-eager to enable CUDA graphs.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Override vLLM max_num_seqs.",
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
    # Keep __file__ pointed at this real Python script. vLLM uses multiprocessing
    # spawn, and spawn re-executes __main__.__file__; pointing at the .ipynb JSON
    # makes the worker try to run notebook JSON as Python.
    module.__file__ = str(Path(__file__).resolve())
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
    adapter_path = checkpoint_dir / "adapter_model.safetensors"
    if adapter_path.exists():
        return
    if not drive_source:
        raise FileNotFoundError(f"Missing local checkpoint and no --drive-source was provided: {checkpoint_dir}")

    if checkpoint_dir.exists():
        print(f"[checkpoint_eval] local checkpoint exists but is incomplete: {checkpoint_dir}", flush=True)
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
    if not adapter_path.exists():
        raise FileNotFoundError(f"Downloaded checkpoint is missing adapter_model.safetensors: {checkpoint_dir}")


def sync_eval_outputs(eval_dir: Path, stage_name: str, drive_results_target: str | None) -> None:
    if not drive_results_target:
        return

    paths = sorted(eval_dir.glob(f"{stage_name}_eval_*"))
    if not paths:
        print(f"[checkpoint_eval] no eval outputs to sync yet for {stage_name}", flush=True)
        return

    remote_root = drive_results_target.rstrip("/")
    for path in paths:
        remote = f"{remote_root}/{path.name}"
        print(f"[checkpoint_eval] syncing {path} -> {remote}", flush=True)
        result = subprocess.run(
            [
                "rclone",
                "copyto",
                str(path),
                remote,
                "--transfers=4",
                "--checkers=4",
                "--drive-chunk-size=128M",
                "--log-level",
                "INFO",
            ],
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"rclone copyto failed for {path} with exit code {result.returncode}")


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


def apply_vllm_overrides(ns: SimpleNamespace, args: argparse.Namespace) -> None:
    if args.max_new_tokens is not None:
        ns.EVAL_MAX_NEW_TOKENS = args.max_new_tokens
        ns.evaluate_model.__globals__["EVAL_MAX_NEW_TOKENS"] = args.max_new_tokens
        print(f"[checkpoint_eval] EVAL_MAX_NEW_TOKENS={ns.EVAL_MAX_NEW_TOKENS}", flush=True)
    if args.max_model_len is not None:
        ns.MAX_SEQ_LENGTH = args.max_model_len
        ns.load_vllm_engine.__globals__["MAX_SEQ_LENGTH"] = args.max_model_len
        print(f"[checkpoint_eval] MAX_SEQ_LENGTH/max_model_len={ns.MAX_SEQ_LENGTH}", flush=True)
    if args.gpu_memory_utilization is not None:
        ns.VLLM_GPU_MEMORY_UTILIZATION = args.gpu_memory_utilization
        ns.load_vllm_engine.__globals__["VLLM_GPU_MEMORY_UTILIZATION"] = args.gpu_memory_utilization
        print(f"[checkpoint_eval] VLLM_GPU_MEMORY_UTILIZATION={ns.VLLM_GPU_MEMORY_UTILIZATION}", flush=True)
    if args.vllm_dtype is not None:
        ns.VLLM_DTYPE = args.vllm_dtype
        ns.load_vllm_engine.__globals__["VLLM_DTYPE"] = args.vllm_dtype
        print(f"[checkpoint_eval] VLLM_DTYPE={ns.VLLM_DTYPE}", flush=True)
    if args.enforce_eager is not None:
        ns.VLLM_ENFORCE_EAGER = args.enforce_eager
        ns.load_vllm_engine.__globals__["VLLM_ENFORCE_EAGER"] = args.enforce_eager
        print(f"[checkpoint_eval] VLLM_ENFORCE_EAGER={ns.VLLM_ENFORCE_EAGER}", flush=True)

    if args.max_num_seqs is None:
        return

    base_load_vllm_engine = ns.load_vllm_engine

    def load_vllm_engine_with_max_num_seqs(model_name_or_path: str):
        import gc as _gc

        if "__main__" in sys.modules:
            sys.modules["__main__"].__file__ = str(Path(__file__).resolve())

        ns.os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        ns.os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        _gc.collect()
        if ns.torch.cuda.is_available():
            ns.torch.cuda.empty_cache()
            ns.torch.cuda.ipc_collect()

        ns.require_vllm()

        from transformers import AutoTokenizer
        from vllm import LLM

        model_path = str(model_name_or_path)
        if Path(model_path).exists():
            ns.fix_tokenizer_regex(model_path)

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = ns.normalize_tokenizer_special_tokens(tokenizer)
        llm = LLM(
            model=model_path,
            tokenizer=model_path,
            trust_remote_code=True,
            dtype=ns.VLLM_DTYPE,
            max_model_len=ns.MAX_SEQ_LENGTH,
            tensor_parallel_size=ns.VLLM_TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=ns.VLLM_GPU_MEMORY_UTILIZATION,
            enforce_eager=ns.VLLM_ENFORCE_EAGER,
            max_num_seqs=args.max_num_seqs,
            generation_config="vllm",
        )
        return llm, tokenizer

    load_vllm_engine_with_max_num_seqs.__globals__.update(base_load_vllm_engine.__globals__)
    ns.load_vllm_engine = load_vllm_engine_with_max_num_seqs
    ns.evaluate_model.__globals__["load_vllm_engine"] = load_vllm_engine_with_max_num_seqs
    print(f"[checkpoint_eval] VLLM max_num_seqs={args.max_num_seqs}", flush=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    args = parse_args()
    ns = load_eval_namespace()
    apply_vllm_overrides(ns, args)
    eval_rows = load_public_eval_rows(args) if args.public_only else ns.eval_set
    args.merged_root.mkdir(parents=True, exist_ok=True)

    if args.eval_base_model:
        model_name_or_path = args.base_model or ns.BASE_MODEL
        stage_name = args.base_stage_name or ("base_public" if args.public_only else "base_eval")
        summary_path = ns.EVAL_DIR / f"{stage_name}_eval_summary.json"
        if args.skip_existing and summary_path.exists():
            print(f"[checkpoint_eval] skipping existing eval summary: {summary_path}", flush=True)
            return

        try:
            print(f"[checkpoint_eval] evaluating base model {model_name_or_path} as {stage_name}", flush=True)
            metrics = ns.evaluate_model(
                model_name_or_path,
                stage_name,
                eval_rows,
                limit=args.limit,
                batch_size=args.batch_size,
            )
            print(f"[checkpoint_eval] {stage_name} metrics: {metrics}", flush=True)
        finally:
            sync_eval_outputs(ns.EVAL_DIR, stage_name, args.drive_results_target)
        return

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
            sync_eval_outputs(ns.EVAL_DIR, stage_name, args.drive_results_target)
            if not args.keep_merged and merged_dir.exists():
                print(f"[checkpoint_eval] removing merged temp model: {merged_dir}", flush=True)
                shutil.rmtree(merged_dir)
                gc.collect()


if __name__ == "__main__":
    main()
