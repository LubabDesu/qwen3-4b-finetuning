#!/usr/bin/env python3
"""Merge and evaluate GRPO LoRA checkpoints on public eval samples."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
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
    parser = argparse.ArgumentParser(description="Evaluate saved GRPO checkpoints.")
    parser.add_argument(
        "--steps",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Checkpoint step numbers to evaluate. Defaults to all checkpoint-N "
            "directories found under --checkpoint-root."
        ),
    )
    parser.add_argument(
        "--eval-base-model",
        action="store_true",
        help="Evaluate the base model directly instead of merging/evaluating GRPO checkpoints.",
    )
    parser.add_argument(
        "--include-base-model",
        action="store_true",
        help="Also evaluate the base model in the same run as checkpoint evals.",
    )
    parser.add_argument(
        "--base-position",
        choices=["first", "last"],
        default="last",
        help="When --include-base-model is set, evaluate the base model before or after checkpoints.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model name/path for --eval-base-model. Defaults to the notebook BASE_MODEL.",
    )
    parser.add_argument(
        "--base-stage-name",
        default=None,
        help="Stage name prefix for --eval-base-model outputs. Defaults to grpo_base_public or grpo_base_eval.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("checkpoints/grpo"),
        help="Local directory containing checkpoint-N folders.",
    )
    parser.add_argument(
        "--drive-source",
        default=None,
        help=(
            "Optional rclone source containing checkpoint-N folders, for example "
            "'gdrive:151B/grpo' or "
            "'gdrive:151B_SP26_Competition/checkpoints/grpo'."
        ),
    )
    parser.add_argument(
        "--drive-results-target",
        default=None,
        help=(
            "Optional rclone destination for eval result files, for example "
            "'gdrive:151B_SP26_Competition/eval/grpo_public'."
        ),
    )
    parser.add_argument(
        "--drive-sync-summaries-only",
        action="store_true",
        help="When syncing eval outputs to Drive, upload only summary JSON files.",
    )
    parser.add_argument(
        "--ignore-drive-sync-errors",
        action="store_true",
        help="Log rclone sync failures without failing the eval run.",
    )
    parser.add_argument("--rclone-transfers", type=int, default=4, help="rclone --transfers value for Drive copies.")
    parser.add_argument("--rclone-checkers", type=int, default=4, help="rclone --checkers value for Drive copies.")
    parser.add_argument("--rclone-drive-chunk-size", default="128M", help="rclone --drive-chunk-size value.")
    parser.add_argument("--rclone-tpslimit", type=float, default=None, help="Optional rclone --tpslimit value.")
    parser.add_argument("--rclone-tpslimit-burst", type=int, default=None, help="Optional rclone --tpslimit-burst value.")
    parser.add_argument("--rclone-retries", type=int, default=10, help="rclone --retries value.")
    parser.add_argument("--rclone-low-level-retries", type=int, default=20, help="rclone --low-level-retries value.")
    parser.add_argument(
        "--checkpoint-sleep-seconds",
        type=float,
        default=0.0,
        help="Sleep this many seconds after each checkpoint eval/sync before moving to the next checkpoint.",
    )
    parser.add_argument(
        "--merged-root",
        type=Path,
        default=Path("checkpoints/eval_merged_grpo"),
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
        "--resume-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Resume from existing *_eval_results.jsonl or *_eval_results.partial.jsonl by skipping completed ids. "
            "Use --no-resume-existing to force a fresh eval."
        ),
    )
    parser.add_argument(
        "--public-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Evaluate on public.jsonl instead of the held-out mixed eval set. "
            "Enabled by default for GRPO checkpoint selection; use --no-public-only "
            "for the notebook held-out eval set."
        ),
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
        default=300,
        help="Deterministically sample N public rows before applying --limit.",
    )
    parser.add_argument("--public-seed", type=int, default=42, help="Seed for deterministic public sampling.")
    parser.add_argument(
        "--public-mcq-n",
        type=int,
        default=None,
        help="Sample exactly this many MCQ (options-bearing) rows from public eval.",
    )
    parser.add_argument(
        "--public-non-mcq-n",
        type=int,
        default=None,
        help="Sample exactly this many non-MCQ rows from public eval.",
    )
    return parser.parse_args()


def load_notebook_cells() -> list[dict[str, Any]]:
    with NOTEBOOK_PATH.open() as f:
        return json.load(f)["cells"]


def exec_cell(cells: list[dict[str, Any]], cell_id: int, namespace: dict[str, Any]) -> None:
    source = "".join(cells[cell_id].get("source", []))
    exec(compile(source, f"{NOTEBOOK_PATH.name}#cell-{cell_id}", "exec"), namespace)


def load_eval_namespace() -> SimpleNamespace:
    os.chdir(REPO_ROOT)
    module_name = "__grpo_checkpoint_eval__"
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


def eval_row_id(row: dict[str, Any], fallback_index: int | None = None) -> str:
    row_id = row.get("id")
    if row_id is not None and str(row_id).strip():
        return str(row_id)
    if fallback_index is None:
        raise ValueError(f"Eval/result row is missing id and no fallback index was provided: {row}")
    return f"__row_index_{fallback_index}"


def dedupe_eval_rows(eval_rows: list[dict[str, Any]], stage_name: str) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique_rows: list[dict[str, Any]] = []
    duplicate_count = 0
    for index, row in enumerate(eval_rows):
        row_id = eval_row_id(row, index)
        if row_id in seen:
            duplicate_count += 1
            continue
        seen.add(row_id)
        unique_rows.append(row)
    if duplicate_count:
        print(
            f"[checkpoint_eval] {stage_name}: skipped {duplicate_count} duplicate eval input ids",
            flush=True,
        )
    return unique_rows


def load_existing_result_rows(eval_dir: Path, stage_name: str) -> list[dict[str, Any]]:
    paths = [
        eval_dir / f"{stage_name}_eval_results.jsonl",
        eval_dir / f"{stage_name}_eval_results.partial.jsonl",
    ]
    rows_by_id: dict[str, dict[str, Any]] = {}
    duplicate_count = 0
    for path in paths:
        if not path.exists():
            continue
        for row in load_jsonl(path):
            try:
                row_id = eval_row_id(row)
            except ValueError:
                duplicate_count += 1
                continue
            if row_id in rows_by_id:
                duplicate_count += 1
                continue
            rows_by_id[row_id] = row
    if duplicate_count:
        print(
            f"[checkpoint_eval] {stage_name}: ignored {duplicate_count} duplicate/malformed existing result rows",
            flush=True,
        )
    return list(rows_by_id.values())


def build_eval_metrics(rows: list[dict[str, Any]], partial: bool = False) -> dict[str, Any]:
    n = len(rows)
    correct = sum(int(bool(row.get("correct"))) for row in rows)
    compliance = sum(int(bool(row.get("format_ok"))) for row in rows)
    total_words = sum(int(row.get("word_count") or 0) for row in rows)
    mcq_rows = [row for row in rows if row.get("is_mcq")]
    multi_rows = [row for row in rows if row.get("is_multi")]
    metrics: dict[str, Any] = {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "correct": correct,
        "mcq_accuracy": (
            sum(int(bool(row.get("correct"))) for row in mcq_rows) / len(mcq_rows) if mcq_rows else None
        ),
        "mcq_total": len(mcq_rows),
        "multi_answer_accuracy": (
            sum(int(bool(row.get("correct"))) for row in multi_rows) / len(multi_rows) if multi_rows else None
        ),
        "multi_answer_total": len(multi_rows),
        "avg_response_words": total_words / n if n else 0.0,
        "boxed_compliance_rate": compliance / n if n else 0.0,
        "inference_backend": "vllm",
    }
    if partial:
        metrics["partial"] = True
    return metrics


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def install_resumable_evaluate_model(ns: SimpleNamespace, args: argparse.Namespace) -> None:
    if not args.resume_existing:
        return

    def evaluate_model_resumable(
        model_name_or_path: str,
        stage_name: str,
        eval_rows: list[dict[str, Any]] | None = None,
        limit: int | None = None,
        batch_size: int = 20,
        report_every_batches: int = 1,
        report_examples: int = 2,
    ) -> dict[str, Any]:
        if eval_rows is None:
            eval_path = ns.EVAL_DIR / "heldout_eval_set.jsonl"
            eval_rows = ns.load_jsonl(eval_path) if eval_path.exists() else ns.build_eval_sets()
        if limit:
            eval_rows = eval_rows[:limit]
        eval_rows = dedupe_eval_rows(eval_rows, stage_name)

        existing_rows = load_existing_result_rows(ns.EVAL_DIR, stage_name)
        eval_ids = {eval_row_id(row, index) for index, row in enumerate(eval_rows)}
        existing_by_id = {
            eval_row_id(row): row
            for row in existing_rows
            if eval_row_id(row) in eval_ids
        }
        pending_rows = [
            row
            for index, row in enumerate(eval_rows)
            if eval_row_id(row, index) not in existing_by_id
        ]

        if existing_by_id:
            print(
                f"[checkpoint_eval] {stage_name}: resuming with {len(existing_by_id)} completed ids; "
                f"{len(pending_rows)} remaining",
                flush=True,
            )

        if not pending_rows:
            rows = [existing_by_id[eval_row_id(row, index)] for index, row in enumerate(eval_rows)]
            metrics = build_eval_metrics(rows)
            out_path = ns.EVAL_DIR / f"{stage_name}_eval_results.jsonl"
            summary_path = ns.EVAL_DIR / f"{stage_name}_eval_summary.json"
            write_jsonl(rows, out_path)
            summary_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
            ns.append_results_log(stage_name, metrics)
            print(f"[checkpoint_eval] {stage_name}: already complete after resume/dedupe", flush=True)
            print(json.dumps(metrics, indent=2))
            return metrics

        llm, tokenizer = ns.load_vllm_engine(model_name_or_path)
        generated_by_id: dict[str, dict[str, Any]] = {}

        try:
            for start in ns.tqdm(range(0, len(pending_rows), batch_size), desc=f"Eval {stage_name}"):
                batch = pending_rows[start : start + batch_size]
                responses = ns.generate_responses_vllm(
                    llm,
                    tokenizer,
                    batch,
                    max_new_tokens=ns.EVAL_MAX_NEW_TOKENS,
                    temperature=ns.EVAL_VLLM_TEMPERATURE,
                )
                batch_result_rows = []

                for item_index, (item, response) in enumerate(zip(batch, responses), start=start):
                    is_mcq = bool(item.get("options"))
                    is_multi = (not is_mcq) and ns.count_ans_blanks(item.get("question", "")) > 1
                    if item.get("eval_source") == "public":
                        ok = ns.score_public_item(item, response)
                    else:
                        ok = ns.score_math_item(item, response)
                    fmt_ok = ns.boxed_format_ok(
                        item.get("question", ""),
                        response,
                        is_mcq=is_mcq,
                        options=item.get("options"),
                    )
                    words = ns.response_word_count(response)
                    row = {
                        "id": item.get("id"),
                        "eval_source": item.get("eval_source"),
                        "is_mcq": is_mcq,
                        "is_multi": is_multi,
                        "question": item.get("question"),
                        "options": item.get("options"),
                        "gold": item.get("answer"),
                        "response": response,
                        "correct": ok,
                        "format_ok": fmt_ok,
                        "word_count": words,
                    }
                    generated_by_id[eval_row_id(item, item_index)] = row
                    batch_result_rows.append(row)

                combined_by_id = {**existing_by_id, **generated_by_id}
                combined_rows = [
                    combined_by_id[eval_row_id(row, index)]
                    for index, row in enumerate(eval_rows)
                    if eval_row_id(row, index) in combined_by_id
                ]
                partial_metrics = build_eval_metrics(combined_rows, partial=True)
                partial_out_path = ns.EVAL_DIR / f"{stage_name}_eval_results.partial.jsonl"
                partial_summary_path = ns.EVAL_DIR / f"{stage_name}_eval_summary.partial.json"
                write_jsonl(combined_rows, partial_out_path)
                partial_summary_path.write_text(json.dumps(partial_metrics, indent=2, ensure_ascii=False))

                batch_index = start // batch_size + 1
                if report_every_batches and batch_index % report_every_batches == 0:
                    ns.print_eval_batch_report(stage_name, combined_rows, batch_result_rows, max_examples=report_examples)
                    print(f"  partial results: {partial_out_path}")
        finally:
            ns.cleanup_vllm(llm)

        final_by_id = {**existing_by_id, **generated_by_id}
        rows = [final_by_id[eval_row_id(row, index)] for index, row in enumerate(eval_rows)]
        metrics = build_eval_metrics(rows)
        out_path = ns.EVAL_DIR / f"{stage_name}_eval_results.jsonl"
        summary_path = ns.EVAL_DIR / f"{stage_name}_eval_summary.json"
        write_jsonl(rows, out_path)
        summary_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
        ns.append_results_log(stage_name, metrics)

        print(json.dumps(metrics, indent=2))
        return metrics

    ns.evaluate_model = evaluate_model_resumable
    print("[checkpoint_eval] resume mode enabled: existing result ids will be skipped", flush=True)


def load_public_eval_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    import random

    rows = load_jsonl(args.public_path)
    rows = [{"eval_source": "public", **row} for row in rows]

    if args.public_mcq_n is not None or args.public_non_mcq_n is not None:
        rng = random.Random(args.public_seed)
        mcq = [r for r in rows if r.get("options")]
        non_mcq = [r for r in rows if not r.get("options")]
        rng.shuffle(mcq)
        rng.shuffle(non_mcq)
        sampled = []
        if args.public_mcq_n is not None:
            sampled += mcq[: args.public_mcq_n]
        if args.public_non_mcq_n is not None:
            sampled += non_mcq[: args.public_non_mcq_n]
        rng.shuffle(sampled)
        rows = sampled
    elif args.public_sample_size is not None and args.public_sample_size < len(rows):
        rng = random.Random(args.public_seed)
        rows = rng.sample(rows, args.public_sample_size)

    if args.limit:
        rows = rows[: args.limit]
    mcq_count = sum(1 for r in rows if r.get("options"))
    print(
        "[checkpoint_eval] public eval rows "
        f"path={args.public_path} n={len(rows)} mcq={mcq_count} non_mcq={len(rows)-mcq_count} "
        f"sample_size={args.public_sample_size} seed={args.public_seed}",
        flush=True,
    )
    return rows


def rclone_common_args(args: argparse.Namespace) -> list[str]:
    cmd = [
        f"--transfers={args.rclone_transfers}",
        f"--checkers={args.rclone_checkers}",
        f"--drive-chunk-size={args.rclone_drive_chunk_size}",
        f"--retries={args.rclone_retries}",
        f"--low-level-retries={args.rclone_low_level_retries}",
        "--log-level",
        "INFO",
    ]
    if args.rclone_tpslimit is not None:
        cmd.append(f"--tpslimit={args.rclone_tpslimit}")
    if args.rclone_tpslimit_burst is not None:
        cmd.append(f"--tpslimit-burst={args.rclone_tpslimit_burst}")
    return cmd


def ensure_local_checkpoint(checkpoint_dir: Path, drive_source: str | None, args: argparse.Namespace) -> None:
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
            *rclone_common_args(args),
        ],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"rclone copy failed for {remote} with exit code {result.returncode}")
    if not adapter_path.exists():
        raise FileNotFoundError(f"Downloaded checkpoint is missing adapter_model.safetensors: {checkpoint_dir}")


def discover_checkpoint_steps(checkpoint_root: Path) -> list[int]:
    if not checkpoint_root.exists():
        return []

    steps: list[int] = []
    for path in checkpoint_root.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            steps.append(int(path.name.removeprefix("checkpoint-")))
        except ValueError:
            continue
    return sorted(steps)


def sync_eval_outputs(eval_dir: Path, stage_name: str, drive_results_target: str | None, args: argparse.Namespace) -> None:
    if not drive_results_target:
        return

    paths = sorted(eval_dir.glob(f"{stage_name}_eval_*"))
    if args.drive_sync_summaries_only:
        paths = [path for path in paths if "_summary" in path.name and path.suffix == ".json"]
    else:
        paths = sorted(
            paths,
            key=lambda path: (
                0 if "_summary" in path.name else 1,
                0 if ".partial" not in path.name else 1,
                path.name,
            ),
        )
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
                *rclone_common_args(args),
            ],
            check=False,
        )
        if result.returncode != 0:
            if args.ignore_drive_sync_errors:
                print(
                    f"[checkpoint_eval] warning: rclone copyto failed for {path} with exit code {result.returncode}",
                    flush=True,
                )
                continue
            raise RuntimeError(f"rclone copyto failed for {path} with exit code {result.returncode}")


def merge_checkpoint(ns: SimpleNamespace, checkpoint_dir: Path, merged_dir: Path) -> Path:
    has_model_weights = (merged_dir / "model.safetensors.index.json").exists() or any(
        merged_dir.glob("model-*.safetensors")
    )
    has_tokenizer = (merged_dir / "tokenizer.json").exists() and (merged_dir / "tokenizer_config.json").exists()
    has_config = (merged_dir / "config.json").exists()
    if has_model_weights and has_tokenizer and has_config:
        print(f"[checkpoint_eval] using existing merged model: {merged_dir.resolve()}", flush=True)
        return merged_dir
    if merged_dir.exists():
        print(f"[checkpoint_eval] removing incomplete merged model before retry: {merged_dir}", flush=True)
        shutil.rmtree(merged_dir)

    from unsloth import FastLanguageModel

    print(f"[checkpoint_eval] merging {checkpoint_dir} -> {merged_dir.resolve()}", flush=True)
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

        model_path_obj = Path(model_name_or_path)
        model_path = str(model_path_obj.resolve()) if model_path_obj.exists() else str(model_name_or_path)
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


def evaluate_base_model(ns: SimpleNamespace, args: argparse.Namespace, eval_rows: list[dict[str, Any]]) -> None:
    model_name_or_path = args.base_model or ns.BASE_MODEL
    stage_name = args.base_stage_name or ("grpo_base_public" if args.public_only else "grpo_base_eval")
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
        sync_eval_outputs(ns.EVAL_DIR, stage_name, args.drive_results_target, args)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    args = parse_args()
    ns = load_eval_namespace()
    apply_vllm_overrides(ns, args)
    install_resumable_evaluate_model(ns, args)
    eval_rows = load_public_eval_rows(args) if args.public_only else ns.eval_set
    args.merged_root.mkdir(parents=True, exist_ok=True)

    if args.eval_base_model:
        evaluate_base_model(ns, args, eval_rows)
        return

    if args.include_base_model and args.base_position == "first":
        evaluate_base_model(ns, args, eval_rows)

    steps = args.steps if args.steps else discover_checkpoint_steps(args.checkpoint_root)
    if not steps:
        raise FileNotFoundError(
            f"No checkpoint-N directories found under {args.checkpoint_root}. "
            "Pass --steps with --drive-source to download specific checkpoints."
        )
    print(f"[checkpoint_eval] GRPO checkpoint steps: {steps}", flush=True)

    for step in steps:
        stage_name = f"grpo_public_ckpt_{step}" if args.public_only else f"grpo_ckpt_{step}"
        summary_path = ns.EVAL_DIR / f"{stage_name}_eval_summary.json"
        if args.skip_existing and summary_path.exists():
            print(f"[checkpoint_eval] skipping existing eval summary: {summary_path}", flush=True)
            continue

        checkpoint_dir = args.checkpoint_root / f"checkpoint-{step}"
        ensure_local_checkpoint(checkpoint_dir, args.drive_source, args)
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
            sync_eval_outputs(ns.EVAL_DIR, stage_name, args.drive_results_target, args)
            if not args.keep_merged and merged_dir.exists():
                print(f"[checkpoint_eval] removing merged temp model: {merged_dir}", flush=True)
                shutil.rmtree(merged_dir)
                gc.collect()
            if args.checkpoint_sleep_seconds > 0:
                print(
                    f"[checkpoint_eval] sleeping {args.checkpoint_sleep_seconds:g}s before next checkpoint",
                    flush=True,
                )
                time.sleep(args.checkpoint_sleep_seconds)

    if args.include_base_model and args.base_position == "last":
        evaluate_base_model(ns, args, eval_rows)


if __name__ == "__main__":
    main()
