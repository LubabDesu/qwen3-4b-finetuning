#!/usr/bin/env python3
"""Generate parallel vLLM reasoning paths for competition private JSONL rows."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "checkpoints/merged_unified_ckpt500_waitle5_boxed"
DEFAULT_INPUT = REPO_ROOT / "competition-data/private.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/private_reasoning_paths/unified500_waitle5_n16.jsonl"

SYSTEM_PROMPT = """You are an expert mathematician. Reason step by step, then put your final answer within \boxed{}.

Rules:
- For MCQ: put only the option letter in \boxed{} (e.g. \boxed{F})
- For multi-part problems with multiple [ANS] blanks: put all answers comma-separated in ONE box in blank order (e.g. \boxed{2, -6, 120})
- Never put multiple \boxed{} -- always exactly one at the very end
- Never round unless the problem explicitly asks you to round
- For decimals, keep full precision
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use vLLM offline inference to generate N diverse reasoning paths per "
            "private competition prompt for later majority voting."
        )
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="HF model ID or local merged model path.")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional tokenizer path. Defaults to --model.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input .jsonl path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output .jsonl path.")
    parser.add_argument("--batch-size", type=int, default=16, help="Number of prompts per vLLM generate call.")
    parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        default=16,
        help="Parallel samples per prompt via SamplingParams.n. Use 8 if throughput or KV cache pressure is high.",
    )
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature.")
    parser.add_argument("--min-p", type=float, default=0.05, help="vLLM min_p token filtering.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Optional nucleus filtering.")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help=(
            "Maximum generated tokens per path. 4096 is the default sweet spot for "
            "A100 throughput; try 5120 or 8192 only if boxed answers are still truncated."
        ),
    )
    parser.add_argument("--max-model-len", type=int, default=8192, help="vLLM max model length.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="vLLM GPU memory target.")
    parser.add_argument("--dtype", default="bfloat16", help="vLLM dtype, e.g. bfloat16, float16, auto.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Use 1 for a single A100.")
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=256,
        help="vLLM scheduler concurrency cap. Should usually be at least batch_size * num_samples.",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Base sampling seed.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N valid rows.")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip question_ids already present in --output.",
    )
    parser.add_argument(
        "--store-rendered-prompt",
        action="store_true",
        help="Store the full chat-template prompt instead of the raw question/options prompt.",
    )
    parser.add_argument(
        "--skip-existing-output-check",
        action="store_true",
        help="Allow writing to an existing output file when --no-resume is used.",
    )
    parser.add_argument(
        "--system-prompt-file",
        type=Path,
        default=None,
        help="Optional text file overriding the built-in system prompt.",
    )
    return parser.parse_args()


def load_completed_ids(path: Path) -> set[Any]:
    completed: set[Any] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"[paths] warning: ignoring malformed existing output line {line_no}", file=sys.stderr)
                continue
            if "question_id" in row:
                completed.add(row["question_id"])
    return completed


def count_nonempty_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def user_prompt_from_row(row: dict[str, Any]) -> str:
    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("missing non-empty string field 'question'")

    options = row.get("options")
    if isinstance(options, list) and options:
        option_lines = ["\nOptions:"]
        for idx, option in enumerate(options):
            label = chr(ord("A") + idx)
            option_lines.append(f"{label}. {option}")
        return question.rstrip() + "\n" + "\n".join(option_lines)
    return question.rstrip()


def question_id_from_row(row: dict[str, Any], fallback: int) -> Any:
    for key in ("id", "question_id"):
        if key in row:
            return row[key]
    return fallback


def render_prompt(tokenizer: Any, user_prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"{system_prompt.strip()}\n\nUser:\n{user_prompt}\n\nAssistant:\n"


def iter_input_batches(
    path: Path,
    tokenizer: Any,
    batch_size: int,
    completed_ids: set[Any],
    limit: int | None,
    store_rendered_prompt: bool,
    system_prompt: str,
) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    valid_seen = 0
    skipped_done = 0
    malformed = 0

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("row is not a JSON object")
                question_id = question_id_from_row(row, line_no - 1)
                if question_id in completed_ids:
                    skipped_done += 1
                    continue
                user_prompt = user_prompt_from_row(row)
            except (json.JSONDecodeError, ValueError) as exc:
                malformed += 1
                print(f"[paths] warning: skipping input line {line_no}: {exc}", file=sys.stderr)
                continue

            rendered_prompt = render_prompt(tokenizer, user_prompt, system_prompt)
            batch.append(
                {
                    "question_id": question_id,
                    "prompt": rendered_prompt if store_rendered_prompt else user_prompt,
                    "rendered_prompt": rendered_prompt,
                }
            )
            valid_seen += 1
            if limit is not None and valid_seen >= limit:
                break
            if len(batch) >= batch_size:
                yield batch
                batch = []

    if batch:
        yield batch
    if skipped_done or malformed:
        print(f"[paths] skipped_completed={skipped_done} malformed={malformed}", flush=True)


def init_vllm(args: argparse.Namespace) -> tuple[Any, Any]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        from transformers import AutoTokenizer
        from vllm import LLM
    except ImportError as exc:
        raise SystemExit(
            "Missing inference dependency. Activate the project venv and ensure transformers/vllm are installed."
        ) from exc

    model_path = Path(args.model)
    model = str(model_path.resolve()) if model_path.exists() else args.model
    tokenizer_path = args.tokenizer or model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=model,
        tokenizer=tokenizer_path,
        trust_remote_code=True,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=False,
        generation_config="vllm",
    )
    return llm, tokenizer


def build_sampling_params(args: argparse.Namespace) -> Any:
    from vllm import SamplingParams

    return SamplingParams(
        n=args.num_samples,
        temperature=args.temperature,
        min_p=args.min_p,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        stop=["<|im_end|>", "<|endoftext|>"],
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    args = parse_args()
    if args.num_samples <= 0:
        raise SystemExit("--num-samples must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.max_num_seqs < args.batch_size:
        print(
            "[paths] warning: --max-num-seqs is lower than --batch-size; throughput may be limited.",
            file=sys.stderr,
        )
    if args.output.exists() and not args.resume and not args.skip_existing_output_check:
        raise SystemExit(
            f"Refusing to append to existing output with --no-resume: {args.output}. "
            "Use --skip-existing-output-check if this is intentional."
        )

    started = time.time()
    system_prompt = SYSTEM_PROMPT
    if args.system_prompt_file is not None:
        system_prompt = args.system_prompt_file.read_text(encoding="utf-8").strip()
    completed_ids = load_completed_ids(args.output) if args.resume else set()
    total_lines = count_nonempty_lines(args.input)
    if args.limit is not None:
        total_lines = min(total_lines, args.limit + len(completed_ids))

    print(f"[paths] model={args.model}", flush=True)
    print(f"[paths] input={args.input} output={args.output}", flush=True)
    print(
        f"[paths] n={args.num_samples} temp={args.temperature} min_p={args.min_p} "
        f"max_tokens={args.max_tokens} batch_size={args.batch_size}",
        flush=True,
    )
    if completed_ids:
        print(f"[paths] resume: found {len(completed_ids)} completed question_ids", flush=True)

    llm, tokenizer = init_vllm(args)
    sampling = build_sampling_params(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    generated_paths = 0
    progress_total = None if args.limit is None else args.limit
    if args.limit is None and not completed_ids:
        progress_total = total_lines

    with args.output.open("a", encoding="utf-8") as out_f, tqdm(total=progress_total, desc="questions") as pbar:
        for batch in iter_input_batches(
            args.input,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            completed_ids=completed_ids,
            limit=args.limit,
            store_rendered_prompt=args.store_rendered_prompt,
            system_prompt=system_prompt,
        ):
            rendered_prompts = [item["rendered_prompt"] for item in batch]
            outputs = llm.generate(rendered_prompts, sampling)
            for item, output in zip(batch, outputs):
                generations = [candidate.text.strip() for candidate in output.outputs]
                out_row = {
                    "question_id": item["question_id"],
                    "prompt": item["prompt"],
                    "generations": generations,
                }
                out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                generated_paths += len(generations)
            out_f.flush()
            processed += len(batch)
            pbar.update(len(batch))
            elapsed = time.time() - started
            pbar.set_postfix(paths=generated_paths, q_per_min=f"{processed / max(elapsed, 1e-9) * 60:.1f}")

    elapsed = time.time() - started
    print(
        f"[paths] done: processed={processed} generated_paths={generated_paths} "
        f"elapsed={elapsed / 60:.2f} min output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
