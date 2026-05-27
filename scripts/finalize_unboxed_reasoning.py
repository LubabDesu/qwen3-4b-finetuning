#!/usr/bin/env python3
"""Finalize unboxed private reasoning traces into boxed-only answers for draft_3."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "competition-data/private.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/private_reasoning_paths/draft_3_finalizer_outputs.jsonl"
DEFAULT_MODEL = REPO_ROOT / "checkpoints/merged_rs_sft_public526_110_spiral2_review10_15"
DEFAULT_GENERATION_FILES = [
    REPO_ROOT / "artifacts/private_reasoning_paths/rs_sft_public526_110_n8_6000.jsonl",
    REPO_ROOT / "artifacts/private_reasoning_paths/rs_sft_public526_110_zero_box_recovery_n4_8000.jsonl",
    REPO_ROOT / "artifacts/private_reasoning_paths/rs_sft_public526_110_still_zero_box_inject80_n4_8000.jsonl",
]

BOX_RE = re.compile(r"\\boxed\s*\{")
FINAL_CUE_RE = re.compile(
    r"(final answer|answers? (?:is|are)|the answers? (?:is|are)|so the answers? (?:is|are)|option [A-H]|therefore)",
    re.I,
)
PANIC_RE = re.compile(r"\b(wait|hold on|maybe|actually|rethink|confus|not sure|however)\b", re.I)

SYSTEM_PROMPT = """You are an answer extraction assistant. Extract the final answer from the provided reasoning. Output exactly one boxed answer and nothing else."""

FINALIZER_TEMPLATE = """Question:
{question}

Extract the final answer from the reasoning below.

Rules:
- Output exactly one \\boxed{{}}.
- Do not explain.
- If multiple [ANS] blanks, output all answers comma-separated in one box.
- If MCQ, output only the option letter.

Reasoning:
{reasoning}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="HF model ID or local merged model path.")
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer path. Defaults to --model.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Private questions JSONL.")
    parser.add_argument(
        "--generation-file",
        type=Path,
        action="append",
        default=None,
        help="Generation JSONL to consider. Repeatable. Defaults to draft_3 base/recovery/injection files.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Finalizer output JSONL.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--reasoning-tail-chars", type=int, default=9000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--store-prompt", action="store_true")
    parser.add_argument(
        "--prefill-box",
        action="store_true",
        help="Append \\boxed{ to the assistant prompt and wrap the continuation as a boxed answer.",
    )
    return parser.parse_args()


def read_questions(path: Path) -> dict[Any, str]:
    questions = {}
    with path.open(encoding="utf-8") as f:
        for fallback, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            qid = row.get("id", row.get("question_id", fallback))
            question = row.get("question", "").rstrip()
            options = row.get("options")
            if isinstance(options, list) and options:
                option_lines = ["\nOptions:"]
                for idx, option in enumerate(options):
                    option_lines.append(f"{chr(ord('A') + idx)}. {option}")
                question += "\n" + "\n".join(option_lines)
            questions[qid] = question
    return questions


def read_completed(path: Path) -> set[Any]:
    done = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "question_id" in row:
                done.add(row["question_id"])
    return done


def reasoning_score(text: str, source_rank: int) -> int:
    tail = text[-5000:]
    score = 0
    score += 12 * len(FINAL_CUE_RE.findall(tail))
    score -= 2 * len(PANIC_RE.findall(tail[-1500:]))
    score += max(0, 10 - source_rank)
    if len(text) < 18000:
        score += 4
    return score


def collect_unboxed_items(generation_files: Iterable[Path], questions: dict[Any, str], tail_chars: int) -> list[dict[str, Any]]:
    by_q: dict[Any, list[dict[str, Any]]] = {}
    for source_rank, path in enumerate(generation_files):
        if not path.exists():
            print(f"[finalizer] warning: missing generation file: {path}", file=sys.stderr, flush=True)
            continue
        source_name = path.name
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                qid = row["question_id"]
                by_q.setdefault(qid, [])
                for gen_idx, gen in enumerate(row.get("generations") or []):
                    by_q[qid].append(
                        {
                            "source": source_name,
                            "source_rank": source_rank,
                            "generation_index": gen_idx,
                            "text": gen or "",
                        }
                    )

    items = []
    for qid, gens in sorted(by_q.items()):
        if qid not in questions:
            continue
        if any(BOX_RE.search(g["text"]) for g in gens):
            continue
        best = max(gens, key=lambda g: reasoning_score(g["text"], g["source_rank"]))
        reasoning = best["text"][-tail_chars:]
        user_prompt = FINALIZER_TEMPLATE.format(question=questions[qid], reasoning=reasoning)
        items.append(
            {
                "question_id": qid,
                "question": questions[qid],
                "source": best["source"],
                "source_generation_index": best["generation_index"],
                "source_reasoning_chars": len(best["text"]),
                "prompt": user_prompt,
            }
        )
    return items


def render_prompt(tokenizer: Any, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"{SYSTEM_PROMPT}\n\nUser:\n{user_prompt}\n\nAssistant:\n"


def init_vllm(args: argparse.Namespace) -> tuple[Any, Any]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer
    from vllm import LLM

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


def build_sampling(args: argparse.Namespace) -> Any:
    from vllm import SamplingParams

    return SamplingParams(
        n=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        min_p=args.min_p,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )


def batched(items: list[dict[str, Any]], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    generation_files = args.generation_file or DEFAULT_GENERATION_FILES
    questions = read_questions(args.input)
    completed = read_completed(args.output) if args.resume else set()
    items = collect_unboxed_items(generation_files, questions, args.reasoning_tail_chars)
    if completed:
        items = [item for item in items if item["question_id"] not in completed]
    if args.limit is not None:
        items = items[: args.limit]

    print(f"[finalizer] generation_files={len(generation_files)}", flush=True)
    print(f"[finalizer] pending_unboxed_items={len(items)} output={args.output}", flush=True)
    if completed:
        print(f"[finalizer] resume: skipped {len(completed)} completed question_ids", flush=True)
    if not items:
        return

    llm, tokenizer = init_vllm(args)
    sampling = build_sampling(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("a", encoding="utf-8") as out_f, tqdm(total=len(items), desc="finalizer") as pbar:
        for batch in batched(items, args.batch_size):
            prompts = [render_prompt(tokenizer, item["prompt"]) for item in batch]
            if args.prefill_box:
                prompts = [prompt + "\n\boxed{" for prompt in prompts]
            outputs = llm.generate(prompts, sampling)
            for item, output in zip(batch, outputs):
                finals = []
                for candidate in output.outputs:
                    text = candidate.text.strip()
                    if args.prefill_box:
                        text = "\\boxed{" + text
                        if not text.rstrip().endswith("}"):
                            text = text.rstrip() + "}"
                    finals.append(text)
                out_row = {
                    "question_id": item["question_id"],
                    "source": item["source"],
                    "source_generation_index": item["source_generation_index"],
                    "source_reasoning_chars": item["source_reasoning_chars"],
                    "finalizer_generations": finals,
                }
                if args.store_prompt:
                    out_row["prompt"] = item["prompt"]
                out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            out_f.flush()
            pbar.update(len(batch))

    print(f"[finalizer] done output={args.output}", flush=True)


if __name__ == "__main__":
    main()
