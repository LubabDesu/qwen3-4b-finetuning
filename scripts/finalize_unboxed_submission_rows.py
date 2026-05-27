#!/usr/bin/env python3
"""Finalize still-unboxed rows from a draft submission CSV."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
BOX_RE = re.compile(r"\\boxed\s*\{")
SYSTEM_PROMPT = """You are an answer extraction assistant. Extract the final answer from the provided reasoning. Output exactly one boxed answer and nothing else."""
TEMPLATE = """Question:
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
    parser.add_argument("--model", default=str(REPO_ROOT / "checkpoints/merged_rs_sft_public526_110_spiral2_review10_15"))
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--questions", type=Path, default=REPO_ROOT / "competition-data/private.jsonl")
    parser.add_argument("--submission", type=Path, default=REPO_ROOT / "artifacts/private_reasoning_paths/submission_draft_3_finalized.csv")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "artifacts/private_reasoning_paths/draft_3_submission_unboxed_finalizer_prefill_outputs.jsonl")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--reasoning-tail-chars", type=int, default=9000)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--store-prompt", action="store_true")
    return parser.parse_args()


def read_questions(path: Path) -> dict[int, str]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for fallback, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            qid = int(row.get("id", row.get("question_id", fallback)))
            question = row.get("question", "").rstrip()
            options = row.get("options")
            if isinstance(options, list) and options:
                lines = ["\nOptions:"]
                for idx, option in enumerate(options):
                    lines.append(f"{chr(ord('A') + idx)}. {option}")
                question += "\n" + "\n".join(lines)
            out[qid] = question
    return out


def read_completed(path: Path) -> set[int]:
    done = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                done.add(int(row["question_id"]))
            except Exception:
                continue
    return done


def collect_items(args: argparse.Namespace, questions: dict[int, str]) -> list[dict[str, Any]]:
    completed = read_completed(args.output) if args.resume else set()
    items = []
    with args.submission.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = int(row["id"])
            response = row["response"] or ""
            if qid in completed or BOX_RE.search(response):
                continue
            question = questions.get(qid)
            if question is None:
                print(f"[submission_finalizer] warning: missing question for id={qid}", file=sys.stderr, flush=True)
                continue
            reasoning = response[-args.reasoning_tail_chars:]
            prompt = TEMPLATE.format(question=question, reasoning=reasoning)
            items.append({"question_id": qid, "prompt": prompt, "source_reasoning_chars": len(response)})
    return items


def render_prompt(tokenizer: Any, user_prompt: str) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]
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
    return SamplingParams(n=1, temperature=args.temperature, top_p=args.top_p, min_p=args.min_p, max_tokens=args.max_tokens, stop=["<|im_end|>", "<|endoftext|>"])


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    questions = read_questions(args.questions)
    items = collect_items(args, questions)
    print(f"[submission_finalizer] pending_unboxed_items={len(items)} output={args.output}", flush=True)
    if not items:
        return
    llm, tokenizer = init_vllm(args)
    sampling = build_sampling(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as out_f, tqdm(total=len(items), desc="submission_finalizer") as pbar:
        for i in range(0, len(items), args.batch_size):
            batch = items[i:i + args.batch_size]
            prompts = [render_prompt(tokenizer, item["prompt"]) + "\n\\boxed{" for item in batch]
            outputs = llm.generate(prompts, sampling)
            for item, output in zip(batch, outputs):
                text = output.outputs[0].text.strip() if output.outputs else ""
                final = "\\boxed{" + text
                if not final.rstrip().endswith("}"):
                    final = final.rstrip() + "}"
                row = {
                    "question_id": item["question_id"],
                    "source_reasoning_chars": item["source_reasoning_chars"],
                    "finalizer_generations": [final],
                }
                if args.store_prompt:
                    row["prompt"] = item["prompt"]
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()
            pbar.update(len(batch))
    print(f"[submission_finalizer] done output={args.output}", flush=True)


if __name__ == "__main__":
    main()
