#!/usr/bin/env python3
"""Generate rejection-sampling SFT rows for MCQ prompts with vLLM."""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SYSTEM_PROMPT = """You are an expert mathematician. Solve the problem step by step.

Final answer rules:
- Use exactly one final \\boxed{}.
- For MCQ, put only the option letter in \\boxed{}.
- Never round intermediate calculations.
- Never round your final answer unless the problem explicitly asks you to round."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun RS-SFT generation for MCQ prompts.")
    parser.add_argument("--model", required=True, help="Model path or Hugging Face id.")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path. Defaults to --model.")
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--out-path", type=Path, required=True, help="Accepted SFT JSONL output.")
    parser.add_argument("--candidate-path", type=Path, required=True, help="All-candidate JSONL output.")
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--num-generations", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_user_problem(question: str, options: list[str] | None) -> str:
    question = str(question).strip()
    if not options:
        return question
    option_lines = [f"{chr(65 + idx)}. {str(option).strip()}" for idx, option in enumerate(options)]
    return f"{question}\n\nOptions:\n" + "\n".join(option_lines)


def build_prompt(tokenizer: Any, row: dict[str, Any]) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_problem(row["question"], row.get("options") or [])},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_all_boxed(text: str) -> list[str]:
    boxes: list[str] = []
    marker = r"\boxed"
    start = 0
    while True:
        i = text.find(marker, start)
        if i < 0:
            break
        j = text.find("{", i)
        if j < 0:
            break
        depth = 0
        for k in range(j, len(text)):
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
                if depth == 0:
                    boxes.append(text[j + 1:k].strip())
                    start = k + 1
                    break
        else:
            break
    return boxes


def normalize_mcq_answer(answer: Any) -> str:
    if isinstance(answer, list):
        answer = answer[0] if answer else ""
    answer = str(answer or "").strip().upper()
    match = re.fullmatch(r"\(?\s*([A-J])\s*\)?", answer)
    return match.group(1) if match else ""


def clean_completion(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"<\|im_end\|>\s*$", "", text).strip()
    boxes = extract_all_boxed(text)
    if boxes:
        last = boxes[-1]
        last_box = rf"\boxed{{{last}}}"
        idx = text.rfind(last_box)
        if idx >= 0:
            text = text[: idx + len(last_box)].strip()
    return text


def target_shape_ok(text: str) -> bool:
    return (
        text.count("<think>") == 1
        and text.count("</think>") == 1
        and len(extract_all_boxed(text)) == 1
        and text.rfind("</think>") < text.rfind(r"\boxed")
    )


def make_sft_row(row: dict[str, Any], completion: str, candidate_index: int, tokens: int) -> dict[str, Any]:
    gold = normalize_mcq_answer(row.get("gold_answer"))
    reasoning = completion.split("</think>", 1)[0].replace("<think>", "").strip()
    return {
        "id": f"rs_sft_{row['id']}",
        "question": row["question"],
        "options": row.get("options") or [],
        "answer": gold,
        "target_answer": gold,
        "reasoning": reasoning,
        "target": completion,
        "source": "rs_sft_grpo",
        "original_source": row.get("source", "public_train"),
        "original_id": row.get("id"),
        "is_mcq": True,
        "n_ans": 1,
        "gold_answer": [gold],
        "rs_sft": {
            "question_type": "mcq",
            "selected_candidate_index": candidate_index,
            "tier": "clean",
            "completion_tokens": tokens,
            "cleaned_tokens": tokens,
            "num_generations": None,
        },
        "rendered_tokens": tokens,
    }


def main() -> None:
    args = parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = load_jsonl(args.input_path)
    processed_ids = set()
    existing_sft_rows = 0
    if args.candidate_path.exists():
        for old_row in load_jsonl(args.candidate_path):
            if old_row.get("id"):
                processed_ids.add(old_row["id"])
    if args.out_path.exists():
        existing_sft_rows = len(load_jsonl(args.out_path))
    if processed_ids:
        original_count = len(rows)
        rows = [row for row in rows if row.get("id") not in processed_ids]
        print(
            f"[mcq-rs] resuming: found {len(processed_ids)} processed prompts, "
            f"{len(rows)} / {original_count} remaining",
            flush=True,
        )
    else:
        print(f"[mcq-rs] starting fresh: {len(rows)} prompts", flush=True)

    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    prompts = [build_prompt(tokenizer, row) for row in rows]

    llm = LLM(
        model=args.model,
        tokenizer=tokenizer_path,
        trust_remote_code=True,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=False,
        generation_config="vllm",
    )
    sampling = SamplingParams(
        n=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        stop=["<|im_end|>"],
        seed=args.seed,
    )

    stats: collections.Counter[str] = collections.Counter()
    accepted_total = existing_sft_rows

    for start in range(0, len(rows), args.batch_size):
        end = min(start + args.batch_size, len(rows))
        print(f"[mcq-rs] generating {start + 1}-{end} / {len(rows)}", flush=True)
        outputs = llm.generate(prompts[start:end], sampling)
        batch_candidate_rows: list[dict[str, Any]] = []
        batch_sft_rows: list[dict[str, Any]] = []
        for row, output in zip(rows[start:end], outputs):
            gold = normalize_mcq_answer(row.get("gold_answer"))
            candidates = []
            selected = None
            for cand_idx, cand in enumerate(output.outputs):
                completion = clean_completion(cand.text)
                boxes = extract_all_boxed(completion)
                extracted = normalize_mcq_answer(boxes[-1] if boxes else "")
                shape_ok = target_shape_ok(completion)
                correct = bool(extracted and extracted == gold)
                accepted = bool(correct and shape_ok)
                reason = "" if accepted else ("incorrect" if not correct else "bad_target_shape")
                tokens = len(cand.token_ids or [])
                candidates.append(
                    {
                        "candidate_index": cand_idx,
                        "accepted": accepted,
                        "tier": "clean" if accepted else "",
                        "reject_reason": reason,
                        "correct": correct,
                        "extracted_answer": extracted,
                        "normalized_answer": extracted,
                        "completion_tokens": tokens,
                        "cleaned_tokens": tokens,
                        "under_preferred_cap": tokens <= args.max_new_tokens,
                        "structure": {
                            "has_think_close": "</think>" in completion,
                            "final_box_count": len(boxes),
                            "total_box_count": len(boxes),
                            "has_text_after_final_box": False,
                        },
                        "completion": cand.text,
                        "cleaned_completion": completion,
                    }
                )
                stats[f"reject:{reason or 'accepted'}"] += 1
                if accepted and selected is None:
                    selected = cand_idx
                    sft_row = make_sft_row(row, completion, cand_idx, tokens)
                    sft_row["rs_sft"]["num_generations"] = args.num_generations
                    batch_sft_rows.append(sft_row)

            batch_candidate_rows.append(
                {
                    "prompt_key": f"{row.get('id')}\t{row.get('question')}",
                    "prompt_index": len(processed_ids) + start + len(batch_candidate_rows),
                    "id": row.get("id"),
                    "source": row.get("source", "public_train"),
                    "input_path": str(args.input_path),
                    "question": row.get("question"),
                    "question_type": "mcq",
                    "preferred_output_tokens": args.max_new_tokens,
                    "max_output_tokens": args.max_new_tokens,
                    "gold": [gold],
                    "options": row.get("options") or [],
                    "accepted_count": sum(1 for c in candidates if c["accepted"]),
                    "selected_candidate_index": selected,
                    "candidates": candidates,
                }
            )
        append_jsonl(args.candidate_path, batch_candidate_rows)
        append_jsonl(args.out_path, batch_sft_rows)
        accepted_total += len(batch_sft_rows)
        print(
            f"[mcq-rs] saved batch: prompts={len(batch_candidate_rows)} "
            f"accepted={len(batch_sft_rows)} total_accepted={accepted_total}",
            flush=True,
        )

    if args.manifest_path:
        manifest = {
            "model": args.model,
            "tokenizer": tokenizer_path,
            "input_paths": [str(args.input_path)],
            "out_path": str(args.out_path),
            "candidate_path": str(args.candidate_path),
            "prompt_count": len(rows) + len(processed_ids),
            "processed_prompt_count": len(processed_ids) + len(rows),
            "sft_record_count": accepted_total,
            "num_generations": args.num_generations,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "stats": dict(stats),
        }
        args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    print(f"[mcq-rs] accepted {accepted_total} total prompts", flush=True)
    print(f"[mcq-rs] wrote {args.out_path}", flush=True)
    print(f"[mcq-rs] wrote {args.candidate_path}", flush=True)


if __name__ == "__main__":
    main()
