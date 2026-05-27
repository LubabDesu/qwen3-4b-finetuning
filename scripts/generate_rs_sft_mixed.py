#!/usr/bin/env python3
"""Generate mixed MCQ/non-MCQ rejection-sampling SFT rows with vLLM."""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import re
import sys
import threading
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import judger as judger_module  # noqa: E402
from generate_rs_sft_mcq import (  # noqa: E402
    CONCLUSION_NUDGE,
    anti_spiral_reject_reason,
    build_user_problem,
    candidate_quality_metrics,
    candidate_score,
    clean_completion,
    extract_all_boxed,
    generate_with_optional_nudge,
    normalize_mcq_answer,
    target_shape_ok,
)
from judger import Judger  # noqa: E402


SYSTEM_PROMPT = """You are an expert mathematician. Reason step by step, then put your final answer within \\boxed{}.

Final answer rules:
- Use exactly one final \\boxed{}.
- For MCQ, put only the option letter in \\boxed{}.
- For multi-part problems with multiple [ANS] blanks, put all answers comma-separated in ONE box in blank order.
- Never round intermediate calculations.
- Never round your final answer unless the problem explicitly asks you to round."""

_JUDGER_LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model path or Hugging Face id.")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path. Defaults to --model.")
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--out-path", type=Path, required=True, help="Accepted SFT JSONL output.")
    parser.add_argument("--candidate-path", type=Path, required=True, help="All-candidate JSONL output.")
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--num-generations", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=3000)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-num-seqs", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--conclusion-nudge-frac", type=float, default=0.8)
    parser.add_argument("--conclusion-nudge", default=CONCLUSION_NUDGE)
    parser.add_argument("--max-spiral-markers", type=int, default=2)
    parser.add_argument("--max-duplicate-lines", type=int, default=1)
    parser.add_argument("--max-repeated-ngram-ratio", type=float, default=0.10)
    parser.add_argument("--strict-extract", action="store_true", help="Use Judger(strict_extract=True) for non-MCQ rows.")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_id(row: dict[str, Any], fallback: int) -> Any:
    return row.get("id", row.get("question_id", fallback))


def gold_answers(row: dict[str, Any]) -> list[str]:
    gold = row.get("gold_answer", row.get("answer", row.get("target_answer", [])))
    if isinstance(gold, list):
        return [str(item).strip() for item in gold if str(item).strip()]
    if gold is None:
        return []
    return [str(gold).strip()]


def is_mcq_row(row: dict[str, Any], gold: list[str]) -> bool:
    options = row.get("options") or []
    if not isinstance(options, list) or not options or len(gold) != 1:
        return False
    return bool(normalize_mcq_answer(gold[0]))


def normalize_final_answer_for_row(row: dict[str, Any], value: str) -> str:
    if is_mcq_row(row, gold_answers(row)):
        return normalize_mcq_answer(value)
    return str(value or "").strip()


def build_prompt(tokenizer: Any, row: dict[str, Any]) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_problem(row["question"], row.get("options") or [])},
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


def strip_boxed_expressions(text: str) -> str:
    out = []
    start = 0
    marker = r"\boxed"
    while True:
        i = text.find(marker, start)
        if i < 0:
            out.append(text[start:])
            break
        out.append(text[start:i])
        j = text.find("{", i)
        if j < 0:
            start = i + len(marker)
            continue
        depth = 0
        for k in range(j, len(text)):
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
                if depth == 0:
                    start = k + 1
                    break
        else:
            start = len(text)
            break
    cleaned = "".join(out)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def repair_to_single_final_box(completion: str, final_answer: str) -> str:
    if "</think>" in completion:
        before, _after = completion.split("</think>", 1)
    else:
        before = completion
    reasoning = before.replace("<think>", "").strip()
    reasoning = strip_boxed_expressions(reasoning)
    return f"<think>\n{reasoning}\n</think>\n\n\\boxed{{{final_answer.strip()}}}"


@contextlib.contextmanager
def disable_judger_alarm() -> Any:
    old_signal = judger_module.signal.signal
    old_alarm = judger_module.signal.alarm
    judger_module.signal.signal = lambda *args, **kwargs: None
    judger_module.signal.alarm = lambda *args, **kwargs: 0
    try:
        yield
    finally:
        judger_module.signal.signal = old_signal
        judger_module.signal.alarm = old_alarm


def judge_completion(
    judger: Judger,
    row: dict[str, Any],
    completion: str,
    gold: list[str],
    boxes: list[str],
) -> tuple[bool, str]:
    extracted = boxes[-1].strip() if boxes else ""
    if is_mcq_row(row, gold):
        pred = normalize_mcq_answer(extracted)
        target = normalize_mcq_answer(gold[0])
        return bool(pred and pred == target), pred

    if not gold:
        return False, extracted
    per_answer_options = [[] for _ in gold]
    try:
        with _JUDGER_LOCK:
            with disable_judger_alarm():
                correct = judger.auto_judge(pred=completion, gold=gold, options=per_answer_options)
    except Exception:
        correct = False
    return bool(correct), extracted


def make_sft_row(
    row: dict[str, Any],
    completion: str,
    candidate_index: int,
    tokens: int,
    final_answer: str,
    selected_score: float | None,
    selected_metrics: dict[str, Any] | None,
    used_conclusion_nudge: bool,
    num_generations: int,
) -> dict[str, Any]:
    rid = row_id(row, 0)
    gold = gold_answers(row)
    reasoning = completion.split("</think>", 1)[0].replace("<think>", "").strip()
    question_type = row.get("question_type") or ("mcq" if is_mcq_row(row, gold) else "open")
    normalized_final = normalize_final_answer_for_row(row, final_answer)
    return {
        "id": f"rs_sft_{rid}",
        "question": row["question"],
        "options": row.get("options") or [],
        "answer": normalized_final,
        "target_answer": normalized_final,
        "reasoning": reasoning,
        "target": completion,
        "source": "rs_sft_mixed",
        "original_source": row.get("source", "public_train"),
        "original_id": rid,
        "question_type": question_type,
        "is_mcq": is_mcq_row(row, gold),
        "n_ans": len(gold),
        "gold_answer": gold,
        "rs_sft": {
            "question_type": question_type,
            "selected_candidate_index": candidate_index,
            "tier": "clean",
            "completion_tokens": tokens,
            "cleaned_tokens": tokens,
            "num_generations": num_generations,
            "selection": {
                "method": "correct_then_anti_spiral_score",
                "quality_score": selected_score,
                "quality_metrics": selected_metrics,
                "used_conclusion_nudge": used_conclusion_nudge,
            },
        },
        "rendered_tokens": tokens,
    }


def main() -> None:
    args = parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = load_jsonl(args.input_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    processed_ids = set()
    existing_sft_rows = 0
    if args.candidate_path.exists():
        for old_row in load_jsonl(args.candidate_path):
            if old_row.get("id") is not None:
                processed_ids.add(old_row["id"])
    if args.out_path.exists():
        existing_sft_rows = len(load_jsonl(args.out_path))
    if processed_ids:
        original_count = len(rows)
        rows = [row for idx, row in enumerate(rows) if row_id(row, idx) not in processed_ids]
        print(f"[mixed-rs] resuming: {len(rows)} / {original_count} prompts remaining", flush=True)
    else:
        print(f"[mixed-rs] starting fresh: {len(rows)} prompts", flush=True)

    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    prompts = [build_prompt(tokenizer, row) for row in rows]
    judger = Judger(strict_extract=args.strict_extract)

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

    stats: collections.Counter[str] = collections.Counter()
    accepted_total = existing_sft_rows

    for start in range(0, len(rows), args.batch_size):
        end = min(start + args.batch_size, len(rows))
        print(f"[mixed-rs] generating {start + 1}-{end} / {len(rows)}", flush=True)
        generated = generate_with_optional_nudge(llm, prompts[start:end], args, SamplingParams)
        batch_candidate_rows: list[dict[str, Any]] = []
        batch_sft_rows: list[dict[str, Any]] = []

        for local_idx, (row, output_candidates) in enumerate(zip(rows[start:end], generated)):
            rid = row_id(row, start + local_idx)
            gold = gold_answers(row)
            row_is_mcq = is_mcq_row(row, gold)
            candidates = []
            selected = None
            selected_score = None
            selected_completion = None
            selected_tokens = None
            selected_metrics = None
            selected_final = None
            selected_used_nudge = False

            for cand_idx, cand in enumerate(output_candidates):
                completion = clean_completion(cand.text)
                boxes = extract_all_boxed(completion)
                correct, extracted = judge_completion(judger, row, completion, gold, boxes)
                normalized_extracted = normalize_final_answer_for_row(row, extracted)
                repaired_completion = repair_to_single_final_box(completion, normalized_extracted) if extracted else completion
                repaired_boxes = extract_all_boxed(repaired_completion)
                shape_ok = target_shape_ok(repaired_completion)
                if not correct and extracted and not row_is_mcq:
                    correct, _ = judge_completion(judger, row, repaired_completion, gold, repaired_boxes)
                metrics = candidate_quality_metrics(repaired_completion)
                anti_spiral_reason = anti_spiral_reject_reason(metrics, args)
                accepted = bool(correct and shape_ok and anti_spiral_reason is None)
                if accepted:
                    reason = ""
                elif not correct:
                    reason = "incorrect"
                elif not shape_ok:
                    reason = "bad_target_shape"
                else:
                    reason = anti_spiral_reason or "quality_filter"

                tokens = len(cand.token_ids or [])
                score = candidate_score(tokens, metrics, cand.used_conclusion_nudge) if accepted else None
                candidates.append(
                    {
                        "candidate_index": cand_idx,
                        "accepted": accepted,
                        "tier": "clean" if accepted else "",
                        "reject_reason": reason,
                        "correct": correct,
                        "extracted_answer": extracted,
                        "normalized_answer": normalized_extracted,
                        "completion_tokens": tokens,
                        "cleaned_tokens": tokens,
                        "under_preferred_cap": tokens <= args.max_new_tokens,
                        "used_conclusion_nudge": cand.used_conclusion_nudge,
                        "quality_score": score,
                        "quality_metrics": metrics,
                        "structure": {
                            "has_think_close": "</think>" in repaired_completion,
                            "final_box_count": len(repaired_boxes),
                            "total_box_count": len(repaired_boxes),
                            "raw_box_count": len(boxes),
                            "has_text_after_final_box": False,
                        },
                        "completion": cand.text,
                        "cleaned_completion": repaired_completion,
                        "raw_cleaned_completion": completion,
                    }
                )
                stats[f"reject:{reason or 'accepted'}"] += 1
                stats[f"type:{'mcq' if row_is_mcq else 'non_mcq'}:{reason or 'accepted'}"] += 1

                if accepted and (selected_score is None or score > selected_score):
                    selected = cand_idx
                    selected_score = score
                    selected_completion = repaired_completion
                    selected_tokens = tokens
                    selected_metrics = metrics
                    selected_final = extracted
                    selected_used_nudge = cand.used_conclusion_nudge

            if selected is not None and selected_completion is not None and selected_tokens is not None:
                batch_sft_rows.append(
                    make_sft_row(
                        row=row,
                        completion=selected_completion,
                        candidate_index=selected,
                        tokens=selected_tokens,
                        final_answer=selected_final or "",
                        selected_score=selected_score,
                        selected_metrics=selected_metrics,
                        used_conclusion_nudge=selected_used_nudge,
                        num_generations=args.num_generations,
                    )
                )

            batch_candidate_rows.append(
                {
                    "prompt_key": f"{rid}\t{row.get('question')}",
                    "prompt_index": len(processed_ids) + start + len(batch_candidate_rows),
                    "id": rid,
                    "source": row.get("source", "public_train"),
                    "input_path": str(args.input_path),
                    "question": row.get("question"),
                    "question_type": row.get("question_type") or ("mcq" if row_is_mcq else "open"),
                    "is_mcq": row_is_mcq,
                    "preferred_output_tokens": args.max_new_tokens,
                    "max_output_tokens": args.max_new_tokens,
                    "gold": gold,
                    "options": row.get("options") or [],
                    "accepted_count": sum(1 for c in candidates if c["accepted"]),
                    "selected_candidate_index": selected,
                    "selection_method": "correct_then_anti_spiral_score",
                    "candidates": candidates,
                }
            )

        append_jsonl(args.candidate_path, batch_candidate_rows)
        append_jsonl(args.out_path, batch_sft_rows)
        accepted_total += len(batch_sft_rows)
        print(
            f"[mixed-rs] saved batch: prompts={len(batch_candidate_rows)} "
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
            "conclusion_nudge_frac": args.conclusion_nudge_frac,
            "conclusion_nudge": args.conclusion_nudge,
            "max_spiral_markers": args.max_spiral_markers,
            "max_duplicate_lines": args.max_duplicate_lines,
            "max_repeated_ngram_ratio": args.max_repeated_ngram_ratio,
            "strict_extract": args.strict_extract,
            "stats": dict(stats),
        }
        args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[mixed-rs] accepted {accepted_total} total prompts", flush=True)
    print(f"[mixed-rs] wrote {args.out_path}", flush=True)
    print(f"[mixed-rs] wrote {args.candidate_path}", flush=True)


if __name__ == "__main__":
    main()
