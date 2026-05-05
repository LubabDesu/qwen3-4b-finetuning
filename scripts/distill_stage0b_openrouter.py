#!/usr/bin/env python3
"""
Build Stage 0b format-distillation data with OpenRouter.

Goal: teach exact output contract:
<think>
concise reasoning
</think>

\\boxed{...}

Uses known answers from course public.jsonl. Teacher writes reasoning only; final
answer is fixed and validated. Heldout eval public IDs are excluded.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PATH = ROOT / "comp_folder/151B_SP26_Competition/data/public.jsonl"
HELDOUT_PATH = ROOT / "artifacts/post_training_curriculum/eval/heldout_eval_set.jsonl"
OUT_DIR = ROOT / "artifacts/post_training_curriculum/datasets"
DEFAULT_OUT = OUT_DIR / "stage0b_openrouter_distilled_records.jsonl"
DEFAULT_REJECTS = OUT_DIR / "stage0b_openrouter_rejects.jsonl"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODEL = "deepseek/deepseek-v4-flash"

# Optional: paste key here if you prefer script-local config.
# Safer default is env var: export OPENROUTER_API_KEY="..."
OPENROUTER_API_KEY_IN_SCRIPT = "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"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(row: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_ans_blanks(question: str) -> int:
    return str(question).count("[ANS]")


def option_labels(n: int) -> List[str]:
    return [chr(65 + i) for i in range(n)]


def normalize_final_answer(answer: Any) -> str:
    if isinstance(answer, (list, tuple)):
        return ", ".join(str(x).strip() for x in answer)
    return str(answer).strip()


def format_options(options: Optional[Sequence[str]]) -> str:
    if not options:
        return ""
    return "\n".join(f"{label}. {str(opt).strip()}" for label, opt in zip(option_labels(len(options)), options))


def build_user_problem(question: str, options: Optional[Sequence[str]]) -> str:
    if options:
        return f"{question}\n\nOptions:\n{format_options(options)}"
    return question


def extract_all_boxed(text: str) -> List[str]:
    text = text or ""
    boxes: List[str] = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        start = match.end()
        depth = 1
        i = start
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            boxes.append(text[start : i - 1].strip())
    return boxes


def extract_after_think(text: str) -> str:
    parts = (text or "").split("</think>")
    return parts[-1].strip() if len(parts) > 1 else (text or "").strip()


def extract_reasoning(text: str) -> str:
    if "<think>" in text and "</think>" in text:
        return text.split("<think>", 1)[1].split("</think>", 1)[0].strip()
    if "</think>" in text:
        return text.split("</think>", 1)[0].replace("<think>", "").strip()
    return text.replace("<think>", "").strip()


def boxed_format_ok(question: str, response: str, is_mcq: bool, options: Optional[Sequence[str]]) -> bool:
    if "<think>" not in response or "</think>" not in response:
        return False
    after = extract_after_think(response)
    boxes = extract_all_boxed(after)
    if len(boxes) != 1:
        return False
    final = boxes[0].strip()
    if is_mcq:
        return final.upper() in set(option_labels(len(options or [])))
    expected = count_ans_blanks(question)
    if expected <= 1:
        return bool(final)
    return len([p.strip() for p in final.split(",") if p.strip()]) == expected


def make_training_record(
    question: str,
    answer: Any,
    reasoning: str,
    options: Optional[Sequence[str]] = None,
    source: str = "",
) -> Dict[str, Any]:
    return {
        "question": question,
        "options": list(options) if options else None,
        "answer": normalize_final_answer(answer),
        "reasoning": reasoning.strip(),
        "source": source,
        "n_ans": count_ans_blanks(question),
        "is_mcq": bool(options),
    }


def heldout_public_ids(path: Path) -> set:
    if not path.exists():
        return set()
    ids = set()
    for row in load_jsonl(path):
        if row.get("eval_source") == "public":
            ids.add(row.get("id"))
    return ids


def classify_public_rows(rows: Sequence[Dict[str, Any]], heldout_ids: set) -> Dict[str, List[Dict[str, Any]]]:
    eligible = [r for r in rows if r.get("id") not in heldout_ids]
    buckets = {
        "mcq10": [],
        "multi": [],
        "table_many": [],
        "single": [],
    }
    table_keys = ("table", "complete the table", "select the most appropriate")
    for row in eligible:
        q = str(row.get("question", ""))
        n_ans = count_ans_blanks(q)
        opts = row.get("options") or []
        q_lower = q.lower()
        if opts and len(opts) >= 7:
            buckets["mcq10"].append(row)
        elif n_ans >= 5 or any(k in q_lower for k in table_keys):
            buckets["table_many"].append(row)
        elif n_ans > 1:
            buckets["multi"].append(row)
        elif n_ans == 1 and not opts:
            buckets["single"].append(row)
    return buckets


def choose_rows(
    buckets: Dict[str, List[Dict[str, Any]]],
    seed: int,
    n_mcq10: int,
    n_multi: int,
    n_table_many: int,
    n_single: int,
) -> List[Tuple[str, Dict[str, Any]]]:
    rng = random.Random(seed)
    plan = [
        ("mcq10", n_mcq10),
        ("multi", n_multi),
        ("table_many", n_table_many),
        ("single", n_single),
    ]
    chosen: List[Tuple[str, Dict[str, Any]]] = []
    for name, n in plan:
        rows = list(buckets.get(name, []))
        rng.shuffle(rows)
        if len(rows) < n:
            print(f"warning: requested {n} {name}, only {len(rows)} available after heldout filter", file=sys.stderr)
        chosen.extend((name, row) for row in rows[: min(n, len(rows))])
    rng.shuffle(chosen)
    return chosen


def teacher_prompt(row: Dict[str, Any], bucket: str) -> List[Dict[str, str]]:
    question = row["question"]
    options = row.get("options")
    answer = normalize_final_answer(row.get("answer"))
    n_ans = count_ans_blanks(question)
    is_mcq = bool(options)
    problem = build_user_problem(question, options)

    system = (
        "You write supervised fine-tuning traces for a small math model.\n"
        "You are given the known correct final answer. Do not change it.\n"
        "Your job: write concise reasoning that leads to that known answer and obey exact format."
    )

    style_hint = "Use enough reasoning to be credible, but avoid repeated arithmetic and avoid long option-by-option analysis."
    if bucket == "mcq10":
        style_hint = "For MCQ, solve directly and compare only necessary options. Final box must contain only the letter."
    elif bucket in {"multi", "table_many"}:
        style_hint = (
            f"The question has {n_ans} [ANS] blanks. Final box must contain exactly {n_ans} answers, "
            "comma-separated, in the same order, with no labels."
        )

    user = f"""Question:
{problem}

Known correct final answer:
{answer}

Write one training trace.

Required output exactly:
<think>
Brief reasoning here.
</think>

\\boxed{{{answer}}}

Rules:
- Do not change the known final answer.
- {style_hint}
- No text before <think>.
- No text after the final box.
- No labels inside the final box.
"""
    if is_mcq:
        user += "- MCQ final box contains only the answer letter.\n"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_openrouter(
    messages: List[Dict[str, str]],
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    app_url: Optional[str] = None,
    app_title: Optional[str] = None,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if app_url:
        headers["HTTP-Referer"] = app_url
    if app_title:
        headers["X-Title"] = app_title
    req = urllib.request.Request(OPENROUTER_URL, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    obj = json.loads(raw)
    return obj["choices"][0]["message"]["content"].strip()


def force_final_box(response: str, answer: str) -> str:
    reasoning = extract_reasoning(response)
    if not reasoning:
        reasoning = f"The required final answer is {answer}. I check the requested answer format and place it in one final box."
    return f"<think>\n{reasoning}\n</think>\n\n\\boxed{{{answer}}}"


def load_done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    for row in load_jsonl(path):
        done.add(row.get("distill_key"))
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description="Distill Stage 0b records with OpenRouter.")
    parser.add_argument("--model", default=os.environ.get("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL))
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY") or OPENROUTER_API_KEY_IN_SCRIPT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--rejects", type=Path, default=DEFAULT_REJECTS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-mcq10", type=int, default=350)
    parser.add_argument("--n-multi", type=int, default=350)
    parser.add_argument("--n-table-many", type=int, default=250)
    parser.add_argument("--n-single", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--app-url", default=os.environ.get("OPENROUTER_APP_URL"))
    parser.add_argument("--app-title", default=os.environ.get("OPENROUTER_APP_TITLE", "cse151b-stage0b-distill"))
    args = parser.parse_args()

    if not args.api_key and not args.dry_run:
        raise SystemExit("Set OPENROUTER_API_KEY or pass --api-key.")

    public = load_jsonl(PUBLIC_PATH)
    heldout_ids = heldout_public_ids(HELDOUT_PATH)
    buckets = classify_public_rows(public, heldout_ids)
    print("available after heldout filter:", {k: len(v) for k, v in buckets.items()})

    chosen = choose_rows(
        buckets,
        seed=args.seed,
        n_mcq10=args.n_mcq10,
        n_multi=args.n_multi,
        n_table_many=args.n_table_many,
        n_single=args.n_single,
    )
    if args.limit is not None:
        chosen = chosen[: args.limit]
    print(f"selected {len(chosen)} rows")

    if args.dry_run:
        for bucket, row in chosen[:5]:
            print("=" * 80)
            print("bucket", bucket, "id", row.get("id"), "answer", normalize_final_answer(row.get("answer")))
            print(teacher_prompt(row, bucket)[1]["content"][:2500])
        return 0

    done = load_done_keys(args.out)
    accepted = rejected = skipped = 0

    for idx, (bucket, row) in enumerate(chosen, 1):
        key = f"public:{row.get('id')}:{bucket}"
        if key in done:
            skipped += 1
            continue

        answer = normalize_final_answer(row.get("answer"))
        messages = teacher_prompt(row, bucket)
        response = ""
        err = ""
        for attempt in range(1, args.retries + 1):
            try:
                response = call_openrouter(
                    messages=messages,
                    model=args.model,
                    api_key=args.api_key,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    app_url=args.app_url,
                    app_title=args.app_title,
                )
                fixed = force_final_box(response, answer)
                if boxed_format_ok(row["question"], fixed, bool(row.get("options")), row.get("options")):
                    record = make_training_record(
                        row["question"],
                        row.get("answer"),
                        extract_reasoning(fixed),
                        row.get("options"),
                        source=f"stage0b_openrouter_{bucket}_{args.model}",
                    )
                    record["distill_key"] = key
                    append_jsonl(record, args.out)
                    accepted += 1
                    break
                err = "format validation failed after forced final box"
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                err = repr(exc)
                wait = min(30, 2**attempt)
                print(f"retry {attempt}/{args.retries} id={row.get('id')} err={err}; sleep {wait}s", file=sys.stderr)
                time.sleep(wait)
        else:
            reject = {
                "distill_key": key,
                "id": row.get("id"),
                "bucket": bucket,
                "answer": row.get("answer"),
                "error": err,
                "response": response,
            }
            append_jsonl(reject, args.rejects)
            rejected += 1

        if idx % 25 == 0:
            print(f"progress {idx}/{len(chosen)} accepted={accepted} rejected={rejected} skipped={skipped}")
        if args.sleep:
            time.sleep(args.sleep)

    print(f"done accepted={accepted} rejected={rejected} skipped={skipped}")
    print(f"out={args.out}")
    print(f"rejects={args.rejects}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
