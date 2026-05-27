#!/usr/bin/env python3
"""Build a stricter RS-SFT JSONL from reviewed RS-SFT rows.

This is meant for small ablations where training hyperparameters stay fixed and
only the data contract changes:
- bounded "wait" loops in the reasoning
- no boxed answers inside the thinking section
- exactly one final boxed answer after </think>
- no text after the final boxed answer
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPO_ROOT
    / "artifacts/grpo_rs_sft/sft/ckpt500_combined_review_cleaned_plus_mcq_longcap_plus_recovered16_waitle20.jsonl"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts/grpo_rs_sft/sft/ckpt500_combined_review_cleaned_plus_mcq_longcap_plus_recovered16_waitle10_boxed.jsonl"
)

WAIT_RE = re.compile(r"\bwait\b", flags=re.I)
BOXED_RE = re.compile(r"\\boxed\s*\{", flags=re.S)
THINK_RE = re.compile(r"^\s*<think>\s*(.*?)\s*</think>\s*(.*)\s*$", flags=re.S)
NOISE_PHRASES = [
    "self-check",
    "self-assessment",
    "self assessment",
    "analysis:",
    "in summary",
    "to summarize",
    "final answer:",
    "final answer is",
    "the final answer is",
    "the correct answer is",
    "correct answer:",
    "the answer is",
    "therefore, the correct choice is",
    "hence, the correct choice",
    "the correct option is",
    "correct option is",
    "correct option:",
    "knowledge point",
    "knowledge tested",
    "self-evaluation",
    "conclusion:",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def word_count(text: str) -> int:
    return len(str(text or "").split())


def extract_boxed_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    start = 0
    while True:
        match = BOXED_RE.search(text, start)
        if not match:
            break
        i = match.end()
        depth = 1
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            spans.append((match.start(), i, text[match.end() : i - 1].strip()))
        start = max(i, match.end())
    return spans


def normalize_final(value: Any) -> str:
    if isinstance(value, list):
        value = ", ".join(str(item).strip() for item in value if str(item).strip())
    text = str(value or "").strip()
    text = re.sub(r"^\s*\\boxed\s*\{(.*)\}\s*$", r"\1", text, flags=re.S).strip()
    return text


def target_parts(row: dict[str, Any]) -> tuple[str, str]:
    target = str(row.get("target") or "")
    match = THINK_RE.match(target)
    reasoning = str(row.get("reasoning") or "").strip()
    final = normalize_final(row.get("target_answer") or row.get("answer") or row.get("gold_answer"))

    if match:
        target_reasoning = match.group(1).strip()
        target_tail = match.group(2).strip()
        boxes = extract_boxed_spans(target_tail)
        if target_reasoning:
            reasoning = target_reasoning
        if boxes:
            final = normalize_final(boxes[-1][2])

    return reasoning.strip(), final.strip()


def render_target(reasoning: str, final: str) -> str:
    return f"<think>\n{reasoning.strip()}\n</think>\n\n\\boxed{{{final.strip()}}}"


def validate_rendered(target: str) -> str | None:
    match = THINK_RE.match(target)
    if not match:
        return "bad_think_structure"
    reasoning = match.group(1)
    tail = match.group(2)
    if r"\boxed" in reasoning:
        return "boxed_in_reasoning"
    boxes = extract_boxed_spans(tail)
    if len(boxes) != 1:
        return "not_exactly_one_box_after_think"
    if tail[boxes[0][1] :].strip():
        return "text_after_box"
    if not boxes[0][2].strip():
        return "empty_box"
    return None


def build(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_jsonl(args.input)
    kept: list[dict[str, Any]] = []
    drops: collections.Counter[str] = collections.Counter()
    wait_counts: list[int] = []
    word_counts: list[int] = []

    for row in rows:
        reasoning, final = target_parts(row)
        if not reasoning:
            drops["empty_reasoning"] += 1
            continue
        if not final:
            drops["empty_final"] += 1
            continue

        wait_count = len(WAIT_RE.findall(reasoning))
        if wait_count > args.max_waits:
            drops["too_many_waits"] += 1
            continue

        words = word_count(reasoning)
        if args.max_reasoning_words is not None and words > args.max_reasoning_words:
            drops["too_many_words"] += 1
            continue
        if args.min_reasoning_words is not None and words < args.min_reasoning_words:
            drops["too_few_words"] += 1
            continue

        if args.reject_panic_preamble and re.search(
            r"\b(?:complex or challenging|difficult to provide a direct|i need to think about it)\b",
            reasoning[:600],
            flags=re.I,
        ):
            drops["panic_preamble"] += 1
            continue

        if args.reject_noise_phrases:
            low_reasoning = reasoning.lower()
            if any(phrase in low_reasoning for phrase in NOISE_PHRASES):
                drops["noise_phrase"] += 1
                continue

        if row.get("is_mcq"):
            valid = {chr(ord("A") + idx) for idx, _ in enumerate(row.get("options") or [])}
            if valid and final.strip().upper() not in valid:
                drops["invalid_mcq_final"] += 1
                continue
            final = final.strip().upper()

        target = render_target(reasoning, final)
        invalid = validate_rendered(target)
        if invalid:
            drops[invalid] += 1
            continue

        new_row = dict(row)
        new_row["reasoning"] = reasoning
        new_row["target_answer"] = final
        new_row["answer"] = final
        new_row["target"] = target
        new_row["rs_sft_trim"] = {
            "source": Path(__file__).name,
            "input": str(args.input),
            "max_waits": args.max_waits,
            "wait_count": wait_count,
            "reasoning_words": words,
            "exactly_one_box_after_think": True,
        }
        kept.append(new_row)
        wait_counts.append(wait_count)
        word_counts.append(words)

    manifest = {
        "input": str(args.input),
        "output": str(args.output),
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "dropped_rows": len(rows) - len(kept),
        "drop_counts": dict(drops.most_common()),
        "max_waits": args.max_waits,
        "min_reasoning_words": args.min_reasoning_words,
        "max_reasoning_words": args.max_reasoning_words,
        "reject_noise_phrases": args.reject_noise_phrases,
        "source_counts": dict(collections.Counter(row.get("source") for row in kept)),
        "mcq_rows": sum(bool(row.get("is_mcq")) for row in kept),
        "non_mcq_rows": sum(not bool(row.get("is_mcq")) for row in kept),
        "wait_counts": summarize_ints(wait_counts),
        "reasoning_words": summarize_ints(word_counts),
    }
    return kept, manifest


def summarize_ints(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "median": None, "p90": None, "max": None, "avg": None}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": ordered[round((len(ordered) - 1) * 0.5)],
        "p90": ordered[round((len(ordered) - 1) * 0.9)],
        "max": ordered[-1],
        "avg": sum(values) / len(values),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--max-waits", type=int, default=10)
    parser.add_argument("--min-reasoning-words", type=int, default=50)
    parser.add_argument("--max-reasoning-words", type=int, default=None)
    parser.add_argument("--reject-noise-phrases", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reject-panic-preamble", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, manifest = build(args)
    write_jsonl(args.output, rows)
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"Wrote manifest to {manifest_path}")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
