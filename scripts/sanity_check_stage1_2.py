#!/usr/bin/env python3
"""Sanity-check the Stage 1-2 r2 dataset before training."""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_stage1_2_r2 import (
    ANCHOR_SOURCE,
    DEFAULT_MAX_RENDERED_TOKENS,
    R2_NOISE_PHRASES,
    assistant_text_for_render,
    ends_abruptly,
    extract_all_boxed,
    validate_record,
    word_count,
)


CHINESE_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_answer(value: Any) -> str:
    return str(value or "").strip().upper()


def option_labels(n: int) -> set[str]:
    return {chr(65 + i) for i in range(n)}


def target_structure_ok(text: str) -> bool:
    think_start = text.find("<think>")
    think_end = text.find("</think>")
    boxed = text.find(r"\boxed{")
    return (
        text.count("<think>") == 1
        and text.count("</think>") == 1
        and text.count(r"\boxed{") == 1
        and 0 <= think_start < think_end < boxed
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("artifacts/post_training_curriculum/datasets/stage1_2_r2_records.jsonl"),
    )
    parser.add_argument(
        "--review-path",
        type=Path,
        default=Path("artifacts/post_training_curriculum/datasets/stage1_2_r2_records_for_review.json"),
    )
    parser.add_argument("--max-rendered-tokens", type=int, default=DEFAULT_MAX_RENDERED_TOKENS)
    parser.add_argument("--show-examples", type=int, default=5)
    args = parser.parse_args()

    rows = load_jsonl(args.input_path)
    review_rows = json.loads(args.review_path.read_text()) if args.review_path.exists() else []

    source_counts = collections.Counter(str(row.get("source", "unknown")) for row in rows)
    failures: dict[str, list[Any]] = collections.defaultdict(list)

    for idx, row in enumerate(rows):
        rendered = assistant_text_for_render(row)
        reasoning = str(row.get("reasoning", "") or "")

        validation_error = validate_record(row)
        if validation_error:
            failures["validation_errors"].append((idx, row.get("source"), validation_error))

        if CHINESE_CHAR_RE.search(str(row.get("question", "") or "")):
            failures["chinese_question"].append((idx, row.get("source")))
        if CHINESE_CHAR_RE.search(reasoning):
            failures["chinese_reasoning"].append((idx, row.get("source")))
        if CHINESE_CHAR_RE.search(rendered):
            failures["chinese_rendered_target"].append((idx, row.get("source")))

        if row.get("source") != ANCHOR_SOURCE:
            if ends_abruptly(reasoning):
                failures["abrupt_reasoning"].append((idx, row.get("source"), reasoning[-160:]))
            if word_count(reasoning) < 50:
                failures["short_reasoning"].append((idx, row.get("source"), word_count(reasoning)))
            if r"\boxed" in reasoning:
                failures["boxed_in_reasoning"].append((idx, row.get("source")))
            if "<think>" in reasoning or "</think>" in reasoning:
                failures["think_tags_in_reasoning"].append((idx, row.get("source")))
            if any(phrase in reasoning.lower() for phrase in R2_NOISE_PHRASES):
                failures["noise_in_reasoning"].append((idx, row.get("source")))

        rendered_tokens = row.get("rendered_tokens")
        if rendered_tokens is None:
            failures["missing_rendered_tokens"].append((idx, row.get("source")))
        elif rendered_tokens > args.max_rendered_tokens:
            failures["rendered_tokens_over_limit"].append((idx, row.get("source"), rendered_tokens))

        if not target_structure_ok(rendered):
            failures["target_structure_errors"].append((idx, row.get("source")))

        options = row.get("options") or []
        if row.get("is_mcq") or options:
            valid = option_labels(len(options))
            answer = normalize_answer(row.get("answer"))
            target_answer = normalize_answer(row.get("target_answer", row.get("answer")))
            boxes = extract_all_boxed(rendered)
            boxed = normalize_answer(boxes[-1] if boxes else "")
            reasons = []
            if not options:
                reasons.append("mcq_without_options")
            if answer not in valid:
                reasons.append("answer_not_option_label")
            if target_answer not in valid:
                reasons.append("target_not_option_label")
            if boxed not in valid:
                reasons.append("box_not_option_label")
            if reasons:
                failures["invalid_mcq"].append((idx, row.get("source"), answer, target_answer, boxed, sorted(valid), reasons))

    if review_rows and len(review_rows) != len(rows):
        failures["review_row_count_mismatch"].append((len(rows), len(review_rows)))

    print(f"Dataset: {args.input_path}")
    print(f"Rows: {len(rows)}")
    print(f"Source counts: {dict(source_counts)}")
    if review_rows:
        print(f"Review rows: {len(review_rows)}")

    hard_failures = {name: values for name, values in failures.items() if values}
    if hard_failures:
        print("SANITY: FAIL")
        for name, values in hard_failures.items():
            print(f"{name}: {len(values)}")
            for example in values[: args.show_examples]:
                print(f"  {example}")
        return 1

    print("SANITY: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
