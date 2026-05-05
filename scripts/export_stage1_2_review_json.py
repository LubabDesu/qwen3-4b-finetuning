#!/usr/bin/env python3
"""Export cleaned Stage 1-2 records into a single review-friendly JSON array."""

from __future__ import annotations

import json
import argparse
from pathlib import Path


INPUT_PATH = Path("artifacts/post_training_curriculum/datasets/stage1_2_clean_records.jsonl")
OUTPUT_PATH = Path("artifacts/post_training_curriculum/datasets/stage1_2_clean_records_for_review.json")


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def render_target(row: dict) -> str:
    if row.get("target"):
        return str(row["target"])
    reasoning = str(row.get("reasoning", "")).strip()
    final = str(row.get("target_answer", row.get("answer", ""))).strip()
    return f"<think>\n{reasoning}\n</think>\n\n\\boxed{{{final}}}"


def build_review_rows(rows: list[dict]) -> list[dict]:
    review_rows = []
    for idx, row in enumerate(rows):
        review_rows.append(
            {
                "review_index": idx,
                "source": row.get("source"),
                "original_source": row.get("original_source"),
                "question": row.get("question"),
                "options": row.get("options"),
                "is_mcq": row.get("is_mcq"),
                "n_ans": row.get("n_ans"),
                "answer": row.get("answer"),
                "target_answer": row.get("target_answer", row.get("answer")),
                "reasoning": row.get("reasoning"),
                "target": row.get("target"),
                "rendered_target": render_target(row),
            }
        )
    return review_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, default=INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    rows = load_jsonl(args.input_path)
    review_rows = build_review_rows(rows)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w") as f:
        json.dump(review_rows, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(review_rows)} rows to {args.output_path}")


if __name__ == "__main__":
    main()
