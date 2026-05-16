#!/usr/bin/env python3
"""Summarize checkpoint eval JSON files and rank them by accuracy."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


STEP_RE = re.compile(r"ckpt_(\d+)_eval_summary(?:\.partial)?\.json$")


def load_summary(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    match = STEP_RE.search(path.name)
    if not match:
        raise ValueError(f"Could not infer checkpoint step from {path.name}")
    data = dict(data)
    data["step"] = int(match.group(1))
    data["summary_path"] = str(path)
    return data


def metric_value(row: dict[str, Any], metric: str) -> float:
    value = row.get(metric)
    if value is None:
        return float("-inf")
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank checkpoint eval summaries.")
    parser.add_argument("--eval-dir", type=Path, default=Path("artifacts/post_training_curriculum/eval"))
    parser.add_argument("--pattern", default="stage1_2_public_ckpt_*_eval_summary.json")
    parser.add_argument("--metric", default="accuracy")
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    paths = sorted(args.eval_dir.glob(args.pattern))
    rows = [load_summary(path) for path in paths]
    rows.sort(key=lambda row: (metric_value(row, args.metric), metric_value(row, "boxed_compliance_rate")), reverse=True)

    if not rows:
        print(f"No summaries matched: {args.eval_dir / args.pattern}")
        return

    fields = [
        "step",
        "n",
        "correct",
        "accuracy",
        "mcq_accuracy",
        "mcq_total",
        "multi_answer_accuracy",
        "multi_answer_total",
        "avg_response_words",
        "boxed_compliance_rate",
        "inference_backend",
        "summary_path",
    ]
    print(",".join(fields))
    for row in rows:
        print(",".join(str(row.get(field, "")) for field in fields))

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fields})
        print(f"Wrote {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
