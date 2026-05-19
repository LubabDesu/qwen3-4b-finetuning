#!/usr/bin/env python3
"""Create a stratified public train/validation split for GRPO calibration."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUBLIC_PATH = ROOT / "data" / "public.jsonl"
DEFAULT_TRAIN_PATH = ROOT / "artifacts" / "grpo" / "public_train_300.jsonl"
DEFAULT_VAL_PATH = ROOT / "artifacts" / "grpo" / "public_val_rest.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def bucket(row: dict) -> str:
    if row.get("options"):
        return "mcq"
    answer = row.get("answer")
    if isinstance(answer, list) and len(answer) > 1:
        return "multi"
    return "single"


def convert_public_row(row: dict) -> dict:
    answer = row.get("answer", [])
    if isinstance(answer, str):
        gold_answer = [answer]
    else:
        gold_answer = [str(item) for item in answer]
    return {
        "id": f"public_{row['id']}",
        "question": row["question"],
        "gold_answer": gold_answer,
        "options": row.get("options") or [],
        "difficulty_level": 3,
        "source": "public_train",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-path", type=Path, default=DEFAULT_PUBLIC_PATH)
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val-path", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--train-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_jsonl(args.public_path)
    rng = random.Random(args.seed)

    buckets: dict[str, list[dict]] = {"mcq": [], "multi": [], "single": []}
    for row in rows:
        buckets[bucket(row)].append(row)

    for rows_in_bucket in buckets.values():
        rng.shuffle(rows_in_bucket)

    total = len(rows)
    raw_targets = {
        name: args.train_size * len(rows_in_bucket) / total
        for name, rows_in_bucket in buckets.items()
    }
    targets = {name: int(value) for name, value in raw_targets.items()}
    remaining = args.train_size - sum(targets.values())
    for name, _ in sorted(
        raw_targets.items(),
        key=lambda item: item[1] - int(item[1]),
        reverse=True,
    )[:remaining]:
        targets[name] += 1

    train_raw = []
    val_raw = []
    for name, rows_in_bucket in buckets.items():
        n = targets[name]
        train_raw.extend(rows_in_bucket[:n])
        val_raw.extend(rows_in_bucket[n:])

    train_ids = {row["id"] for row in train_raw}
    val_raw = [row for row in rows if row["id"] not in train_ids]
    train_raw.sort(key=lambda row: row["id"])
    val_raw.sort(key=lambda row: row["id"])

    write_jsonl(args.train_path, [convert_public_row(row) for row in train_raw])
    write_jsonl(args.val_path, val_raw)

    print(f"public total: {len(rows)}")
    print(f"train: {len(train_raw)} -> {args.train_path}")
    print(f"val: {len(val_raw)} -> {args.val_path}")
    for name in ("mcq", "multi", "single"):
        train_count = sum(bucket(row) == name for row in train_raw)
        val_count = sum(bucket(row) == name for row in val_raw)
        print(f"{name}: train={train_count} val={val_count}")


if __name__ == "__main__":
    main()
