#!/usr/bin/env python3
"""Merge draft_3 finalizer boxed answers into submission_draft_3.csv."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

BOX_START_RE = re.compile(r"\\boxed\s*\{")


def extract_first_boxed(text: str) -> str | None:
    if not text:
        return None
    m = BOX_START_RE.search(text)
    if not m:
        return None
    start = m.end()
    depth = 1
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "{" and (i == 0 or text[i - 1] != "\\"):
            depth += 1
        elif ch == "}" and (i == 0 or text[i - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
        i += 1
    return text[start:].strip() or None


def boxed_response(answer: str) -> str:
    return f"\\boxed{{{answer.strip()}}}"


def has_boxed(text: str) -> bool:
    return bool(BOX_START_RE.search(text or ""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("artifacts/private_reasoning_paths/submission_draft_3.csv"))
    parser.add_argument("--finalizer", type=Path, default=Path("artifacts/private_reasoning_paths/draft_3_finalizer_prefill_outputs.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/private_reasoning_paths/submission_draft_3_finalized.csv"))
    parser.add_argument("--log", type=Path, default=Path("artifacts/private_reasoning_paths/submission_draft_3_finalized_merge_log.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    finalizer_answers = {}
    multi_box = 0
    missing_box = 0
    with args.finalizer.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = int(row["question_id"])
            generations = row.get("finalizer_generations") or []
            if not generations:
                missing_box += 1
                continue
            text = generations[0]
            if len(BOX_START_RE.findall(text)) > 1:
                multi_box += 1
            answer = extract_first_boxed(text)
            if answer is None:
                missing_box += 1
                continue
            finalizer_answers[qid] = boxed_response(answer)

    rows = []
    stats = {
        "base_rows": 0,
        "base_boxed": 0,
        "base_unboxed": 0,
        "finalizer_rows": len(finalizer_answers),
        "finalizer_multi_box_outputs": multi_box,
        "finalizer_missing_box_outputs": missing_box,
        "replaced_unboxed": 0,
        "unboxed_without_finalizer": 0,
        "output_boxed": 0,
        "output_unboxed": 0,
    }
    log_entries = []

    with args.base.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = int(row["id"])
            resp = row["response"]
            stats["base_rows"] += 1
            if has_boxed(resp):
                stats["base_boxed"] += 1
            else:
                stats["base_unboxed"] += 1
                replacement = finalizer_answers.get(qid)
                if replacement:
                    resp = replacement
                    stats["replaced_unboxed"] += 1
                    log_entries.append({"qid": qid, "replacement": replacement})
                else:
                    stats["unboxed_without_finalizer"] += 1
                    log_entries.append({"qid": qid, "replacement": None})
            if has_boxed(resp):
                stats["output_boxed"] += 1
            else:
                stats["output_unboxed"] += 1
            rows.append({"id": str(qid), "response": resp})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "response"], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    with args.log.open("w", encoding="utf-8") as f:
        for entry in log_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(json.dumps(stats, indent=2, sort_keys=True))
    print(f"output={args.output}")
    print(f"log={args.log}")


if __name__ == "__main__":
    main()
