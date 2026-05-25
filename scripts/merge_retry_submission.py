#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

BOX_RE = re.compile(r"\\boxed\s*\{")

def parse_args():
    p = argparse.ArgumentParser(description="Merge boxed retry traces into an existing submission CSV.")
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--retry", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()

def has_box(text: str) -> bool:
    return bool(BOX_RE.search(text or ""))

def main():
    args = parse_args()
    retry_best = {}
    total_retry = 0
    boxed_retry = 0
    for line in args.retry.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        qid = str(row["question_id"])
        gens = row.get("generations") or []
        total_retry += 1
        chosen = next((g for g in gens if has_box(g)), None)
        if chosen is not None:
            retry_best[qid] = chosen
            boxed_retry += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    replaced = 0
    kept = 0
    with args.base.open(newline="", encoding="utf-8") as in_f, args.output.open("w", newline="", encoding="utf-8") as out_f:
        reader = csv.DictReader(in_f)
        writer = csv.DictWriter(out_f, fieldnames=["id", "response"], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in reader:
            qid = row["id"]
            if qid in retry_best and not has_box(row.get("response", "")):
                row["response"] = retry_best[qid]
                replaced += 1
            else:
                kept += 1
            writer.writerow({"id": qid, "response": row["response"]})
    print(f"retry_rows={total_retry} retry_with_box={boxed_retry} replaced={replaced} kept={kept} output={args.output}")

if __name__ == "__main__":
    main()
