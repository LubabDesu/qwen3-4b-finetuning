#!/usr/bin/env python3
"""
Convert generated reasoning paths to submission CSV with majority voting.

Reads JSONL with multiple generations per question, extracts final answers
using judger.py, performs order-invariant majority voting, and outputs a 
pristine, quote-escaped CSV with one full response trace per question.
"""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from judger import Judger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert reasoning paths to submission CSV with majority voting."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL file with generations (from generate_private_reasoning_paths.py)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV file for submission (id, response columns)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print details for each question",
    )
    return parser.parse_args()


def extract_answer(judger: Judger, response: str) -> str:
    """Extract boxed answer from response using Judger."""
    try:
        ans = judger.extract_boxed_answer(response)
        return ans if ans is not None else ""
    except Exception:
        # Fallback to empty string if parsing crashes
        return ""


def normalize_answer(answer: str) -> str:
    """Normalize an extracted answer for vote aggregation while preserving blank order."""
    if not answer:
        return ""
    # Preserve comma-separated order because private scoring expects blank order.
    parts = [p.strip().lower() for p in re.split(r"[,;]+", answer) if p.strip()]
    return ", ".join(parts)


def find_majority_key(voting_keys: list[str]) -> str | None:
    """
    Find the most common normalized answer key.
    Returns None if there is a tie for first place or no clear consensus.
    """
    if not voting_keys:
        return None
    counter = Counter(voting_keys)
    top_two = counter.most_common(2)

    # If there are multiple answers and the top two have the exact same vote count, it's a tie
    if len(top_two) > 1 and top_two[0][1] == top_two[1][1]:
        return None

    return top_two[0][0]


def main():
    args = parse_args()
    judger = Judger()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    successful = 0
    no_majority = 0

    print(f"[submission] Reading from: {args.input}", flush=True)
    print(f"[submission] Writing to: {args.output}", flush=True)

    with args.input.open("r", encoding="utf-8") as in_f, args.output.open(
        "w", encoding="utf-8", newline=""
    ) as out_f:
        
        # CRITICAL FIX: Use csv.QUOTE_ALL to ensure multi-line reasoning traces
        # containing commas, quotes, and backslashes do not break row parsing.
        writer = csv.DictWriter(
            out_f, fieldnames=["id", "response"], quoting=csv.QUOTE_ALL
        )
        writer.writeheader()

        for line_no, line in enumerate(in_f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                question_id = row["question_id"]
                generations = row["generations"]

                if not generations:
                    print(f"[submission] warning: line {line_no} has no generations", flush=True)
                    continue

                # 1. Extract raw targets via your judger module
                extracted_answers = [extract_answer(judger, gen) for gen in generations]

                # 2. Convert to normalized strings strictly for vote aggregation
                normalized_keys = [normalize_answer(ans) for ans in extracted_answers]

                # 3. Aggregate and determine consensus key
                majority_key = find_majority_key(normalized_keys)

                if majority_key is None or majority_key == "":
                    if args.verbose:
                        print(f"[submission] Q{question_id}: tie or no majority answer, falling back to first trace", flush=True)
                    # Safe fallback path: prefer a trace that has an extractable boxed answer.
                    selected_response = next((gen for gen in generations if "\\boxed" in gen), generations[0])
                    no_majority += 1
                else:
                    # 4. Map back: Find the first trace that maps to the winning consensus key
                    selected_response = None
                    for gen, norm_key in zip(generations, normalized_keys):
                        if norm_key == majority_key:
                            selected_response = gen
                            break
                    
                    # Fallback guard
                    if selected_response is None:
                        selected_response = generations[0]

                # 5. Safe dictionary line serialization
                writer.writerow({"id": question_id, "response": selected_response})

                if args.verbose and majority_key is not None:
                    print(
                        f"[submission] Q{question_id}: votes={Counter(normalized_keys).most_common(3)} selected_key={majority_key}",
                        flush=True,
                    )

                processed += 1
                successful += 1

            except (json.JSONDecodeError, KeyError) as e:
                print(f"[submission] error: line {line_no}: {e}", flush=True)
                continue

    print(
        f"[submission] done: processed={processed} successful={successful} no_majority={no_majority} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()