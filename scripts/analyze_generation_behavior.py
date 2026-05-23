#!/usr/bin/env python3
"""Diagnose bounded-verification behavior in eval generation JSONL files.

The competition score says whether the final answer was accepted. This script
answers a different question: did the model solve, verify in a bounded way, and
then stop cleanly?
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
from pathlib import Path
from typing import Any


BOXED_RE = re.compile(r"\\boxed\s*\{", flags=re.S)
WAIT_RE = re.compile(
    r"\b(?:wait|hold on|hang on|actually|let me recheck|let me check again|double[- ]check)\b",
    flags=re.I,
)
PANIC_RE = re.compile(
    r"\b(?:complex or challenging|difficult to provide a direct|i need to think about it|"
    r"not sure|i'm confused|this is tricky)\b",
    flags=re.I,
)
VERIFY_RE = re.compile(
    r"\b(?:check|verify|verification|sanity check|substitute|plug(?:ging)? back|"
    r"confirm|validate|units|edge case)\b",
    flags=re.I,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def word_count(text: str) -> int:
    return len(str(text or "").split())


def find_box_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
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
            spans.append((match.start(), i))
        start = max(i, match.end())
    return spans


def after_think(text: str) -> str:
    parts = str(text or "").split("</think>")
    return parts[-1] if len(parts) > 1 else str(text or "")


def ratio(numer: int, denom: int) -> float:
    return numer / denom if denom else 0.0


def quantile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * q)
    return ordered[idx]


def analyze_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    response = str(row.get("response") or row.get("target") or "")
    response_words = int(row.get("word_count") or word_count(response))
    box_spans = find_box_spans(response)
    final_section = after_think(response)

    final_box_is_last = False
    post_box_words = None
    post_box_wait_count = 0
    if box_spans:
        _, end = box_spans[-1]
        tail = response[end:].strip()
        post_box_words = word_count(tail)
        post_box_wait_count = len(WAIT_RE.findall(tail))
        final_box_is_last = post_box_words <= args.max_post_box_words

    wait_count = len(WAIT_RE.findall(response))
    verify_count = len(VERIFY_RE.findall(response))
    panic_count = len(PANIC_RE.findall(response))
    post_think_words = word_count(final_section)

    too_long = response_words > args.max_words
    too_many_waits = wait_count > args.max_waits
    too_many_boxes = len(box_spans) > args.max_boxes
    lacks_box = not box_spans
    lacks_verification = verify_count < args.min_verify_markers
    post_answer_loop = bool(box_spans and post_box_wait_count > 0)
    panic_preamble = bool(PANIC_RE.search(response[: args.panic_scan_chars]))

    ok = not any(
        [
            too_long,
            too_many_waits,
            too_many_boxes,
            lacks_box,
            lacks_verification,
            not final_box_is_last,
            post_answer_loop,
            panic_preamble,
        ]
    )

    flags = []
    for name, value in [
        ("too_long", too_long),
        ("too_many_waits", too_many_waits),
        ("too_many_boxes", too_many_boxes),
        ("lacks_box", lacks_box),
        ("lacks_verification", lacks_verification),
        ("box_not_terminal", bool(box_spans and not final_box_is_last)),
        ("post_answer_loop", post_answer_loop),
        ("panic_preamble", panic_preamble),
    ]:
        if value:
            flags.append(name)

    return {
        "id": row.get("id"),
        "correct": bool(row.get("correct")),
        "format_ok": bool(row.get("format_ok")),
        "is_mcq": bool(row.get("is_mcq")),
        "is_multi": bool(row.get("is_multi")),
        "word_count": response_words,
        "wait_count": wait_count,
        "verify_count": verify_count,
        "panic_count": panic_count,
        "boxed_count": len(box_spans),
        "post_box_words": post_box_words if post_box_words is not None else "",
        "post_think_words": post_think_words,
        "bounded_verification_ok": ok,
        "flags": ";".join(flags),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    words = [int(row["word_count"]) for row in rows]
    waits = [int(row["wait_count"]) for row in rows]
    verifies = [int(row["verify_count"]) for row in rows]
    flag_counts = collections.Counter(
        flag for row in rows for flag in str(row.get("flags") or "").split(";") if flag
    )

    return {
        "n": n,
        "bounded_verification_ok": sum(row["bounded_verification_ok"] for row in rows),
        "bounded_verification_rate": ratio(sum(row["bounded_verification_ok"] for row in rows), n),
        "accuracy": ratio(sum(row["correct"] for row in rows), n),
        "format_ok_rate": ratio(sum(row["format_ok"] for row in rows), n),
        "avg_words": sum(words) / n if n else 0.0,
        "median_words": quantile(words, 0.5),
        "p90_words": quantile(words, 0.9),
        "avg_wait_count": sum(waits) / n if n else 0.0,
        "p90_wait_count": quantile(waits, 0.9),
        "avg_verify_count": sum(verifies) / n if n else 0.0,
        "flag_counts": dict(flag_counts.most_common()),
    }


def print_summary(name: str, summary: dict[str, Any]) -> None:
    n = int(summary["n"])
    ok = int(summary["bounded_verification_ok"])
    print(f"{name}")
    print(f"  rows: {n}")
    print(f"  bounded_verification_ok: {ok}/{n} ({summary['bounded_verification_rate']:.1%})")
    print(f"  accuracy: {summary['accuracy']:.1%}")
    print(f"  format_ok: {summary['format_ok_rate']:.1%}")
    print(
        "  words: "
        f"avg={summary['avg_words']:.1f}, "
        f"median={summary['median_words']}, "
        f"p90={summary['p90_words']}"
    )
    print(
        "  waits: "
        f"avg={summary['avg_wait_count']:.2f}, "
        f"p90={summary['p90_wait_count']}"
    )
    print(f"  verification markers avg: {summary['avg_verify_count']:.2f}")
    print("  top flags:")
    for flag, count in list(summary["flag_counts"].items())[:12]:
        print(f"    {flag}: {count}/{n} ({ratio(count, n):.1%})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Eval result JSONL files.")
    parser.add_argument("--max-words", type=int, default=1400)
    parser.add_argument("--max-waits", type=int, default=6)
    parser.add_argument("--max-boxes", type=int, default=1)
    parser.add_argument("--max-post-box-words", type=int, default=8)
    parser.add_argument("--min-verify-markers", type=int, default=1)
    parser.add_argument("--panic-scan-chars", type=int, default=600)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    all_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for path in args.paths:
        analyzed = [analyze_row(row, args) for row in load_jsonl(path)]
        summaries[str(path)] = summarize(analyzed)
        print_summary(str(path), summaries[str(path)])
        print()
        for row in analyzed:
            row = dict(row)
            row["path"] = str(path)
            all_rows.append(row)

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "path",
            "id",
            "correct",
            "format_ok",
            "is_mcq",
            "is_multi",
            "word_count",
            "wait_count",
            "verify_count",
            "panic_count",
            "boxed_count",
            "post_box_words",
            "post_think_words",
            "bounded_verification_ok",
            "flags",
        ]
        with args.out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Wrote row diagnostics to {args.out_csv}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summaries, indent=2, ensure_ascii=False))
        print(f"Wrote summary JSON to {args.out_json}")


if __name__ == "__main__":
    main()
