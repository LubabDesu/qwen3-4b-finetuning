#!/usr/bin/env python3
"""Deterministically standardize final boxed answers in a submission CSV.

This postprocessor does not solve problems or invent answer values. It extracts
answer strings already present in the model response, chooses the most plausible
final answer serialization, and appends one final \boxed{...} answer.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for p in (ROOT, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from judger import Judger  # noqa: E402

BOX_RE = re.compile(r"\\boxed\s*\{")
FINAL_MARKER_RE = re.compile(
    r"(?:final\s+answers?|final\s+answer|answer\s*:|answers\s+are|therefore|thus|hence)",
    flags=re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True, help="JSONL with question/options for expected answer counts.")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--diff-jsonl", type=Path, default=None)
    parser.add_argument("--append-label", default="Final standardized answer:")
    parser.add_argument(
        "--conservative",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only apply high-confidence serialization repairs; skip risky plain-text/last-n guesses.",
    )
    parser.add_argument(
        "--only-shape-fixes",
        action="store_true",
        help="Only change rows where the original judger extraction count is not the expected count and the repaired count is expected.",
    )
    parser.add_argument(
        "--allowed-sources",
        default=None,
        help="Comma-separated recovery source allowlist, e.g. final_boxes:single,final_boxes:multi_dedupe,all_boxes:last_mcq.",
    )
    return parser.parse_args()


def load_questions(path: Path) -> dict[str, dict[str, Any]]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            out[str(row.get("id", row.get("question_id", idx)))] = row
    return out


def expected_count(row: dict[str, Any] | None) -> int:
    if not row:
        return 1
    question = str(row.get("question") or "")
    count = question.count("[ANS]")
    return max(1, count)


def is_mcq(row: dict[str, Any] | None, n_expected: int) -> bool:
    return bool(row and n_expected == 1 and isinstance(row.get("options"), list) and row.get("options"))


def boxed_spans(text: str) -> list[tuple[int, int, str]]:
    spans = []
    start = 0
    while True:
        m = BOX_RE.search(text, start)
        if not m:
            break
        i = m.end()
        depth = 1
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            spans.append((m.start(), i, text[m.end(): i - 1].strip()))
        start = max(i, m.end())
    return spans


def final_section(text: str) -> str:
    think_end = text.rfind("</think>")
    section = text[think_end + len("</think>"):] if think_end >= 0 else text
    matches = list(FINAL_MARKER_RE.finditer(section))
    if matches:
        return section[matches[-1].start():]
    # Last chunk is often enough and avoids earlier worked examples.
    return section[-2500:]


def top_level_split(text: str) -> list[str]:
    parts = []
    cur = []
    depth = 0
    for ch in str(text or ""):
        if ch in "{[<":
            depth += 1
        elif ch in "}]>" and depth > 0:
            depth -= 1
        if ch in ",;" and depth == 0:
            part = "".join(cur).strip()
            if part:
                parts.append(part)
            cur = []
        else:
            cur.append(ch)
    part = "".join(cur).strip()
    if part:
        parts.append(part)
    return parts


def normalize_answer_text(ans: str) -> str:
    ans = str(ans or "").strip()
    ans = re.sub(r"^\$+|\$+$", "", ans).strip()
    ans = ans.replace("\\left", "").replace("\\right", "")
    ans = ans.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    ans = re.sub(r"\\(?:text|mathrm|mathbf)\{([^{}]*)\}", r"\1", ans)
    ans = re.sub(r"\s+", " ", ans).strip()
    return ans.strip(" .。")


def normalize_mcq(ans: str) -> str:
    ans = normalize_answer_text(ans).strip().upper()
    m = re.match(r"^\(?\s*([A-J])\b", ans)
    return m.group(1) if m else ans


def dedupe_repeated_sequence(parts: list[str], n_expected: int) -> list[str]:
    if n_expected <= 0 or len(parts) <= n_expected or len(parts) % n_expected != 0:
        return parts
    first = [normalize_answer_text(p).replace(" ", "") for p in parts[:n_expected]]
    chunks = [parts[i:i+n_expected] for i in range(0, len(parts), n_expected)]
    if all([normalize_answer_text(p).replace(" ", "") for p in chunk] == first for chunk in chunks):
        return parts[:n_expected]
    return parts


def answer_variants(response: str, n_expected: int, mcq: bool, conservative: bool = True) -> list[tuple[str, str]]:
    variants = []

    def add(source: str, value: str) -> None:
        value = normalize_mcq(value) if mcq else normalize_answer_text(value)
        if value and (source, value) not in variants:
            variants.append((source, value))

    section = final_section(response)
    full_boxes = boxed_spans(response)
    sec_boxes = boxed_spans(section)

    for source, spans in (("final_boxes", sec_boxes), ("all_boxes", full_boxes)):
        contents = [normalize_answer_text(c) for _s, _e, c in spans if normalize_answer_text(c)]
        if not contents:
            continue
        if mcq:
            add(source + ":last_mcq", contents[-1])
            continue
        # Single final box may already contain all answers, or duplicated sequences.
        for content in reversed(contents):
            parts = dedupe_repeated_sequence(top_level_split(content), n_expected)
            if len(parts) == n_expected:
                add(source + ":single", ", ".join(parts))
                break
        # Multiple boxes: use last expected number of boxes.
        if n_expected > 1 and len(contents) >= n_expected:
            parts = dedupe_repeated_sequence(contents, n_expected)
            if len(parts) == n_expected:
                add(source + ":multi_dedupe", ", ".join(parts))
            if not conservative:
                add(source + ":last_n", ", ".join(contents[-n_expected:]))
        # Multiple boxes with repeated final group.
        if n_expected > 1:
            split_all = []
            for content in contents:
                split_all.extend(top_level_split(content))
            split_all = dedupe_repeated_sequence(split_all, n_expected)
            if len(split_all) == n_expected:
                add(source + ":split_dedupe", ", ".join(split_all))

    # Labelled final text fallback, e.g. A: 32 / B: 96 / C: 52.
    if not mcq and n_expected > 1:
        labeled = re.findall(
            r"(?:^|[\n\-•*]\s*)(?:part\s*)?(?:[A-Z]|\d+)\s*[\).:]\s*([^\n;]+)",
            section,
            flags=re.I,
        )
        cleaned = []
        for item in labeled:
            item = re.sub(r"^\s*(?:is|=)\s*", "", item.strip(), flags=re.I)
            item = re.split(r"\s{2,}|(?:\s+-\s+)", item)[0].strip()
            item = normalize_answer_text(item)
            if item:
                cleaned.append(item)
        if len(cleaned) >= n_expected:
            add("labeled:last_n", ", ".join(cleaned[-n_expected:]))

    # Plain final answer fallback after final marker.
    if not conservative and not variants and not mcq:
        m = re.search(r"(?:final\s+answers?|answers?\s*(?:are|:))\s*([^\n]+)", section, flags=re.I)
        if m:
            value = normalize_answer_text(m.group(1))
            if len(top_level_split(value)) == n_expected:
                add("plain_final", value)

    return variants


def extraction_count(judger: Judger, response: str) -> int:
    try:
        ans = judger.extract_ans(response)
        if not ans:
            return 0
        return len(judger.split_by_comma(ans))
    except Exception:
        return 0


def append_standard_box(response: str, answer: str, label: str) -> str:
    answer = normalize_answer_text(answer)
    return response.rstrip() + f"\n\n{label}\n\\boxed{{{answer}}}"


def main() -> None:
    args = parse_args()
    questions = load_questions(args.questions)
    judger = Judger(strict_extract=False)

    stats = {
        "rows": 0,
        "changed": 0,
        "shape_saved": 0,
        "already_good_shape": 0,
        "no_variant": 0,
        "source_counts": {},
    }
    diffs = []

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.diff_jsonl:
        args.diff_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with args.input_csv.open(newline="", encoding="utf-8") as in_f, args.output_csv.open("w", newline="", encoding="utf-8") as out_f:
        reader = csv.DictReader(in_f)
        writer = csv.DictWriter(out_f, fieldnames=reader.fieldnames or ["id", "response"], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in reader:
            stats["rows"] += 1
            rid = str(row.get("id", ""))
            q = questions.get(rid)
            n_expected = expected_count(q)
            mcq = is_mcq(q, n_expected)
            response = row.get("response", "")
            before_count = extraction_count(judger, response)
            variants = answer_variants(response, n_expected, mcq, conservative=args.conservative)
            if args.allowed_sources:
                allowed = {item.strip() for item in args.allowed_sources.split(",") if item.strip()}
                variants = [item for item in variants if item[0] in allowed]
            selected = variants[0] if variants else None
            new_response = response
            after_count = before_count
            if selected:
                source, answer = selected
                candidate_response = append_standard_box(response, answer, args.append_label)
                after_count = extraction_count(judger, candidate_response)
                is_shape_fix = before_count != n_expected and after_count == n_expected
                if (not args.only_shape_fixes) or is_shape_fix:
                    new_response = candidate_response
                    stats["source_counts"][source] = stats["source_counts"].get(source, 0) + 1
                    if new_response != response:
                        stats["changed"] += 1
                if is_shape_fix:
                    stats["shape_saved"] += 1
                    diffs.append({
                        "id": rid,
                        "expected_count": n_expected,
                        "before_count": before_count,
                        "after_count": after_count,
                        "source": source,
                        "answer": answer,
                        "old_tail": response[-800:],
                        "new_tail": candidate_response[-800:],
                    })
                elif before_count == n_expected:
                    stats["already_good_shape"] += 1
            else:
                stats["no_variant"] += 1
            out = dict(row)
            out["response"] = new_response
            writer.writerow(out)

    if args.diff_jsonl:
        with args.diff_jsonl.open("w", encoding="utf-8") as f:
            for diff in diffs:
                f.write(json.dumps(diff, ensure_ascii=False) + "\n")
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
