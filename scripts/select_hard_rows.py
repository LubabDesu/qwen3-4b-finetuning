#!/usr/bin/env python3
"""Select hard rows from a private submission for targeted regeneration."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from judger import Judger  # noqa: E402

BOX_START_RE = re.compile(r"\\boxed\s*\{")
LETTER_RE = re.compile(r"\b[A-J]\b", re.I)
NONE_WORDS = {"", "none", "null", "n/a", "na"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--submission",
        type=Path,
        default=REPO_ROOT / "artifacts/private_reasoning_paths/submission_draft_8_reasoning_restored.csv",
        help="CSV with id,response columns.",
    )
    ap.add_argument("--private", type=Path, default=REPO_ROOT / "competition-data/private.jsonl")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT)
    ap.add_argument("--prefix", default="hard_rows")
    return ap.parse_args()


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\\boxed{", "").replace("boxed{", "")
    text = text.replace("\\", "")
    text = re.sub(r"[{}$]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" .,:;()[]")


def read_private(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for fallback, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            qid = int(row.get("id", row.get("question_id", fallback)))
            rows[qid] = row
    return rows


def split_top_level(answer: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for i, ch in enumerate(answer or ""):
        if ch in "([{" and (i == 0 or answer[i - 1] != "\\"):
            depth += 1
        elif ch in ")]}" and (i == 0 or answer[i - 1] != "\\") and depth > 0:
            depth -= 1
        if ch in ",;" and depth == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
        else:
            buf.append(ch)
    part = "".join(buf).strip()
    if part:
        parts.append(part)
    return parts


def extract_boxed_contents_after_think(response: str) -> list[str]:
    think_end = response.rfind("</think>")
    text = response[think_end + len("</think>") :] if think_end >= 0 else response
    out: list[str] = []
    pos = 0
    while True:
        m = BOX_START_RE.search(text, pos)
        if not m:
            break
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
                    out.append(text[start:i].strip())
                    pos = i + 1
                    break
            i += 1
        else:
            break
    return out


def option_value_to_letter(answer: str, options: list[Any]) -> str | None:
    na = normalize_text(answer).lower()
    if not na:
        return None
    for i, opt in enumerate(options):
        if na == normalize_text(opt).lower():
            return chr(ord("A") + i)
    return None


def mcq_maps_to_valid(answer: str, options: list[Any]) -> bool:
    ans = normalize_text(answer)
    if not ans:
        return False
    letters = re.findall(r"(?<![A-Za-z])([A-J])(?![A-Za-z])", ans, re.I)
    if letters:
        return True
    return option_value_to_letter(ans, options) is not None


def ends_with_clean_boxed(response: str) -> bool:
    tail = response.strip()
    # Allow trailing punctuation/quotes after final closing brace, but no prose.
    return re.search(r"\\boxed\s*\{[^{}]*\}\s*$", tail, re.S) is not None


def answer_is_random_sentence(answer: str) -> bool:
    stripped = normalize_text(answer)
    if not stripped:
        return True
    words = re.findall(r"[A-Za-z]+", stripped)
    if len(words) > 12:
        return True
    if len(stripped) > 220:
        return True
    return False


def question_allows_none(question: str) -> bool:
    return bool(re.search(r"\b(no solution|none|does not exist|impossible|no such|undefined)\b", question, re.I))


def numeric_only(answer: str) -> bool:
    return bool(re.fullmatch(r"\s*-?\d+(?:\.\d+)?(?:/\d+)?%?\s*", answer or ""))


def packed_interval_mismatch(question: str, answer: str, n_ans: int) -> bool:
    if n_ans < 2:
        return False
    q = question.lower()
    has_separate_bounds = bool(
        re.search(r"\[ans\]\s*<[^\n]{0,120}<\s*\[ans\]", q)
        or re.search(r"\[\s*\[ans\]\s*,\s*\[ans\]\s*\]", q)
        or re.search(r"from\s+\[ans\]\s+to\s+\[ans\]", q)
        or re.search(r"between\s+\[ans\]\s+and\s+\[ans\]", q)
    )
    if not has_separate_bounds:
        return False
    ans = (answer or "").strip()
    if re.fullmatch(r"[\[\(].+,.+[\]\)]", ans) and len(split_top_level(ans)) == 1:
        return True
    return False


def score_row(judger: Judger, qrow: dict[str, Any], response: str) -> dict[str, Any]:
    question = qrow.get("question") or ""
    options = qrow.get("options")
    has_options = isinstance(options, list) and bool(options)
    n_ans = question.count("[ANS]")
    word_count = len(re.findall(r"\S+", response or ""))
    wait_count = len(re.findall(r"\bwait\b", response or "", re.I))
    flags: list[str] = []
    score = 0

    try:
        extracted = judger.extract_ans(response) or ""
    except Exception:
        extracted = ""

    extracted_norm = normalize_text(extracted)
    parts = split_top_level(extracted)

    if answer_is_random_sentence(extracted):
        flags.append("bad_extract")
        score += 10

    if n_ans > 0 and len(parts) != n_ans:
        flags.append("component_mismatch")
        score += 10

    if has_options and not mcq_maps_to_valid(extracted, options):
        flags.append("mcq_invalid")
        score += 9

    if "</think>" not in response and not ends_with_clean_boxed(response):
        flags.append("clipped_no_final")
        score += 8

    if ("boxed{" in response and "\\boxed{" not in response) or "\x08oxed" in response:
        flags.append("malformed_box")
        score += 8

    if extracted_norm.lower() in NONE_WORDS and not question_allows_none(question):
        flags.append("none_invalid")
        score += 7

    if re.search(r"\b(formula|model|equation|function|regression|line|polynomial)\b", question, re.I) and numeric_only(extracted):
        flags.append("formula_numeric_only")
        score += 5

    if re.search(r"\b(remainder|mod|modulo|divided by|units digit|last digit)\b", question, re.I) and numeric_only(extracted):
        # Numeric may be correct; suspicious mostly when trace is long enough to contain intermediates.
        flags.append("modulo_suspicious")
        score += 5

    if packed_interval_mismatch(question, extracted, n_ans):
        flags.append("packed_interval_mismatch")
        score += 5

    boxed_after = extract_boxed_contents_after_think(response)
    if len(set(normalize_text(x) for x in boxed_after if normalize_text(x))) > 1:
        flags.append("multiple_conflicting_boxes")
        score += 4

    if word_count > 2500:
        flags.append("very_long")
        score += 2

    if wait_count > 20:
        flags.append("many_waits")
        score += 1

    return {
        "hard_score": score,
        "reason_flags": ";".join(flags),
        "extracted": extracted,
        "num_ans_blanks": n_ans,
        "has_options": has_options,
        "word_count": word_count,
        "wait_count": wait_count,
        "response_tail": (response or "")[-1000:].replace("\r", " "),
        "question": question,
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "hard_score",
        "reason_flags",
        "extracted",
        "num_ans_blanks",
        "has_options",
        "word_count",
        "wait_count",
        "response_tail",
        "question",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    private = read_private(args.private)
    judger = Judger()
    rows: list[dict[str, Any]] = []
    flag_counts: Counter[str] = Counter()
    total = 0

    with args.submission.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["id", "response"]:
            raise SystemExit(f"expected id,response columns, got {reader.fieldnames}")
        for row in reader:
            total += 1
            qid = int(row["id"])
            if qid not in private:
                raise SystemExit(f"submission id {qid} not found in private file")
            scored = score_row(judger, private[qid], row["response"] or "")
            if scored["hard_score"] > 0:
                out = {"id": qid, **scored}
                rows.append(out)
                for flag in scored["reason_flags"].split(";"):
                    if flag:
                        flag_counts[flag] += 1

    rows.sort(key=lambda r: (-int(r["hard_score"]), int(r["id"])))
    tier1 = [r for r in rows if int(r["hard_score"]) >= 8]
    tier2 = [r for r in rows if 5 <= int(r["hard_score"]) < 8]

    write_rows(args.out_dir / f"{args.prefix}_all.csv", rows)
    write_rows(args.out_dir / f"{args.prefix}_tier1.csv", tier1)
    write_rows(args.out_dir / f"{args.prefix}_tier2.csv", tier2)

    print(f"total rows: {total}")
    print(f"flagged rows: {len(rows)}")
    print(f"tier1 count: {len(tier1)}")
    print(f"tier2 count: {len(tier2)}")
    print("count by flag:")
    for flag, count in flag_counts.most_common():
        print(f"  {flag}: {count}")
    print(f"wrote: {args.out_dir / (args.prefix + '_all.csv')}")
    print(f"wrote: {args.out_dir / (args.prefix + '_tier1.csv')}")
    print(f"wrote: {args.out_dir / (args.prefix + '_tier2.csv')}")


if __name__ == "__main__":
    main()
