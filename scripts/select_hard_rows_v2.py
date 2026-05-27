#!/usr/bin/env python3
"""Select hard rows v2 from a canonical private submission."""
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
REASONING_TEXT_RE = re.compile(
    r"####|\b(step|calculate|margin of error|therefore|because)\b", re.I
)
BAD_EXTRACT_WORDS = {"", "none", "null", "n/a", "na"}
TIER2A_FLAGS = {"packed_separate_blanks", "modulo_suspicious", "formula_numeric_only"}
TIER2B_FLAGS = {"incomplete_unstable_reasoning"}



def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--submission",
        type=Path,
        default=REPO_ROOT / "artifacts/private_reasoning_paths/submission_draft_9_canonical.csv",
        help="CSV with id,response columns.",
    )
    ap.add_argument("--private", type=Path, default=REPO_ROOT / "competition-data/private.jsonl")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/private_reasoning_paths",
    )
    ap.add_argument("--prefix", default="hard_rows_v2")
    return ap.parse_args()


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


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\\boxed{", "").replace("boxed{", "")
    text = re.sub(r"[{}$]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" .,:;")


def strip_outer_wrappers(text: str) -> str:
    out = str(text or "").strip()
    changed = True
    while changed and len(out) >= 2:
        changed = False
        pairs = {"[": "]", "(": ")", "{": "}"}
        if out[0] in pairs and out[-1] == pairs[out[0]]:
            out = out[1:-1].strip()
            changed = True
    return out


def split_top_level(answer: str, flatten_outer: bool = False) -> list[str]:
    text = strip_outer_wrappers(answer) if flatten_outer else str(answer or "")
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for i, ch in enumerate(text):
        if ch in "([{" and (i == 0 or text[i - 1] != "\\"):
            depth += 1
        elif ch in ")]}" and (i == 0 or text[i - 1] != "\\") and depth > 0:
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


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def wait_count(text: str) -> int:
    return len(re.findall(r"\bwait\b", text or "", re.I))


def final_reasoning_tail(response: str, chars: int = 900) -> str:
    text = response or ""
    final_match = None
    for final_match in BOX_START_RE.finditer(text):
        pass
    if final_match:
        text = text[: final_match.start()]
    return text[-chars:].replace("\r", " ")


def has_options(row: dict[str, Any]) -> bool:
    options = row.get("options")
    return isinstance(options, list) and bool(options)


def option_value_to_letter(answer: str, options: list[Any]) -> str | None:
    normalized = normalize_text(answer).lower().replace("\\", "")
    if not normalized:
        return None
    for i, opt in enumerate(options):
        opt_norm = normalize_text(opt).lower().replace("\\", "")
        if normalized == opt_norm:
            return chr(ord("A") + i)
    return None


def mcq_valid_or_mappable(answer: str, options: list[Any]) -> bool:
    ans = normalize_text(answer)
    if not ans:
        return False
    if re.fullmatch(r"[A-J](?:\s*,\s*[A-J])*|[A-J]+", ans, re.I):
        return True
    return option_value_to_letter(ans, options) is not None


def numeric_value(answer: str) -> float | None:
    try:
        return float(str(answer).strip().replace(",", "").replace("%", ""))
    except Exception:
        return None


def requested_modulus(question: str) -> int | None:
    q = str(question or "").lower()
    patterns = [
        r"(?:remainder when|modulo|mod)\D{0,80}(\d+)",
        r"divided by\s+(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            return int(match.group(1))
    return None


def modulo_repair_maps_to_option(answer: str, question: str, options: list[Any]) -> bool:
    value = numeric_value(answer)
    modulus = requested_modulus(question)
    if value is None or modulus is None or abs(value - round(value)) > 1e-9:
        return False
    reduced = str(int(round(value)) % modulus)
    return option_value_to_letter(reduced, options) is not None


def has_unbalanced_braces(text: str) -> bool:
    depth = 0
    for i, ch in enumerate(text or ""):
        if ch == "{" and (i == 0 or text[i - 1] != "\\"):
            depth += 1
        elif ch == "}" and (i == 0 or text[i - 1] != "\\"):
            depth -= 1
            if depth < 0:
                return True
    return depth != 0


def corrupted_latex(text: str) -> bool:
    raw = str(text or "")
    if "\x08oxed" in raw:
        return True
    if "boxed{" in raw and "\\boxed{" not in raw:
        return True
    if has_unbalanced_braces(raw):
        return True
    if re.search(r"\\(?:frac|dfrac|tfrac)(?!\s*\{)", raw):
        return True
    if re.search(r"\\(?:frac|dfrac|tfrac)\s*\{[^{}]+\}\s*\{[^{}]+\}\s*[-+]?\d", raw):
        return True
    if re.search(r"\\sqrt(?!\s*(?:\{|\[))", raw):
        return True
    return False


def corrupted_boxed_syntax(text: str) -> bool:
    raw = str(text or "")
    return "\x08oxed" in raw or ("boxed{" in raw and "\\boxed{" not in raw)


def packed_separate_blank_question(question: str) -> bool:
    q = str(question or "")
    return bool(
        re.search(r"\[\s*\[ANS\]\s*,\s*\[ANS\]\s*\]", q, re.I)
        or re.search(r"\[ANS\]\s*<[^\n]{0,120}<\s*\[ANS\]", q, re.I)
        or re.search(r"from\s+\[ANS\]\s+to\s+\[ANS\]", q, re.I)
        or re.search(r"between\s+\[ANS\]\s+and\s+\[ANS\]", q, re.I)
        or ("Covariance=[ANS]" in q and "Correlation=[ANS]" in q)
    )


def answer_is_packed(answer: str) -> bool:
    text = str(answer or "").strip()
    if not re.fullmatch(r"[\[\(].+[\]\)]", text, re.S):
        return False
    return len(split_top_level(text, flatten_outer=True)) >= 2


def all_plain_numbers(parts: list[str]) -> bool:
    if not parts:
        return False
    return all(re.fullmatch(r"\s*-?\d+(?:\.\d+)?(?:/\d+)?%?\s*", p or "") for p in parts)


def incomplete_unstable_reasoning(response: str, tail: str, words: int, waits: int) -> bool:
    lower_tail = tail.lower()
    tail_unstable = bool(
        re.search(r"\b(wait|actually|reconsider|mistake|let me check|however)\b", lower_tail)
    )
    return (
        ("</think>" not in response and words > 2500 and waits > 20)
        or (words > 3000 and waits > 30)
        or tail_unstable
    )


def score_row(judger: Judger, qrow: dict[str, Any], response: str) -> dict[str, Any]:
    question = str(qrow.get("question") or "")
    options = qrow.get("options") if isinstance(qrow.get("options"), list) else []
    num_blanks = question.count("[ANS]")
    words = word_count(response)
    waits = wait_count(response)
    tail = final_reasoning_tail(response)
    flags: list[str] = []
    tiers: list[int] = []

    try:
        extracted = judger.extract_ans(response) or ""
    except Exception:
        extracted = ""

    extracted_norm = normalize_text(extracted)
    flattened_parts = split_top_level(extracted, flatten_outer=True)

    if (
        not extracted_norm
        or extracted_norm.lower() in BAD_EXTRACT_WORDS
        or word_count(extracted) > 20
        or REASONING_TEXT_RE.search(extracted)
    ):
        flags.append("bad_extract")
        tiers.append(1)

    if num_blanks > 1 and len(flattened_parts) < num_blanks:
        flags.append("underfilled_after_flatten")
        tiers.append(1)

    modulo_repairable = modulo_repair_maps_to_option(extracted, question, options) if has_options(qrow) else False

    if has_options(qrow) and not mcq_valid_or_mappable(extracted, options) and not modulo_repairable:
        flags.append("mcq_invalid_unmappable")
        tiers.append(1)

    if corrupted_latex(extracted) or corrupted_boxed_syntax(response):
        flags.append("corrupted_latex")
        tiers.append(1)

    if packed_separate_blank_question(question) and answer_is_packed(extracted):
        flags.append("packed_separate_blanks")
        tiers.append(2)

    if re.search(r"\b(remainder|mod|modulo|divided by|units digit)\b", question, re.I):
        if numeric_value(extracted) is not None:
            if has_options(qrow):
                option_values = {normalize_text(opt) for opt in options}
                if normalize_text(extracted) not in option_values:
                    flags.append("modulo_suspicious")
                    tiers.append(2)
            elif requested_modulus(question) is not None:
                flags.append("modulo_suspicious")
                tiers.append(2)

    if (
        num_blanks > 1
        and re.search(r"\b(formula|model|equation|function)\b", question, re.I)
        and all_plain_numbers(flattened_parts)
    ):
        flags.append("formula_numeric_only")
        tiers.append(2)

    if incomplete_unstable_reasoning(response, tail, words, waits):
        flags.append("incomplete_unstable_reasoning")
        tiers.append(2)

    if words > 3500:
        flags.append("very_long")
        tiers.append(3)

    if waits > 40:
        flags.append("many_waits")
        tiers.append(3)

    if not tiers:
        tier = 0
    elif 1 in tiers:
        tier = 1
    elif 2 in tiers:
        tier = 2
    else:
        tier = 3

    hard_score = sum(10 if t == 1 else 5 if t == 2 else 1 for t in tiers)

    missing_think = "</think>" not in response
    unstable_rank_score = words + waits + (1000 if missing_think else 0)

    return {
        "tier": tier,
        "hard_score": hard_score,
        "flags": ";".join(flags),
        "extracted": extracted,
        "num_blanks": num_blanks,
        "has_options": has_options(qrow),
        "word_count": words,
        "wait_count": waits,
        "missing_think": missing_think,
        "unstable_rank_score": unstable_rank_score,
        "response_tail": (response or "")[-1000:].replace("\r", " "),
        "question": question,
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "tier",
        "hard_score",
        "flags",
        "extracted",
        "num_blanks",
        "has_options",
        "word_count",
        "wait_count",
        "missing_think",
        "unstable_rank_score",
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
    tier_counts: Counter[int] = Counter()
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
            if int(scored["hard_score"]) > 0:
                out = {"id": qid, **scored}
                rows.append(out)
                tier_counts[int(scored["tier"])] += 1
                for flag in scored["flags"].split(";"):
                    if flag:
                        flag_counts[flag] += 1

    rows.sort(key=lambda r: (int(r["tier"]), -int(r["hard_score"]), int(r["id"])))
    tier1 = [r for r in rows if int(r["tier"]) == 1]
    tier2 = [r for r in rows if int(r["tier"]) == 2]
    tier3 = [r for r in rows if int(r["tier"]) == 3]

    def flag_set(row: dict[str, Any]) -> set[str]:
        return {f for f in str(row["flags"]).split(";") if f}

    tier2a = [r for r in tier2 if flag_set(r) & TIER2A_FLAGS]
    tier2b = [
        r
        for r in tier2
        if not (flag_set(r) & TIER2A_FLAGS) and flag_set(r) <= TIER2B_FLAGS
    ]
    tier2b.sort(key=lambda r: (-int(r["unstable_rank_score"]), -int(r["word_count"]), -int(r["wait_count"]), int(r["id"])))
    tier2b_top100 = tier2b[:100]
    regen_plan = sorted(tier1 + tier2a + tier2b_top100, key=lambda r: (int(r["tier"]), -int(r["hard_score"]), int(r["id"])))

    write_rows(args.out_dir / f"{args.prefix}_all.csv", rows)
    write_rows(args.out_dir / f"{args.prefix}_tier1.csv", tier1)
    write_rows(args.out_dir / f"{args.prefix}_tier2.csv", tier2)
    write_rows(args.out_dir / f"{args.prefix}_tier2a.csv", tier2a)
    write_rows(args.out_dir / f"{args.prefix}_tier2b.csv", tier2b)
    write_rows(args.out_dir / f"{args.prefix}_tier2b_top100.csv", tier2b_top100)
    write_rows(args.out_dir / f"{args.prefix}_tier3.csv", tier3)
    write_rows(args.out_dir / f"{args.prefix}_regen_plan_150ish.csv", regen_plan)

    print(f"total rows: {total}")
    print(f"flagged rows: {len(rows)}")
    print("count by tier:")
    for tier in (1, 2, 3):
        print(f"  tier{tier}: {tier_counts[tier]}")
    print(f"  tier2a: {len(tier2a)}")
    print(f"  tier2b: {len(tier2b)}")
    print(f"  tier2b_top100: {len(tier2b_top100)}")
    print(f"  regen_plan_150ish: {len(regen_plan)}")
    print("count by flag:")
    for flag, count in flag_counts.most_common():
        print(f"  {flag}: {count}")
    for suffix in ("all", "tier1", "tier2", "tier2a", "tier2b", "tier2b_top100", "tier3", "regen_plan_150ish"):
        print(f"wrote: {args.out_dir / (args.prefix + '_' + suffix + '.csv')}")


if __name__ == "__main__":
    main()
