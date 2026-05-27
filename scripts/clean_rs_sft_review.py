#!/usr/bin/env python3
"""Clean reviewed GRPO rejection-sampled SFT traces.

The cleaner treats the source SFT JSONL files as the source of truth and uses
the pretty review JSON only as advisory metadata. It writes a new cleaned SFT
JSONL plus readable review/report artifacts without mutating the inputs.
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from judger import Judger  # noqa: E402


DEFAULT_REVIEW = ROOT / "artifacts/grpo_rs_sft/reviews/ckpt500_combined_dedup_review.pretty.json"
DEFAULT_SFT_INPUTS = [
    ROOT / "artifacts/grpo_rs_sft/sft/ckpt500_mixed500.jsonl",
    ROOT / "artifacts/grpo_rs_sft/sft/ckpt500_public_train300.jsonl",
]
DEFAULT_CLEANED_JSONL = ROOT / "artifacts/grpo_rs_sft/sft/ckpt500_combined_review_cleaned.jsonl"
DEFAULT_CLEANED_PRETTY = ROOT / "artifacts/grpo_rs_sft/reviews/ckpt500_combined_review_cleaned.pretty.json"
DEFAULT_DROPPED_PRETTY = ROOT / "artifacts/grpo_rs_sft/reviews/ckpt500_combined_review_dropped.pretty.json"
DEFAULT_MANUAL_PRETTY = ROOT / "artifacts/grpo_rs_sft/reviews/ckpt500_combined_review_needs_manual.pretty.json"
DEFAULT_REPORT = ROOT / "artifacts/grpo_rs_sft/reviews/ckpt500_combined_review_cleanup_report.md"


PROMPT_LEAKAGE_PATTERNS = [
    r"\bthe user wants\b",
    r"\buser wants\b",
    r"\bthe user asked\b",
    r"\bthe prompt asks\b",
    r"\bfinal answer rules\b",
    r"\bdo not write any text after\b",
    r"\bI should output\b",
    r"\bI need to output\b",
    r"\bboxed answer should be\b",
]

USER_KEEP_DECISIONS = {"keep", "accept", "accept, change gold", "accept change gold"}

BAD_SUFFIX_PATTERNS = [
    r"^\s*Wait,?\s+",
    r"^\s*Let me check again\b",
    r"^\s*Let'?s check again\b",
    r"^\s*The key (?:was|is)\b",
    r"^\s*I don'?t see any mistakes\b",
    r"^\s*I think that'?s it\b",
    r"^\s*That seems (?:right|correct)\b",
    r"^\s*So (?:the )?(?:final )?answer (?:is|should be)\b",
    r"^\s*(?:The )?(?:final )?answer (?:is|should be)\b",
    r"^\s*So the boxed answer\b",
    r"^\s*But the user wants\b",
    r"^\s*The answer should be boxed\b",
]

INCOMPLETE_TAIL_PATTERNS = [
    r"(?:^|\n)\s*(?:So\s+)?(?:the\s+)?(?:final\s+)?answer\s+(?:should\s+be|is)\s*[\.:]?\s*$",
    r"(?:^|\n)\s*So\s+the\s*$",
    r"(?:^|\n)\s*The\s*$",
    r"(?:^|\n)\s*Wait,?\s*$",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-path", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--sft-input", type=Path, action="append", default=None)
    parser.add_argument("--cleaned-jsonl", type=Path, default=DEFAULT_CLEANED_JSONL)
    parser.add_argument("--cleaned-pretty", type=Path, default=DEFAULT_CLEANED_PRETTY)
    parser.add_argument("--dropped-pretty", type=Path, default=DEFAULT_DROPPED_PRETTY)
    parser.add_argument("--manual-pretty", type=Path, default=DEFAULT_MANUAL_PRETTY)
    parser.add_argument(
        "--manual-label-path",
        type=Path,
        default=DEFAULT_MANUAL_PRETTY,
        help="Optional prior manual-review JSON whose labels should override the broad review file.",
    )
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict-extract", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wait-trim-threshold", type=int, default=8)
    parser.add_argument("--wait-manual-threshold", type=int, default=12)
    parser.add_argument("--wait-severe-threshold", type=int, default=20)
    parser.add_argument("--simple-wait-threshold", type=int, default=5)
    parser.add_argument("--max-final-answer-words", type=int, default=12)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_review_fields(record: dict[str, Any]) -> dict[str, Any]:
    raw_review = record.get("review")
    review_obj: dict[str, Any] = {}
    decision = ""
    notes = record.get("review_notes")

    if isinstance(raw_review, dict):
        review_obj = raw_review
        decision = str(
            raw_review.get("decision")
            or raw_review.get("label")
            or raw_review.get("verdict")
            or ""
        ).strip().lower()
        notes = raw_review.get("notes", raw_review.get("note", notes))
    elif isinstance(raw_review, str):
        decision = raw_review.strip().lower()
        review_obj = {"decision": decision}

    if not decision:
        raw_decision = record.get("review_decision")
        if isinstance(raw_decision, str):
            decision = raw_decision.strip().lower()

    return {"review": review_obj, "decision": decision, "notes": notes}


def load_review_metadata(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if not path.exists():
        return {}, {"loaded": False, "error": "missing"}
    text = path.read_text()
    repaired = re.sub(r",(\s*[}\]])", r"\1", text)
    try:
        data = json.loads(repaired)
    except json.JSONDecodeError as exc:
        return {}, {"loaded": False, "error": str(exc)}

    meta_by_id: dict[str, dict[str, Any]] = {}
    for record in data.get("records", []):
        row_id = str(record.get("id") or "")
        if not row_id:
            continue
        normalized_review = normalize_review_fields(record)
        meta_by_id[row_id] = {
            "readable_index": record.get("readable_index"),
            "run": record.get("run"),
            "selected_answer": record.get("selected_answer"),
            "style_flags": record.get("style_flags") or {},
            "review": normalized_review["review"],
            "decision": normalized_review["decision"],
            "notes": normalized_review["notes"],
        }
    return meta_by_id, {"loaded": True, "records": len(meta_by_id)}


def merge_review_metadata(
    base: dict[str, dict[str, Any]],
    override: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    merged = {row_id: dict(meta) for row_id, meta in base.items()}
    for row_id, meta in override.items():
        decision = str(meta.get("decision") or "").strip().lower()
        notes = meta.get("notes")
        current = merged.setdefault(row_id, {})
        current.update({k: v for k, v in meta.items() if v is not None})
    return merged


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = str(row.get("id") or "")
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        deduped.append(row)
    return deduped


def normalize_gold(row: dict[str, Any]) -> list[str]:
    raw = row.get("gold_answer", row.get("gold", row.get("answer", [])))
    if isinstance(raw, list):
        return [str(item).strip() for item in raw]
    return [str(raw).strip()]


def normalize_options(row: dict[str, Any]) -> list[str]:
    options = row.get("options") or []
    if isinstance(options, list):
        return [str(item) for item in options]
    return []


def extract_boxed_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    start = 0
    while True:
        idx = text.find(r"\boxed{", start)
        if idx < 0:
            break
        pos = idx + len(r"\boxed{")
        depth = 1
        chars: list[str] = []
        while pos < len(text) and depth:
            ch = text[pos]
            if ch == "{":
                depth += 1
                chars.append(ch)
            elif ch == "}":
                depth -= 1
                if depth:
                    chars.append(ch)
            else:
                chars.append(ch)
            pos += 1
        if depth != 0:
            break
        spans.append((idx, pos, "".join(chars).strip()))
        start = pos
    return spans


def remove_boxed(text: str) -> str:
    spans = extract_boxed_spans(text)
    if not spans:
        return text
    pieces: list[str] = []
    last = 0
    for start, end, _ in spans:
        pieces.append(text[last:start])
        last = end
    pieces.append(text[last:])
    return "".join(pieces)


def split_target(target: str) -> tuple[str, str, str] | None:
    before, sep, after = target.partition("</think>")
    if not sep:
        return None
    final_boxes = extract_boxed_spans(after)
    if not final_boxes:
        return None
    start, end, answer = final_boxes[-1]
    if after[end:].strip():
        return None
    reasoning = before.replace("<think>", "", 1).strip()
    return reasoning, answer.strip(), after[:start].strip()


def canonical_completion(reasoning: str, answer: str) -> str:
    return f"<think>\n{reasoning.strip()}\n</think>\n\n\\boxed{{{answer.strip()}}}"


def normalize_spaces(text: str) -> str:
    text = re.sub(r"<\|im_(?:start|end)\|>", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_incomplete_tail(text: str) -> tuple[str, bool]:
    out = text.rstrip()
    changed = False
    while True:
        new = out
        for pattern in INCOMPLETE_TAIL_PATTERNS:
            new = re.sub(pattern, "", new, flags=re.I).rstrip()
        if new == out:
            return out, changed
        out = new
        changed = True


def wait_count(text: str) -> int:
    return len(re.findall(r"\bwait\b", text, flags=re.I))


def reasoning_word_count(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", text))


def has_prompt_leakage(text: str) -> bool:
    return any(re.search(pattern, text, flags=re.I) for pattern in PROMPT_LEAKAGE_PATTERNS)


def is_yes_no_gold(gold: list[str]) -> str | None:
    if len(gold) != 1:
        return None
    raw = gold[0].strip()
    value = raw.lower()
    if value == "yes":
        return "Yes"
    if value == "no":
        return "No"
    if value == "true":
        return "True"
    if value == "false":
        return "False"
    return None


def answer_boolish(answer: str) -> str | None:
    value = answer.strip().lower()
    value = value.strip("$ .,:;")
    if value in {"yes", "true", "1"}:
        return "Yes"
    if value in {"no", "false", "0"}:
        return "No"
    return None


def yes_no_semantic(surface: str | None) -> str | None:
    if surface is None:
        return None
    return answer_boolish(surface)


def is_simple_row(row: dict[str, Any], yes_no: str | None) -> bool:
    if yes_no:
        return True
    question = str(row.get("question") or "")
    if row.get("is_mcq") or row.get("options"):
        return False
    if len(normalize_gold(row)) != 1:
        return False
    simple_markers = [
        r"\bsolve\b",
        r"\bcalculate\b",
        r"\bfind\b",
        r"\bwhat is\b",
        r"\bsimplify\b",
        r"\bprobability\b",
    ]
    return reasoning_word_count(question) <= 80 and any(re.search(p, question, flags=re.I) for p in simple_markers)


def answer_mentions_in_paragraph(paragraph: str, answer: str, yes_no: str | None) -> bool:
    lower = paragraph.lower()
    if yes_no:
        if yes_no == "Yes":
            patterns = [
                r"\b(?:answer|boxed)\b.{0,60}\byes\b",
                r"\b(?:therefore|so|hence).{0,100}\b(?:exists|true|holds|possible|valid)\b",
                r"\b(?:such a function|there) (?:does )?exists?\b",
                r"\bthe statement is true\b",
                r"\bthis is true\b",
            ]
        else:
            patterns = [
                r"\b(?:answer|boxed)\b.{0,60}\b(?:no|false)\b",
                r"\b(?:therefore|so|hence).{0,100}\b(?:does not exist|not true|false|fails|invalid)\b",
                r"\bthe statement is false\b",
                r"\bthis is false\b",
                r"\bdoes not exist\b",
            ]
        return any(re.search(pattern, lower) for pattern in patterns)
    if re.search(r"\b(?:answer|boxed)\b.{0,80}\b(?:is|should be)\b", lower):
        return True
    answer_norm = answer.strip().strip("$")
    if answer_norm and answer_norm in paragraph:
        return True
    plain = answer_norm.replace("\\frac", "frac").replace("{", "").replace("}", "")
    return bool(plain and plain in paragraph.replace("\\frac", "frac").replace("{", "").replace("}", ""))


def reasoning_supports_yes_no(reasoning: str, expected: str) -> bool:
    lower = reasoning.lower()
    paragraphs = [p.strip().lower() for p in re.split(r"\n\s*\n", reasoning) if p.strip()]
    tail = "\n\n".join(paragraphs[-4:]) if paragraphs else lower
    if expected == "Yes":
        negative_patterns = [
            r"\banswer is no\b",
            r"\bconclusion is no\b",
            r"\bhence,? the answer is no\b",
            r"\btherefore,? the answer is no\b",
            r"\bnot in the set\b",
            r"\bdoes not exist\b",
            r"\bis not .*integrable\b",
            r"\bthe statement is false\b",
        ]
        positive_patterns = [
            r"\banswer is yes\b",
            r"\bconclusion is yes\b",
            r"\btherefore,? (?:the answer is )?yes\b",
            r"\bso,? (?:the answer is )?yes\b",
            r"\bsuch (?:a )?.* exists\b",
            r"\bthere exists\b",
            r"\bis lebesgue integrable\b",
            r"\bis integrable\b",
            r"\bthe statement is true\b",
        ]
    else:
        negative_patterns = [
            r"\banswer is yes\b",
            r"\bconclusion is yes\b",
            r"\btherefore,? (?:the answer is )?yes\b",
            r"\bthere exists\b",
            r"\bis integrable\b",
            r"\bthe statement is true\b",
        ]
        positive_patterns = [
            r"\banswer is no\b",
            r"\bconclusion is no\b",
            r"\bhence,? the answer is no\b",
            r"\btherefore,? the answer is no\b",
            r"\bdoes not exist\b",
            r"\bis not .*integrable\b",
            r"\bthe statement is false\b",
        ]
    if any(re.search(pattern, tail) for pattern in negative_patterns):
        return False
    return any(re.search(pattern, tail) for pattern in positive_patterns)


def is_bad_suffix_paragraph(paragraph: str) -> bool:
    return any(re.search(pattern, paragraph, flags=re.I) for pattern in BAD_SUFFIX_PATTERNS) or has_prompt_leakage(paragraph)


def truncate_suffix(reasoning: str, answer: str, yes_no: str | None, force: bool) -> tuple[str, bool, str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", reasoning) if p.strip()]
    if len(paragraphs) <= 2:
        return reasoning, False, ""

    first_answer_idx: int | None = None
    for idx, paragraph in enumerate(paragraphs):
        if answer_mentions_in_paragraph(paragraph, answer, yes_no):
            first_answer_idx = idx
            break

    leakage_idx: int | None = None
    for idx, paragraph in enumerate(paragraphs):
        if has_prompt_leakage(paragraph):
            leakage_idx = idx
            break

    cut_idx: int | None = None
    reason = ""
    if leakage_idx is not None:
        cut_idx = leakage_idx
        reason = "prompt_leakage_suffix"

    if first_answer_idx is not None:
        checks_allowed = 1
        for idx in range(first_answer_idx + 1, len(paragraphs)):
            paragraph = paragraphs[idx]
            if is_bad_suffix_paragraph(paragraph):
                candidate_cut = idx
                if cut_idx is None or candidate_cut < cut_idx:
                    cut_idx = candidate_cut
                    reason = "post_answer_loop"
                break
            if re.search(r"\b(check|verify|confirm|mistake|correct|again)\b", paragraph, flags=re.I):
                checks_allowed -= 1
                if checks_allowed < 0:
                    candidate_cut = idx
                    if cut_idx is None or candidate_cut < cut_idx:
                        cut_idx = candidate_cut
                        reason = "extra_verification_suffix"
                    break

    if cut_idx is None and force and first_answer_idx is not None and len(paragraphs) - first_answer_idx > 3:
        cut_idx = first_answer_idx + 2
        reason = "forced_wait_suffix"

    if cut_idx is None and force:
        midpoint = max(1, len(paragraphs) // 2)
        for idx in range(midpoint, len(paragraphs)):
            if is_bad_suffix_paragraph(paragraphs[idx]):
                cut_idx = idx
                reason = "late_bad_suffix"
                break

    if cut_idx is None or cut_idx <= 0:
        return reasoning, False, ""

    truncated = "\n\n".join(paragraphs[:cut_idx]).strip()
    if reasoning_word_count(truncated) < 40:
        return reasoning, False, ""
    return truncated, truncated != reasoning.strip(), reason


def final_answer_corrupted(answer: str, max_words: int) -> str | None:
    if not answer.strip():
        return "empty_final_answer"
    lowered = answer.lower()
    if any(token in lowered for token in ["</think>", "\\boxed", "user wants", "answer should", "\n"]):
        return "final_box_contains_reasoning_or_markup"
    if reasoning_word_count(answer) > max_words:
        return "final_answer_too_wordy"
    return None


def judge_correct(judger: Judger, completion: str, gold: list[str], options: list[str]) -> bool:
    option_payload = [options for _ in gold]
    try:
        return bool(judger.auto_judge(pred=completion, gold=list(gold), options=option_payload))
    except Exception:
        return False


def clean_one(
    row: dict[str, Any],
    review_meta: dict[str, Any],
    judger: Judger,
    args: argparse.Namespace,
    readable_index: int,
) -> dict[str, Any]:
    row_id = str(row.get("id") or "")
    gold = normalize_gold(row)
    options = normalize_options(row)
    target = str(row.get("target") or "")
    parsed = split_target(target)
    actions: list[str] = []
    flags: list[str] = []
    decision = review_meta.get("decision", "")

    base_record = {
        "readable_index": review_meta.get("readable_index") or readable_index,
        "id": row_id,
        "original_source": row.get("original_source"),
        "original_id": row.get("original_id"),
        "question_type": (row.get("rs_sft") or {}).get("question_type"),
        "is_mcq": bool(row.get("is_mcq") or row.get("options")),
        "question": row.get("question"),
        "options": options,
        "gold_answer": gold,
        "review_decision": decision or None,
        "review_notes": review_meta.get("notes"),
    }

    if decision == "drop":
        return {
            "status": "drop",
            "reason": "user_review_drop",
            "record": {**base_record, "source_target": target},
        }

    if parsed is None:
        return {
            "status": "manual",
            "reason": "target_not_safely_parseable",
            "record": {**base_record, "source_target": target},
        }

    reasoning, answer, _ = parsed
    reasoning = remove_boxed(reasoning)
    reasoning = normalize_spaces(reasoning)
    reasoning, stripped_tail = strip_incomplete_tail(reasoning)
    if stripped_tail:
        actions.append("stripped_incomplete_tail")

    corrupted = final_answer_corrupted(answer, args.max_final_answer_words)
    if decision in {"unsure", "missing target", "missing_target"}:
        reason = "user_review_unsure" if decision == "unsure" else "user_review_missing_target"
        return {
            "status": "manual",
            "reason": reason,
            "record": {**base_record, "boxed_answer": answer, "target": target},
        }

    if corrupted:
        if corrupted == "final_box_contains_reasoning_or_markup":
            return {"status": "drop", "reason": corrupted, "record": {**base_record, "boxed_answer": answer}}
        return {
            "status": "manual",
            "reason": corrupted,
            "record": {**base_record, "boxed_answer": answer, "target": target},
        }

    yes_no = is_yes_no_gold(gold)
    yes_no_expected = yes_no_semantic(yes_no)
    original_answer = answer
    if yes_no:
        boolish = answer_boolish(answer)
        if boolish == yes_no_expected:
            answer = yes_no
            if answer != original_answer:
                actions.append("canonicalized_yes_no")
        elif boolish is None and answer.strip().lower() == yes_no.lower():
            answer = yes_no
        elif reasoning_supports_yes_no(reasoning, yes_no_expected or yes_no):
            answer = yes_no
            actions.append("rescued_yes_no_from_reasoning")
        else:
            return {
                "status": "manual",
                "reason": "suspicious_yes_no_answer",
                "record": {
                    **base_record,
                    "boxed_answer": original_answer,
                    "canonical_gold": yes_no,
                    "source_target": target,
                },
            }

    wc = wait_count(reasoning)
    simple = is_simple_row(row, yes_no)
    force_trim = (
        wc >= args.wait_trim_threshold
        or (simple and wc >= args.simple_wait_threshold)
        or decision in {"edit", "truncate", "edit/drop"}
        or has_prompt_leakage(reasoning)
    )
    severe_wait = wc >= args.wait_severe_threshold
    high_wait = wc >= args.wait_manual_threshold

    cleaned_reasoning, truncated, truncate_reason = truncate_suffix(reasoning, answer, yes_no, force_trim)
    if truncated:
        reasoning = cleaned_reasoning
        actions.append("truncated_suffix")
        if truncate_reason:
            actions.append(truncate_reason)

    if has_prompt_leakage(reasoning):
        return {
            "status": "manual" if not severe_wait else "drop",
            "reason": "unresolved_prompt_leakage",
            "record": {**base_record, "wait_count": wc, "target": canonical_completion(reasoning, answer)},
        }

    if extract_boxed_spans(reasoning):
        return {
            "status": "manual",
            "reason": "boxed_answer_remains_inside_reasoning",
            "record": {**base_record, "target": canonical_completion(reasoning, answer)},
        }

    if high_wait and not truncated:
        flags.append("high_wait_untrimmed")
    if severe_wait and not truncated:
        return {
            "status": "manual",
            "reason": "severe_wait_without_safe_truncation",
            "record": {**base_record, "wait_count": wc, "target": canonical_completion(reasoning, answer)},
        }

    cleaned = canonical_completion(reasoning, answer)
    final_boxes = extract_boxed_spans(cleaned.split("</think>", 1)[1])
    if len(final_boxes) != 1 or cleaned.split("</think>", 1)[1][final_boxes[-1][1] :].strip():
        return {"status": "manual", "reason": "canonical_format_validation_failed", "record": base_record}

    user_accepted = decision in USER_KEEP_DECISIONS
    if not user_accepted and not judge_correct(judger, cleaned, gold, options):
        return {
            "status": "manual",
            "reason": "judger_rejected_cleaned_target",
            "record": {**base_record, "boxed_answer": answer, "target": cleaned},
        }
    if user_accepted:
        actions.append("user_review_accept_change_gold" if "change gold" in decision else "user_review_keep")

    clean_row = copy.deepcopy(row)
    clean_row["answer"] = answer
    clean_row["target_answer"] = answer
    clean_row["reasoning"] = reasoning
    clean_row["target"] = cleaned
    if "change gold" in decision:
        clean_row["gold_answer"] = [answer]
    clean_row["rs_sft_cleanup"] = {
        "source": "clean_rs_sft_review.py",
        "readable_index": base_record["readable_index"],
        "actions": actions,
        "flags": flags,
        "original_boxed_answer": original_answer,
        "wait_count_before": wc,
        "wait_count_after": wait_count(reasoning),
        "reasoning_words_after": reasoning_word_count(reasoning),
        "review_decision": decision or None,
    }

    review_record = {
        **base_record,
        "selected_answer": answer,
        "boxed_answer_in_target": answer,
        "cleanup": clean_row["rs_sft_cleanup"],
        "target": cleaned,
    }
    return {"status": "clean", "reason": "", "row": clean_row, "record": review_record}


def bucket_wait_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    buckets = {"0-4": 0, "5-7": 0, "8-11": 0, "12-19": 0, "20+": 0}
    for record in records:
        wc = int(((record.get("cleanup") or {}).get("wait_count_before")) or record.get("wait_count") or 0)
        if wc < 5:
            buckets["0-4"] += 1
        elif wc < 8:
            buckets["5-7"] += 1
        elif wc < 12:
            buckets["8-11"] += 1
        elif wc < 20:
            buckets["12-19"] += 1
        else:
            buckets["20+"] += 1
    return buckets


def report_examples(records: list[dict[str, Any]], limit: int = 8) -> list[str]:
    return [
        f"- {record.get('readable_index')}: `{record.get('id')}`"
        for record in records[:limit]
    ]


def build_report(
    *,
    source_count: int,
    deduped_count: int,
    cleaned: list[dict[str, Any]],
    dropped: list[dict[str, Any]],
    manual: list[dict[str, Any]],
    review_status: dict[str, Any],
    dry_run: bool,
) -> str:
    clean_records = [item["record"] for item in cleaned]
    action_counts: collections.Counter[str] = collections.Counter()
    for item in cleaned:
        for action in item["record"].get("cleanup", {}).get("actions", []):
            action_counts[action] += 1
    drop_reasons = collections.Counter(item["reason"] for item in dropped)
    manual_reasons = collections.Counter(item["reason"] for item in manual)
    high_wait = [
        item["record"]
        for item in cleaned
        if (item["record"].get("cleanup") or {}).get("wait_count_before", 0) >= 12
    ]
    yes_no = [item["record"] for item in cleaned if "canonicalized_yes_no" in item["record"].get("cleanup", {}).get("actions", [])]
    truncated = [item["record"] for item in cleaned if "truncated_suffix" in item["record"].get("cleanup", {}).get("actions", [])]
    leakage = [
        item["record"]
        for item in cleaned
        if any("prompt_leakage" in action for action in item["record"].get("cleanup", {}).get("actions", []))
    ]

    lines = [
        "# ckpt500 RS-SFT Cleanup Report",
        "",
        f"Mode: {'dry-run' if dry_run else 'write'}",
        f"Review metadata: {review_status}",
        "",
        "## Counts",
        "",
        f"- Source rows: {source_count}",
        f"- Deduped input rows: {deduped_count}",
        f"- Cleaned rows: {len(cleaned)}",
        f"- Dropped rows: {len(dropped)}",
        f"- Manual review rows: {len(manual)}",
        f"- Truncated rows: {action_counts.get('truncated_suffix', 0)}",
        f"- Yes/no canonicalizations: {action_counts.get('canonicalized_yes_no', 0)}",
        f"- Prompt-leakage rescues: {len(leakage)}",
        f"- Prompt-leakage drops/manual: {drop_reasons.get('unresolved_prompt_leakage', 0) + manual_reasons.get('unresolved_prompt_leakage', 0)}",
        "",
        "## Wait Buckets",
        "",
    ]
    for bucket, count in bucket_wait_counts(clean_records).items():
        lines.append(f"- {bucket}: {count}")

    lines.extend(["", "## Cleaned Action Counts", ""])
    for action, count in action_counts.most_common():
        lines.append(f"- {action}: {count}")

    lines.extend(["", "## Drop Reasons", ""])
    for reason, count in drop_reasons.most_common():
        lines.append(f"- {reason}: {count}")
    if not drop_reasons:
        lines.append("- none")

    lines.extend(["", "## Manual Reasons", ""])
    for reason, count in manual_reasons.most_common():
        lines.append(f"- {reason}: {count}")
    if not manual_reasons:
        lines.append("- none")

    example_groups = [
        ("Truncated Examples", truncated),
        ("Yes/No Canonicalized Examples", yes_no),
        ("High-Wait Kept Examples", high_wait),
        ("Dropped Examples", [item["record"] for item in dropped]),
        ("Manual Examples", [item["record"] for item in manual]),
    ]
    for title, records in example_groups:
        lines.extend(["", f"## {title}", ""])
        lines.extend(report_examples(records) or ["- none"])

    return "\n".join(lines) + "\n"


def validate_cleaned_rows(rows: list[dict[str, Any]], judger: Judger) -> list[str]:
    errors: list[str] = []
    for idx, row in enumerate(rows, start=1):
        target = str(row.get("target") or "")
        parsed = split_target(target)
        if parsed is None:
            errors.append(f"{idx}:{row.get('id')}: target parse failed")
            continue
        reasoning, _, _ = parsed
        if extract_boxed_spans(reasoning):
            errors.append(f"{idx}:{row.get('id')}: boxed answer inside reasoning")
        after = target.split("</think>", 1)[1]
        boxes = extract_boxed_spans(after)
        if len(boxes) != 1:
            errors.append(f"{idx}:{row.get('id')}: final box count {len(boxes)}")
        elif after[boxes[-1][1] :].strip():
            errors.append(f"{idx}:{row.get('id')}: text after final box")
        cleanup_actions = (row.get("rs_sft_cleanup") or {}).get("actions", [])
        if any(action in cleanup_actions for action in ["user_review_keep", "user_review_accept_change_gold"]):
            continue
        if not judge_correct(judger, target, normalize_gold(row), normalize_options(row)):
            errors.append(f"{idx}:{row.get('id')}: judger rejected")
    return errors


def main() -> None:
    args = parse_args()
    input_paths = [resolve_path(path) for path in (args.sft_input or DEFAULT_SFT_INPUTS)]
    source_rows: list[dict[str, Any]] = []
    for path in input_paths:
        source_rows.extend(load_jsonl(path))
    rows = dedupe_rows(source_rows)

    review_meta, review_status = load_review_metadata(resolve_path(args.review_path))
    manual_label_path = resolve_path(args.manual_label_path)
    if manual_label_path.exists() and manual_label_path != resolve_path(args.review_path):
        manual_meta, manual_status = load_review_metadata(manual_label_path)
        for meta in manual_meta.values():
            if not str(meta.get("decision") or "").strip():
                meta["decision"] = "keep"
                meta["notes"] = meta.get("notes") or "unlabelled in manual review file; treated as keep"
        review_meta = merge_review_metadata(review_meta, manual_meta)
        review_status = {
            **review_status,
            "manual_label_path": str(manual_label_path.relative_to(ROOT)),
            "manual_label_records": manual_status.get("records", 0),
        }
    judger = Judger(strict_extract=args.strict_extract)

    cleaned: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    manual: list[dict[str, Any]] = []
    clean_jsonl_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        row_id = str(row.get("id") or "")
        result = clean_one(row, review_meta.get(row_id, {}), judger, args, idx)
        if result["status"] == "clean":
            cleaned.append({"reason": result["reason"], "record": result["record"]})
            clean_jsonl_rows.append(result["row"])
        elif result["status"] == "drop":
            dropped.append({"reason": result["reason"], "record": result["record"]})
        else:
            manual.append({"reason": result["reason"], "record": result["record"]})

    validation_errors = validate_cleaned_rows(clean_jsonl_rows, judger)
    if validation_errors:
        raise SystemExit("cleaned row validation failed:\n" + "\n".join(validation_errors[:50]))

    report = build_report(
        source_count=len(source_rows),
        deduped_count=len(rows),
        cleaned=cleaned,
        dropped=dropped,
        manual=manual,
        review_status=review_status,
        dry_run=args.dry_run,
    )

    print(report)
    if args.dry_run:
        return

    write_jsonl(resolve_path(args.cleaned_jsonl), clean_jsonl_rows)
    write_json(resolve_path(args.cleaned_pretty), {
        "summary": {
            "record_count": len(cleaned),
            "source_rows": len(source_rows),
            "deduped_input_rows": len(rows),
        },
        "records": [item["record"] for item in cleaned],
    })
    write_json(resolve_path(args.dropped_pretty), {
        "summary": {"record_count": len(dropped)},
        "records": [{**item["record"], "drop_reason": item["reason"]} for item in dropped],
    })
    write_json(resolve_path(args.manual_pretty), {
        "summary": {"record_count": len(manual)},
        "records": [{**item["record"], "manual_reason": item["reason"]} for item in manual],
    })
    resolve_path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
    resolve_path(args.report_path).write_text(report)


if __name__ == "__main__":
    main()
