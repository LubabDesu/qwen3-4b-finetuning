#!/usr/bin/env python3
"""Build the cleaned Stage 1-2 SFT dataset from cached Stage 1 and Stage 0 data."""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path
from typing import Any


CLEAN_SOURCE_CAPS = {
    "numina_synthetic_math": 3000,
    "numina_orca_math": 2500,
    "numina_synthetic_amc": 800,
    "numina_gsm8k": 200,
    "openmathreasoning_easy": 300,
}

ANCHOR_SOURCE = "stage0_format_anchor"
ANCHOR_CAP = 1000
MIN_ACCEPTED_ROWS = 6000
PREFERRED_MIN_ROWS = 8000
PREFERRED_MAX_ROWS = 9300

NOISE_PHRASES = [
    "self-check",
    "self-assessment",
    "self assessment",
    "this question examines",
    "this problem examines",
    "this problem tests",
    "analysis:",
    "in summary",
    "to summarize",
    "key point",
    "key concept",
    "final answer:",
    "final answer is",
    "the final answer is",
    "the correct answer is",
    "correct answer:",
    "the answer is",
    "therefore, the correct choice is",
    "hence, the correct choice",
    "the correct option is",
    "correct option is",
    "correct option:",
    "knowledge point",
    "knowledge tested",
    "self-evaluation",
    "conclusion:",
]

MCQ_OPTION_RE = re.compile(
    r"(?im)(?:^|\n)\s*(?:[-*]\s*)?(?:\*\*)?\(?([A-J])\)?(?:\*\*)?\s*(?:[).:])"
)

MCQ_OPTION_LINE_RE = re.compile(
    r"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?\(?([A-J])\)?(?:\*\*)?\s*(?:[).:])\s*(.+?)\s*$"
)

MCQ_INLINE_LABEL_RE = re.compile(r"\\textbf\{\(?([A-J])\)?\}")
CHINESE_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")

EARLY_POISON_PHRASES = [
    "self-check",
    "let me evaluate each option",
    "option a:",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", str(text or "")))


def normalize_text_answer(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"\\(?:text|textbf|mathrm|mathbf)\{([^{}]*)\}", r"\1", text)
    return text.strip()


def normalize_for_option_match(text: str) -> str:
    text = normalize_text_answer(text)
    text = text.replace("$", "")
    text = re.sub(r"\\left|\\right", "", text)
    text = re.sub(r"\s+", "", text)
    text = text.strip(" .,:;")
    return text.lower()


def normalize_for_noise(text: str) -> str:
    text = str(text or "").lower()
    text = text.replace("*", "").replace("_", "")
    return text


def is_embedded_mcq(row: dict[str, Any]) -> bool:
    if row.get("is_mcq") or row.get("options"):
        return True
    question = str(row.get("question", ""))
    letters = [m.group(1).upper() for m in MCQ_OPTION_RE.finditer(question)]
    letters.extend(m.group(1).upper() for m in MCQ_INLINE_LABEL_RE.finditer(question))
    return len(set(letters)) >= 3 and "A" in letters and "B" in letters


def embedded_options(row: dict[str, Any]) -> dict[str, str]:
    if row.get("options"):
        return {
            chr(65 + idx): str(option).strip()
            for idx, option in enumerate(row.get("options") or [])
            if idx < 10
        }
    options: dict[str, str] = {}
    for match in MCQ_OPTION_LINE_RE.finditer(str(row.get("question", ""))):
        options[match.group(1).upper()] = match.group(2).strip()
    question = str(row.get("question", ""))
    inline_matches = list(MCQ_INLINE_LABEL_RE.finditer(question))
    for idx, match in enumerate(inline_matches):
        label = match.group(1).upper()
        end = inline_matches[idx + 1].start() if idx + 1 < len(inline_matches) else len(question)
        option_text = question[match.end() : end]
        option_text = option_text.replace(r"\qquad", " ").strip()
        if option_text:
            options[label] = option_text
    return options


def canonical_target_answer(row: dict[str, Any]) -> str:
    answer = normalize_text_answer(row.get("answer", ""))
    if is_embedded_mcq(row):
        match = re.match(r"\s*\(?([A-J])\)?(?:\s*[).:])?", answer)
        if match:
            return match.group(1).upper()
        match = re.search(r"\b([A-J])\s*[\).:]", answer)
        if match:
            return match.group(1).upper()
        match = re.fullmatch(r"\s*([A-J])\s*", answer)
        if match:
            return match.group(1).upper()
        normalized_answer = normalize_for_option_match(answer)
        for label, option_text in embedded_options(row).items():
            normalized_option = normalize_for_option_match(option_text)
            if normalized_answer and normalized_answer == normalized_option:
                return label
    return answer.strip()


def replace_boxed_with_inner(text: str) -> str:
    text = str(text or "")
    out: list[str] = []
    i = 0
    marker = r"\boxed{"
    while i < len(text):
        start = text.find(marker, i)
        if start < 0:
            out.append(text[i:])
            break
        out.append(text[i:start])
        j = start + len(marker)
        depth = 1
        while j < len(text) and depth:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            inner = text[start + len(marker) : j - 1]
            out.append(normalize_text_answer(inner))
            i = j
        else:
            out.append(text[start:])
            break
    return "".join(out)


def first_phrase_pos(text_lower: str, phrases: list[str], start_at: int = 0) -> tuple[int, str | None]:
    best_pos = -1
    best_phrase: str | None = None
    for phrase in phrases:
        pos = text_lower.find(phrase.lower(), start_at)
        if pos >= 0 and (best_pos < 0 or pos < best_pos):
            best_pos = pos
            best_phrase = phrase
    return best_pos, best_phrase


def truncate_at_noise(solution: str) -> tuple[str, str | None]:
    lower = solution.lower()
    best_pos = -1
    best_phrase: str | None = None
    for phrase in NOISE_PHRASES:
        if phrase == "conclusion:":
            match = re.search(r"\*{0,2}conclusion\*{0,2}\s*:", lower[100:])
            pos = 100 + match.start() if match else -1
        elif phrase in {"key point", "key concept"}:
            words = re.escape(phrase).replace(r"\ ", r"\s+")
            match = re.search(rf"\*{{0,2}}{words}\*{{0,2}}", lower[100:])
            pos = 100 + match.start() if match else -1
        else:
            pos = lower.find(phrase.lower(), 100)
        if pos >= 0 and (best_pos < 0 or pos < best_pos):
            best_pos = pos
            best_phrase = phrase
    if best_pos >= 0:
        return solution[:best_pos].strip(), best_phrase
    return solution.strip(), None


def validate_answer(answer: Any) -> tuple[bool, str]:
    answer_text = str(answer or "").strip()
    if not answer_text:
        return False, "empty_answer"
    if len(answer_text) >= 50:
        return False, "answer_too_long"
    if "\n" in answer_text or "\r" in answer_text:
        return False, "answer_has_newline"
    return True, ""


def clean_solution(raw_solution: str, min_words: int = 50) -> tuple[str | None, str, str | None]:
    solution = str(raw_solution or "").strip()
    if not solution:
        return None, "empty_solution", None
    if "<think>" in solution or "</think>" in solution:
        return None, "think_tags", None

    lower = normalize_for_noise(solution)
    early_pos, early_phrase = first_phrase_pos(lower[:100], EARLY_POISON_PHRASES)
    if early_pos >= 0:
        return None, f"early_noise:{early_phrase}", early_phrase

    solution = replace_boxed_with_inner(solution)
    solution, trunc_phrase = truncate_at_noise(solution)
    solution = re.sub(r"(?:^|\n)\s*\d+\.\s*$", "", solution).strip()
    solution = re.sub(r"[ \t]+", " ", solution)
    solution = re.sub(r"\n{3,}", "\n\n", solution).strip()

    wc = word_count(solution)
    if wc < min_words:
        return None, "too_short", trunc_phrase
    if CHINESE_CHAR_RE.search(solution):
        return None, "chinese_characters", trunc_phrase
    remaining_noise_pos, remaining_noise_phrase = first_phrase_pos(normalize_for_noise(solution), NOISE_PHRASES)
    if remaining_noise_pos >= 0:
        return None, f"noise_remains:{remaining_noise_phrase}", trunc_phrase
    if r"\boxed" in solution:
        return None, "boxed_remains", trunc_phrase
    if "<think>" in solution or "</think>" in solution:
        return None, "think_tags_after_clean", trunc_phrase

    return solution, "", trunc_phrase


def source_allowed(row: dict[str, Any]) -> bool:
    return str(row.get("source", "")) in CLEAN_SOURCE_CAPS


def clean_stage1_rows(
    raw_rows: list[dict[str, Any]],
    *,
    seed: int,
    min_words: int = 50,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    shuffled = list(raw_rows)
    rng.shuffle(shuffled)

    accepted_by_source: dict[str, list[dict[str, Any]]] = {source: [] for source in CLEAN_SOURCE_CAPS}
    drop_counts: collections.Counter[str] = collections.Counter()
    trunc_counts: collections.Counter[str] = collections.Counter()

    for row in shuffled:
        source = str(row.get("source", ""))
        if source == "math_qa_easy":
            drop_counts["math_qa_easy_dropped"] += 1
            continue
        if source not in CLEAN_SOURCE_CAPS:
            drop_counts[f"source_not_allowed:{source}"] += 1
            continue
        if len(accepted_by_source[source]) >= CLEAN_SOURCE_CAPS[source]:
            drop_counts[f"cap_reached:{source}"] += 1
            continue

        answer_ok, answer_reason = validate_answer(row.get("answer"))
        if not answer_ok:
            drop_counts[answer_reason] += 1
            continue

        if source == "openmathreasoning_easy" and word_count(row.get("reasoning", "")) >= 400:
            drop_counts["omr_over_400_preclean"] += 1
            continue

        cleaned, reason, trunc_phrase = clean_solution(str(row.get("reasoning", "")), min_words=min_words)
        if reason:
            drop_counts[reason] += 1
            continue
        if trunc_phrase:
            trunc_counts[trunc_phrase] += 1

        answer = str(row.get("answer", "")).strip()
        target_answer = canonical_target_answer(row)
        if not target_answer or len(target_answer) >= 50 or "\n" in target_answer or "\r" in target_answer:
            drop_counts["bad_target_answer"] += 1
            continue
        is_mcq = is_embedded_mcq(row)
        if is_mcq and not re.fullmatch(r"[A-J]", target_answer):
            drop_counts["bad_mcq_target_answer"] += 1
            continue
        cleaned_row = {
            "question": row.get("question", ""),
            "options": row.get("options"),
            "answer": answer,
            "target_answer": target_answer,
            "reasoning": cleaned,
            "source": source,
            "original_source": source,
            "n_ans": int(row.get("n_ans") or str(row.get("question", "")).count("[ANS]")),
            "is_mcq": is_mcq,
        }
        accepted_by_source[source].append(cleaned_row)

    records: list[dict[str, Any]] = []
    for source, cap in CLEAN_SOURCE_CAPS.items():
        rows = accepted_by_source[source][:cap]
        records.extend(rows)

    manifest = {
        "min_words": min_words,
        "source_caps": CLEAN_SOURCE_CAPS,
        "accepted_by_source": {source: len(rows) for source, rows in accepted_by_source.items()},
        "drop_counts": dict(drop_counts),
        "truncation_counts": dict(trunc_counts),
    }
    return records, manifest


def build_stage0_anchors(anchor_rows: list[dict[str, Any]], *, seed: int, n: int = ANCHOR_CAP) -> list[dict[str, Any]]:
    if not anchor_rows:
        raise ValueError("No Stage 0 anchor rows available")
    rng = random.Random(seed)
    shuffled = list(anchor_rows)
    rng.shuffle(shuffled)
    anchors: list[dict[str, Any]] = []
    for idx in range(n):
        base = dict(shuffled[idx % len(shuffled)])
        base["original_source"] = base.get("source", "")
        base["original_bucket"] = base.get("bucket", "")
        base["source"] = ANCHOR_SOURCE
        base["anchor_index"] = idx
        anchors.append(base)
    return anchors


def assert_clean_records(records: list[dict[str, Any]]) -> None:
    for idx, row in enumerate(records):
        if row.get("source") == ANCHOR_SOURCE:
            if not row.get("target"):
                raise AssertionError(f"anchor row {idx} missing target")
            continue
        solution = str(row.get("reasoning", ""))
        answer = str(row.get("answer", ""))
        target_answer = str(row.get("target_answer", answer))
        wc = word_count(solution)
        if r"\boxed" in solution:
            raise AssertionError(f"row {idx} still has boxed wrapper")
        if "<think>" in solution or "</think>" in solution:
            raise AssertionError(f"row {idx} still has think tags")
        if wc < 50:
            raise AssertionError(f"row {idx} word count below minimum: {wc}")
        if CHINESE_CHAR_RE.search(solution):
            raise AssertionError(f"row {idx} contains Chinese characters")
        if not (0 < len(answer) < 50):
            raise AssertionError(f"row {idx} bad answer length: {len(answer)}")
        if "\n" in answer or "\r" in answer:
            raise AssertionError(f"row {idx} answer has newline")
        if not (0 < len(target_answer) < 50):
            raise AssertionError(f"row {idx} bad target answer length: {len(target_answer)}")
        if "\n" in target_answer or "\r" in target_answer:
            raise AssertionError(f"row {idx} target answer has newline")


def build_stage1_2_clean_records_from_cache(
    *,
    raw_path: Path,
    anchor_path: Path,
    out_path: Path,
    manifest_path: Path | None = None,
    seed: int = 151,
    force_rebuild: bool = False,
) -> list[dict[str, Any]]:
    if out_path.exists() and not force_rebuild:
        return load_jsonl(out_path)

    raw_rows = load_jsonl(raw_path)
    clean_records, manifest = clean_stage1_rows(raw_rows, seed=seed)

    anchors = build_stage0_anchors(load_jsonl(anchor_path), seed=seed + 77, n=ANCHOR_CAP)
    records = clean_records + anchors
    rng = random.Random(seed + 123)
    rng.shuffle(records)

    assert_clean_records(records)

    source_counts = collections.Counter(str(row.get("source", "unknown")) for row in records)
    word_counts_by_source: dict[str, dict[str, int]] = {}
    for source in source_counts:
        if source == ANCHOR_SOURCE:
            continue
        counts = sorted(word_count(row.get("reasoning", "")) for row in records if row.get("source") == source)
        if counts:
            word_counts_by_source[source] = {
                "min": counts[0],
                "median": counts[len(counts) // 2],
                "max": counts[-1],
            }

    manifest.update(
        {
            "raw_path": str(raw_path),
            "anchor_path": str(anchor_path),
            "out_path": str(out_path),
            "preferred_row_range": [PREFERRED_MIN_ROWS, PREFERRED_MAX_ROWS],
            "minimum_row_count": MIN_ACCEPTED_ROWS,
            "total_records": len(records),
            "source_counts": dict(source_counts),
            "word_counts_by_source": word_counts_by_source,
            "assertions_passed": True,
        }
    )

    write_jsonl(records, out_path)
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    return records


def assistant_target(row: dict[str, Any]) -> str:
    if row.get("target"):
        return str(row["target"])
    final = row.get("target_answer", row.get("answer", ""))
    return f"<think>\n{row.get('reasoning', '').strip()}\n</think>\n\n\\boxed{{{final}}}"


def print_samples(records: list[dict[str, Any]], *, seed: int, per_source: int = 1, limit: int = 900) -> None:
    by_source: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in records:
        by_source[str(row.get("source", "unknown"))].append(row)
    for source in sorted(by_source):
        sample_rows = by_source[source][:]
        random.Random(seed + len(source)).shuffle(sample_rows)
        print("\n" + "=" * 100)
        print(f"{source}: showing {min(per_source, len(sample_rows))} / {len(sample_rows)}")
        for row in sample_rows[:per_source]:
            print("-" * 80)
            print(
                f"answer={row.get('answer')!r} target_answer={row.get('target_answer', row.get('answer'))!r} "
                f"n_ans={row.get('n_ans')} is_mcq={row.get('is_mcq')} original={row.get('original_source')}"
            )
            target = assistant_target(row)
            print(target[:limit] + (" ..." if len(target) > limit else ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-path", type=Path, default=Path("artifacts/post_training_curriculum/datasets/stage1_2_records.jsonl"))
    parser.add_argument("--anchor-path", type=Path, default=Path("artifacts/post_training_curriculum/datasets/stage0b_final_finetune.jsonl"))
    parser.add_argument("--out-path", type=Path, default=Path("artifacts/post_training_curriculum/datasets/stage1_2_clean_records.jsonl"))
    parser.add_argument("--manifest-path", type=Path, default=Path("artifacts/post_training_curriculum/datasets/stage1_2_clean_manifest.json"))
    parser.add_argument("--seed", type=int, default=151)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--print-samples", action="store_true")
    args = parser.parse_args()

    records = build_stage1_2_clean_records_from_cache(
        raw_path=args.raw_path,
        anchor_path=args.anchor_path,
        out_path=args.out_path,
        manifest_path=args.manifest_path,
        seed=args.seed,
        force_rebuild=args.force_rebuild,
    )
    manifest = json.loads(args.manifest_path.read_text())
    print(f"Built {len(records)} clean Stage 1-2 records: {args.out_path}")
    print("Source counts:", manifest["source_counts"])
    print("Accepted by source before anchors:", manifest["accepted_by_source"])
    print("Top drop counts:", dict(collections.Counter(manifest["drop_counts"]).most_common(20)))
    print("Word counts by source:", manifest["word_counts_by_source"])
    if args.print_samples:
        print_samples(records, seed=args.seed)


if __name__ == "__main__":
    main()
