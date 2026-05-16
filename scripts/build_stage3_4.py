#!/usr/bin/env python3
"""Build the Stage 3-4 hard free-form math SFT dataset.

Sources:
- zwhe99/DeepMath-103K, difficulty >= 5, shortest clean r1_solution trace.
- simplescaling/s1K-1.1, verified DeepSeek trace first, Gemini fallback.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from datasets import load_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_stage1_2_r2 import (  # noqa: E402
    CHINESE_CHAR_RE,
    DEFAULT_TOKENIZER_NAME_OR_PATH,
    R2_NOISE_PHRASES,
    SYSTEM_PROMPT,
    build_user_problem,
    elapsed_seconds,
    ends_abruptly,
    extract_all_boxed,
    load_tokenizer_for_filter,
    make_assistant_content,
    normalize_text_answer,
    rendered_token_length,
    replace_boxed_with_inner,
    strip_outer_think,
    strip_trailing_boxed,
    word_count,
)
from scripts.export_stage1_2_review_json import build_review_rows  # noqa: E402


DEFAULT_OUT_PATH = Path("artifacts/post_training_curriculum/datasets/stage3_4_records.jsonl")
DEFAULT_MANIFEST_PATH = Path("artifacts/post_training_curriculum/datasets/stage3_4_manifest.json")
DEFAULT_REVIEW_PATH = Path("artifacts/post_training_curriculum/datasets/stage3_4_records_for_review.json")

DEEPMATH_SOURCE = "deepmath_103k"
S1K_SOURCE = "s1k_1_1"
DEFAULT_DEEPMATH_CAP = 15000
DEFAULT_S1K_CAP = 1000
DEFAULT_MIN_DEEPMATH_DIFFICULTY = 5.0
DEFAULT_MAX_RENDERED_TOKENS = 10000
DEFAULT_MIN_REASONING_WORDS = 50

MCQ_MARKERS_RE = re.compile(
    r"(?is)(?:answer choices?|multiple choice|choose the correct|which of the following|"
    r"\b[A-H]\s*[).]\s+.*\b[B-H]\s*[).]\s+)"
)
MCQ_INLINE_OPTION_RE = re.compile(r"(?is)(?:^|\s)\(?[A-H]\)?[).]\s+")
LETTER_ANSWER_RE = re.compile(r"^\(?[A-H]\)?$")


def log(message: str) -> None:
    print(f"[stage3_4] {message}", flush=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_review(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(build_review_rows(list(rows)), f, ensure_ascii=False, indent=2)


def preview_field(value: Any, limit: int = 180) -> str:
    text = str(value or "").replace("\n", "\\n").replace("\r", "\\r")
    return text if len(text) <= limit else text[:limit] + "..."


def looks_like_mcq(question: Any, answer: Any = "") -> bool:
    text = str(question or "")
    if MCQ_MARKERS_RE.search(text):
        return True
    if len(MCQ_INLINE_OPTION_RE.findall(text.replace("\n", " "))) >= 3:
        return True
    return bool(LETTER_ANSWER_RE.fullmatch(str(answer or "").strip().upper()))


def normalize_final_answer(value: Any) -> tuple[str | None, str]:
    text = normalize_text_answer(value)
    boxes = [box.strip() for box in extract_all_boxed(text) if box.strip()]
    if boxes:
        text = boxes[-1]
    text = re.sub(r"\\boxed\s*\{\s*\}", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .")
    if not text:
        return None, "empty_answer"
    if CHINESE_CHAR_RE.search(text):
        return None, "chinese_answer"
    if "\n" in text or "\r" in text:
        return None, "multiline_answer"
    if len(text) > 240:
        return None, "answer_too_long"
    return text, ""


def clean_reasoning_trace(value: Any, *, min_words: int = DEFAULT_MIN_REASONING_WORDS) -> tuple[str | None, str]:
    text = strip_outer_think(str(value or ""))
    text = strip_trailing_boxed(text)
    text = replace_boxed_with_inner(text)
    text = re.sub(r"</?think>\s*", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if not text:
        return None, "empty_reasoning"
    if "<think>" in text or "</think>" in text:
        return None, "think_tags"
    if r"\boxed" in text:
        return None, "boxed_remains"
    if CHINESE_CHAR_RE.search(text):
        return None, "chinese_reasoning"
    if any(phrase in text.lower() for phrase in R2_NOISE_PHRASES):
        return None, "noise_phrase"
    if word_count(text) < min_words:
        return None, "too_short"
    if ends_abruptly(text):
        return None, "abrupt_tail"
    return text, ""


def choose_shortest_clean_trace(
    traces: Sequence[Any],
    drop_counts: collections.Counter[str],
    *,
    source_prefix: str,
    min_words: int,
) -> tuple[str | None, str]:
    candidates: list[tuple[int, str]] = []
    reasons: collections.Counter[str] = collections.Counter()
    for trace in traces:
        cleaned, reason = clean_reasoning_trace(trace, min_words=min_words)
        if cleaned:
            candidates.append((word_count(cleaned), cleaned))
        else:
            reasons[reason] += 1
    if not candidates:
        for reason, count in reasons.items():
            drop_counts[f"{source_prefix}:{reason}"] += count
        return None, "no_clean_trace"
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1], ""


def validate_record(row: dict[str, Any], *, min_words: int) -> str | None:
    question = str(row.get("question", ""))
    reasoning = str(row.get("reasoning", ""))
    answer = str(row.get("target_answer", row.get("answer", "")))
    rendered = make_assistant_content(reasoning, answer)

    if not question.strip():
        return "empty_question"
    if CHINESE_CHAR_RE.search(question):
        return "chinese_question"
    if looks_like_mcq(question, answer):
        return "mcq_like"
    if CHINESE_CHAR_RE.search(rendered):
        return "chinese_rendered_target"
    if rendered.count("<think>") != 1 or rendered.count("</think>") != 1:
        return "bad_think_tag_count"
    if rendered.count(r"\boxed{") != 1 or not rendered.rstrip().endswith(f"\\boxed{{{answer}}}"):
        return "bad_boxed_structure"
    if word_count(reasoning) < min_words:
        return "reasoning_too_short"
    if r"\boxed" in reasoning:
        return "boxed_in_reasoning"
    return None


def maybe_add_rendered_tokens(
    record: dict[str, Any],
    tokenizer: Any | None,
    max_rendered_tokens: int | None,
) -> tuple[dict[str, Any] | None, str]:
    if tokenizer is None or not max_rendered_tokens:
        return record, ""
    try:
        length = rendered_token_length(tokenizer, record)
    except Exception as exc:  # pragma: no cover - depends on tokenizer/runtime
        return None, f"rendered_tokenize_error:{type(exc).__name__}"
    if length > max_rendered_tokens:
        return None, "rendered_too_long"
    record = dict(record)
    record["rendered_tokens"] = length
    return record, ""


def load_deepmath_records(
    seed: int,
    drop_counts: collections.Counter[str],
    *,
    cap: int | None,
    min_difficulty: float,
    tokenizer: Any | None,
    max_rendered_tokens: int | None,
    min_words: int,
    log_every: int,
) -> list[dict[str, Any]]:
    start = time.monotonic()
    log("loading DeepMath split: zwhe99/DeepMath-103K train")
    ds = load_dataset("zwhe99/DeepMath-103K", split="train")
    rows = list(ds)
    random.Random(seed).shuffle(rows)
    log(f"loaded DeepMath rows={len(rows)} in {elapsed_seconds(start)}; shuffling with seed={seed}")

    accepted: list[dict[str, Any]] = []
    target_label = "uncapped" if cap is None else str(cap)
    scan_start = time.monotonic()
    for seen, row in enumerate(rows, 1):
        if cap is not None and len(accepted) >= cap:
            break
        if log_every > 0 and seen % log_every == 0:
            log(
                "DeepMath progress "
                f"seen={seen}/{len(rows)} accepted={len(accepted)}/{target_label} "
                f"drops={sum(v for k, v in drop_counts.items() if k.startswith('deepmath:'))} "
                f"elapsed={elapsed_seconds(scan_start)}"
            )

        try:
            difficulty = float(row.get("difficulty"))
        except (TypeError, ValueError):
            drop_counts["deepmath:bad_difficulty"] += 1
            continue
        if difficulty < min_difficulty:
            drop_counts["deepmath:difficulty_too_low"] += 1
            continue

        question = str(row.get("question", "")).strip()
        answer, reason = normalize_final_answer(row.get("final_answer"))
        if reason:
            drop_counts[f"deepmath:{reason}"] += 1
            continue
        if looks_like_mcq(question, answer):
            drop_counts["deepmath:mcq_like"] += 1
            continue
        if CHINESE_CHAR_RE.search(question):
            drop_counts["deepmath:chinese_question"] += 1
            continue

        traces = [row.get("r1_solution_1"), row.get("r1_solution_2"), row.get("r1_solution_3")]
        reasoning, reason = choose_shortest_clean_trace(
            traces,
            drop_counts,
            source_prefix="deepmath",
            min_words=min_words,
        )
        if reason:
            drop_counts[f"deepmath:{reason}"] += 1
            continue

        record = {
            "question": question,
            "options": None,
            "answer": answer,
            "target_answer": answer,
            "reasoning": reasoning,
            "source": DEEPMATH_SOURCE,
            "original_source": f"deepmath:{row.get('topic', 'unknown')}",
            "n_ans": question.count("[ANS]"),
            "is_mcq": False,
            "difficulty": difficulty,
            "topic": row.get("topic"),
        }
        record, reason = maybe_add_rendered_tokens(record, tokenizer, max_rendered_tokens)
        if reason:
            drop_counts[f"deepmath:{reason}"] += 1
            continue
        accepted.append(record)

    log(f"finished DeepMath accepted={len(accepted)} elapsed={elapsed_seconds(scan_start)}")
    return accepted


def s1k_trace_and_attempt(row: dict[str, Any]) -> tuple[str | None, str | None, str]:
    deepseek_grade = str(row.get("deepseek_grade", "")).strip().lower()
    gemini_grade = str(row.get("gemini_grade", "")).strip().lower()
    if deepseek_grade == "yes":
        return row.get("deepseek_thinking_trajectory"), row.get("deepseek_attempt"), "deepseek"
    if deepseek_grade == "no" and gemini_grade == "yes":
        return row.get("gemini_thinking_trajectory"), row.get("gemini_attempt"), "gemini"
    return None, None, "unverified"


def load_s1k_records(
    seed: int,
    drop_counts: collections.Counter[str],
    *,
    cap: int | None,
    tokenizer: Any | None,
    max_rendered_tokens: int | None,
    min_words: int,
    log_every: int,
) -> list[dict[str, Any]]:
    start = time.monotonic()
    log("loading s1K split: simplescaling/s1K-1.1 train")
    ds = load_dataset("simplescaling/s1K-1.1", split="train")
    rows = list(ds)
    random.Random(seed).shuffle(rows)
    log(f"loaded s1K rows={len(rows)} in {elapsed_seconds(start)}; shuffling with seed={seed}")

    accepted: list[dict[str, Any]] = []
    target_label = "uncapped" if cap is None else str(cap)
    scan_start = time.monotonic()
    for seen, row in enumerate(rows, 1):
        if cap is not None and len(accepted) >= cap:
            break
        if log_every > 0 and seen % log_every == 0:
            log(
                "s1K progress "
                f"seen={seen}/{len(rows)} accepted={len(accepted)}/{target_label} "
                f"drops={sum(v for k, v in drop_counts.items() if k.startswith('s1k:'))} "
                f"elapsed={elapsed_seconds(scan_start)}"
            )

        trace, attempt, trace_source = s1k_trace_and_attempt(row)
        if trace_source == "unverified":
            drop_counts["s1k:unverified"] += 1
            continue

        question = str(row.get("question", "")).strip()
        boxes = [box.strip() for box in extract_all_boxed(str(attempt or "")) if box.strip()]
        if not boxes:
            drop_counts["s1k:no_boxed_attempt_answer"] += 1
            continue
        answer, reason = normalize_final_answer(boxes[-1])
        if reason:
            drop_counts[f"s1k:{reason}"] += 1
            continue
        if looks_like_mcq(question, answer):
            drop_counts["s1k:mcq_like"] += 1
            continue
        if CHINESE_CHAR_RE.search(question):
            drop_counts["s1k:chinese_question"] += 1
            continue

        reasoning, reason = clean_reasoning_trace(trace, min_words=min_words)
        if reason:
            drop_counts[f"s1k:{trace_source}:{reason}"] += 1
            continue

        record = {
            "question": question,
            "options": None,
            "answer": answer,
            "target_answer": answer,
            "reasoning": reasoning,
            "source": S1K_SOURCE,
            "original_source": f"s1k:{trace_source}",
            "n_ans": question.count("[ANS]"),
            "is_mcq": False,
            "s1k_trace_source": trace_source,
        }
        record, reason = maybe_add_rendered_tokens(record, tokenizer, max_rendered_tokens)
        if reason:
            drop_counts[f"s1k:{reason}"] += 1
            continue
        accepted.append(record)

    log(f"finished s1K accepted={len(accepted)} elapsed={elapsed_seconds(scan_start)}")
    return accepted


def build_stage3_4_records(
    *,
    out_path: Path = DEFAULT_OUT_PATH,
    manifest_path: Path | None = DEFAULT_MANIFEST_PATH,
    review_path: Path | None = DEFAULT_REVIEW_PATH,
    seed: int = 151,
    force_rebuild: bool = False,
    tokenizer_name_or_path: str | None = DEFAULT_TOKENIZER_NAME_OR_PATH,
    max_rendered_tokens: int | None = DEFAULT_MAX_RENDERED_TOKENS,
    deepmath_cap: int | None = DEFAULT_DEEPMATH_CAP,
    s1k_cap: int | None = DEFAULT_S1K_CAP,
    min_deepmath_difficulty: float = DEFAULT_MIN_DEEPMATH_DIFFICULTY,
    min_reasoning_words: int = DEFAULT_MIN_REASONING_WORDS,
    dry_run: bool = False,
    log_every: int = 1000,
) -> list[dict[str, Any]]:
    build_start = time.monotonic()
    log(
        "build requested "
        f"out_path={out_path} force_rebuild={force_rebuild} "
        f"tokenizer={tokenizer_name_or_path} max_rendered_tokens={max_rendered_tokens} "
        f"deepmath_cap={deepmath_cap if deepmath_cap is not None else 'uncapped'} "
        f"s1k_cap={s1k_cap if s1k_cap is not None else 'uncapped'} "
        f"min_deepmath_difficulty={min_deepmath_difficulty} dry_run={dry_run}"
    )
    if out_path.exists() and not force_rebuild:
        records = load_jsonl(out_path)
        log(f"loaded cached records={len(records)} from {out_path}")
        return records

    drop_counts: collections.Counter[str] = collections.Counter()
    tokenizer = load_tokenizer_for_filter(tokenizer_name_or_path) if tokenizer_name_or_path and max_rendered_tokens else None

    deepmath_records = load_deepmath_records(
        seed,
        drop_counts,
        cap=deepmath_cap,
        min_difficulty=min_deepmath_difficulty,
        tokenizer=tokenizer,
        max_rendered_tokens=max_rendered_tokens,
        min_words=min_reasoning_words,
        log_every=log_every,
    )
    s1k_records = load_s1k_records(
        seed + 17,
        drop_counts,
        cap=s1k_cap,
        tokenizer=tokenizer,
        max_rendered_tokens=max_rendered_tokens,
        min_words=min_reasoning_words,
        log_every=log_every,
    )

    records = deepmath_records + s1k_records
    random.Random(seed + 123).shuffle(records)

    validation_examples: list[dict[str, Any]] = []
    valid_records: list[dict[str, Any]] = []
    for idx, row in enumerate(records):
        reason = validate_record(row, min_words=min_reasoning_words)
        if reason:
            drop_counts[f"validation:{reason}"] += 1
            if len(validation_examples) < 20:
                validation_examples.append(
                    {
                        "row_index": idx,
                        "source": row.get("source"),
                        "reason": reason,
                        "question_preview": preview_field(row.get("question")),
                        "answer_preview": preview_field(row.get("target_answer", row.get("answer"))),
                    }
                )
            continue
        valid_records.append(row)
    records = valid_records

    source_counts = collections.Counter(str(row.get("source", "unknown")) for row in records)
    word_counts_by_source: dict[str, dict[str, int]] = {}
    token_counts_by_source: dict[str, dict[str, int]] = {}
    for source in source_counts:
        words = sorted(word_count(row.get("reasoning", "")) for row in records if row.get("source") == source)
        if words:
            word_counts_by_source[source] = {
                "min": words[0],
                "median": words[len(words) // 2],
                "p95": words[int(0.95 * (len(words) - 1))],
                "max": words[-1],
            }
        tokens = sorted(
            int(row["rendered_tokens"])
            for row in records
            if row.get("source") == source and row.get("rendered_tokens") is not None
        )
        if tokens:
            token_counts_by_source[source] = {
                "min": tokens[0],
                "median": tokens[len(tokens) // 2],
                "p95": tokens[int(0.95 * (len(tokens) - 1))],
                "max": tokens[-1],
            }

    manifest = {
        "total_records": len(records),
        "source_counts": dict(source_counts),
        "requested_source_caps": {
            DEEPMATH_SOURCE: deepmath_cap,
            S1K_SOURCE: s1k_cap,
        },
        "accepted_before_validation": {
            DEEPMATH_SOURCE: len(deepmath_records),
            S1K_SOURCE: len(s1k_records),
        },
        "min_deepmath_difficulty": min_deepmath_difficulty,
        "min_reasoning_words": min_reasoning_words,
        "max_rendered_tokens": max_rendered_tokens,
        "drop_counts": dict(drop_counts),
        "validation_error_examples": validation_examples,
        "word_counts_by_source": word_counts_by_source,
        "rendered_token_counts_by_source": token_counts_by_source,
        "system_prompt": SYSTEM_PROMPT,
        "out_path": str(out_path),
        "review_path": str(review_path) if review_path else None,
        "dry_run": dry_run,
    }

    if dry_run:
        log(f"dry run enabled; not writing records to {out_path}")
    else:
        log(f"writing {len(records)} records to {out_path}")
        write_jsonl(records, out_path)
        if review_path is not None:
            log(f"writing review JSON to {review_path}")
            write_review(records, review_path)
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        log(f"writing manifest to {manifest_path}")
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    log(f"build finished in {elapsed_seconds(build_start)}; final_records={len(records)}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 3-4 DeepMath+s1K hard free-form SFT records.")
    parser.add_argument("--out-path", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--review-path", type=Path, default=DEFAULT_REVIEW_PATH)
    parser.add_argument("--seed", type=int, default=151)
    parser.add_argument("--tokenizer-name-or-path", default=DEFAULT_TOKENIZER_NAME_OR_PATH)
    parser.add_argument("--max-rendered-tokens", type=int, default=DEFAULT_MAX_RENDERED_TOKENS)
    parser.add_argument("--deepmath-cap", type=int, default=DEFAULT_DEEPMATH_CAP, help="Use 0 for uncapped.")
    parser.add_argument("--s1k-cap", type=int, default=DEFAULT_S1K_CAP, help="Use 0 for uncapped.")
    parser.add_argument("--min-deepmath-difficulty", type=float, default=DEFAULT_MIN_DEEPMATH_DIFFICULTY)
    parser.add_argument("--min-reasoning-words", type=int, default=DEFAULT_MIN_REASONING_WORDS)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-every", type=int, default=1000, help="Print scan progress every N rows; use 0 to disable.")
    args = parser.parse_args()

    deepmath_cap = None if args.deepmath_cap == 0 else args.deepmath_cap
    s1k_cap = None if args.s1k_cap == 0 else args.s1k_cap

    records = build_stage3_4_records(
        out_path=args.out_path,
        manifest_path=args.manifest_path,
        review_path=args.review_path,
        seed=args.seed,
        force_rebuild=args.force_rebuild,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        max_rendered_tokens=args.max_rendered_tokens,
        deepmath_cap=deepmath_cap,
        s1k_cap=s1k_cap,
        min_deepmath_difficulty=args.min_deepmath_difficulty,
        min_reasoning_words=args.min_reasoning_words,
        dry_run=args.dry_run,
        log_every=args.log_every,
    )
    manifest = json.loads(args.manifest_path.read_text())
    print(f"Built {len(records)} Stage 3-4 records: {args.out_path}")
    print("Source counts:", manifest["source_counts"])
    print("Accepted before validation:", manifest["accepted_before_validation"])
    print("Top drop counts:", dict(collections.Counter(manifest["drop_counts"]).most_common(20)))
    print("Word counts by source:", manifest["word_counts_by_source"])
    print("Rendered token counts by source:", manifest["rendered_token_counts_by_source"])


if __name__ == "__main__":
    main()
