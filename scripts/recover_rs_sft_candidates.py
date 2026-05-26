#!/usr/bin/env python3
"""Recover correct RS-SFT rows from candidate JSONL with robust final-answer boxing.

This is a deterministic serialization/extraction repair pass. It does not compute
new answers; it only standardizes answer strings already present in model output.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for p in (ROOT, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import judger as judger_module  # noqa: E402
from generate_rs_sft_mixed import (  # noqa: E402
    gold_answers,
    is_mcq_row,
    make_sft_row,
    normalize_final_answer_for_row,
    repair_to_single_final_box,
)
from generate_rs_sft_mcq import (  # noqa: E402
    anti_spiral_reject_reason,
    candidate_quality_metrics,
    candidate_score,
    extract_all_boxed,
    normalize_mcq_answer,
    target_shape_ok,
)
from judger import Judger  # noqa: E402

_JUDGER_LOCK = threading.Lock()

FINAL_MARKER_RE = re.compile(
    r"(?:final\s+answers?|answer\s*:|answers\s+are|therefore|thus|hence)",
    flags=re.I,
)
BOX_RE = re.compile(r"\\boxed\s*\{")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-rows", type=Path, required=True, help="Original problem JSONL with gold answers.")
    parser.add_argument("--candidate-path", type=Path, required=True, help="Mixed RS candidate JSONL.")
    parser.add_argument("--out-path", type=Path, required=True, help="Recovered SFT JSONL.")
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--max-spiral-markers", type=int, default=2)
    parser.add_argument("--max-duplicate-lines", type=int, default=1)
    parser.add_argument("--max-repeated-ngram-ratio", type=float, default=0.10)
    parser.add_argument("--strict-extract", action="store_true")
    parser.add_argument("--keep-incorrect", action="store_true", help="Write best recovered row even if gold validation fails. Use only for private/no-gold inspection.")
    parser.add_argument("--log-every", type=int, default=25, help="Print progress every N candidate rows. Use 0 to disable periodic logs.")
    parser.add_argument("--slow-row-seconds", type=float, default=10.0, help="Print a row-level warning when recovery takes at least this many seconds. Use 0 to disable.")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def log(message: str) -> None:
    print(f"[recover-rs] {message}", flush=True)


def row_id(row: dict[str, Any], fallback: int) -> Any:
    return row.get("id", row.get("question_id", fallback))


def load_problem_map(path: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for idx, row in enumerate(load_jsonl(path)):
        out[str(row_id(row, idx))] = row
    return out


@contextlib.contextmanager
def disable_judger_alarm() -> Any:
    old_signal = judger_module.signal.signal
    old_alarm = judger_module.signal.alarm
    judger_module.signal.signal = lambda *args, **kwargs: None
    judger_module.signal.alarm = lambda *args, **kwargs: 0
    try:
        yield
    finally:
        judger_module.signal.signal = old_signal
        judger_module.signal.alarm = old_alarm


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


def top_level_split_commas(text: str) -> list[str]:
    parts: list[str] = []
    cur: list[str] = []
    depth = 0
    for ch in str(text or ""):
        if ch in "{[":
            depth += 1
        elif ch in "}]" and depth > 0:
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


def normalize_candidate_answer(ans: str) -> str:
    ans = str(ans or "").strip()
    ans = re.sub(r"^\$+|\$+$", "", ans).strip()
    ans = ans.replace("\\left", "").replace("\\right", "")
    ans = ans.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    ans = re.sub(r"\\(?:text|mathrm|mathbf)\{([^{}]*)\}", r"\1", ans)
    ans = re.sub(r"\s+", " ", ans).strip()
    ans = ans.strip(" .。")
    return ans


def answer_key(ans: str) -> str:
    ans = normalize_candidate_answer(ans).lower()
    ans = ans.replace(" ", "")
    ans = ans.replace("*", "")
    ans = ans.replace("\\cdot", "")
    ans = ans.replace("$", "")
    return ans


def relaxed_literal_match(answer: str, gold: list[str], row: dict[str, Any]) -> bool:
    if not gold:
        return False
    if is_mcq_row(row, gold):
        return normalize_mcq_answer(answer) == normalize_mcq_answer(gold[0])
    parts = top_level_split_commas(answer)
    if len(parts) != len(gold):
        return False
    return all(answer_key(a) == answer_key(g) for a, g in zip(parts, gold))


def final_section(text: str) -> str:
    search_start = max(text.rfind("</think>"), 0)
    after_think = text[search_start + len("</think>"):] if "</think>" in text else text
    matches = list(FINAL_MARKER_RE.finditer(after_think))
    if matches:
        return after_think[matches[-1].start():]
    return after_think


def candidate_answer_strings(text: str, expected_count: int | None) -> list[tuple[str, str]]:
    variants: list[tuple[str, str]] = []
    full_spans = boxed_spans(text)
    section = final_section(text)
    section_spans = boxed_spans(section)

    def add(source: str, value: str) -> None:
        value = normalize_candidate_answer(value)
        if value and (source, value) not in variants:
            variants.append((source, value))

    # Prefer a single box that already contains all expected comma-separated answers.
    for source, spans in (("final_single_box", section_spans), ("single_box", full_spans)):
        for _s, _e, content in reversed(spans):
            if expected_count is None or expected_count <= 1 or len(top_level_split_commas(content)) == expected_count:
                add(source, content)

    # Common false negative: final answer rendered as \boxed{a}, \boxed{b}, ...
    for source, spans in (("final_multi_box", section_spans), ("multi_box", full_spans)):
        if expected_count and len(spans) >= expected_count:
            add(source, ", ".join(content for _s, _e, content in spans[-expected_count:]))
        if len(spans) > 1:
            add(source + "_all", ", ".join(content for _s, _e, content in spans))

    # If the model writes "A: 32, B: 96, C: 52" near the end, recover the values.
    if expected_count and expected_count > 1:
        labeled = re.findall(
            r"(?:^|[\n\-•*]\s*)(?:part\s*)?(?:[A-Z]|\d+)\s*[\).:]\s*([^\n;]+)",
            section,
            flags=re.I,
        )
        cleaned = []
        for item in labeled:
            item = re.split(r"(?:,\s*(?:and\s*)?$|\s{2,})", item.strip())[0].strip()
            item = re.sub(r"^(?:is|=)\s*", "", item, flags=re.I).strip()
            if item:
                cleaned.append(item)
        if len(cleaned) >= expected_count:
            add("labeled_lines", ", ".join(cleaned[-expected_count:]))

    return variants


def judge_answer(judger: Judger, row: dict[str, Any], completion: str, answer: str) -> bool:
    gold = gold_answers(row)
    if not gold:
        return False
    if relaxed_literal_match(answer, gold, row):
        return True
    repaired = repair_to_single_final_box(completion, normalize_final_answer_for_row(row, answer))
    try:
        with _JUDGER_LOCK:
            with disable_judger_alarm():
                return bool(judger.auto_judge(pred=repaired, gold=gold, options=[[] for _ in gold]))
    except Exception:
        return False


def recover_best_candidate(args: argparse.Namespace, judger: Judger, row: dict[str, Any], candidate_row: dict[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    gold = gold_answers(row)
    expected = len(gold) if gold else None
    diagnostics: list[dict[str, Any]] = []
    best = None
    best_score = None

    for cand in candidate_row.get("candidates", []):
        raw_completion = cand.get("raw_cleaned_completion") or cand.get("cleaned_completion") or cand.get("completion") or ""
        variants = candidate_answer_strings(raw_completion, expected)
        for source, answer in variants:
            normalized = normalize_final_answer_for_row(row, answer)
            repaired = repair_to_single_final_box(raw_completion, normalized)
            metrics = candidate_quality_metrics(repaired)
            anti_spiral = anti_spiral_reject_reason(metrics, args)
            shape_ok = target_shape_ok(repaired)
            if (anti_spiral is not None or not shape_ok) and not args.keep_incorrect:
                correct = False
            else:
                correct = judge_answer(judger, row, repaired, normalized)
            accepted = bool((correct or args.keep_incorrect) and shape_ok and anti_spiral is None)
            tokens = int(cand.get("completion_tokens") or cand.get("cleaned_tokens") or 0)
            score = candidate_score(tokens, metrics, bool(cand.get("used_conclusion_nudge")))
            diagnostics.append(
                {
                    "candidate_index": cand.get("candidate_index"),
                    "source": source,
                    "answer": normalized,
                    "correct": correct,
                    "accepted": accepted,
                    "anti_spiral_reason": anti_spiral,
                    "score": score,
                    "metrics": metrics,
                }
            )
            if accepted and (best_score is None or score > best_score):
                best_score = score
                best = {
                    "candidate_index": cand.get("candidate_index", 0),
                    "tokens": tokens,
                    "answer": normalized,
                    "completion": repaired,
                    "score": score,
                    "metrics": metrics,
                    "used_conclusion_nudge": bool(cand.get("used_conclusion_nudge")),
                    "recovery_source": source,
                }
    if best is None:
        return None, diagnostics

    sft_row = make_sft_row(
        row=row,
        completion=best["completion"],
        candidate_index=int(best["candidate_index"]),
        tokens=int(best["tokens"]),
        final_answer=str(best["answer"]),
        selected_score=float(best["score"]),
        selected_metrics=best["metrics"],
        used_conclusion_nudge=bool(best["used_conclusion_nudge"]),
        num_generations=len(candidate_row.get("candidates", [])),
    )
    sft_row["rs_sft"]["selection"]["method"] = "recover_answer_then_anti_spiral_score"
    sft_row["rs_sft"]["selection"]["recovery_source"] = best["recovery_source"]
    return sft_row, diagnostics


def main() -> None:
    args = parse_args()
    start_time = time.monotonic()
    log(f"loading input rows: {args.input_rows}")
    problems = load_problem_map(args.input_rows)
    log(f"loading candidate rows: {args.candidate_path}")
    candidate_rows = load_jsonl(args.candidate_path)
    judger = Judger(strict_extract=args.strict_extract)
    log(
        "starting recovery: "
        f"candidates={len(candidate_rows)} problems={len(problems)} "
        f"max_spiral_markers={args.max_spiral_markers} "
        f"max_duplicate_lines={args.max_duplicate_lines} "
        f"max_repeated_ngram_ratio={args.max_repeated_ngram_ratio}"
    )

    recovered: list[dict[str, Any]] = []
    diagnostics_rows: list[dict[str, Any]] = []
    stats: collections.Counter[str] = collections.Counter()

    for idx, cand_row in enumerate(candidate_rows, start=1):
        row_start = time.monotonic()
        rid = str(cand_row.get("id"))
        row = problems.get(rid)
        if row is None:
            stats["missing_problem"] += 1
            diagnostics_rows.append({"id": rid, "accepted": False, "diagnostics": [], "note": "missing_problem"})
            if args.log_every and idx % args.log_every == 0:
                elapsed = time.monotonic() - start_time
                log(f"progress {idx}/{len(candidate_rows)} recovered={len(recovered)} missing={stats['missing_problem']} elapsed={elapsed:.1f}s")
            continue
        sft_row, diagnostics = recover_best_candidate(args, judger, row, cand_row)
        diagnostics_rows.append({"id": rid, "accepted": sft_row is not None, "diagnostics": diagnostics[:20]})
        if sft_row is None:
            stats["not_recovered"] += 1
        else:
            recovered.append(sft_row)
            stats["recovered"] += 1
            stats[f"recovered_type:{'mcq' if sft_row.get('is_mcq') else 'non_mcq'}"] += 1
            source = sft_row.get("rs_sft", {}).get("selection", {}).get("recovery_source", "unknown")
            stats[f"source:{source}"] += 1
        row_elapsed = time.monotonic() - row_start
        if args.slow_row_seconds and row_elapsed >= args.slow_row_seconds:
            log(
                "slow row "
                f"{idx}/{len(candidate_rows)} id={rid} "
                f"accepted={sft_row is not None} variants={len(diagnostics)} "
                f"elapsed={row_elapsed:.1f}s"
            )
        if args.log_every and idx % args.log_every == 0:
            elapsed = time.monotonic() - start_time
            rate = idx / elapsed if elapsed > 0 else 0.0
            log(
                f"progress {idx}/{len(candidate_rows)} "
                f"recovered={len(recovered)} not_recovered={stats['not_recovered']} "
                f"missing={stats['missing_problem']} rate={rate:.2f} rows/s"
            )

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"writing recovered rows: {args.out_path}")
    with args.out_path.open("w", encoding="utf-8") as f:
        for row in recovered:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    elapsed = time.monotonic() - start_time
    manifest = {
        "input_rows": str(args.input_rows),
        "candidate_path": str(args.candidate_path),
        "out_path": str(args.out_path),
        "candidate_prompt_count": len(candidate_rows),
        "recovered_rows": len(recovered),
        "elapsed_seconds": round(elapsed, 3),
        "stats": dict(stats),
    }
    if args.manifest_path:
        args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        diag_path = args.manifest_path.with_suffix(".diagnostics.jsonl")
        with diag_path.open("w", encoding="utf-8") as f:
            for row in diagnostics_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["diagnostics_path"] = str(diag_path)
        log(f"wrote manifest: {args.manifest_path}")
        log(f"wrote diagnostics: {diag_path}")
    log(f"done recovered={len(recovered)} elapsed={elapsed:.1f}s")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
