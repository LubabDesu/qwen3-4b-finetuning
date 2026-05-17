#!/usr/bin/env python3
"""Postprocess eval result JSONL files with more lenient capability scoring.

This does not replace the strict Kaggle-style score. It adds a diagnostic
"capability" score for common formatting/parser misses:
- ``\boxed[9]`` instead of ``\boxed{9}``
- MCQ value boxed instead of option letter
- no box, but prose ends with "answer is X" or "= X"
- exact-vs-decimal numeric equivalence with a small tolerance
- multiple final boxes for multi-part answers
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import re
import signal
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from judger import Judger  # noqa: E402

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional for this diagnostic script.
    tqdm = None


BOXED_CURLY_RE = re.compile(r"\\boxed\s*\{")
BOXED_SQUARE_RE = re.compile(r"\\boxed\s*\[([^\[\]]+)\]", flags=re.S)
ANSWER_IS_RE = re.compile(
    r"(?is)(?:final\s+answer|the\s+answer|answer|correct\s+answer)\s*(?:is|:)\s*"
    r"(.{1,240}?)(?:$|[.\n])"
)
TRAILING_EQUALS_RE = re.compile(r"(?is)=\s*([^\n=]{1,160}?)\s*(?:$|[.\n])")
OPTION_PROSE_RE = re.compile(r"(?is)(?:option|choice)\s*\(?([A-J])\)?")
LETTER_RE = re.compile(r"^\(?\s*([A-J])\s*\)?$")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_boxed_syntax(text: str) -> str:
    return BOXED_SQUARE_RE.sub(lambda m: f"\\boxed{{{m.group(1).strip()}}}", str(text or ""))


def normalize_text(value: Any) -> str:
    text = str(value or "").strip()
    text = text.strip("$")
    text = re.sub(r"\\(?:left|right)\s*", "", text)
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = re.sub(r"\\(?:text|textbf|mathrm|mathbf)\{([^{}]*)\}", r"\1", text)
    text = text.replace("−", "-").replace("–", "-").replace("×", "*").replace("÷", "/")
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" .,:;")


def normalize_for_text_match(value: Any) -> str:
    text = normalize_text(value).lower()
    text = text.replace("\\,", "").replace("\\;", "").replace("\\:", "").replace("\\!", "")
    text = text.replace("\\quad", "").replace("\\qquad", "")
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\s+", "", text)
    return text.strip(" .,:;")


def extract_after_think(text: str) -> str:
    parts = str(text or "").split("</think>")
    return parts[-1].strip() if len(parts) > 1 else str(text or "").strip()


def extract_all_boxed(text: str, *, final_section_only: bool = True) -> list[str]:
    text = normalize_boxed_syntax(text)
    search_text = extract_after_think(text) if final_section_only else text
    boxes: list[str] = []
    start = 0
    while True:
        match = BOXED_CURLY_RE.search(search_text, start)
        if not match:
            break
        i = match.end()
        depth = 1
        while i < len(search_text) and depth:
            if search_text[i] == "{":
                depth += 1
            elif search_text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            boxes.append(normalize_text(search_text[match.end() : i - 1]))
        start = max(i, match.end())
    if boxes or not final_section_only:
        return [box for box in boxes if box]
    return extract_all_boxed(text, final_section_only=False)


def split_top_level_commas(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    parts: list[str] = []
    start = 0
    depth = 0
    for idx, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}" and depth > 0:
            depth -= 1
        elif char == "," and depth == 0:
            part = text[start:idx].strip()
            if part:
                parts.append(part)
            start = idx + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def gold_list(row: dict[str, Any]) -> list[str]:
    gold = row.get("gold")
    if isinstance(gold, list):
        return [normalize_text(item) for item in gold]
    return [normalize_text(gold)]


def clean_candidate(value: str) -> str:
    text = normalize_text(value)
    text = re.sub(r"(?is)^\\boxed\s*\{(.+)\}$", r"\1", text).strip()
    text = re.sub(r"(?is)^[$\\\[\]\(\)\s]+|[$\\\[\]\(\)\s]+$", "", text).strip()
    text = re.sub(r"(?is)\s*(?:is the answer|is our answer)\s*$", "", text).strip()
    return normalize_text(text)


def prose_fallback_candidates(response: str) -> list[str]:
    text = normalize_boxed_syntax(str(response or ""))
    after = extract_after_think(text)
    candidates: list[str] = []

    for match in ANSWER_IS_RE.finditer(after):
        candidates.append(clean_candidate(match.group(1)))
    for match in TRAILING_EQUALS_RE.finditer(after):
        candidates.append(clean_candidate(match.group(1)))
    for match in OPTION_PROSE_RE.finditer(after):
        candidates.append(match.group(1).upper())

    # A final bare sentence like "Therefore, 9." is common enough to be useful,
    # but keep it last so explicit answer phrases win first.
    tail = re.sub(r"(?is).*?(?:therefore|thus|hence|so)[,:]?\s*", "", after).strip()
    if len(tail) <= 120:
        candidates.append(clean_candidate(tail))

    return dedupe([candidate for candidate in candidates if candidate])


def answer_candidates(response: str) -> list[tuple[str, str]]:
    boxes = extract_all_boxed(response)
    candidates: list[tuple[str, str]] = []
    if boxes:
        candidates.append(("boxed_last", boxes[-1]))
        if len(boxes) > 1:
            candidates.append(("multi_box_join", ", ".join(boxes)))
            candidates.extend(("boxed_each", box) for box in boxes)
    else:
        candidates.extend(("prose_fallback", candidate) for candidate in prose_fallback_candidates(response))
    return [(mode, candidate) for mode, candidate in candidates if candidate]


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = normalize_for_text_match(value)
        if key and key not in seen:
            seen.add(key)
            out.append(value)
    return out


def parse_numeric(value: str) -> float | None:
    text = normalize_text(value)
    if not text:
        return None
    text = text.replace(",", "")
    if not re.search(r"\d", text):
        return None
    
    # fast path - try direct float first
    try:
        return float(text)
    except ValueError:
        pass
    
    # skip sympy for obviously symbolic expressions
    # only call sympy if it looks like a simple latex fraction/sqrt
    if len(text) > 80:
        return None
    if re.search(r"[A-Za-z]{3,}", text.replace("\\frac", "").replace("\\sqrt", "").replace("\\pi", "")):
        return None
    
    # only now call expensive sympy
    try:
        import sympy as sp
        from sympy.parsing.latex import parse_latex
        expr = parse_latex(text)
        return float(sp.N(expr.subs(parse_latex("\\pi"), math.pi)))
    except Exception:
        return None

class Timeout:
    def __init__(self, seconds: int):
        self.seconds = seconds
        self.previous: Any = None

    def __enter__(self) -> None:
        if self.seconds <= 0:
            return
        self.previous = signal.getsignal(signal.SIGALRM)

        def handler(signum: int, frame: Any) -> None:
            raise TimeoutError("capability postprocess row timed out")

        signal.signal(signal.SIGALRM, handler)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if self.seconds <= 0:
            return False
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self.previous)
        return False


def numeric_close(pred: str, gold: str, *, abs_tol: float, rel_tol: float) -> bool:
    pred_value = parse_numeric(pred)
    gold_value = parse_numeric(gold)
    if pred_value is None or gold_value is None:
        return False
    if not math.isfinite(pred_value) or not math.isfinite(gold_value):
        return False
    return abs(pred_value - gold_value) <= max(abs_tol, rel_tol * max(1.0, abs(gold_value)))


def lenient_parts_match(pred: str, gold_items: list[str], *, abs_tol: float, rel_tol: float) -> bool:
    pred_parts = split_top_level_commas(pred)
    if len(pred_parts) != len(gold_items):
        return False
    for pred_part, gold_part in zip(pred_parts, gold_items):
        if normalize_for_text_match(pred_part) == normalize_for_text_match(gold_part):
            continue
        if numeric_close(pred_part, gold_part, abs_tol=abs_tol, rel_tol=rel_tol):
            continue
        return False
    return True


def judge_candidate(
    judger: Judger,
    candidate: str,
    gold_items: list[str],
    *,
    abs_tol: float,
    rel_tol: float,
) -> bool:
    if not candidate:
        return False
    pred = f"\\boxed{{{candidate}}}"
    try:
        if judger.auto_judge(pred=pred, gold=gold_items, options=[[]] * len(gold_items)):
            return True
    except Exception:
        pass
    return lenient_parts_match(candidate, gold_items, abs_tol=abs_tol, rel_tol=rel_tol)


def option_label_for_value(
    candidate: str,
    options: list[Any],
    *,
    abs_tol: float,
    rel_tol: float,
) -> str:
    match = LETTER_RE.fullmatch(normalize_text(candidate).upper())
    if match:
        return match.group(1)

    candidate_norm = normalize_for_text_match(candidate)
    for idx, option in enumerate(options):
        label = chr(65 + idx)
        option_text = normalize_text(option)
        if candidate_norm and candidate_norm == normalize_for_text_match(option_text):
            return label
        if numeric_close(candidate, option_text, abs_tol=abs_tol, rel_tol=rel_tol):
            return label
    return ""


def score_row(
    row: dict[str, Any],
    judger: Judger,
    *,
    abs_tol: float,
    rel_tol: float,
) -> tuple[bool, str, str]:
    strict_ok = bool(row.get("correct", False))
    if strict_ok:
        return True, "strict_correct", ""

    response = normalize_boxed_syntax(str(row.get("response", "")))
    candidates = answer_candidates(response)
    if not candidates:
        return False, "unrescued", ""

    if row.get("is_mcq") or row.get("options"):
        options = list(row.get("options") or [])
        gold = normalize_text(row.get("gold")).upper()
        for mode, candidate in candidates:
            label = option_label_for_value(candidate, options, abs_tol=abs_tol, rel_tol=rel_tol)
            if label == gold:
                if mode == "prose_fallback":
                    return True, "prose_fallback", candidate
                if LETTER_RE.fullmatch(normalize_text(candidate).upper()):
                    return True, "boxed_syntax" if "\\boxed[" in str(row.get("response", "")) else "mcq_letter_recovered", candidate
                return True, "mcq_value_match", candidate
        return False, "unrescued", ""

    gold_items = gold_list(row)
    for mode, candidate in candidates:
        if judge_candidate(judger, candidate, gold_items, abs_tol=abs_tol, rel_tol=rel_tol):
            if mode == "multi_box_join":
                return True, "multi_box_join", candidate
            if mode == "prose_fallback":
                return True, "prose_fallback", candidate
            if "\\boxed[" in str(row.get("response", "")):
                return True, "boxed_syntax", candidate
            if len(gold_items) == len(split_top_level_commas(candidate)) and any(
                numeric_close(p, g, abs_tol=abs_tol, rel_tol=rel_tol)
                for p, g in zip(split_top_level_commas(candidate), gold_items)
            ):
                return True, "exact_decimal_numeric", candidate
            return True, "judger_lenient", candidate
    return False, "unrescued", ""


def empty_split() -> dict[str, Any]:
    return {"n": 0, "strict_correct": 0, "capability_correct": 0, "rescued": 0}


def add_split(stats: dict[str, Any], row: dict[str, Any], strict_ok: bool, capability_ok: bool) -> None:
    stats["n"] += 1
    stats["strict_correct"] += int(strict_ok)
    stats["capability_correct"] += int(capability_ok)
    stats["rescued"] += int(capability_ok and not strict_ok)


def rate(num: int, den: int) -> float:
    return num / den if den else 0.0


def finalize_split(stats: dict[str, Any]) -> dict[str, Any]:
    n = int(stats["n"])
    return {
        **stats,
        "strict_accuracy": rate(int(stats["strict_correct"]), n),
        "capability_accuracy": rate(int(stats["capability_correct"]), n),
    }


def process_file(
    path: Path,
    *,
    abs_tol: float,
    rel_tol: float,
    write_rescued: bool,
    row_timeout_seconds: int,
    show_progress: bool,
) -> dict[str, Any]:
    rows = load_jsonl(path)
    judger = Judger(strict_extract=False)
    breakdown: collections.Counter[str] = collections.Counter()
    split_stats = {"mcq": empty_split(), "non_mcq": empty_split(), "multi": empty_split()}
    rescued_rows: list[dict[str, Any]] = []
    progress_rows = rows
    if show_progress and tqdm is not None:
        progress_rows = tqdm(rows, desc=f"postprocess {path.name}", unit="row")
    elif show_progress:
        print(f"[postprocess] {path}: {len(rows)} rows", flush=True)

    strict_correct = capability_correct = rescued = 0
    for index, row in enumerate(progress_rows, start=1):
        strict_ok = bool(row.get("correct", False))
        try:
            with Timeout(row_timeout_seconds):
                capability_ok, mode, candidate = score_row(row, judger, abs_tol=abs_tol, rel_tol=rel_tol)
        except TimeoutError:
            capability_ok, mode, candidate = strict_ok, "timeout" if not strict_ok else "strict_correct", ""
        except Exception as exc:
            capability_ok, mode, candidate = strict_ok, f"error:{type(exc).__name__}", ""
        strict_correct += int(strict_ok)
        capability_correct += int(capability_ok)
        rescued += int(capability_ok and not strict_ok)
        if capability_ok and not strict_ok:
            breakdown[mode] += 1
        elif mode.startswith("error:") or mode == "timeout":
            breakdown[mode] += 1

        split_name = "mcq" if row.get("is_mcq") or row.get("options") else "non_mcq"
        add_split(split_stats[split_name], row, strict_ok, capability_ok)
        if row.get("is_multi"):
            add_split(split_stats["multi"], row, strict_ok, capability_ok)

        if capability_ok and not strict_ok:
            rescued_rows.append(
                {
                    "id": row.get("id"),
                    "mode": mode,
                    "candidate": candidate,
                    "gold": row.get("gold"),
                    "is_mcq": bool(row.get("is_mcq") or row.get("options")),
                    "is_multi": bool(row.get("is_multi")),
                }
            )
        if show_progress and tqdm is None and (index == len(rows) or index % 25 == 0):
            print(
                f"[postprocess] {path.name}: {index}/{len(rows)} rows, "
                f"strict={strict_correct}, capability={capability_correct}, rescued={rescued}",
                flush=True,
            )

    n = len(rows)
    report = {
        "input_path": str(path),
        "n": n,
        "strict_correct": strict_correct,
        "capability_correct": capability_correct,
        "strict_accuracy": rate(strict_correct, n),
        "capability_accuracy": rate(capability_correct, n),
        "rescued_count": rescued,
        "breakdown_by_failure_mode": dict(breakdown),
        "splits": {name: finalize_split(stats) for name, stats in split_stats.items()},
    }

    if write_rescued:
        rescued_path = path.with_name(path.stem + "_capability_rescues.jsonl")
        with rescued_path.open("w") as f:
            for row in rescued_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        report["rescued_rows_path"] = str(rescued_path)

    return report


def aggregate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if len(reports) == 1:
        return reports[0]
    total = {
        "input_path": [report["input_path"] for report in reports],
        "n": sum(report["n"] for report in reports),
        "strict_correct": sum(report["strict_correct"] for report in reports),
        "capability_correct": sum(report["capability_correct"] for report in reports),
        "rescued_count": sum(report["rescued_count"] for report in reports),
        "breakdown_by_failure_mode": dict(
            sum((collections.Counter(report["breakdown_by_failure_mode"]) for report in reports), collections.Counter())
        ),
        "files": reports,
    }
    total["strict_accuracy"] = rate(total["strict_correct"], total["n"])
    total["capability_accuracy"] = rate(total["capability_correct"], total["n"])
    return total


def expand_inputs(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        matches = sorted(glob.glob(value))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(value))
    return paths


def print_summary_table(reports: list[dict[str, Any]]) -> None:
    col_w = 42
    print("\n" + "=" * 100, flush=True)
    print(f"{'FILE':<{col_w}}  {'N':>5}  {'STRICT':>7}  {'CAPAB':>7}  {'RESCUED':>7}  {'MCQ_CAP':>8}  {'NON-MCQ_CAP':>11}", flush=True)
    print("-" * 100, flush=True)
    for r in reports:
        name = Path(r["input_path"]).stem.replace("_eval_results", "")
        n = r["n"]
        strict = r["strict_accuracy"]
        cap = r["capability_accuracy"]
        rescued = r["rescued_count"]
        mcq_cap = r.get("splits", {}).get("mcq", {}).get("capability_accuracy")
        non_mcq_cap = r.get("splits", {}).get("non_mcq", {}).get("capability_accuracy")
        mcq_str = f"{mcq_cap:.1%}" if mcq_cap is not None else "  —  "
        non_mcq_str = f"{non_mcq_cap:.1%}" if non_mcq_cap is not None else "  —  "
        print(
            f"{name:<{col_w}}  {n:>5}  {strict:>7.1%}  {cap:>7.1%}  {rescued:>7}  {mcq_str:>8}  {non_mcq_str:>11}",
            flush=True,
        )
    print("=" * 100 + "\n", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore eval result JSONL files with capability-oriented postprocessing.")
    parser.add_argument(
        "inputs",
        nargs="*",
        default=["artifacts/post_training_curriculum/eval/*_eval_results.jsonl"],
        help="Eval result JSONL paths or globs.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON report output path.")
    parser.add_argument("--abs-tol", type=float, default=5e-3, help="Absolute tolerance for exact-vs-decimal rescue.")
    parser.add_argument("--rel-tol", type=float, default=1e-2, help="Relative tolerance for exact-vs-decimal rescue.")
    parser.add_argument("--row-timeout-seconds", type=int, default=5, help="Max seconds spent rescoring one row.")
    parser.add_argument("--write-rescued", action="store_true", help="Write rescued row details next to each input file.")
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show per-row progress bars. Use --no-progress for quiet JSON-only output.",
    )
    args = parser.parse_args()

    paths = [path for path in expand_inputs(args.inputs) if path.exists()]
    if not paths:
        raise FileNotFoundError("No eval result JSONL files matched the requested inputs.")

    print(f"[postprocess] processing {len(paths)} file(s)", flush=True)
    reports: list[dict[str, Any]] = []
    for path in paths:
        report = process_file(
            path,
            abs_tol=args.abs_tol,
            rel_tol=args.rel_tol,
            write_rescued=args.write_rescued,
            row_timeout_seconds=args.row_timeout_seconds,
            show_progress=args.progress,
        )
        reports.append(report)
        mcq_s = report.get("splits", {}).get("mcq", {})
        non_mcq_s = report.get("splits", {}).get("non_mcq", {})
        print(
            f"[postprocess] {path.name}: n={report['n']} "
            f"strict={report['strict_accuracy']:.1%} capability={report['capability_accuracy']:.1%} "
            f"rescued={report['rescued_count']} "
            f"mcq_cap={mcq_s.get('capability_accuracy', 0):.1%}({mcq_s.get('n', 0)}) "
            f"non_mcq_cap={non_mcq_s.get('capability_accuracy', 0):.1%}({non_mcq_s.get('n', 0)})",
            flush=True,
        )

    if len(reports) > 1:
        print_summary_table(reports)

    report = aggregate_reports(reports)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(report, indent=2, ensure_ascii=False)
        args.out.write_text(text + "\n")
        print(f"[postprocess] report saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
