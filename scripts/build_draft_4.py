#!/usr/bin/env python3
"""
Build submission_draft_4.csv from submission_draft_3.csv by:
  1. Replacing 43 unboxed rows with injection-rescued generations that already contain \\boxed{}.
  2. For remaining unboxed rows, running a deterministic postprocessor that
     scans the model's reasoning for a committed answer and appends \\boxed{answer}.
"""

import csv
import json
import re
import sys
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
DRAFT3 = "artifacts/private_reasoning_paths/submission_draft_3.csv"
INJECT = "artifacts/private_reasoning_paths/rs_sft_public526_110_still_zero_box_inject80_n4_8000.jsonl"
DRAFT4 = "artifacts/private_reasoning_paths/submission_draft_4.csv"

# ── helpers ────────────────────────────────────────────────────────────────

def has_boxed(text: str) -> bool:
    return bool(re.search(r"\\boxed\{", text))


def _is_valid_answer(s: str) -> bool:
    """Return True if s looks like a plausible answer value."""
    if not s:
        return False
    s = s.strip()
    # Single MCQ option letter
    if re.fullmatch(r"[A-J]", s):
        return True
    # Number (possibly with decimal, fraction, negative, percent, commas)
    if re.fullmatch(r"-?[0-9][0-9,]*\.?[0-9]*(?:/[0-9]+)?%?", s):
        return True
    # LaTeX fraction like \frac{1}{2}
    if re.search(r"\\frac\{", s):
        return True
    # Short alphanumeric (e.g. "June 1908", "True", "False", "11 ft")
    if len(s) <= 30 and any(c.isdigit() for c in s):
        return True
    if s.lower() in ("true", "false", "yes", "no", "none"):
        return True
    return False


def _clean_extracted(s: str) -> str:
    """Clean up an extracted answer string."""
    s = s.strip()
    # Remove trailing punctuation
    s = s.rstrip(".,;:!? ")
    # Remove wrapping $ signs
    s = s.strip("$")
    # Remove thousand separators in pure numbers
    if re.match(r"^[0-9,]+\.?[0-9]*$", s):
        s = s.replace(",", "")
    s = s.strip()
    return s


def extract_answer_from_reasoning(text: str) -> str | None:
    """
    Try to deterministically extract the model's intended answer from its
    reasoning trace, even though it never wrote \\boxed{}.

    Returns the extracted answer string, or None if nothing confident found.
    """

    def _try(candidate: str) -> str | None:
        c = _clean_extracted(candidate)
        return c if _is_valid_answer(c) else None

    # ── Strategy 1: </think> tag present → model thought it was done ──────
    think_idx = text.rfind("</think>")
    if think_idx >= 0:
        pre_think = text[:think_idx]
        tail = pre_think[-500:]

        # "answer is X" right before </think>
        m = re.search(
            r"(?:the\s+)?answer\s+(?:is|should\s+be|would\s+be|=)\s*"
            r"[\$\\]*\s*(?:\\boxed\{)?([A-J]|-?[0-9][0-9,]*\.?[0-9]*(?:/[0-9]+)?)(?:\})?",
            tail, re.IGNORECASE
        )
        if m:
            v = _try(m.group(1))
            if v:
                return v

    # ── Strategy 2: "the answer is X" near the end ────────────────────────
    tail = text[-2000:]

    # Tight pattern: "answer is" followed by an option letter or number
    matches = list(re.finditer(
        r"(?:the\s+)?answer\s+(?:is|should\s+be|=)\s*"
        r"[\$\\]*(?:\\boxed\{)?([A-J]|-?[0-9][0-9,]*\.?[0-9]*(?:/[0-9]+)?(?:\\?%)?)"
        r"(?:\})?",
        tail, re.IGNORECASE
    ))
    if matches:
        v = _try(matches[-1].group(1))
        if v:
            return v

    # ── Strategy 3: "which is option X" ───────────────────────────────────
    m = re.search(r"which\s+is\s+option\s+([A-J])", tail, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # ── Strategy 4: "correct option/answer/choice is X" ──────────────────
    m = re.search(
        r"(?:correct|right)\s+(?:option|answer|choice)\s+is\s+([A-J])\b",
        tail, re.IGNORECASE
    )
    if m:
        return m.group(1).upper()

    # ── Strategy 5: "option X" as the last option mention ─────────────────
    option_mentions = list(re.finditer(
        r"option\s+([A-J])\b", tail, re.IGNORECASE
    ))
    if option_mentions:
        return option_mentions[-1].group(1).upper()

    # ── Strategy 6: "approximately X" or "≈ X" ───────────────────────────
    matches = list(re.finditer(
        r"(?:approximately|≈|about|roughly|is\s+equal\s+to)\s+"
        r"[\$]*(-?[0-9][0-9,]*\.?[0-9]*(?:/[0-9]+)?)[\$]*",
        tail, re.IGNORECASE
    ))
    if matches:
        v = _try(matches[-1].group(1))
        if v:
            return v

    # ── Strategy 7: "= X" or "is X" before "Wait/Let's check" ───────────
    m = re.search(
        r"(?:is\s+|=\s*)[\$]*([A-J]|-?[0-9][0-9,]*\.?[0-9]*(?:/[0-9]+)?)"
        r"[\$]*[\.\s,]*(?:Wait|Let'?s\s+(?:check|verify|double))",
        text[-3000:], re.IGNORECASE
    )
    if m:
        v = _try(m.group(1))
        if v:
            return v

    # ── Strategy 8: Last standalone number in the text ────────────────────
    last_num = list(re.finditer(
        r"(?<!\w)(-?[0-9][0-9,]*\.?[0-9]*(?:/[0-9]+)?)(?!\w)",
        text[-1000:]
    ))
    if last_num:
        v = _try(last_num[-1].group(1))
        if v:
            return v

    # ── Strategy 9: Last MCQ letter ──────────────────────────────────────
    last_letter = list(re.finditer(
        r"(?:^|\s)([A-J])(?:\s*[\.\),;:]|\s*$)",
        text[-500:], re.MULTILINE
    ))
    if last_letter:
        return last_letter[-1].group(1).upper()

    return None


# ── main ───────────────────────────────────────────────────────────────────

def main():
    # 1. Load injection-rescued generations
    rescued = {}
    with open(INJECT) as f:
        for line in f:
            row = json.loads(line)
            qid = row["question_id"]
            boxed_gens = [g for g in row["generations"] if has_boxed(g)]
            if boxed_gens:
                rescued[qid] = max(boxed_gens, key=len)

    print(f"[1] Loaded {len(rescued)} rescued generations from injection pass")

    # 2. Read draft 3, merge injection + postprocess, write draft 4
    total = 0
    replaced_inject = 0
    postprocessed = 0
    postprocess_failed = 0
    boxed_before = 0
    boxed_after = 0

    rows_out = []
    postprocess_log = []  # for debugging

    with open(DRAFT3, "r") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            total += 1
            qid = int(row["id"])
            resp = row["response"]

            was_boxed = has_boxed(resp)
            if was_boxed:
                boxed_before += 1

            if qid in rescued and not was_boxed:
                # Step 1: replace with injection generation
                resp = rescued[qid]
                replaced_inject += 1
            elif not was_boxed:
                # Step 2: deterministic postprocessing
                answer = extract_answer_from_reasoning(resp)
                if answer:
                    resp = resp + f"\n\n\\boxed{{{answer}}}"
                    postprocessed += 1
                    postprocess_log.append({"qid": qid, "extracted": answer})
                else:
                    postprocess_failed += 1
                    postprocess_log.append({"qid": qid, "extracted": None})

            if has_boxed(resp):
                boxed_after += 1

            rows_out.append({"id": str(qid), "response": resp})

    # Write draft 4
    with open(DRAFT4, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=["id", "response"], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for r in rows_out:
            writer.writerow(r)

    # Write postprocess log for review
    log_path = DRAFT4.replace(".csv", "_postprocess_log.jsonl")
    with open(log_path, "w") as f:
        for entry in postprocess_log:
            f.write(json.dumps(entry) + "\n")

    print(f"\n[2] Draft 3 → Draft 4 merge complete")
    print(f"  Total rows:              {total}")
    print(f"  Boxed before:            {boxed_before}")
    print(f"  Replaced (injection):    {replaced_inject}")
    print(f"  Postprocessed (heuristic): {postprocessed}")
    print(f"  Postprocess failed:      {postprocess_failed}")
    print(f"  Boxed after:             {boxed_after}")
    print(f"  Still unboxed:           {total - boxed_after}")
    print(f"\n  Written to: {DRAFT4}")
    print(f"  Postprocess log: {log_path}")


if __name__ == "__main__":
    main()
