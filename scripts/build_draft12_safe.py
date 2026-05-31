#!/usr/bin/env python3
"""Build draft12 as a conservative repair/canonicalization pass."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from judger import Judger  # noqa: E402

BOX_START_RE = re.compile(r"\\boxed\s*\{")
BAD_WORD_RE = re.compile(r"####|\b(step|calculate|therefore|because)\b", re.I)
BAD_ANSWERS = {"", "none", "null", "n/a", "na"}


REPAIR_PROMPT = """Previous extracted answer is invalid.

Question has {num_blanks} [ANS] blanks.
Previous answer gave {num_components} components:
{bad_answer}

Re-read the question and reasoning.
Output exactly {num_blanks} comma-separated answers in blank order.
Use exactly one \\boxed{{}}.
No explanation.
If options are given, output only option letter(s).
If final operation is needed, apply it.

Question:
{question}

Options:
{options_if_any}

Reasoning:
{reasoning}
"""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--base",
        type=Path,
        default=REPO_ROOT / "submission_drafts/submission_draft_10_ultrasafe.csv",
    )
    ap.add_argument("--private", type=Path, default=REPO_ROOT / "competition-data/private.jsonl")
    ap.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "submission_drafts/submission_draft_12.csv",
    )
    ap.add_argument(
        "--audit",
        type=Path,
        default=REPO_ROOT / "submission_drafts/audit_draft12.csv",
    )
    ap.add_argument("--repair-model", default="", help="Optional vLLM model path for one invalid-answer repair pass.")
    ap.add_argument("--repair-tokenizer", default=None)
    ap.add_argument("--repair-max-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=12288)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--dtype", default="bfloat16")
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
    text = str(value or "").strip()
    text = text.replace("\\boxed{", "").replace("boxed{", "")
    text = re.sub(r"[{}$]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" .,:;")


def find_matching_brace(text: str, open_idx: int) -> int | None:
    depth = 0
    for i in range(open_idx, len(text)):
        ch = text[i]
        if ch == "{" and (i == 0 or text[i - 1] != "\\"):
            depth += 1
        elif ch == "}" and (i == 0 or text[i - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return i
    return None


def extract_all_boxed(text: str) -> list[tuple[int, int, str]]:
    boxes: list[tuple[int, int, str]] = []
    for match in BOX_START_RE.finditer(text or ""):
        open_idx = (text or "").find("{", match.start())
        if open_idx < 0:
            continue
        close_idx = find_matching_brace(text or "", open_idx)
        if close_idx is None:
            continue
        boxes.append((match.start(), close_idx + 1, (text or "")[open_idx + 1 : close_idx]))
    return boxes


def strip_all_boxed(text: str) -> str:
    boxes = extract_all_boxed(text)
    if not boxes:
        return (text or "").rstrip()
    parts: list[str] = []
    last = 0
    for start, end, _ in boxes:
        parts.append((text or "")[last:start])
        last = end
    parts.append((text or "")[last:])
    cleaned = "".join(parts)
    cleaned = re.sub(r"(?im)^\s*(?:Final answer|Answer|Final)\s*:\s*$", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).rstrip()
    return cleaned


def append_single_final_box(response: str, answer: str) -> str:
    cleaned = strip_all_boxed(response)
    return (cleaned.rstrip() + f"\n\nFinal answer: \\boxed{{{answer}}}").lstrip()


def strip_outer_wrappers(text: str) -> str:
    out = str(text or "").strip()
    pairs = {"[": "]", "(": ")", "{": "}"}
    changed = True
    while changed and len(out) >= 2:
        changed = False
        if out[0] in pairs and out[-1] == pairs[out[0]]:
            out = out[1:-1].strip()
            changed = True
    return out


def split_top_level(answer: str, flatten_outer: bool = False) -> list[str]:
    text = strip_outer_wrappers(answer) if flatten_outer else str(answer or "").strip()
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


def bad_latex(text: str) -> bool:
    raw = str(text or "")
    if "\x08oxed" in raw or ("boxed{" in raw and "\\boxed{" not in raw):
        return True
    if has_unbalanced_braces(raw):
        return True
    if re.search(r"\\(?:frac|dfrac|tfrac)(?!\s*\{)", raw):
        return True
    if re.search(r"\\sqrt(?!\s*(?:\{|\[))", raw):
        return True
    return False


def options_text(row: dict[str, Any]) -> str:
    options = row.get("options")
    if not isinstance(options, list) or not options:
        return ""
    return "\n".join(f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(options))


def has_options(row: dict[str, Any]) -> bool:
    options = row.get("options")
    return isinstance(options, list) and bool(options)


def option_letters(answer: str) -> str | None:
    ans = normalize_text(answer)
    if re.fullmatch(r"[A-J](?:\s*,\s*[A-J])*|[A-J]+", ans, re.I):
        parts = re.findall(r"[A-J]", ans, re.I)
        if "," in ans:
            return ", ".join(p.upper() for p in parts)
        return "".join(p.upper() for p in parts)
    return None


def option_value_to_letter(answer: str, options: list[Any]) -> str | None:
    normalized = normalize_text(answer).lower().replace("\\", "")
    if not normalized:
        return None
    for i, option in enumerate(options or []):
        opt_norm = normalize_text(option).lower().replace("\\", "")
        if normalized == opt_norm:
            return chr(ord("A") + i)
    return None


def numeric_value(answer: str) -> float | None:
    text = str(answer or "").strip().replace(",", "").replace("%", "")
    try:
        value = float(text)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def requested_modulus(question: str) -> int | None:
    q = str(question or "").lower()
    patterns = [
        r"(?:remainder when|remainder of|modulo|mod|units digit)\D{0,100}(\d+)",
        r"(?:divided by)\s+(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            return int(match.group(1))
    return None


def question_implies_percent(question: str) -> bool:
    q = str(question or "").lower()
    return "[ans]%" in q or "[ans] percent" in q or "percentage" in q or "as a percent" in q




def interval_blanks_require_interval_components(question: str) -> bool:
    q = str(question or "").lower()
    if q.count("[ans]") <= 1:
        return False
    interval_terms = ["interval notation", "confidence interval", "interval estimate", "interval estimates"]
    return any(term in q for term in interval_terms)


def looks_like_interval_component(part: str) -> bool:
    text = str(part or "").strip()
    if not text:
        return False
    if not re.search(r"[\[\(].+,.+[\]\)]", text):
        return False
    return len(split_top_level(text, flatten_outer=True)) >= 2

def allows_extra_components(question: str) -> bool:
    q = str(question or "").lower()
    patterns = [
        "separate multiple answers by commas",
        "separate your answers by commas",
        "separate answers by commas",
        "find all",
        "list all",
        "all values",
        "all solutions",
        "all answers",
        "all possible",
        "select all",
        "check all",
        "more than one answer",
        "which statements",
        "which of the following statements",
    ]
    return any(pattern in q for pattern in patterns)


def validate_answer(answer: str, row: dict[str, Any]) -> tuple[bool, str]:
    ans = str(answer or "").strip()
    norm = normalize_text(ans)
    question = str(row.get("question") or "")
    num_blanks = question.count("[ANS]")
    flattened_parts = split_top_level(ans, flatten_outer=True)
    if not norm or norm.lower() in BAD_ANSWERS:
        return False, "empty"
    if word_count(ans) > 20:
        return False, "too_many_words"
    if BAD_WORD_RE.search(ans):
        return False, "reasoning_text"
    if bad_latex(ans):
        return False, "bad_latex"
    if has_options(row) and option_letters(ans) is None:
        return False, "mcq_not_letters"
    if num_blanks > 1:
        if interval_blanks_require_interval_components(question):
            interval_parts = split_top_level(ans, flatten_outer=False)
            if len(interval_parts) != num_blanks:
                return False, "interval_component_count_mismatch"
            if not all(looks_like_interval_component(part) for part in interval_parts):
                return False, "interval_component_shape"
        component_count = len(flattened_parts)
        if component_count < num_blanks:
            return False, "underfilled_after_flatten"
        if component_count > num_blanks and not allows_extra_components(question):
            return False, "overfilled_after_flatten"
    return True, "valid"


def deterministic_repair(answer: str, row: dict[str, Any]) -> tuple[str | None, str]:
    ans = str(answer or "").strip()
    question = str(row.get("question") or "")
    num_blanks = question.count("[ANS]")
    options = row.get("options") if isinstance(row.get("options"), list) else []

    if has_options(row):
        letters = option_letters(ans)
        if letters:
            return letters, "canonicalize_mcq_letters"
        mapped = option_value_to_letter(ans, options)
        if mapped:
            return mapped, "mcq_value_to_letter"

    modulus = requested_modulus(question)
    value = numeric_value(ans)
    if modulus and value is not None and abs(value - round(value)) < 1e-9:
        reduced = str(int(round(value)) % modulus)
        if has_options(row):
            mapped = option_value_to_letter(reduced, options)
            if mapped:
                return mapped, "modulo_to_option"
        if normalize_text(reduced) != normalize_text(ans):
            return reduced, "apply_modulo"

    if num_blanks > 1:
        flattened = strip_outer_wrappers(ans)
        parts = split_top_level(flattened)
        if len(parts) == num_blanks and flattened != ans:
            return ", ".join(p.strip() for p in parts), "strip_outer_brackets_multi"
        if len(parts) == num_blanks and re.fullmatch(r"[\[\(].+[\]\)]", ans, re.S):
            return ", ".join(p.strip() for p in parts), "flatten_packed_separate_blanks"

    if ans.endswith("%") and question_implies_percent(question):
        stripped = ans.rstrip("%").strip()
        if stripped:
            return stripped, "strip_redundant_percent"

    return None, "no_deterministic_repair"


def render_prompt(tokenizer: Any, prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You extract final answers exactly."},
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return "System: You extract final answers exactly.\n\nUser:\n" + prompt + "\n\nAssistant:\n"


class RepairExtractor:
    def __init__(self, args: argparse.Namespace):
        if not args.repair_model:
            self.llm = None
            self.tokenizer = None
            return
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from transformers import AutoTokenizer
        from vllm import LLM

        model_path = Path(args.repair_model)
        model = str(model_path.resolve()) if model_path.exists() else args.repair_model
        tok_path = args.repair_tokenizer or model
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm = LLM(
            model=model,
            tokenizer=tok_path,
            trust_remote_code=True,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=8,
            enforce_eager=False,
            generation_config="vllm",
        )
        self.max_tokens = args.repair_max_tokens

    def available(self) -> bool:
        return self.llm is not None and self.tokenizer is not None

    def repair(self, prompt: str) -> str:
        if not self.available():
            return ""
        from vllm import SamplingParams

        rendered = render_prompt(self.tokenizer, prompt)
        params = SamplingParams(
            n=1,
            temperature=0.0,
            top_p=1.0,
            max_tokens=self.max_tokens,
            seed=99991,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        output = self.llm.generate([rendered], params)[0]
        return output.outputs[0].text.strip() if output.outputs else ""


def extract_answer(judger: Judger, response: str) -> str:
    try:
        return judger.extract_ans(response) or ""
    except Exception:
        return ""


def main() -> None:
    args = parse_args()
    judger = Judger()
    private = read_private(args.private)
    extractor = RepairExtractor(args)

    out_rows: list[dict[str, str]] = []
    audit_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    with args.base.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["id", "response"]:
            raise SystemExit(f"expected id,response columns, got {reader.fieldnames}")
        for row in reader:
            qid = int(row["id"])
            if qid not in private:
                raise SystemExit(f"id {qid} not found in private file")
            qrow = private[qid]
            response = row["response"] or ""
            old_answer = extract_answer(judger, response).strip()
            old_valid, old_invalid_reason = validate_answer(old_answer, qrow)
            chosen_answer = old_answer
            reason = "canonicalized_only"

            repaired, repair_reason = deterministic_repair(old_answer, qrow)
            if repaired:
                repaired_valid, repaired_invalid_reason = validate_answer(repaired, qrow)
                if repaired_valid:
                    chosen_answer = repaired
                    reason = repair_reason
                elif not old_valid:
                    reason = f"deterministic_repair_invalid:{repaired_invalid_reason}"

            chosen_valid, chosen_invalid_reason = validate_answer(chosen_answer, qrow)
            if not chosen_valid and extractor.available():
                num_blanks = str(qrow.get("question") or "").count("[ANS]")
                if num_blanks == 0:
                    num_blanks = 1
                prompt = REPAIR_PROMPT.format(
                    num_blanks=num_blanks,
                    num_components=len(split_top_level(chosen_answer, flatten_outer=True)),
                    bad_answer=chosen_answer,
                    question=qrow.get("question", ""),
                    options_if_any=options_text(qrow),
                    reasoning=response,
                )
                repaired_response = extractor.repair(prompt)
                repaired_answer = extract_answer(judger, repaired_response).strip()
                repaired_valid, repaired_invalid_reason = validate_answer(repaired_answer, qrow)
                if repaired_valid:
                    chosen_answer = repaired_answer
                    reason = "repair_extractor"
                    chosen_valid = True
                else:
                    reason = f"invalid_no_change_after_repair_extractor:{repaired_invalid_reason}"
                    chosen_valid = False
                    chosen_invalid_reason = repaired_invalid_reason
            elif not chosen_valid:
                reason = f"invalid_no_repair_model:{chosen_invalid_reason}"

            final_response = append_single_final_box(response, chosen_answer)
            extracted_after = extract_answer(judger, final_response).strip()
            verified = extracted_after == chosen_answer.strip()
            if not verified:
                failures.append(
                    {
                        "id": qid,
                        "chosen_answer": chosen_answer,
                        "extracted_after": extracted_after,
                        "reason": reason,
                    }
                )

            changed = normalize_text(chosen_answer) != normalize_text(old_answer)
            out_rows.append({"id": str(qid), "response": final_response})
            audit_rows.append(
                {
                    "id": qid,
                    "old_answer": old_answer,
                    "new_answer": chosen_answer,
                    "changed": changed,
                    "reason": reason,
                    "old_valid": old_valid,
                    "old_invalid_reason": old_invalid_reason,
                    "verified": verified,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "response"], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(out_rows)

    args.audit.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "old_answer",
        "new_answer",
        "changed",
        "reason",
        "old_valid",
        "old_invalid_reason",
        "verified",
    ]
    with args.audit.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(audit_rows)

    changed_count = sum(1 for r in audit_rows if r["changed"])
    invalid_remaining = sum(1 for r in audit_rows if str(r["reason"]).startswith("invalid_no"))
    print(
        json.dumps(
            {
                "rows": len(out_rows),
                "changed": changed_count,
                "invalid_remaining": invalid_remaining,
                "verification_failures": len(failures),
                "output": str(args.output),
                "audit": str(args.audit),
            },
            indent=2,
        )
    )
    if failures:
        raise SystemExit("canonical verification failed: " + json.dumps(failures[:10], ensure_ascii=False))


if __name__ == "__main__":
    main()
