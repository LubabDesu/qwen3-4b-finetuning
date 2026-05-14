#!/usr/bin/env python3
"""Build the Stage 1-2 r2 SFT dataset from OpenR1, MathInstruct aqua_rat, and Stage 0 anchors."""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Sequence

from datasets import load_dataset


OPENR1_TARGET = 6500
MATHINSTRUCT_MAX = 1500
ANCHOR_CAP = 1000
ANCHOR_SOURCE = "stage0_format_anchor"
DEFAULT_TOKENIZER_NAME_OR_PATH = "Qwen/Qwen3-4B-Thinking-2507"
DEFAULT_MAX_RENDERED_TOKENS = 7000
PREFERRED_MIN_ROWS = 8000
PREFERRED_MAX_ROWS = 9500
MIN_ACCEPTED_ROWS = 7000

R2_SOURCE_CAPS = {
    "openr1_default": OPENR1_TARGET,
    "mathinstruct_aqua_rat": MATHINSTRUCT_MAX,
}

R2_NOISE_PHRASES = [
    "self-check",
    "self-assessment",
    "self assessment",
    "analysis:",
    "in summary",
    "to summarize",
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

ABRUPT_END_WORDS = {"we", "thus", "therefore", "hence", "so", "but", "and", "or", "then", "method"}
ABRUPT_END_PHRASES = [
    "that would mean",
    "which would mean",
    "would mean",
    "that means",
    "this means",
    "it means",
    "the answer would be",
    "the answer is",
    "answer is",
    "we get",
    "we have",
    "equals",
    "is equal to",
    "hold on",
    "wait",
    "let me check",
    "check my calculations",
]
MCQ_ANSWER_RE = re.compile(
    r"(?i)(?:the\s+answer\s+is|answer\s*:|so\s+the\s+answer\s+is|correct\s+answer\s*:?)\s*\(?([A-J])\)?\b"
)
THINK_BLOCK_RE = re.compile(r"^\s*<think>\s*(.*?)\s*</think>\s*$", flags=re.S)
TRAILING_BOX_RE = re.compile(r"\s*\\boxed\s*\{", flags=re.S)
INLINE_OPTION_RE = re.compile(r"\(([A-J])\)\s*(.*?)(?=(?:\s*\([A-J]\)\s*)|$)", flags=re.S)
CHINESE_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")


def log(message: str) -> None:
    print(f"[stage1_2_r2] {message}", flush=True)


def elapsed_seconds(start_time: float) -> str:
    return f"{time.monotonic() - start_time:.1f}s"


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


def option_labels(n: int) -> list[str]:
    return [chr(65 + i) for i in range(n)]


def normalize_mcq_letter_answer(answer: Any) -> str:
    text = normalize_text_answer(answer).strip().upper()
    match = re.fullmatch(r"\(?\s*([A-J])\s*\)?", text)
    return match.group(1) if match else text


def format_options(options: Sequence[str] | None) -> str:
    if not options:
        return ""
    return "\n".join(f"{label}. {str(opt).strip()}" for label, opt in zip(option_labels(len(options)), options))


def build_user_problem(question: str, options: Sequence[str] | None = None) -> str:
    if options:
        return f"{question}\n\nOptions:\n{format_options(options)}"
    return question


SYSTEM_PROMPT = """You are an expert mathematician. Solve the problem step by step and put your final answer within \\boxed{}.

For multi-part questions with [ANS] blanks: put all answers comma-separated in one \\boxed{}.
For MCQ: identify the correct option and put only the letter in \\boxed{}.
Never round intermediate calculations.
Never round your final answer unless the problem explicitly asks you to round.
Keep full precision in all numerical answers."""


def extract_all_boxed(text: str) -> list[str]:
    text = str(text or "")
    boxes: list[str] = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        start = match.end()
        depth = 1
        i = start
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            boxes.append(text[start : i - 1].strip())
    return boxes


def strip_outer_think(text: str) -> str:
    text = str(text or "").strip()
    match = THINK_BLOCK_RE.match(text)
    return match.group(1).strip() if match else text


def strip_trailing_boxed(text: str) -> str:
    text = str(text or "").strip()
    last = None
    for match in re.finditer(r"\\boxed\s*\{", text):
        last = match
    if last is None:
        return text
    start = last.start()
    depth = 1
    i = last.end()
    while i < len(text) and depth:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return text
    tail = text[i:].strip()
    if tail:
        return text
    return text[:start].rstrip()


def strip_answer_tail(text: str) -> str:
    text = str(text or "").strip()
    patterns = [
        r"(?is)\s*(?:therefore|thus|hence|so)?[,:]?\s*the\s+final\s+answer\s+is\b.*$",
        r"(?is)\s*(?:therefore|thus|hence|so)?[,:]?\s*the\s+correct\s+answer\s+is\b.*$",
        r"(?is)\s*(?:therefore|thus|hence|so)?[,:]?\s*the\s+answer\s+is\b.*$",
        r"(?is)\s*answer\s*:\s*.*$",
        r"(?is)\s*correct\s+answer\s*:\s*.*$",
    ]
    for pattern in patterns:
        newer = re.sub(pattern, "", text).strip()
        if newer != text:
            text = newer
    return text.strip()


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


def clean_stage0_assistant_content(text: str) -> str:
    text = str(text or "")
    replacements = [
        (r"\s*This confirms the requested (?:value|answer|expression|computation|result)\.\s*", " "),
        (
            r"\s*The computation also checks the ordering requested, which is enough to identify the requested result\.\s*",
            " The final answer is listed in the required order. ",
        ),
        (r"\s*The computation also checks the ordering requested\.\s*", " The final answer is listed in order. "),
        (r"\brequested\b", "required"),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text, flags=re.I)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def make_assistant_content(reasoning: str, final_answer: Any) -> str:
    final_raw = normalize_text_answer(final_answer)
    final_boxes = [box.strip() for box in extract_all_boxed(final_raw) if box.strip()]
    final = final_boxes[-1] if final_boxes else re.sub(r"\\boxed\s*\{\s*\}", "", final_raw).strip()
    reasoning = re.sub(r"</?think>\s*", "", str(reasoning or "")).strip()
    if not reasoning:
        reasoning = "I solve the problem carefully and keep the final answer in the required format."
    return f"<think>\n{reasoning}\n</think>\n\n\\boxed{{{final}}}"


def assistant_text_for_render(row: dict[str, Any]) -> str:
    if row.get("target"):
        return clean_stage0_assistant_content(row["target"])
    return clean_stage0_assistant_content(
        make_assistant_content(row.get("reasoning", ""), row.get("target_answer", row.get("answer", "")))
    )


def rendered_token_length(tokenizer: Any, row: dict[str, Any]) -> int:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_problem(row["question"], row.get("options"))},
        {"role": "assistant", "content": assistant_text_for_render(row)},
    ]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return len(tokenizer(full_text, add_special_tokens=False).input_ids)


def load_tokenizer_for_filter(model_name_or_path: str) -> Any:
    from transformers import AutoTokenizer

    start = time.monotonic()
    log(f"loading tokenizer for rendered-token filter: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    eos_candidates = [tokenizer.eos_token, "<|im_end|>", "<|endoftext|>"]
    for token in eos_candidates:
        if token is None:
            continue
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None:
            tokenizer.eos_token = token
            tokenizer.pad_token = token
            tokenizer.padding_side = "right"
            log(f"loaded tokenizer in {elapsed_seconds(start)}; eos/pad={token!r}")
            return tokenizer
    raise ValueError("Could not find a valid EOS token in tokenizer vocabulary")


def ends_abruptly(text: str) -> bool:
    text = str(text or "").strip()
    if not text:
        return True
    normalized = re.sub(r"[\s.。,:;，；：!?！？]+$", "", text.lower()).strip()
    if any(normalized.endswith(phrase) for phrase in ABRUPT_END_PHRASES):
        return True
    last = re.findall(r"[A-Za-z]+", text[-20:])
    return bool(last) and last[-1].lower() in ABRUPT_END_WORDS


def parse_options(question: str) -> list[str] | None:
    question = str(question or "")
    if "Answer Choices:" in question:
        question = question.split("Answer Choices:", 1)[1]
    matches = INLINE_OPTION_RE.findall(question.replace("\n", " "))
    if len(matches) < 2:
        return None
    options = []
    expected = ord(matches[0][0])
    for label, text in matches:
        if ord(label) != expected:
            return None
        expected += 1
        cleaned = re.sub(r"\s+", " ", text).strip(" ;")
        options.append(cleaned)
    return options if len(options) >= 2 else None


def normalize_openr1_answer(answer: Any) -> str:
    text = normalize_text_answer(answer)
    boxes = [b.strip() for b in extract_all_boxed(text) if b.strip()]
    if boxes:
        text = boxes[-1]
    return text.strip()


def clean_reasoning(reasoning: str, *, min_words: int = 50) -> tuple[str | None, str]:
    text = strip_outer_think(reasoning)
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
    if any(phrase in text.lower() for phrase in R2_NOISE_PHRASES):
        return None, "noise_phrase"
    if CHINESE_CHAR_RE.search(text):
        return None, "chinese_characters"
    wc = word_count(text)
    if wc < min_words:
        return None, "too_short"
    if ends_abruptly(text):
        return None, "abrupt_tail"
    return text, ""


def extract_mcq_letter(output: str) -> str:
    output = str(output or "")
    boxes = [b.strip() for b in extract_all_boxed(output) if b.strip()]
    if boxes and re.fullmatch(r"[A-J]", normalize_text_answer(boxes[-1]).upper()):
        return normalize_text_answer(boxes[-1]).upper()
    match = MCQ_ANSWER_RE.search(output)
    return match.group(1).upper() if match else ""


def remove_mcq_answer_sentence(output: str) -> str:
    text = str(output or "").strip()
    patterns = [
        r"(?is)\s*(?:therefore|thus|hence|so)?[,:]?\s*the\s+answer\s+is\s+\(?[A-J]\)?\.?\s*$",
        r"(?is)\s*(?:therefore|thus|hence|so)?[,:]?\s*answer\s*:\s*\(?[A-J]\)?\.?\s*$",
        r"(?is)\s*(?:therefore|thus|hence|so)?[,:]?\s*the\s+correct\s+answer\s+is\s+\(?[A-J]\)?\.?\s*$",
    ]
    for pattern in patterns:
        newer = re.sub(pattern, "", text).strip()
        if newer != text:
            text = newer
    return text


def is_codey_output(text: str) -> bool:
    low = str(text or "").lower()
    bad_markers = ["print(", "options =", "index =", "answers =", "```python", "def ", "return "]
    return any(marker in low for marker in bad_markers)


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


def preview_field(value: Any, limit: int = 180) -> str:
    text = str(value or "").replace("\n", "\\n").replace("\r", "\\r")
    return text if len(text) <= limit else text[:limit] + "..."


def validate_record(row: dict[str, Any]) -> str | None:
    if CHINESE_CHAR_RE.search(str(row.get("question", ""))):
        return "chinese_question"

    if row.get("source") == ANCHOR_SOURCE:
        target = str(row.get("target", ""))
        if target.count("<think>") != 1 or target.count("</think>") != 1:
            return "anchor_bad_think_tag_count"
        return None

    reasoning = str(row.get("reasoning", ""))
    answer = str(row.get("answer", ""))
    target_answer = str(row.get("target_answer", answer))

    if reasoning.count("<think>") or reasoning.count("</think>"):
        return "think_tags"
    if r"\boxed" in reasoning:
        return "boxed_remains"
    if any(phrase in reasoning.lower() for phrase in R2_NOISE_PHRASES):
        return "noise_phrase_remains"
    if CHINESE_CHAR_RE.search(reasoning):
        return "chinese_characters"
    if word_count(reasoning) < 50:
        return "reasoning_too_short"
    if row.get("is_mcq") or row.get("options"):
        if not re.fullmatch(r"[A-J]", target_answer):
            return "non_letter_mcq_target_answer"
        if not row.get("options"):
            return "mcq_without_options"
    if not answer or len(answer) >= 80 or "\n" in answer or "\r" in answer:
        return "invalid_answer_field"
    if not target_answer or len(target_answer) >= 80 or "\n" in target_answer or "\r" in target_answer:
        return "invalid_target_answer_field"
    return None


def filter_valid_records(
    records: list[dict[str, Any]],
    drop_counts: collections.Counter[str],
    *,
    max_examples: int = 20,
    log_every: int = 1000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    start = time.monotonic()

    log(f"validating {len(records)} assembled records")
    for idx, row in enumerate(records, 1):
        reason = validate_record(row)
        if reason is None:
            valid.append(row)
        else:
            drop_counts[f"validation:{reason}"] += 1
            if len(examples) < max_examples:
                examples.append(
                    {
                        "row_index": idx - 1,
                        "source": row.get("source"),
                        "original_source": row.get("original_source"),
                        "reason": reason,
                        "question_preview": preview_field(row.get("question")),
                        "answer_preview": preview_field(row.get("answer")),
                        "target_answer_preview": preview_field(row.get("target_answer", row.get("answer"))),
                    }
                )

        if log_every > 0 and (idx % log_every == 0 or idx == len(records)):
            log(f"validation progress {idx}/{len(records)} kept={len(valid)} elapsed={elapsed_seconds(start)}")

    return valid, examples


def filter_rendered_token_lengths(
    records: list[dict[str, Any]],
    drop_counts: collections.Counter[str],
    *,
    tokenizer: Any | None,
    max_rendered_tokens: int | None,
    max_examples: int = 20,
    log_every: int = 1000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if tokenizer is None or max_rendered_tokens is None or max_rendered_tokens <= 0:
        log("rendered-token filter disabled")
        return records, {"enabled": False}

    kept: list[dict[str, Any]] = []
    lengths_by_source: dict[str, list[int]] = collections.defaultdict(list)
    examples: list[dict[str, Any]] = []
    start = time.monotonic()

    log(f"rendered-token filtering {len(records)} records with max_rendered_tokens={max_rendered_tokens}")
    for idx, row in enumerate(records, 1):
        source = str(row.get("source", "unknown"))
        try:
            length = rendered_token_length(tokenizer, row)
        except Exception as exc:
            drop_counts[f"{source}:rendered_tokenize_error"] += 1
            if len(examples) < max_examples:
                examples.append(
                    {
                        "row_index": idx - 1,
                        "source": source,
                        "reason": "rendered_tokenize_error",
                        "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                        "question_preview": preview_field(row.get("question")),
                    }
                )
            continue

        lengths_by_source[source].append(length)
        if length > max_rendered_tokens:
            drop_counts[f"{source}:rendered_too_long"] += 1
            if len(examples) < max_examples:
                examples.append(
                    {
                        "row_index": idx - 1,
                        "source": source,
                        "reason": "rendered_too_long",
                        "rendered_tokens": length,
                        "max_rendered_tokens": max_rendered_tokens,
                        "question_preview": preview_field(row.get("question")),
                    }
                )
            continue

        row = dict(row)
        row["rendered_tokens"] = length
        kept.append(row)

        if log_every > 0 and (idx % log_every == 0 or idx == len(records)):
            log(
                "rendered-token progress "
                f"{idx}/{len(records)} kept={len(kept)} dropped={idx - len(kept)} elapsed={elapsed_seconds(start)}"
            )

    stats_by_source: dict[str, dict[str, int]] = {}
    for source, values in lengths_by_source.items():
        values = sorted(values)
        if values:
            stats_by_source[source] = {
                "min": values[0],
                "median": values[len(values) // 2],
                "p95": values[int(0.95 * (len(values) - 1))],
                "max": values[-1],
            }

    return kept, {
        "enabled": True,
        "max_rendered_tokens": max_rendered_tokens,
        "dropped": len(records) - len(kept),
        "token_lengths_by_source": stats_by_source,
        "drop_examples": examples,
    }


def choose_openr1_generation(row: dict[str, Any]) -> tuple[str | None, str]:
    generations = list(row.get("generations") or [])
    complete = list(row.get("is_reasoning_complete") or [])
    correct = list(row.get("correctness_math_verify") or [])
    candidates = []
    for idx, generation in enumerate(generations):
        if idx >= len(complete) or idx >= len(correct):
            continue
        if correct[idx] is True and complete[idx] is True:
            cleaned, reason = clean_reasoning(generation)
            if cleaned:
                candidates.append((word_count(cleaned), cleaned))
    if not candidates:
        return None, "no_clean_verified_generation"
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1], ""


def load_openr1_rows(
    seed: int,
    drop_counts: collections.Counter[str],
    *,
    cap: int | None = OPENR1_TARGET,
    tokenizer: Any | None = None,
    max_rendered_tokens: int | None = None,
    log_every: int = 1000,
) -> list[dict[str, Any]]:
    start = time.monotonic()
    log("loading OpenR1 dataset split: open-r1/OpenR1-Math-220k/default train")
    ds = load_dataset("open-r1/OpenR1-Math-220k", "default", split="train")
    rows = list(ds)
    log(f"loaded OpenR1 rows={len(rows)} in {elapsed_seconds(start)}; shuffling with seed={seed}")
    random.Random(seed).shuffle(rows)
    accepted: list[dict[str, Any]] = []
    scan_start = time.monotonic()
    target_label = "uncapped" if cap is None else str(cap)
    log(f"scanning OpenR1 rows for target={target_label}")
    for seen, row in enumerate(rows, 1):
        if cap is not None and len(accepted) >= cap:
            break
        if log_every > 0 and seen % log_every == 0:
            log(
                "OpenR1 progress "
                f"seen={seen}/{len(rows)} accepted={len(accepted)}/{target_label} "
                f"drops={sum(v for k, v in drop_counts.items() if k.startswith('openr1:'))} "
                f"elapsed={elapsed_seconds(scan_start)}"
            )
        reasoning, reason = choose_openr1_generation(row)
        if reason:
            drop_counts[f"openr1:{reason}"] += 1
            continue
        answer = normalize_openr1_answer(row.get("answer"))
        if not answer:
            drop_counts["openr1:empty_answer"] += 1
            continue
        question = str(row.get("problem", "")).strip()
        if CHINESE_CHAR_RE.search(question):
            drop_counts["openr1:chinese_question"] += 1
            continue
        options = parse_options(question)
        if options:
            answer_letter = normalize_mcq_letter_answer(answer)
            valid_labels = set(option_labels(len(options)))
            if answer_letter not in valid_labels:
                drop_counts["openr1:mcq_answer_not_option_label"] += 1
                continue
            answer = answer_letter
            target_answer = answer_letter
            is_mcq = True
        else:
            target_answer = answer
            is_mcq = False
        record = {
            "question": question,
            "options": options,
            "answer": answer,
            "target_answer": target_answer,
            "reasoning": reasoning,
            "source": "openr1_default",
            "original_source": f"openr1:{row.get('source', 'unknown')}",
            "n_ans": question.count("[ANS]"),
            "is_mcq": is_mcq,
            "openr1_question_type": row.get("question_type"),
            "openr1_problem_type": row.get("problem_type"),
            "openr1_uuid": row.get("uuid"),
        }
        if tokenizer is not None and max_rendered_tokens:
            try:
                length = rendered_token_length(tokenizer, record)
            except Exception:
                drop_counts["openr1:rendered_tokenize_error"] += 1
                continue
            if length > max_rendered_tokens:
                drop_counts["openr1:rendered_too_long"] += 1
                continue
            record["rendered_tokens"] = length
        accepted.append(record)
        if log_every > 0 and cap is not None and len(accepted) >= cap:
            log(
                "OpenR1 progress "
                f"seen={seen}/{len(rows)} accepted={len(accepted)}/{target_label} "
                f"drops={sum(v for k, v in drop_counts.items() if k.startswith('openr1:'))} "
                f"elapsed={elapsed_seconds(scan_start)}"
            )
    log(f"finished OpenR1 accepted={len(accepted)} elapsed={elapsed_seconds(scan_start)}")
    return accepted


def load_mathinstruct_rows(
    seed: int,
    drop_counts: collections.Counter[str],
    *,
    cap: int | None = MATHINSTRUCT_MAX,
    tokenizer: Any | None = None,
    max_rendered_tokens: int | None = None,
    log_every: int = 1000,
) -> list[dict[str, Any]]:
    start = time.monotonic()
    log("loading MathInstruct dataset split: TIGER-Lab/MathInstruct train")
    ds = load_dataset("TIGER-Lab/MathInstruct", split="train")
    rows = [row for row in ds if str(row.get("source")) == "data/CoT/aqua_rat.json"]
    log(f"loaded MathInstruct aqua_rat rows={len(rows)} in {elapsed_seconds(start)}; shuffling with seed={seed}")
    random.Random(seed).shuffle(rows)
    accepted: list[dict[str, Any]] = []
    scan_start = time.monotonic()
    target_label = "uncapped" if cap is None else str(cap)
    log(f"scanning MathInstruct aqua_rat rows for max={target_label}")
    for seen, row in enumerate(rows, 1):
        if cap is not None and len(accepted) >= cap:
            break
        if log_every > 0 and seen % log_every == 0:
            log(
                "MathInstruct progress "
                f"seen={seen}/{len(rows)} accepted={len(accepted)}/{target_label} "
                f"drops={sum(v for k, v in drop_counts.items() if k.startswith('mathinstruct:'))} "
                f"elapsed={elapsed_seconds(scan_start)}"
            )
        question = str(row.get("instruction", "")).strip()
        if CHINESE_CHAR_RE.search(question):
            drop_counts["mathinstruct:chinese_question"] += 1
            continue
        output = str(row.get("output", "")).strip()
        options = parse_options(question)
        answer = extract_mcq_letter(output)
        if not question or not output or not options:
            drop_counts["mathinstruct:missing_question_or_options"] += 1
            continue
        if not answer:
            drop_counts["mathinstruct:answer_extract_fail"] += 1
            continue
        if answer not in {chr(65 + i) for i in range(len(options))}:
            drop_counts["mathinstruct:answer_not_in_options"] += 1
            continue
        if is_codey_output(output):
            drop_counts["mathinstruct:codey_output"] += 1
            continue
        reasoning = remove_mcq_answer_sentence(output)
        reasoning, reason = clean_reasoning(reasoning)
        if reason:
            drop_counts[f"mathinstruct:{reason}"] += 1
            continue
        record = {
            "question": question,
            "options": options,
            "answer": answer,
            "target_answer": answer,
            "reasoning": reasoning,
            "source": "mathinstruct_aqua_rat",
            "original_source": "mathinstruct:data/CoT/aqua_rat.json",
            "n_ans": question.count("[ANS]"),
            "is_mcq": True,
        }
        if tokenizer is not None and max_rendered_tokens:
            try:
                length = rendered_token_length(tokenizer, record)
            except Exception:
                drop_counts["mathinstruct:rendered_tokenize_error"] += 1
                continue
            if length > max_rendered_tokens:
                drop_counts["mathinstruct:rendered_too_long"] += 1
                continue
            record["rendered_tokens"] = length
        accepted.append(record)
        if log_every > 0 and cap is not None and len(accepted) >= cap:
            log(
                "MathInstruct progress "
                f"seen={seen}/{len(rows)} accepted={len(accepted)}/{target_label} "
                f"drops={sum(v for k, v in drop_counts.items() if k.startswith('mathinstruct:'))} "
                f"elapsed={elapsed_seconds(scan_start)}"
            )
    log(f"finished MathInstruct accepted={len(accepted)} elapsed={elapsed_seconds(scan_start)}")
    return accepted


def build_stage1_2_r2_records(
    *,
    anchor_path: Path,
    out_path: Path,
    manifest_path: Path | None = None,
    seed: int = 151,
    force_rebuild: bool = False,
    tokenizer_name_or_path: str | None = None,
    max_rendered_tokens: int | None = DEFAULT_MAX_RENDERED_TOKENS,
    include_stage0_anchors: bool = True,
    openr1_cap: int | None = OPENR1_TARGET,
    mathinstruct_cap: int | None = MATHINSTRUCT_MAX,
    dry_run: bool = False,
    log_every: int = 1000,
) -> list[dict[str, Any]]:
    build_start = time.monotonic()
    log(
        "build requested "
        f"out_path={out_path} force_rebuild={force_rebuild} "
        f"tokenizer={tokenizer_name_or_path} max_rendered_tokens={max_rendered_tokens} "
        f"include_stage0_anchors={include_stage0_anchors} "
        f"openr1_cap={openr1_cap if openr1_cap is not None else 'uncapped'} "
        f"mathinstruct_cap={mathinstruct_cap if mathinstruct_cap is not None else 'uncapped'} "
        f"dry_run={dry_run}"
    )
    if out_path.exists() and not force_rebuild:
        records = load_jsonl(out_path)
        log(f"loaded cached records={len(records)} from {out_path}")
        return records

    drop_counts: collections.Counter[str] = collections.Counter()
    tokenizer = load_tokenizer_for_filter(tokenizer_name_or_path) if tokenizer_name_or_path and max_rendered_tokens else None
    openr1_records = load_openr1_rows(
        seed,
        drop_counts,
        cap=openr1_cap,
        tokenizer=tokenizer,
        max_rendered_tokens=max_rendered_tokens,
        log_every=log_every,
    )
    mathinstruct_records = load_mathinstruct_rows(
        seed + 17,
        drop_counts,
        cap=mathinstruct_cap,
        tokenizer=tokenizer,
        max_rendered_tokens=max_rendered_tokens,
        log_every=log_every,
    )

    if openr1_cap is not None and len(openr1_records) < openr1_cap:
        drop_counts["openr1:shortfall"] += openr1_cap - len(openr1_records)

    openr1_target = len(openr1_records)
    anchor_target = ANCHOR_CAP if include_stage0_anchors else 0
    if mathinstruct_cap is None:
        mathinstruct_target = len(mathinstruct_records)
    else:
        mathinstruct_target = min(mathinstruct_cap, max(1000, PREFERRED_MIN_ROWS - anchor_target - openr1_target))
    selected_mathinstruct = mathinstruct_records[:mathinstruct_target]
    if mathinstruct_cap is not None and len(selected_mathinstruct) < 1000:
        drop_counts["mathinstruct:shortfall"] += 1000 - len(selected_mathinstruct)

    if include_stage0_anchors:
        log(f"loading Stage 0 anchors from {anchor_path}")
        anchors = build_stage0_anchors(load_jsonl(anchor_path), seed=seed + 77, n=ANCHOR_CAP)
    else:
        log("skipping Stage 0 anchors because include_stage0_anchors=False")
        anchors = []
    records = openr1_records + selected_mathinstruct + anchors
    random.Random(seed + 123).shuffle(records)
    log(
        "assembled records before validation "
        f"openr1={len(openr1_records)} mathinstruct={len(selected_mathinstruct)} anchors={len(anchors)} total={len(records)}"
    )

    records_before_validation = len(records)
    records, validation_error_examples = filter_valid_records(records, drop_counts, log_every=log_every)
    validation_error_count = records_before_validation - len(records)
    final_records_after_validation = len(records)
    log(f"validation complete kept={len(records)} dropped={validation_error_count}")

    records_before_token_filter = len(records)
    records, token_filter_manifest = filter_rendered_token_lengths(
        records,
        drop_counts,
        tokenizer=tokenizer,
        max_rendered_tokens=max_rendered_tokens,
        log_every=log_every,
    )
    rendered_token_drop_count = records_before_token_filter - len(records)
    log(f"rendered-token filtering complete kept={len(records)} dropped={rendered_token_drop_count}")

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

    manifest = {
        "source_caps": R2_SOURCE_CAPS,
        "requested_source_caps": {
            "openr1_default": openr1_cap,
            "mathinstruct_aqua_rat": mathinstruct_cap,
        },
        "include_stage0_anchors": include_stage0_anchors,
        "stage0_anchor_cap": ANCHOR_CAP,
        "preferred_row_range": [PREFERRED_MIN_ROWS, PREFERRED_MAX_ROWS],
        "minimum_row_count": MIN_ACCEPTED_ROWS,
        "total_records": len(records),
        "records_before_validation": records_before_validation,
        "final_records_after_validation": final_records_after_validation,
        "records_before_token_filter": records_before_token_filter,
        "rendered_token_drop_count": rendered_token_drop_count,
        "rendered_token_filter": token_filter_manifest,
        "source_counts": dict(source_counts),
        "accepted_by_source": {
            "openr1_default": len(openr1_records),
            "mathinstruct_aqua_rat": len(selected_mathinstruct),
            ANCHOR_SOURCE: len(anchors),
        },
        "drop_counts": dict(drop_counts),
        "validation_error_count": validation_error_count,
        "validation_error_examples": validation_error_examples,
        "word_counts_by_source": word_counts_by_source,
        "anchor_path": str(anchor_path),
        "out_path": str(out_path),
        "assertions_passed": validation_error_count == 0,
        "dry_run": dry_run,
    }

    if dry_run:
        log(f"dry run enabled; not writing records to {out_path}")
    else:
        log(f"writing {len(records)} records to {out_path}")
        write_jsonl(records, out_path)
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        log(f"writing manifest to {manifest_path}")
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    log(f"build finished in {elapsed_seconds(build_start)}; final_records={len(records)}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-path", type=Path, default=Path("artifacts/post_training_curriculum/datasets/stage0b_final_finetune.jsonl"))
    parser.add_argument("--out-path", type=Path, default=Path("artifacts/post_training_curriculum/datasets/stage1_2_r2_records.jsonl"))
    parser.add_argument("--manifest-path", type=Path, default=Path("artifacts/post_training_curriculum/datasets/stage1_2_r2_manifest.json"))
    parser.add_argument("--seed", type=int, default=151)
    parser.add_argument("--tokenizer-name-or-path", default=DEFAULT_TOKENIZER_NAME_OR_PATH)
    parser.add_argument("--max-rendered-tokens", type=int, default=DEFAULT_MAX_RENDERED_TOKENS)
    parser.add_argument("--openr1-cap", type=int, default=OPENR1_TARGET, help="Max OpenR1 rows to accept; use 0 for uncapped.")
    parser.add_argument("--mathinstruct-cap", type=int, default=MATHINSTRUCT_MAX, help="Max MathInstruct aqua_rat rows to accept; use 0 for uncapped.")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Build and write manifest only; do not overwrite the dataset JSONL.")
    parser.add_argument(
        "--exclude-stage0-anchors",
        action="store_true",
        help="Do not include Stage 0B format-anchor rows in the Stage 1-2 dataset.",
    )
    parser.add_argument("--log-every", type=int, default=1000, help="Print loop progress every N scanned records; use 0 to disable.")
    args = parser.parse_args()
    openr1_cap = None if args.openr1_cap == 0 else args.openr1_cap
    mathinstruct_cap = None if args.mathinstruct_cap == 0 else args.mathinstruct_cap

    records = build_stage1_2_r2_records(
        anchor_path=args.anchor_path,
        out_path=args.out_path,
        manifest_path=args.manifest_path,
        seed=args.seed,
        force_rebuild=args.force_rebuild,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        max_rendered_tokens=args.max_rendered_tokens,
        include_stage0_anchors=not args.exclude_stage0_anchors,
        openr1_cap=openr1_cap,
        mathinstruct_cap=mathinstruct_cap,
        dry_run=args.dry_run,
        log_every=args.log_every,
    )
    manifest = json.loads(args.manifest_path.read_text())
    print(f"Built {len(records)} Stage 1-2 r2 records: {args.out_path}")
    print("Source counts:", manifest["source_counts"])
    print("Accepted by source before anchors:", manifest["accepted_by_source"])
    print("Top drop counts:", dict(collections.Counter(manifest["drop_counts"]).most_common(20)))
    print("Word counts by source:", manifest["word_counts_by_source"])
    print("Rendered token filter:", manifest.get("rendered_token_filter"))


if __name__ == "__main__":
    main()
