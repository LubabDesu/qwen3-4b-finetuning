#!/usr/bin/env python3
"""Build a rejection-sampled SFT dataset from a GRPO checkpoint.

This script generates several completions per training prompt, keeps candidates
that the competition-style judger marks correct, canonicalizes recoverable
answers into:

<think>
...
</think>

\boxed{answer}

and writes a Stage-1/2-style JSONL that can be used for a short assistant-only
SFT repair pass.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from judger import Judger  # noqa: E402


SYSTEM_PROMPT = """You are an expert mathematician. Solve the problem step by step.

Final answer rules:
- Use exactly one final \\boxed{}.
- For multi-part questions with multiple [ANS] blanks, put all answers comma-separated in that one box, in blank order.
- Wrong: \\boxed{2}, \\boxed{4}, \\boxed{120}, \\boxed{-6}
- Right: \\boxed{2, 4, 120, -6}
- For MCQ, put only the option letter in \\boxed{}.
- Never round intermediate calculations.
- Never round your final answer unless the problem explicitly asks you to round.
- For decimal numerical answers, keep enough digits to match the unrounded value.
- When you have determined the answer, stop reasoning immediately and end with the final box.
- Do not write any text after the final \\boxed{}."""


DEFAULT_INPUTS = [
    ROOT / "artifacts" / "grpo" / "public_train_300.jsonl",
    ROOT / "data" / "public_train_300.jsonl",
    ROOT / "artifacts" / "grpo" / "deepmath_filtered_problems.jsonl",
]
DEFAULT_OUT = ROOT / "artifacts" / "grpo_rs_sft" / "rs_sft_records.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "vLLM-loadable model path/name. If omitted, pass --lora-checkpoint "
            "and the script will merge it into --merged-model-dir first."
        ),
    )
    parser.add_argument(
        "--lora-checkpoint",
        type=Path,
        default=None,
        help="Optional Unsloth/PEFT LoRA checkpoint directory to merge before generation.",
    )
    parser.add_argument(
        "--merged-model-dir",
        type=Path,
        default=ROOT / "artifacts" / "grpo_rs_sft" / "merged_for_rs_sft",
        help="Where to write/reuse the merged model when --lora-checkpoint is set.",
    )
    parser.add_argument(
        "--drive-source",
        default=None,
        help=(
            "Optional rclone source containing checkpoint-N folders. If the local "
            "--lora-checkpoint is missing adapter_model.safetensors, the script "
            "copies {drive-source}/{checkpoint-name} into --lora-checkpoint first."
        ),
    )
    parser.add_argument("--rclone-transfers", type=int, default=1)
    parser.add_argument("--rclone-checkers", type=int, default=1)
    parser.add_argument("--rclone-drive-chunk-size", default="128M")
    parser.add_argument("--rclone-tpslimit", type=float, default=1.0)
    parser.add_argument("--rclone-tpslimit-burst", type=int, default=1)
    parser.add_argument("--rclone-drive-pacer-min-sleep", default="5s")
    parser.add_argument("--rclone-drive-pacer-burst", type=int, default=1)
    parser.add_argument("--rclone-retries", type=int, default=20)
    parser.add_argument("--rclone-low-level-retries", type=int, default=50)
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer path/name. Defaults to --model.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        action="append",
        default=None,
        help=(
            "Training-side JSONL prompt pool. Can be repeated. Defaults to "
            "artifacts/grpo/public_train_300.jsonl when present plus DeepMath."
        ),
    )
    parser.add_argument("--out-path", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--candidate-path",
        type=Path,
        default=None,
        help="Optional explicit candidate JSONL path. Defaults to <out_stem>_candidates.jsonl next to --out-path.",
    )
    parser.add_argument(
        "--partial-path",
        type=Path,
        default=None,
        help="Optional explicit partial SFT JSONL path. Defaults to <out_stem>.partial.jsonl next to --out-path.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional explicit manifest JSON path. Defaults to --out-path with .manifest.json suffix.",
    )
    parser.add_argument("--sample-size", type=int, default=500, help="Number of prompts to sample.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-generations", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=6144)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--vllm-dtype", default="bfloat16")
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help=(
            "Fallback hard tokenizer cap on cleaned assistant completion tokens. "
            "By default the script uses tiered caps by question type."
        ),
    )
    parser.add_argument(
        "--preferred-output-tokens",
        type=int,
        default=None,
        help=(
            "Fallback preferred tokenizer cap for candidate ranking. By default "
            "the script uses tiered caps by question type."
        ),
    )
    parser.add_argument(
        "--flat-output-caps",
        action="store_true",
        help="Use --preferred-output-tokens/--max-output-tokens for every question type.",
    )
    parser.add_argument("--mcq-preferred-output-tokens", type=int, default=768)
    parser.add_argument("--mcq-max-output-tokens", type=int, default=1536)
    parser.add_argument("--single-preferred-output-tokens", type=int, default=2048)
    parser.add_argument("--single-max-output-tokens", type=int, default=3072)
    parser.add_argument("--multi-preferred-output-tokens", type=int, default=3072)
    parser.add_argument("--multi-max-output-tokens", type=int, default=4096)
    parser.add_argument(
        "--max-prompts-per-source",
        type=int,
        default=None,
        help="Optional cap per source value after loading.",
    )
    parser.add_argument(
        "--strict-extract",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use strict answer extraction while judging generated candidates.",
    )
    parser.add_argument("--report-every", type=int, default=25)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load/sample prompts and tokenizer, then exit before vLLM generation.",
    )
    parser.add_argument(
        "--resume-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Resume from an existing *_candidates.jsonl by skipping prompt rows "
            "whose candidates were already generated."
        ),
    )
    args = parser.parse_args()
    if not args.model and not args.lora_checkpoint:
        parser.error("one of --model or --lora-checkpoint is required")
    return args


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_gold(row: dict[str, Any]) -> list[str]:
    raw = row.get("gold_answer", row.get("gold", row.get("answer", [])))
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return [str(raw)]


def normalize_options(row: dict[str, Any]) -> list[str]:
    options = row.get("options") or []
    if isinstance(options, str):
        try:
            parsed = json.loads(options)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except Exception:
            return []
    if isinstance(options, list):
        return [str(item) for item in options]
    return []


def format_question(question: str, options: list[str] | None = None) -> str:
    question = str(question).strip()
    if not options:
        return question
    lines = [f"{chr(ord('A') + idx)}. {str(option).strip()}" for idx, option in enumerate(options)]
    return f"{question}\n\nOptions:\n" + "\n".join(lines)


def build_prompt(question: str, options: list[str] | None = None) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{format_question(question, options)}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def load_prompt_pool(args: argparse.Namespace) -> list[dict[str, Any]]:
    input_paths = args.input_path
    if input_paths is None:
        input_paths = [path for path in DEFAULT_INPUTS if path.exists()]
    rows: list[dict[str, Any]] = []
    per_source_counts: collections.Counter[str] = collections.Counter()

    for raw_path in input_paths:
        path = resolve_path(raw_path)
        if not path.exists():
            print(f"[rs-sft] warning: missing input path skipped: {path}", flush=True)
            continue
        loaded = load_jsonl(path)
        print(f"[rs-sft] loaded {len(loaded)} rows from {path}", flush=True)
        for row in loaded:
            source = str(row.get("source") or path.stem)
            if args.max_prompts_per_source is not None and per_source_counts[source] >= args.max_prompts_per_source:
                continue
            question = str(row.get("question", "")).strip()
            if not question:
                continue
            item = dict(row)
            item["_input_path"] = str(path)
            item["_gold"] = normalize_gold(row)
            item["_options"] = normalize_options(row)
            item["_source"] = source
            rows.append(item)
            per_source_counts[source] += 1

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.sample_size and args.sample_size > 0:
        rows = rows[: args.sample_size]
    print(f"[rs-sft] sampled prompt rows: {len(rows)}", flush=True)
    return rows


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
        if depth == 0:
            spans.append((idx, pos, "".join(chars).strip()))
            start = pos
        else:
            break
    return spans


def strip_boxed(text: str) -> str:
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


def truncate_reasoning(text: str) -> str:
    cut_patterns = [
        r"\*\*Final Answer\*\*",
        r"### Final Answer",
        r"## Final Answer",
        r"Final Answer",
        r"Final Output",
        r"Final Result",
        r"Therefore,\s*$",
    ]
    out = text
    for pattern in cut_patterns:
        match = re.search(pattern, out, flags=re.I)
        if match:
            out = out[: match.start()]
            break
    return out


def trim_dangling_answer_sentence(text: str) -> str:
    patterns = [
        r"(?:So\s+)?(?:the\s+)?answer\s+(?:should\s+be|is)\s*[\.:]?\s*$",
        r"(?:So\s+)?(?:the\s+)?final\s+answer\s+(?:should\s+be|is)\s*[\.:]?\s*$",
        r"(?:So\s+)?(?:the\s+)?result\s+(?:should\s+be|is)\s*[\.:]?\s*$",
    ]
    out = text.rstrip()
    for pattern in patterns:
        out = re.sub(pattern, "", out, flags=re.I).rstrip()
    return out


def clean_reasoning(completion: str) -> str:
    if "</think>" in completion:
        reasoning = completion.split("</think>", 1)[0]
    else:
        first_box = completion.find(r"\boxed{")
        reasoning = completion[:first_box] if first_box >= 0 else completion
    reasoning = re.sub(r"</?think>\s*", "", reasoning)
    reasoning = strip_boxed(reasoning)
    reasoning = truncate_reasoning(reasoning)
    reasoning = re.sub(r"<\|im_(?:start|end)\|>", "", reasoning)
    reasoning = re.sub(r"\n{3,}", "\n\n", reasoning)
    reasoning = re.sub(r"[ \t]{2,}", " ", reasoning)
    reasoning = trim_dangling_answer_sentence(reasoning)
    reasoning = reasoning.strip()
    if not reasoning:
        reasoning = "I solve the problem and stop once the final answer is determined."
    return reasoning


def normalize_final_answer(extracted: str, gold: list[str], options: list[str]) -> str:
    answer = str(extracted or "").strip()
    answer = answer.strip("$").strip()
    is_letter_gold = bool(gold) and all(re.fullmatch(r"[A-Ja-j]", str(item).strip()) for item in gold)
    if (options or is_letter_gold) and len(gold) == 1:
        match = re.search(r"\b([A-Ja-j])\b", answer)
        if match:
            return match.group(1).upper()
    return answer


def canonical_completion(reasoning: str, answer: str) -> str:
    return f"<think>\n{reasoning.strip()}\n</think>\n\n\\boxed{{{answer.strip()}}}"


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def rendered_token_count(tokenizer: Any, row: dict[str, Any]) -> int:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_question(row["question"], row.get("options") or [])},
        {"role": "assistant", "content": row["target"]},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return token_count(tokenizer, rendered)


def question_type(row: dict[str, Any]) -> str:
    if row["_options"]:
        return "mcq"
    if len(row["_gold"]) > 1:
        return "multi"
    return "single"


def output_caps_for_row(args: argparse.Namespace, row: dict[str, Any]) -> tuple[str, int, int]:
    qtype = question_type(row)
    if args.flat_output_caps:
        if args.max_output_tokens is None or args.preferred_output_tokens is None:
            raise ValueError("--flat-output-caps requires --max-output-tokens and --preferred-output-tokens")
        return qtype, args.max_output_tokens, args.preferred_output_tokens

    if qtype == "mcq":
        hard_cap = args.mcq_max_output_tokens
        preferred_cap = args.mcq_preferred_output_tokens
    elif qtype == "multi":
        hard_cap = args.multi_max_output_tokens
        preferred_cap = args.multi_preferred_output_tokens
    else:
        hard_cap = args.single_max_output_tokens
        preferred_cap = args.single_preferred_output_tokens
    return qtype, hard_cap, preferred_cap


def judge_correct(judger: Judger, completion: str, gold: list[str], options: list[str]) -> bool:
    try:
        return bool(judger.auto_judge(pred=completion, gold=list(gold), options=list(options)))
    except Exception:
        return False


def classify_structure(completion: str) -> dict[str, Any]:
    before, sep, after = completion.partition("</think>")
    final_region = after if sep else completion
    before_boxes = extract_boxed_spans(before)
    final_boxes = extract_boxed_spans(final_region)
    all_boxes = extract_boxed_spans(completion)
    text_after_final = ""
    if final_boxes:
        text_after_final = final_region[final_boxes[-1][1] :].strip()
    elif all_boxes:
        text_after_final = completion[all_boxes[-1][1] :].strip()
    return {
        "has_think_close": bool(sep),
        "pre_think_box_count": len(before_boxes),
        "final_box_count": len(final_boxes),
        "total_box_count": len(all_boxes),
        "has_text_after_final_box": bool(text_after_final),
    }


def build_candidate_record(
    *,
    tokenizer: Any,
    judger: Judger,
    row: dict[str, Any],
    completion: str,
    candidate_index: int,
    max_output_tokens: int,
    preferred_output_tokens: int,
) -> dict[str, Any]:
    gold = row["_gold"]
    options = row["_options"]
    structure = classify_structure(completion)
    correct = judge_correct(judger, completion, gold, options)
    extracted = judger.extract_ans(completion)

    accepted = False
    reject_reason = ""
    cleaned = ""
    answer = ""
    completion_tokens = token_count(tokenizer, completion)
    cleaned_tokens = None

    if not correct:
        reject_reason = "incorrect"
    elif not extracted:
        reject_reason = "correct_but_no_recoverable_answer"
    else:
        answer = normalize_final_answer(extracted, gold, options)
        reasoning = clean_reasoning(completion)
        cleaned = canonical_completion(reasoning, answer)
        cleaned_tokens = token_count(tokenizer, cleaned)
        if cleaned_tokens > max_output_tokens:
            reject_reason = "cleaned_too_long"
        else:
            accepted = True
            reject_reason = ""

    clean_tier = "rejected"
    if accepted:
        clean_tier = "clean" if (
            structure["has_think_close"]
            and structure["pre_think_box_count"] == 0
            and structure["final_box_count"] == 1
            and structure["total_box_count"] == 1
            and not structure["has_text_after_final_box"]
            and completion_tokens <= max_output_tokens
        ) else "postprocessed"

    return {
        "candidate_index": candidate_index,
        "accepted": accepted,
        "tier": clean_tier,
        "reject_reason": reject_reason,
        "correct": correct,
        "extracted_answer": extracted,
        "normalized_answer": answer,
        "completion_tokens": completion_tokens,
        "cleaned_tokens": cleaned_tokens,
        "under_preferred_cap": bool(cleaned_tokens is not None and cleaned_tokens <= preferred_output_tokens),
        "structure": structure,
        "completion": completion,
        "cleaned_completion": cleaned,
    }


def select_best(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    accepted = [c for c in candidates if c["accepted"]]
    if not accepted:
        return None
    tier_rank = {"clean": 0, "postprocessed": 1}
    return min(
        accepted,
        key=lambda c: (
            0 if c["under_preferred_cap"] else 1,
            tier_rank.get(c["tier"], 9),
            int(c["cleaned_tokens"] or 10**9),
            int(c["completion_tokens"]),
            int(c["candidate_index"]),
        ),
    )


def prompt_key(row: dict[str, Any], index: int) -> str:
    source = str(row.get("_source") or row.get("source") or "")
    row_id = str(row.get("id") or index)
    question = str(row.get("question") or "")
    return f"{index}\t{source}\t{row_id}\t{question}"


def candidate_log_key(log: dict[str, Any]) -> str | None:
    key = log.get("prompt_key")
    if isinstance(key, str) and key:
        return key
    index = log.get("prompt_index")
    if index is None:
        return None
    source = str(log.get("source") or "")
    row_id = str(log.get("id") or index)
    question = str(log.get("question") or "")
    return f"{index}\t{source}\t{row_id}\t{question}"


def load_candidate_logs(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    logs: dict[str, dict[str, Any]] = {}
    for log in load_jsonl(path):
        key = candidate_log_key(log)
        if key is not None:
            logs[key] = log
    return logs


def sft_record_from_candidate_log(
    *,
    tokenizer: Any,
    row: dict[str, Any],
    global_idx: int,
    log: dict[str, Any],
) -> dict[str, Any] | None:
    candidates = log.get("candidates") or []
    selected = select_best(candidates)
    if selected is None:
        return None
    qtype = str(log.get("question_type") or question_type(row))
    answer = selected["normalized_answer"]
    reasoning = clean_reasoning(selected["completion"])
    target = canonical_completion(reasoning, answer)
    cleaned_tokens = token_count(tokenizer, target)
    record = {
        "id": f"rs_sft_{row.get('id', global_idx)}",
        "question": row["question"],
        "options": row["_options"],
        "answer": answer,
        "target_answer": answer,
        "reasoning": reasoning,
        "target": target,
        "source": "rs_sft_grpo",
        "original_source": row.get("_source"),
        "original_id": row.get("id"),
        "is_mcq": bool(row["_options"]),
        "n_ans": len(row["_gold"]),
        "gold_answer": row["_gold"],
        "rs_sft": {
            "question_type": qtype,
            "selected_candidate_index": selected["candidate_index"],
            "tier": selected["tier"],
            "completion_tokens": selected["completion_tokens"],
            "cleaned_tokens": cleaned_tokens,
            "preferred_output_tokens": log.get("preferred_output_tokens"),
            "max_output_tokens": log.get("max_output_tokens"),
            "num_generations": len(candidates),
        },
    }
    try:
        record["rendered_tokens"] = rendered_token_count(tokenizer, record)
    except Exception:
        record["rendered_tokens"] = None
    return record


def rebuild_sft_rows_from_logs(
    *,
    tokenizer: Any,
    prompt_rows: list[dict[str, Any]],
    candidate_logs_by_key: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(prompt_rows):
        log = candidate_logs_by_key.get(prompt_key(row, index))
        if log is None:
            continue
        record = sft_record_from_candidate_log(
            tokenizer=tokenizer,
            row=row,
            global_idx=index,
            log=log,
        )
        if record is not None:
            rows.append(record)
    return rows


def build_stats_from_candidate_logs(candidate_logs: list[dict[str, Any]]) -> collections.Counter[str]:
    stats: collections.Counter[str] = collections.Counter()
    for log in candidate_logs:
        qtype = str(log.get("question_type") or "unknown")
        stats[f"prompt_type:{qtype}"] += 1
        candidates = log.get("candidates") or []
        selected = select_best(candidates)
        if selected is None:
            stats["prompts_without_accepted"] += 1
            stats[f"prompts_without_accepted:{qtype}"] += 1
        else:
            stats["prompts_with_accepted"] += 1
            stats[f"prompts_with_accepted:{qtype}"] += 1
            stats[f"selected_tier:{selected['tier']}"] += 1
            stats[f"selected_type:{qtype}"] += 1
        for candidate in candidates:
            if candidate.get("accepted"):
                stats["accepted_candidates"] += 1
                stats[f"accepted_tier:{candidate['tier']}"] += 1
                stats[f"accepted_type:{qtype}"] += 1
            else:
                reject_reason = candidate.get("reject_reason", "unknown")
                stats[f"reject:{reject_reason}"] += 1
                stats[f"reject:{qtype}:{reject_reason}"] += 1
    return stats


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def has_merged_weights(path: Path) -> bool:
    has_weights = (
        (path / "model.safetensors.index.json").exists()
        or (path / "model.safetensors").exists()
        or any(path.glob("model-*.safetensors"))
    )
    return has_weights and (path / "config.json").exists()


def rclone_common_args(args: argparse.Namespace) -> list[str]:
    cmd = [
        f"--transfers={args.rclone_transfers}",
        f"--checkers={args.rclone_checkers}",
        f"--drive-chunk-size={args.rclone_drive_chunk_size}",
        f"--drive-pacer-min-sleep={args.rclone_drive_pacer_min_sleep}",
        f"--drive-pacer-burst={args.rclone_drive_pacer_burst}",
        f"--retries={args.rclone_retries}",
        f"--low-level-retries={args.rclone_low_level_retries}",
        "--log-level",
        "INFO",
    ]
    if args.rclone_tpslimit is not None:
        cmd.append(f"--tpslimit={args.rclone_tpslimit}")
    if args.rclone_tpslimit_burst is not None:
        cmd.append(f"--tpslimit-burst={args.rclone_tpslimit_burst}")
    return cmd


def ensure_local_lora_checkpoint(checkpoint: Path, args: argparse.Namespace) -> None:
    adapter_path = checkpoint / "adapter_model.safetensors"
    if adapter_path.exists():
        print(f"[rs-sft] using local LoRA checkpoint: {checkpoint}", flush=True)
        return
    if not args.drive_source:
        raise FileNotFoundError(
            f"LoRA checkpoint is missing adapter_model.safetensors and no --drive-source was provided: {checkpoint}"
        )

    remote = f"{args.drive_source.rstrip('/')}/{checkpoint.name}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    print(f"[rs-sft] downloading {remote} -> {checkpoint}", flush=True)
    result = subprocess.run(
        [
            "rclone",
            "copy",
            remote,
            str(checkpoint),
            *rclone_common_args(args),
        ],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"rclone copy failed for {remote} with exit code {result.returncode}")
    if not adapter_path.exists():
        raise FileNotFoundError(f"Downloaded checkpoint is missing adapter_model.safetensors: {checkpoint}")


def ensure_merged_model(args: argparse.Namespace) -> str:
    if args.lora_checkpoint is None:
        assert args.model is not None
        return args.model

    checkpoint = resolve_path(args.lora_checkpoint)
    merged_dir = resolve_path(args.merged_model_dir)
    if has_merged_weights(merged_dir):
        print(f"[rs-sft] using existing merged model: {merged_dir}", flush=True)
        return str(merged_dir)
    ensure_local_lora_checkpoint(checkpoint, args)

    print(f"[rs-sft] merging LoRA checkpoint {checkpoint} -> {merged_dir}", flush=True)
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(checkpoint),
        max_seq_length=args.max_model_len,
        dtype=None,
        load_in_4bit=True,
        trust_remote_code=args.trust_remote_code,
    )
    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
    return str(merged_dir)


def main() -> None:
    args = parse_args()
    args.out_path = resolve_path(args.out_path)
    model_name = ensure_merged_model(args)
    tokenizer_name = args.tokenizer or model_name

    prompt_rows = load_prompt_pool(args)
    if not prompt_rows:
        raise FileNotFoundError("No prompt rows loaded. Pass --input-path with existing JSONL files.")

    from transformers import AutoTokenizer

    print(f"[rs-sft] loading tokenizer: {tokenizer_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.dry_run:
        print("[rs-sft] dry run complete; skipping vLLM generation", flush=True)
        return

    candidate_path = (
        resolve_path(args.candidate_path)
        if args.candidate_path is not None
        else args.out_path.with_name(args.out_path.stem + "_candidates.jsonl")
    )
    partial_sft_path = (
        resolve_path(args.partial_path)
        if args.partial_path is not None
        else args.out_path.with_name(args.out_path.stem + ".partial.jsonl")
    )
    manifest_path = (
        resolve_path(args.manifest_path)
        if args.manifest_path is not None
        else args.out_path.with_suffix(".manifest.json")
    )

    candidate_logs_by_key = load_candidate_logs(candidate_path) if args.resume_existing else {}
    if candidate_logs_by_key:
        print(
            f"[rs-sft] resuming from {candidate_path}: "
            f"{len(candidate_logs_by_key)} prompt candidate logs found",
            flush=True,
        )

    pending_indices = [
        index
        for index, row in enumerate(prompt_rows)
        if prompt_key(row, index) not in candidate_logs_by_key
    ]
    if not pending_indices:
        sft_rows = rebuild_sft_rows_from_logs(
            tokenizer=tokenizer,
            prompt_rows=prompt_rows,
            candidate_logs_by_key=candidate_logs_by_key,
        )
        write_jsonl(args.out_path, sft_rows)
        print(f"[rs-sft] all prompts already have candidates; rebuilt {args.out_path}", flush=True)
        return

    from vllm import LLM, SamplingParams

    print(f"[rs-sft] loading vLLM model: {model_name}", flush=True)
    llm = LLM(
        model=model_name,
        tokenizer=tokenizer_name,
        trust_remote_code=args.trust_remote_code,
        dtype=args.vllm_dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        generation_config="vllm",
    )
    sampling = SamplingParams(
        n=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )
    judger = Judger(strict_extract=args.strict_extract)

    print(
        f"[rs-sft] pending prompt rows: {len(pending_indices)}/{len(prompt_rows)}",
        flush=True,
    )

    prompts = [build_prompt(row["question"], row["_options"]) for row in prompt_rows]
    for batch_start in range(0, len(pending_indices), args.batch_size):
        batch_indices = pending_indices[batch_start : batch_start + args.batch_size]
        batch_rows = [prompt_rows[index] for index in batch_indices]
        batch_prompts = [prompts[index] for index in batch_indices]
        outputs = llm.generate(batch_prompts, sampling)

        for global_idx, row, request_output in zip(batch_indices, batch_rows, outputs):
            qtype, hard_cap, preferred_cap = output_caps_for_row(args, row)
            candidates = [
                build_candidate_record(
                    tokenizer=tokenizer,
                    judger=judger,
                    row=row,
                    completion=out.text,
                    candidate_index=i,
                    max_output_tokens=hard_cap,
                    preferred_output_tokens=preferred_cap,
                )
                for i, out in enumerate(request_output.outputs)
            ]
            selected = select_best(candidates)
            candidate_logs_by_key[prompt_key(row, global_idx)] = {
                "prompt_key": prompt_key(row, global_idx),
                "prompt_index": global_idx,
                "id": row.get("id"),
                "source": row.get("_source"),
                "input_path": row.get("_input_path"),
                "question": row.get("question"),
                "question_type": qtype,
                "preferred_output_tokens": preferred_cap,
                "max_output_tokens": hard_cap,
                "gold": row["_gold"],
                "options": row["_options"],
                "accepted_count": sum(int(c["accepted"]) for c in candidates),
                "selected_candidate_index": None if selected is None else selected["candidate_index"],
                "candidates": candidates,
            }

        ordered_candidate_logs = [
            candidate_logs_by_key[prompt_key(row, index)]
            for index, row in enumerate(prompt_rows)
            if prompt_key(row, index) in candidate_logs_by_key
        ]
        sft_rows = rebuild_sft_rows_from_logs(
            tokenizer=tokenizer,
            prompt_rows=prompt_rows,
            candidate_logs_by_key=candidate_logs_by_key,
        )
        stats = build_stats_from_candidate_logs(ordered_candidate_logs)
        write_jsonl(candidate_path, ordered_candidate_logs)
        write_jsonl(partial_sft_path, sft_rows)

        done = min(batch_start + args.batch_size, len(pending_indices))
        total_done = len(candidate_logs_by_key)
        if args.report_every and (done % args.report_every == 0 or done == len(pending_indices)):
            print(
                f"[rs-sft] progress {total_done}/{len(prompt_rows)} "
                f"kept={len(sft_rows)} accepted_candidates={stats['accepted_candidates']}",
                flush=True,
            )

    ordered_candidate_logs = [
        candidate_logs_by_key[prompt_key(row, index)]
        for index, row in enumerate(prompt_rows)
        if prompt_key(row, index) in candidate_logs_by_key
    ]
    sft_rows = rebuild_sft_rows_from_logs(
        tokenizer=tokenizer,
        prompt_rows=prompt_rows,
        candidate_logs_by_key=candidate_logs_by_key,
    )
    stats = build_stats_from_candidate_logs(ordered_candidate_logs)
    write_jsonl(args.out_path, sft_rows)
    write_jsonl(candidate_path, ordered_candidate_logs)

    manifest = {
        "model": model_name,
        "tokenizer": tokenizer_name,
        "input_paths": [str(resolve_path(path)) for path in (args.input_path or DEFAULT_INPUTS) if resolve_path(path).exists()],
        "out_path": str(args.out_path),
        "candidate_path": str(candidate_path),
        "sample_size": args.sample_size,
        "prompt_count": len(prompt_rows),
        "sft_record_count": len(sft_rows),
        "num_generations": args.num_generations,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "flat_output_caps": args.flat_output_caps,
        "fallback_max_output_tokens": args.max_output_tokens,
        "fallback_preferred_output_tokens": args.preferred_output_tokens,
        "tiered_output_caps": {
            "mcq": {
                "preferred_output_tokens": args.mcq_preferred_output_tokens,
                "max_output_tokens": args.mcq_max_output_tokens,
            },
            "single": {
                "preferred_output_tokens": args.single_preferred_output_tokens,
                "max_output_tokens": args.single_max_output_tokens,
            },
            "multi": {
                "preferred_output_tokens": args.multi_preferred_output_tokens,
                "max_output_tokens": args.multi_max_output_tokens,
            },
        },
        "stats": dict(stats),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"[rs-sft] wrote SFT records: {args.out_path} ({len(sft_rows)})", flush=True)
    print(f"[rs-sft] wrote candidate log: {candidate_path}", flush=True)
    print(f"[rs-sft] wrote manifest: {manifest_path}", flush=True)
    print(json.dumps(manifest["stats"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
