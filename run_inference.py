#!/usr/bin/env python3
"""Reproducible inference pipeline for the CSE151B math Kaggle submission.

The pipeline is deliberately conservative after generation: it canonicalizes the
model's own final answer, applies only guarded option-letter formatting fixes,
and uses regeneration/extraction only for structurally bad outputs.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x: Iterable[Any], **_: Any) -> Iterable[Any]:
        return x


REPO_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from build_draft12_safe import append_single_final_box, extract_all_boxed, split_top_level  # noqa: E402
from judger import Judger  # noqa: E402


STANDARD_SYSTEM_PROMPT = (
    "Solve step by step. If your reasoning exceeds ~6000 tokens, STOP immediately, "
    "write 'Therefore, the final answer is \\boxed{...}', and close </think>. "
    "Your final line after </think> must be exactly one \\boxed{}. MCQ: single capital letter. "
    "Multiple [ANS] blanks: all answers comma-separated in ONE box, in blank order. Never round unless asked."
)

HARD_SYSTEM_PROMPT = STANDARD_SYSTEM_PROMPT + (
    "\n\nHard-problem checklist: verify divisibility/remainder answers by substitution, "
    "check boundary cases for digit/base/sequence problems, and commit to a final boxed answer before the token limit."
)

EXTRACTOR_SYSTEM_PROMPT = (
    "Extract the final answer already present in the solution. Do NOT solve or recompute. "
    "Output exactly one \\boxed{} and nothing else. Options -> letter(s). "
    "Multiple [ANS] blanks -> answers in blank order, using the same values the solution reached."
)

HARD_KEYWORDS = [
    "prime", "composite", "divisor", "divisible", "divisibility", "gcd", "lcm",
    "coprime", "relatively prime", "mod", "modulo", "congruent", "remainder",
    "integer", "positive integer", "factor", "multiple", "digit", "base", "last digit",
    "perfect square", "perfect cube", "sequence", "recurrence", "a_n", "define an algorithm",
    "find smallest", "find largest", "prove", "determine all",
]

BAD_BOX_TERMS = [
    "####", "step", "calculate", "therefore", "because", "not sure", "maybe",
    "we need", "previous answer", "final answer is",
]

NULLISH = {"", "nan", "none", "null", "n/a", "na"}


@dataclass
class GenConfig:
    max_model_len: int
    max_tokens: int
    temperature: float
    top_p: float
    batch_size: int
    gpu_memory_utilization: float
    rope_scaling_json: str = ""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", "--input_jsonl", dest="input_jsonl", type=Path, required=True)
    ap.add_argument("--output", "--output_csv", dest="output_csv", type=Path, required=True)
    ap.add_argument("--model_paths", "--model-paths", nargs="+",
                    default=["LubabDesu2/qwen3-4b-thinking-2507-waitle5"],
                    help="First path is used for all generation stages. Defaults to the HF waitle5 model.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--audit-dir", type=Path, default=None)
    ap.add_argument("--expected-rows", type=int, default=0,
                    help="0 means expect exactly the input row count.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Smoke-test on the first N rows after loading input.")
    ap.add_argument("--skip-extractor", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Route rows and write routing audit only.")

    ap.add_argument("--standard-max-model-len", type=int, default=20480)
    ap.add_argument("--standard-max-tokens", type=int, default=15000)
    ap.add_argument("--standard-batch-size", type=int, default=16)
    ap.add_argument("--standard-gpu-memory-utilization", type=float, default=0.92)

    ap.add_argument("--hard-max-model-len", type=int, default=49152)
    ap.add_argument("--hard-max-tokens", type=int, default=40000)
    ap.add_argument("--hard-batch-size", type=int, default=2)
    ap.add_argument("--hard-gpu-memory-utilization", type=float, default=0.98)
    ap.add_argument("--hard-rope-scaling-json", default='{"rope_type":"yarn","factor":1.5,"original_max_position_embeddings":32768}')

    ap.add_argument("--safety-max-tokens", type=int, default=40000)
    ap.add_argument("--extractor-max-model-len", type=int, default=24576)
    ap.add_argument("--extractor-batch-size", type=int, default=8)
    ap.add_argument("--extractor-gpu-memory-utilization", type=float, default=0.90)

    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--spiral-wait-threshold", type=int, default=40)
    ap.add_argument("--spiral-char-threshold", type=int, default=22000)
    return ap.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for fallback_id, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            row["id"] = int(row.get("id", row.get("question_id", fallback_id)))
            if "question" not in row:
                raise ValueError(f"row id={row['id']} has no question field")
            rows.append(row)
    return rows


def resolve_model(path: str) -> str:
    p = Path(path)
    if p.exists():
        return str(p.resolve())
    rp = REPO_ROOT / path
    if rp.exists():
        return str(rp.resolve())
    return path


def row_question(row: dict[str, Any]) -> str:
    return str(row.get("question", "") or "")


def num_blanks(row: dict[str, Any]) -> int:
    return row_question(row).count("[ANS]")


def options_text(row: dict[str, Any]) -> str:
    opts = row.get("options")
    if isinstance(opts, list) and opts:
        return "\n\nOptions:\n" + "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(opts))
    return ""


def has_options_text(question: str) -> bool:
    return bool(re.search(r"(?m)(?:^|\s)[A-J][.)]\s+", question or ""))


def is_multiselect(question: str) -> bool:
    q = (question or "").lower()
    return any(s in q for s in ["select all", "all that apply", "more than one", "there may be more than one"])


def route_row(row: dict[str, Any]) -> tuple[str, str]:
    q = row_question(row).lower()
    hits = [k for k in HARD_KEYWORDS if k in q]
    return ("HARD", ";".join(hits)) if hits else ("STANDARD", "")


def last_boxed(text: str) -> str:
    boxes = extract_all_boxed(text or "")
    return boxes[-1][2].strip() if boxes else ""


def extract_answer(judger: Judger, response: str) -> str:
    try:
        ans = (judger.extract_ans(response) or "").strip()
    except Exception:
        ans = ""
    return ans or last_boxed(response)


def component_parts(ans: str) -> list[str]:
    return [p.strip() for p in split_top_level(ans or "", flatten_outer=True) if p.strip()]


def component_count(ans: str) -> int:
    return len(component_parts(ans)) if (ans or "").strip() else 0


def bad_prose(ans: str) -> bool:
    low = (ans or "").strip().lower()
    return any(t in low for t in BAD_BOX_TERMS)


def duplicate_spam(ans: str, n_blanks: int) -> bool:
    parts = component_parts(ans)
    if len(parts) < 2:
        return False
    if len(parts) % 2 == 0 and parts[: len(parts) // 2] == parts[len(parts) // 2:]:
        return True
    if n_blanks > 0 and len(parts) > n_blanks and len(parts) % n_blanks == 0:
        chunks = [parts[i: i + n_blanks] for i in range(0, len(parts), n_blanks)]
        return all(chunk == chunks[0] for chunk in chunks)
    return False


def validate_answer(ans: str, row: dict[str, Any], require_exact_multiblank: bool = False) -> tuple[bool, str]:
    s = (ans or "").strip()
    low = s.lower()
    n = num_blanks(row)
    q = row_question(row)
    if low in NULLISH:
        return False, "empty"
    if bad_prose(s):
        return False, "prose"
    if len(s.split()) > 30 or len(s) > 500:
        return False, "too_long"
    if duplicate_spam(s, n):
        return False, "duplicate_spam"
    if n <= 1 and has_options_text(q):
        clean = re.sub(r"[\s,]+", "", s).upper()
        if is_multiselect(q):
            if not re.fullmatch(r"[A-J]+", clean):
                return False, "mcq_not_letters"
        elif not re.fullmatch(r"[A-J]", clean):
            return False, "mcq_not_letter"
    comps = component_count(s)
    if n > 1 and comps < n:
        return False, "underfilled"
    if require_exact_multiblank and n > 1 and comps != n:
        return False, f"component_count_{comps}_expected_{n}"
    return True, "valid"


def collapse_single_blank_letters(ans: str, row: dict[str, Any]) -> str | None:
    if num_blanks(row) > 1:
        return None
    parts = component_parts(ans)
    if len(parts) < 2:
        return None
    letters = [p.strip().upper() for p in parts]
    if all(re.fullmatch(r"[A-J]", p) for p in letters):
        return "".join(letters)
    return None


def collapse_multiblank_letter_run(ans: str, row: dict[str, Any]) -> str | None:
    n = num_blanks(row)
    parts = component_parts(ans)
    if n < 2 or len(parts) <= n:
        return None
    best: list[str] | None = None
    for i in range(len(parts)):
        if not re.fullmatch(r"[A-J]", parts[i].strip().upper()):
            continue
        j = i
        while j < len(parts) and re.fullmatch(r"[A-J]", parts[j].strip().upper()):
            j += 1
        if j - i >= 2:
            collapsed = parts[:i] + ["".join(p.strip().upper() for p in parts[i:j])] + parts[j:]
            if len(collapsed) == n:
                best = collapsed
                break
    return ", ".join(best) if best else None


def deterministic_repair(ans: str, row: dict[str, Any]) -> tuple[str, str]:
    for name, fn in [
        ("single_blank_letter_collapse", collapse_single_blank_letters),
        ("multiblank_letter_run_collapse", collapse_multiblank_letter_run),
    ]:
        fixed = fn(ans, row)
        if fixed and fixed != ans:
            return fixed, name
    return ans, "none"


def canonicalize_response(response: str, row: dict[str, Any], judger: Judger) -> tuple[str, str, str, bool]:
    old_ans = extract_answer(judger, response)
    final_ans, repair = deterministic_repair(old_ans, row)
    if not final_ans:
        return response, old_ans, repair, False
    fixed = append_single_final_box(response, final_ans)
    extracted = extract_answer(judger, fixed)
    ok = extracted.strip() == final_ans.strip()
    return fixed, final_ans, repair, ok


def needs_safety_regen(response: str, row: dict[str, Any], judger: Judger, force_spiral: bool = False) -> tuple[bool, str]:
    ans = extract_answer(judger, response)
    n = num_blanks(row)
    if "</think>" not in response:
        return True, "truncated"
    if not ans.strip():
        return True, "empty_box"
    if bad_prose(ans):
        return True, "prose_box"
    if n >= 2 and component_count(ans) < n:
        return True, "underfilled"
    if force_spiral:
        return True, "standard_spiral_reroute"
    return False, "keep"


def render_prompt(tokenizer: Any, row: dict[str, Any], system_prompt: str, enable_thinking: bool = True) -> str:
    user = f"Question:\n{row_question(row)}{options_text(row)}"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def render_extractor_prompt(tokenizer: Any, row: dict[str, Any], response: str, current_answer: str) -> str:
    user = (
        f"Question:\n{row_question(row)}{options_text(row)}\n\n"
        f"Solution:\n{response}\n\n"
        f"Current extracted answer:\n{current_answer}\n\n"
        "Return exactly one line: \\boxed{...}"
    )
    messages = [{"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT}, {"role": "user", "content": user}]
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt + "\\boxed{"


def init_vllm(model_path: str, args: argparse.Namespace, cfg: GenConfig):
    from transformers import AutoTokenizer
    from vllm import LLM

    model = resolve_model(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, Any] = {}
    if cfg.rope_scaling_json:
        kwargs["hf_overrides"] = {"rope_scaling": json.loads(cfg.rope_scaling_json)}
    llm = LLM(
        model=model,
        tokenizer=model,
        trust_remote_code=True,
        dtype=args.dtype,
        max_model_len=cfg.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_num_seqs=max(1, cfg.batch_size),
        generation_config="vllm",
        **kwargs,
    )
    return llm, tokenizer


def free_vllm(llm: Any) -> None:
    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def generate_stage(
    rows: list[dict[str, Any]],
    model_path: str,
    args: argparse.Namespace,
    cfg: GenConfig,
    system_prompt: str,
    seed_offset: int,
    stage_name: str,
    audit_path: Path,
) -> dict[int, str]:
    if not rows:
        return {}
    from vllm import SamplingParams

    print(json.dumps({"stage": stage_name, "rows": len(rows), "max_tokens": cfg.max_tokens, "max_model_len": cfg.max_model_len}))
    llm, tokenizer = init_vllm(model_path, args, cfg)
    outputs: dict[int, str] = {}
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a", encoding="utf-8") as f:
        for start in tqdm(range(0, len(rows), cfg.batch_size), desc=stage_name):
            batch = rows[start: start + cfg.batch_size]
            prompts = [render_prompt(tokenizer, r, system_prompt, enable_thinking=True) for r in batch]
            params = [
                SamplingParams(
                    n=1,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    max_tokens=cfg.max_tokens,
                    seed=args.seed + int(r["id"]) * 100 + seed_offset,
                    stop=["<|im_end|>", "<|endoftext|>"],
                )
                for r in batch
            ]
            gen = llm.generate(prompts, params)
            for row, out in zip(batch, gen):
                text = out.outputs[0].text if out.outputs else ""
                outputs[int(row["id"])] = text
                f.write(json.dumps({"id": int(row["id"]), "stage": stage_name, "response": text}, ensure_ascii=False) + "\n")
                f.flush()
    free_vllm(llm)
    return outputs


def run_extractor(
    rows: list[dict[str, Any]],
    responses: dict[int, str],
    model_path: str,
    args: argparse.Namespace,
    audit_path: Path,
) -> dict[int, str]:
    if not rows:
        return {}
    from vllm import SamplingParams

    cfg = GenConfig(
        max_model_len=args.extractor_max_model_len,
        max_tokens=512,
        temperature=0.0,
        top_p=1.0,
        batch_size=args.extractor_batch_size,
        gpu_memory_utilization=args.extractor_gpu_memory_utilization,
    )
    print(json.dumps({"stage": "extractor", "rows": len(rows)}))
    llm, tokenizer = init_vllm(model_path, args, cfg)
    judger = Judger()
    outputs: dict[int, str] = {}
    with audit_path.open("a", encoding="utf-8") as f:
        for start in tqdm(range(0, len(rows), cfg.batch_size), desc="extractor"):
            batch = rows[start: start + cfg.batch_size]
            prompts = [render_extractor_prompt(tokenizer, r, responses[int(r["id"])], extract_answer(judger, responses[int(r["id"])])) for r in batch]
            params = SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=512, stop=["}", "<|im_end|>", "<|endoftext|>"])
            gen = llm.generate(prompts, params)
            for row, out in zip(batch, gen):
                suffix = out.outputs[0].text if out.outputs else ""
                text = "\\boxed{" + suffix.strip() + "}"
                outputs[int(row["id"])] = text
                f.write(json.dumps({"id": int(row["id"]), "stage": "extractor", "response": text}, ensure_ascii=False) + "\n")
                f.flush()
    free_vllm(llm)
    return outputs


def write_csv(path: Path, rows: list[dict[str, Any]], responses: dict[int, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "response"], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            resp = responses[int(row["id"])].replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
            writer.writerow({"id": int(row["id"]), "response": resp})


def validate_final(path: Path, input_rows: list[dict[str, Any]], audit_dir: Path) -> dict[str, Any]:
    judger = Judger()
    seen: set[int] = set()
    empty_extract = 0
    nullish_response = 0
    physical_lines = sum(1 for _ in path.open(encoding="utf-8"))
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        out_rows = list(reader)
    for rec in out_rows:
        rid = int(rec["id"])
        seen.add(rid)
        resp = (rec.get("response") or "").replace("\\n", "\n")
        if not resp.strip() or resp.strip().lower() in NULLISH:
            nullish_response += 1
        if not extract_answer(judger, resp):
            empty_extract += 1
    summary = {
        "output": str(path),
        "rows": len(out_rows),
        "expected_rows": len(input_rows),
        "unique_ids": len(seen),
        "physical_lines": physical_lines,
        "nullish_response": nullish_response,
        "empty_extract": empty_extract,
    }
    (audit_dir / "final_validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if len(out_rows) != len(input_rows) or len(seen) != len(input_rows) or nullish_response or empty_extract:
        raise RuntimeError(f"final validation failed: {summary}")
    return summary


def run_inference(input_jsonl: str | Path, output_csv: str | Path, model_paths: list[str], seed: int = 42) -> None:
    argv = ["--input", str(input_jsonl), "--output", str(output_csv), "--seed", str(seed), "--model_paths", *model_paths]
    main(parse_args_from(argv))


def parse_args_from(argv: list[str]) -> argparse.Namespace:
    old = sys.argv
    try:
        sys.argv = [old[0], *argv]
        return parse_args()
    finally:
        sys.argv = old


def main(args: argparse.Namespace | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = args or parse_args()
    args.input_jsonl = Path(args.input_jsonl)
    args.output_csv = Path(args.output_csv)
    audit_dir = Path(args.audit_dir) if args.audit_dir else args.output_csv.parent / (args.output_csv.stem + "_audit")
    audit_dir.mkdir(parents=True, exist_ok=True)
    args.seed = int(args.seed)

    rows = read_jsonl(args.input_jsonl)
    if args.limit:
        rows = rows[:args.limit]
    expected = args.expected_rows or len(rows)
    if len(rows) != expected or len({int(r["id"]) for r in rows}) != len(rows):
        raise ValueError("input rows are missing ids, duplicated, or not expected length")
    model_path = args.model_paths[0]
    judger = Judger()

    routing: list[dict[str, Any]] = []
    standard_rows: list[dict[str, Any]] = []
    hard_rows: list[dict[str, Any]] = []
    for row in rows:
        route, keywords = route_row(row)
        (hard_rows if route == "HARD" else standard_rows).append(row)
        routing.append({"id": int(row["id"]), "route": route, "keywords": keywords})

    with (audit_dir / "routing_decisions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "route", "keywords"])
        writer.writeheader()
        writer.writerows(routing)
    print(json.dumps({"input_rows": len(rows), "standard": len(standard_rows), "hard": len(hard_rows), "audit_dir": str(audit_dir)}, indent=2))
    if args.dry_run:
        return

    generation_audit = audit_dir / "generation_candidates.jsonl"
    if generation_audit.exists():
        generation_audit.unlink()

    standard_cfg = GenConfig(
        max_model_len=args.standard_max_model_len,
        max_tokens=args.standard_max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.standard_batch_size,
        gpu_memory_utilization=args.standard_gpu_memory_utilization,
    )
    hard_cfg = GenConfig(
        max_model_len=args.hard_max_model_len,
        max_tokens=args.hard_max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.hard_batch_size,
        gpu_memory_utilization=args.hard_gpu_memory_utilization,
        rope_scaling_json=args.hard_rope_scaling_json,
    )

    responses: dict[int, str] = {}
    responses.update(generate_stage(standard_rows, model_path, args, standard_cfg, STANDARD_SYSTEM_PROMPT, 0, "standard", generation_audit))
    responses.update(generate_stage(hard_rows, model_path, args, hard_cfg, HARD_SYSTEM_PROMPT, 10000, "hard", generation_audit))

    canonical_audit: list[dict[str, Any]] = []
    standard_spiral_ids: set[int] = set()
    for row in standard_rows:
        rid = int(row["id"])
        resp = responses[rid]
        wait_count = len(re.findall(r"(?i)\bwait\b", resp))
        if wait_count > args.spiral_wait_threshold or len(resp) > args.spiral_char_threshold:
            standard_spiral_ids.add(rid)
    for row in rows:
        rid = int(row["id"])
        fixed, ans, repair, extract_ok = canonicalize_response(responses[rid], row, judger)
        responses[rid] = fixed
        canonical_audit.append({"id": rid, "answer": ans, "repair": repair, "extract_ok": extract_ok})

    safety_rows: list[dict[str, Any]] = []
    safety_reasons: dict[int, str] = {}
    for row in rows:
        rid = int(row["id"])
        need, reason = needs_safety_regen(responses[rid], row, judger, force_spiral=rid in standard_spiral_ids)
        if need:
            safety_rows.append(row)
            safety_reasons[rid] = reason
    print(json.dumps({"stage": "safety_select", "rows": len(safety_rows), "reasons": {r: list(safety_reasons.values()).count(r) for r in sorted(set(safety_reasons.values()))}}, indent=2))

    safety_cfg = GenConfig(
        max_model_len=args.hard_max_model_len,
        max_tokens=args.safety_max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.hard_batch_size,
        gpu_memory_utilization=args.hard_gpu_memory_utilization,
        rope_scaling_json=args.hard_rope_scaling_json,
    )
    safety_outputs = generate_stage(safety_rows, model_path, args, safety_cfg, HARD_SYSTEM_PROMPT, 20000, "safety_regen", generation_audit)
    regen_audit: list[dict[str, Any]] = []
    for row in safety_rows:
        rid = int(row["id"])
        candidate = safety_outputs.get(rid, "")
        cand_fixed, cand_ans, cand_repair, cand_extract_ok = canonicalize_response(candidate, row, judger)
        valid, valid_reason = validate_answer(cand_ans, row, require_exact_multiblank=True)
        accepted = "</think>" in candidate and valid and cand_extract_ok
        regen_audit.append({
            "id": rid, "reason": safety_reasons.get(rid, ""), "candidate_answer": cand_ans,
            "candidate_valid": valid, "candidate_reason": valid_reason, "accepted": accepted,
            "repair": cand_repair,
        })
        if accepted:
            responses[rid] = cand_fixed

    extractor_audit: list[dict[str, Any]] = []
    extractor_rows: list[dict[str, Any]] = []
    if not args.skip_extractor:
        for row in rows:
            rid = int(row["id"])
            need, reason = needs_safety_regen(responses[rid], row, judger, force_spiral=False)
            if reason in {"empty_box", "prose_box", "underfilled", "truncated"}:
                extractor_rows.append(row)
        extractor_outputs = run_extractor(extractor_rows, responses, model_path, args, generation_audit)
        for row in extractor_rows:
            rid = int(row["id"])
            raw = extractor_outputs.get(rid, "")
            ans = last_boxed(raw) or extract_answer(judger, raw)
            ans, repair = deterministic_repair(ans, row)
            valid, reason = validate_answer(ans, row, require_exact_multiblank=True)
            accepted = valid and bool(ans)
            extractor_audit.append({"id": rid, "answer": ans, "valid": valid, "reason": reason, "accepted": accepted, "repair": repair})
            if accepted:
                responses[rid] = append_single_final_box(responses[rid], ans)

    # Final canonicalization after any accepted extractor rows.
    for row in rows:
        rid = int(row["id"])
        fixed, _, _, _ = canonicalize_response(responses[rid], row, judger)
        responses[rid] = fixed

    write_csv(args.output_csv, rows, responses)
    with (audit_dir / "canonicalization_audit.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "answer", "repair", "extract_ok"])
        w.writeheader(); w.writerows(canonical_audit)
    with (audit_dir / "regen_audit.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["id", "reason", "candidate_answer", "candidate_valid", "candidate_reason", "accepted", "repair"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(regen_audit)
    with (audit_dir / "extractor_audit.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["id", "answer", "valid", "reason", "accepted", "repair"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(extractor_audit)
    summary = validate_final(args.output_csv, rows, audit_dir)
    summary.update({
        "safety_selected": len(safety_rows),
        "safety_accepted": sum(1 for r in regen_audit if r["accepted"]),
        "extractor_selected": len(extractor_rows) if not args.skip_extractor else 0,
        "extractor_accepted": sum(1 for r in extractor_audit if r["accepted"]),
        "elapsed_sec": round(time.time() - START_TIME, 1),
    })
    print(json.dumps(summary, indent=2))


START_TIME = time.time()


if __name__ == "__main__":
    main()
