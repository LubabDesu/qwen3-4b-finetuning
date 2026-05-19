"""
GRPO training for math reasoning with soft curriculum sampling.

Usage:
    python scripts/train_grpo.py --model Qwen/Qwen3-4B-Thinking-2507
    python scripts/train_grpo.py --model checkpoints/grpo/checkpoint-1900 --run-name grpo-ckpt1900

Reads artifacts/grpo/filtered_problems.jsonl produced by filter_grpo_data.py.
Saves checkpoints to checkpoints/grpo/ every 100 steps.
Backs up to Google Drive via rclone after each checkpoint.
Use --use-drive-path to copy checkpoints directly to mounted Google Drive.
"""

import argparse
import collections
import json
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import GRPOConfig, GRPOTrainer

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from judger import Judger  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────

FILTERED_DATA_PATH = ROOT / "artifacts" / "grpo" / "filtered_problems.jsonl"
DEEPMATH_FILTERED_DATA_PATH = ROOT / "artifacts" / "grpo" / "deepmath_filtered_problems.jsonl"
INFERENCE_PARTIAL_DATA_PATH = ROOT / "artifacts" / "grpo" / "inference_filtered_problems_partial.jsonl"
PUBLIC_GRPO_TRAIN_PATH = ROOT / "artifacts" / "grpo" / "public_train_300.jsonl"
CHECKPOINT_DIR = ROOT / "checkpoints" / "grpo"
DRIVE_CHECKPOINT_DIR = Path("/content/drive/MyDrive/151B_SP26_Competition/checkpoints/grpo")

SYSTEM_PROMPT = """You are an expert mathematician. Solve the problem step by step.

Final answer rules:
- Use exactly one final \\boxed{}.
- For multi-part questions with multiple [ANS] blanks, put all answers comma-separated in that one box, in blank order.
- Wrong: \\boxed{2}, \\boxed{4}, \\boxed{120}, \\boxed{-6}
- Right: \\boxed{2, 4, 120, -6}
- For MCQ, put only the option letter in \\boxed{}.
- Never round intermediate calculations.
- Never round your final answer unless the problem explicitly asks you to round.
- For decimal numerical answers, keep enough digits to match the unrounded value."""

TOTAL_STEPS = 1000
LOGGING_STEPS = 25
SAVE_STEPS = 100
LR = 3e-7
TEMPERATURE = 1.0
KL_COEF = 0.01
GROUP_SIZE = 8
MAX_PROMPT_TOKENS = 2048
MAX_NEW_TOKENS = 6144
REWARD_VALID_WRONG = 0.08
REWARD_MISSING_THINK = -0.10
REWARD_NO_FINAL_BOX = -0.10
REWARD_EMPTY_FINAL_BOX = -0.20
REWARD_MULTI_FINAL_BOX = -0.30
REWARD_MULTI_ANSWER_MULTI_FINAL_BOX = -0.40
REWARD_PRE_THINK_BOX_PENALTY = 0.10
REWARD_PRE_THINK_BOX_PENALTY_CAP = 0.25
REWARD_MULTI_ANSWER_COUNT_BONUS = 0.03
REWARD_MULTI_ANSWER_COUNT_PENALTY = 0.05
REWARD_MCQ_LETTER_BONUS = 0.03
REWARD_MCQ_SHAPE_PENALTY = 0.05
REWARD_WRONG_VALID_MAX = 0.12
REWARD_FLOOR = -0.60
REWARD_DIAGNOSTIC_EVERY = 200
STOP_FLAT_STEPS = 200       # stop if reward/mean flat for this many consecutive steps
STOP_KL_MAX = 10.0          # stop if kl_divergence exceeds this
STOP_LENGTH_MAX = 7000      # stop if mean response length exceeds this (tokens)
STOP_STD_MIN = 0.05         # diagnostic only; low reward/std should not stop GRPO

_strict_judger = Judger(strict_extract=True)
_reward_stats: collections.Counter[str] = collections.Counter()
_reward_total = 0
_reward_sum = 0.0


# ── Prompt formatting ────────────────────────────────────────────────────────

def format_question(question: str, options: list[str] | None = None) -> str:
    question = str(question).strip()
    if not options:
        return question
    option_lines = []
    for idx, option in enumerate(options):
        label = chr(ord("A") + idx)
        option_lines.append(f"{label}. {str(option).strip()}")
    return f"{question}\n\nOptions:\n" + "\n".join(option_lines)


def build_prompt(question: str, options: list[str] | None = None) -> str:
    question_text = format_question(question, options)
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ── Curriculum dataset construction ─────────────────────────────────────────

def load_filtered_problems(path: Path = FILTERED_DATA_PATH) -> list[dict]:
    problems = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def load_deepmath_problems() -> list[dict]:
    problems = load_filtered_problems(DEEPMATH_FILTERED_DATA_PATH)
    print(f"Loaded {len(problems)} DeepMath problems from {DEEPMATH_FILTERED_DATA_PATH}")
    return problems


def load_inference_partial() -> list[dict]:
    if INFERENCE_PARTIAL_DATA_PATH.exists():
        problems = load_filtered_problems(INFERENCE_PARTIAL_DATA_PATH)
        print(f"Loaded {len(problems)} partial inference problems from {INFERENCE_PARTIAL_DATA_PATH}")
        return problems
    print(f"No partial inference file found at {INFERENCE_PARTIAL_DATA_PATH}")
    return []


def load_public_grpo_train(path: Path = PUBLIC_GRPO_TRAIN_PATH) -> list[dict]:
    if not path.exists():
        print(f"No public GRPO train split found at {path}")
        return []
    problems = load_filtered_problems(path)
    print(f"Loaded {len(problems)} public GRPO train problems from {path}")
    return problems


def curriculum_weights(difficulties: np.ndarray, t: float) -> np.ndarray:
    """
    At t=0 (start): weight proportional to level (level 5=easiest gets highest weight).
    At t=1 (end): weight proportional to (6-level) (level 1=hardest gets highest weight).
    Linear interpolation between the two.

    difficulties: array of difficulty_level values in [1, 5]
    t: training progress in [0, 1]
    """
    w_easy = difficulties.astype(float)           # high for easy (level 5)
    w_hard = (6.0 - difficulties).astype(float)   # high for hard (level 1)
    weights = (1.0 - t) * w_easy + t * w_hard
    weights = np.clip(weights, 1e-6, None)
    return weights / weights.sum()


def build_curriculum_dataset(
    problems: list[dict],
    total_steps: int,
    batch_size_per_step: int,
) -> Dataset:
    """
    Pre-sample total_steps * batch_size_per_step problems using time-varying
    curriculum weights. Problems appear in order of increasing difficulty.
    """
    difficulties = np.array([p["difficulty_level"] for p in problems])
    n = len(problems)
    n_samples = total_steps * batch_size_per_step

    rng = np.random.default_rng(seed=42)
    indices = []
    for step in range(total_steps):
        t = step / max(total_steps - 1, 1)
        weights = curriculum_weights(difficulties, t)
        batch_idx = rng.choice(n, size=batch_size_per_step, p=weights, replace=True)
        indices.extend(batch_idx.tolist())

    rows = []
    for idx in indices:
        p = problems[idx]
        options = p.get("options", [])
        rows.append({
            "prompt": build_prompt(p["question"], options),
            "gold_answer": json.dumps(p["gold_answer"]),
            "options": json.dumps(options),
        })

    return Dataset.from_list(rows)


def build_mixed_curriculum_dataset(
    curriculum_problems: list[dict],
    public_problems: list[dict],
    public_mix_ratio: float,
    total_steps: int,
    batch_size_per_step: int,
) -> Dataset:
    """
    Pre-sample training prompts. Public rows are used as a calibration stream;
    non-public rows keep the easy-to-hard difficulty curriculum.
    """
    public_mix_ratio = min(max(public_mix_ratio, 0.0), 1.0)
    if not public_problems or public_mix_ratio <= 0:
        return build_curriculum_dataset(curriculum_problems, total_steps, batch_size_per_step)

    difficulties = np.array([p["difficulty_level"] for p in curriculum_problems])
    n_curriculum = len(curriculum_problems)
    n_public = len(public_problems)
    rng = np.random.default_rng(seed=42)
    public_order = rng.permutation(n_public).tolist()
    public_cursor = 0
    n_samples = total_steps * batch_size_per_step
    n_public_samples = int(round(n_samples * public_mix_ratio))
    public_sample_positions = set(
        int(pos) for pos in rng.choice(n_samples, size=n_public_samples, replace=False)
    )

    rows = []
    public_count = 0
    curriculum_count = 0
    sample_pos = 0
    for step in range(total_steps):
        t = step / max(total_steps - 1, 1)
        weights = curriculum_weights(difficulties, t)
        for _ in range(batch_size_per_step):
            use_public = sample_pos in public_sample_positions
            sample_pos += 1
            if use_public:
                if public_cursor >= len(public_order):
                    public_order = rng.permutation(n_public).tolist()
                    public_cursor = 0
                p = public_problems[public_order[public_cursor]]
                public_cursor += 1
                public_count += 1
            else:
                idx = int(rng.choice(n_curriculum, p=weights))
                p = curriculum_problems[idx]
                curriculum_count += 1

            options = p.get("options", [])
            rows.append({
                "prompt": build_prompt(p["question"], options),
                "gold_answer": json.dumps(p["gold_answer"]),
                "options": json.dumps(options),
            })

    print(
        f"Mixed curriculum dataset sampled {public_count} public prompts and "
        f"{curriculum_count} curriculum prompts"
    )
    return Dataset.from_list(rows)


# ── Reward function ──────────────────────────────────────────────────────────

def _judge_strict(completion: str, gold: list[str], options: list[str]) -> bool:
    try:
        return _strict_judger.auto_judge(pred=completion, gold=gold, options=options)
    except Exception:
        return False


def _extract_boxed_values(text: str) -> list[str]:
    values = []
    start = 0
    while True:
        match = re.search(r"\\boxed\s*\{", text[start:])
        if not match:
            break
        pos = start + match.end()
        depth = 1
        chars = []
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
            values.append("".join(chars).strip())
            start = pos
        else:
            break
    return values


def _split_think_sections(completion: str) -> tuple[str, str | None]:
    marker = "</think>"
    if marker not in completion:
        return completion, None
    before, after = completion.split(marker, 1)
    return before, after


def _split_top_level_commas(text: str) -> list[str]:
    parts = []
    current = []
    stack = []
    quote: str | None = None
    escape = False
    pairs = {"(": ")", "[": "]", "{": "}"}
    closers = set(pairs.values())

    for ch in text:
        if escape:
            current.append(ch)
            escape = False
            continue
        if ch == "\\":
            current.append(ch)
            escape = True
            continue
        if quote:
            current.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in {"'", '"'}:
            quote = ch
            current.append(ch)
            continue
        if ch in pairs:
            stack.append(pairs[ch])
            current.append(ch)
            continue
        if ch in closers:
            if stack and ch == stack[-1]:
                stack.pop()
            current.append(ch)
            continue
        if ch == "," and not stack:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        current.append(ch)

    part = "".join(current).strip()
    if part:
        parts.append(part)
    return parts


def _length_penalty(n_chars: int) -> float:
    """Approximate char-to-token ratio ~4:1 for math text."""
    n_tokens_approx = n_chars / 4.0
    if n_tokens_approx <= 5000:
        return 0.0
    if n_tokens_approx <= 6000:
        return 0.10 * (n_tokens_approx - 5000) / 1000
    if n_tokens_approx <= 7000:
        return 0.10 + 0.15 * (n_tokens_approx - 6000) / 1000
    if n_tokens_approx <= 8192:
        return 0.25 + 0.25 * (n_tokens_approx - 7000) / 1192
    return 0.50


def _analyze_completion(completion: str, gold: list[str], options: list[str]) -> dict[str, Any]:
    before_think, after_think = _split_think_sections(completion)
    pre_think_boxes = _extract_boxed_values(before_think)
    final_boxes = _extract_boxed_values(after_think or "")
    final_box = final_boxes[0].strip() if len(final_boxes) == 1 else ""
    return {
        "has_think": after_think is not None,
        "pre_think_box_count": len(pre_think_boxes),
        "final_boxes": final_boxes,
        "final_box": final_box,
        "final_box_count": len(final_boxes),
        "is_multi_answer": len(gold) > 1,
        "is_mcq": bool(options),
        "length_penalty": _length_penalty(len(completion)),
    }


def _structure_base_reward(analysis: dict[str, Any]) -> tuple[float | None, str]:
    if not analysis["has_think"]:
        return REWARD_MISSING_THINK, "missing_think"
    final_box_count = analysis["final_box_count"]
    if final_box_count == 0:
        return REWARD_NO_FINAL_BOX, "no_final_box"
    if final_box_count > 1:
        if analysis["is_multi_answer"]:
            return REWARD_MULTI_ANSWER_MULTI_FINAL_BOX, "multi_answer_multi_final_box"
        return REWARD_MULTI_FINAL_BOX, "multi_final_box"
    if not analysis["final_box"].strip():
        return REWARD_EMPTY_FINAL_BOX, "empty_final_box"
    return None, "valid_structure"


def _shape_adjustment(analysis: dict[str, Any], gold: list[str]) -> tuple[float, float, list[str]]:
    adjustment = 0.0
    penalties = 0.0
    tags = []
    final_box = analysis["final_box"].strip()

    if analysis["is_multi_answer"]:
        pred_parts = _split_top_level_commas(final_box)
        if len(pred_parts) == len(gold):
            adjustment += REWARD_MULTI_ANSWER_COUNT_BONUS
            tags.append("multi_count_match")
        else:
            adjustment -= REWARD_MULTI_ANSWER_COUNT_PENALTY
            penalties -= REWARD_MULTI_ANSWER_COUNT_PENALTY
            tags.append("multi_count_mismatch")

    if analysis["is_mcq"]:
        if re.fullmatch(r"[A-Ja-j]", final_box):
            adjustment += REWARD_MCQ_LETTER_BONUS
            tags.append("mcq_letter")
        else:
            adjustment -= REWARD_MCQ_SHAPE_PENALTY
            penalties -= REWARD_MCQ_SHAPE_PENALTY
            tags.append("mcq_bad_shape")

    pre_think_penalty = min(
        analysis["pre_think_box_count"] * REWARD_PRE_THINK_BOX_PENALTY,
        REWARD_PRE_THINK_BOX_PENALTY_CAP,
    )
    if pre_think_penalty:
        adjustment -= pre_think_penalty
        penalties -= pre_think_penalty
        tags.append("pre_think_box")

    if analysis["length_penalty"]:
        adjustment -= analysis["length_penalty"]
        penalties -= analysis["length_penalty"]
        tags.append("length_penalized")

    return adjustment, penalties, tags


def _record_reward_stats(tags: list[str], reward: float) -> None:
    global _reward_total, _reward_sum
    _reward_total += 1
    _reward_sum += reward
    _reward_stats.update(tags)
    if _reward_total % REWARD_DIAGNOSTIC_EVERY != 0:
        return

    avg_reward = _reward_sum / max(_reward_total, 1)
    key_order = [
        "correct",
        "valid_wrong",
        "missing_think",
        "no_final_box",
        "empty_final_box",
        "multi_final_box",
        "multi_answer_multi_final_box",
        "multi_count_match",
        "multi_count_mismatch",
        "mcq_letter",
        "mcq_bad_shape",
        "pre_think_box",
        "length_penalized",
    ]
    counts = " ".join(f"{key}={_reward_stats[key]}" for key in key_order if _reward_stats[key])
    print(f"[reward] n={_reward_total} avg={avg_reward:.4f} {counts}", flush=True)


def compute_reward(completion: str, gold: list[str], options: list[str]) -> float:
    analysis = _analyze_completion(completion, gold, options)
    malformed_reward, structure_tag = _structure_base_reward(analysis)
    correct_strict = _judge_strict(completion, gold, options)

    if malformed_reward is not None:
        reward = malformed_reward - analysis["length_penalty"]
        tags = [structure_tag]
        if analysis["length_penalty"]:
            tags.append("length_penalized")
    else:
        adjustment, penalties, tags = _shape_adjustment(analysis, gold)
        tags.append(structure_tag)
        if correct_strict:
            reward = 1.0 + penalties
            tags.append("correct")
        else:
            reward = min(REWARD_WRONG_VALID_MAX, REWARD_VALID_WRONG + adjustment)
            tags.append("valid_wrong")

    reward = max(REWARD_FLOOR, float(reward))
    _record_reward_stats(tags, reward)
    return reward


def reward_fn(
    prompts: list[str],
    completions: list[str],
    gold_answer: list[str],
    options: list[str],
    **kwargs: Any,
) -> list[float]:
    rewards = []
    for completion, gold_json, opts_json in zip(completions, gold_answer, options):
        gold = json.loads(gold_json)
        opts = json.loads(opts_json)
        rewards.append(compute_reward(completion, gold, opts))
    return rewards


# ── Stop conditions callback ─────────────────────────────────────────────────

def _get_log_val(logs: dict, *keys: str) -> float | None:
    """Try multiple key names, return first match. TRL 0.12 metric names vary."""
    for k in keys:
        v = logs.get(k)
        if v is not None:
            return float(v)
    return None


class StopConditionCallback(TrainerCallback):
    """
    Checks early-stop conditions on every log event:
    - reward/mean flat for STOP_FLAT_STEPS consecutive steps
    - kl_divergence > STOP_KL_MAX
    - mean response length > STOP_LENGTH_MAX tokens
    - reward/std < STOP_STD_MIN is logged but does not stop training
    """

    def __init__(self, eval_steps: int) -> None:
        self.eval_steps = eval_steps
        self._reward_history: collections.deque[float] = collections.deque(
            maxlen=STOP_FLAT_STEPS // eval_steps + 1
        )

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> TrainerControl:
        if not logs:
            return control

        step = state.global_step
        # TRL 0.12 logs rewards as "reward", "reward_std", kl as "kl",
        # completion lengths as "completion_length" or "response_length"
        reward_mean = _get_log_val(logs, "reward", "reward_mean", "train/reward_mean")
        reward_std = _get_log_val(logs, "reward_std", "train/reward_std")
        kl = _get_log_val(logs, "kl", "train/kl", "kl_divergence")
        resp_len = _get_log_val(
            logs, "completion_length", "response_length", "train/response_length"
        )

        reason = None

        if reward_mean is not None:
            self._reward_history.append(reward_mean)
            if len(self._reward_history) >= self._reward_history.maxlen:
                span = max(self._reward_history) - min(self._reward_history)
                if span < 0.005:
                    reason = f"reward/mean flat ~{STOP_FLAT_STEPS} steps (span={span:.4f})"

        if reward_std is not None and reward_std < STOP_STD_MIN:
            print(
                f"\n[StopCondition] reward/std={reward_std:.4f} < {STOP_STD_MIN}; "
                "continuing"
            )

        if kl is not None and kl > STOP_KL_MAX:
            reason = f"kl={kl:.2f} > {STOP_KL_MAX}"

        if resp_len is not None and resp_len > STOP_LENGTH_MAX:
            reason = f"response_length={resp_len:.0f} > {STOP_LENGTH_MAX}"

        if reason:
            print(f"\n[StopCondition] Stopping at step {step}: {reason}")
            control.should_training_stop = True

        return control


# ── rclone backup ────────────────────────────────────────────────────────────

class RcloneBackupCallback(TrainerCallback):
    """Syncs checkpoint directory to Google Drive after each save."""

    def __init__(self, gdrive_remote: str, local_dir: Path, use_drive_path: bool = False) -> None:
        self.gdrive_remote = gdrive_remote
        self.local_dir = local_dir
        self.use_drive_path = use_drive_path

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        step = state.global_step
        checkpoint_path = self.local_dir / f"checkpoint-{step}"
        if self.use_drive_path:
            target_path = DRIVE_CHECKPOINT_DIR / f"checkpoint-{step}"
            print(f"\n[drive] Copying checkpoint-{step} to {target_path}...")
            try:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(checkpoint_path, target_path, dirs_exist_ok=True)
                print("[drive] Backup complete.")
            except Exception as exc:
                print(f"[drive] Warning: backup failed: {exc}")
            return

        cmd = [
            "rclone", "copy",
            str(checkpoint_path),
            f"{self.gdrive_remote}/grpo/checkpoint-{step}",
            "--progress",
        ]
        print(f"\n[rclone] Backing up checkpoint-{step} to {self.gdrive_remote}...")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                print(f"[rclone] Warning: {result.stderr[:200]}")
            else:
                print(f"[rclone] Backup complete.")
        except subprocess.TimeoutExpired:
            print("[rclone] Warning: backup timed out after 300s")
        except FileNotFoundError:
            print("[rclone] Warning: rclone not found, skipping backup")


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # WandB init
    wandb.init(
        project="cse151b-grpo",
        name=args.run_name,
        config={
            "model": args.model,
            "lr": LR,
            "temperature": TEMPERATURE,
            "kl_coef": KL_COEF,
            "group_size": args.group_size,
            "total_steps": args.max_steps,
            "max_new_tokens": args.max_new_tokens,
            "public_mix_ratio": args.public_mix_ratio,
        },
    )

    # Load problems and build curriculum dataset
    if args.deepmath_only:
        problems = load_deepmath_problems()
        print(f"Loaded {len(problems)} total problems for DeepMath-only mode")
    else:
        problems = load_filtered_problems(FILTERED_DATA_PATH)
        print(f"Loaded {len(problems)} filtered problems from {FILTERED_DATA_PATH}")
    if args.include_inference_partial:
        inference_partial = load_inference_partial()
        problems.extend(inference_partial)
        print(f"Loaded {len(problems)} total problems after adding partial inference rows")
    public_problems = load_public_grpo_train(args.public_train_path)

    # One prompt per step; GRPO samples num_generations completions for that prompt.
    batch_per_step = 1
    dataset = build_mixed_curriculum_dataset(
        problems,
        public_problems,
        args.public_mix_ratio,
        args.max_steps,
        batch_per_step,
    )
    print("Dataset built")
    print(f"Built curriculum dataset: {len(dataset)} rows")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print("Tokenizer loaded")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # GRPOConfig uses completion length naming for generation tokens.
    grpo_config = GRPOConfig(
        output_dir=str(CHECKPOINT_DIR),
        run_name=args.run_name,
        learning_rate=LR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.group_size,    # rollouts per prompt
        generation_batch_size=args.group_size,
        max_prompt_length=MAX_PROMPT_TOKENS,
        max_completion_length=args.max_new_tokens,
        temperature=TEMPERATURE,
        beta=KL_COEF,                       # KL penalty coefficient
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_mem,
        max_steps=args.max_steps,
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        report_to="wandb",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        dataloader_num_workers=0,
        remove_unused_columns=False,        # keep gold_answer, options columns
        seed=42,
    )
    print("Config created")

    # Callbacks
    callbacks = [
        StopConditionCallback(eval_steps=LOGGING_STEPS),
    ]
    if args.use_drive_path or args.gdrive_remote:
        callbacks.append(
            RcloneBackupCallback(
                args.gdrive_remote,
                CHECKPOINT_DIR,
                use_drive_path=args.use_drive_path,
            )
        )

    # Trainer
    trainer = GRPOTrainer(
        model=args.model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks,
    )
    print("Trainer created")

    print(f"Starting GRPO training: {args.model}")
    print(f"  group_size={args.group_size}, lr={LR}, kl_coef={KL_COEF}")
    print(f"  max_new_tokens={args.max_new_tokens}, total_steps={args.max_steps}")
    print(f"  log every {LOGGING_STEPS} steps, save every {SAVE_STEPS} steps")

    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO training for math reasoning")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-4B-Thinking-2507",
        help="Model ID or checkpoint path",
    )
    parser.add_argument("--run-name", default="grpo-base", help="WandB run name")
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=GROUP_SIZE,
        help="Number of completions to generate per prompt",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=TOTAL_STEPS,
        help="Maximum number of GRPO training steps",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help="Maximum completion tokens generated per GRPO rollout",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default="",
        help="Trainer checkpoint directory to resume from, e.g. checkpoints/grpo/checkpoint-300.",
    )
    parser.add_argument(
        "--vllm-gpu-mem",
        type=float,
        default=0.6,
        help="GPU memory utilization target for colocated vLLM",
    )
    parser.add_argument(
        "--deepmath-only",
        action="store_true",
        help=f"Train using only DeepMath filtered data from {DEEPMATH_FILTERED_DATA_PATH}",
    )
    parser.add_argument(
        "--include-inference-partial",
        action="store_true",
        help=(
            "Also include legacy rows from inference_filtered_problems_partial.jsonl. "
            "Disabled by default so --deepmath-only is actually DeepMath-only."
        ),
    )
    parser.add_argument(
        "--public-train-path",
        type=Path,
        default=PUBLIC_GRPO_TRAIN_PATH,
        help="Optional public-style GRPO train split to mix into training.",
    )
    parser.add_argument(
        "--public-mix-ratio",
        type=float,
        default=0.0,
        help="Probability that each GRPO prompt comes from --public-train-path.",
    )
    parser.add_argument(
        "--gdrive-remote",
        default="gdrive:151B",
        help="rclone remote:path for checkpoint backup (empty to disable)",
    )
    parser.add_argument(
        "--use-drive-path",
        action="store_true",
        help=f"Copy checkpoints directly to mounted Drive path: {DRIVE_CHECKPOINT_DIR}",
    )
    args = parser.parse_args()
    main(args)
