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
import contextlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
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
import judger as judger_module  # noqa: E402
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
REWARD_MULTI_ANSWER_COUNT_BONUS = 0.03
REWARD_MULTI_ANSWER_COUNT_PENALTY = 0.05
REWARD_MCQ_LETTER_BONUS = 0.03
REWARD_MCQ_SHAPE_PENALTY = 0.05
REWARD_FLOOR = -0.60
REWARD_CAP = 1.00
REWARD_DIAGNOSTIC_EVERY = 200
STOP_FLAT_STEPS = 200       # stop if reward/mean flat for this many consecutive steps
STOP_KL_MAX = 10.0          # stop if kl_divergence exceeds this
STOP_LENGTH_MAX = 7000      # stop if mean response length exceeds this (tokens)
STOP_STD_MIN = 0.05         # diagnostic only; low reward/std should not stop GRPO

_judger_local = threading.local()
_judger_signal_lock = threading.Lock()
_reward_stats: collections.Counter[str] = collections.Counter()
_reward_total = 0
_reward_sum = 0.0
_reward_phase = "long"


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


def filter_by_min_difficulty(problems: list[dict], min_level: int | None) -> list[dict]:
    if min_level is None:
        return problems
    filtered = [p for p in problems if int(p.get("difficulty_level", 0)) >= min_level]
    print(
        f"Filtered curriculum by difficulty_level >= {min_level}: "
        f"{len(filtered)}/{len(problems)} rows"
    )
    return filtered


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
    warmup_public_mix_ratio: float,
    public_warmup_steps: int,
    total_steps: int,
    batch_size_per_step: int,
) -> Dataset:
    """
    Pre-sample training prompts. Public rows are used as a calibration stream;
    non-public rows keep the easy-to-hard difficulty curriculum.
    """
    public_mix_ratio = min(max(public_mix_ratio, 0.0), 1.0)
    warmup_public_mix_ratio = min(max(warmup_public_mix_ratio, 0.0), 1.0)
    public_warmup_steps = max(public_warmup_steps, 0)
    if not public_problems or (public_mix_ratio <= 0 and warmup_public_mix_ratio <= 0):
        return build_curriculum_dataset(curriculum_problems, total_steps, batch_size_per_step)

    difficulties = np.array([p["difficulty_level"] for p in curriculum_problems])
    n_curriculum = len(curriculum_problems)
    n_public = len(public_problems)
    rng = np.random.default_rng(seed=42)
    public_order = rng.permutation(n_public).tolist()
    public_cursor = 0
    public_sample_positions = set()
    for step in range(total_steps):
        ratio = warmup_public_mix_ratio if step < public_warmup_steps else public_mix_ratio
        for inner in range(batch_size_per_step):
            sample_pos = step * batch_size_per_step + inner
            if rng.random() < ratio:
                public_sample_positions.add(sample_pos)

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
        f"{curriculum_count} curriculum prompts "
        f"(warmup_public_mix_ratio={warmup_public_mix_ratio}, "
        f"public_warmup_steps={public_warmup_steps}, public_mix_ratio={public_mix_ratio})"
    )
    return Dataset.from_list(rows)


# ── Reward function ──────────────────────────────────────────────────────────

def _get_official_judger() -> Judger:
    judger = getattr(_judger_local, "official_judger", None)
    if judger is None:
        judger = Judger(strict_extract=False)
        _judger_local.official_judger = judger
    return judger


@contextlib.contextmanager
def _disable_judger_alarm() -> Any:
    old_signal = judger_module.signal.signal
    old_alarm = judger_module.signal.alarm
    judger_module.signal.signal = lambda *args, **kwargs: None
    judger_module.signal.alarm = lambda *args, **kwargs: 0
    try:
        yield
    finally:
        judger_module.signal.signal = old_signal
        judger_module.signal.alarm = old_alarm


def _judge_official(completion: str, gold: list[str], options: list[str]) -> bool:
    try:
        with _judger_signal_lock:
            with _disable_judger_alarm():
                return _get_official_judger().auto_judge(
                    pred=completion,
                    gold=gold,
                    options=options,
                )
    except Exception:
        return False


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


def _approx_tokens(completion: str) -> float:
    """Cheap token proxy for reward shaping."""
    return max(1.0, len(completion) / 4.0)


def _joined_box_answer(boxes: list[str]) -> str:
    return ", ".join(str(box).strip() for box in boxes if str(box).strip())


def _analyze_completion(completion: str, gold: list[str], options: list[str]) -> dict[str, Any]:
    before_think, after_think = _split_think_sections(completion)
    official_judger = _get_official_judger()
    boxes_before = official_judger.extract_all_boxed(before_think)
    boxes_after = official_judger.extract_all_boxed(after_think or "")
    boxes_full = official_judger.extract_all_boxed(completion)
    raw_box_count = completion.count("\\boxed{")
    official_extract = official_judger.extract_ans(completion)
    has_usable_box = bool(boxes_before or boxes_after or boxes_full)
    has_think = after_think is not None
    ideal_box = has_think and bool(boxes_after)
    fallback_box_missing_think = not has_think and has_usable_box
    prethink_box = has_think and not boxes_after and bool(boxes_before)
    correct = _judge_official(completion, gold, options)
    contiguous_count = len(boxes_before) + len(boxes_after)
    return {
        "completion": completion,
        "tokens": _approx_tokens(completion),
        "correct": correct,
        "official_extract": official_extract,
        "official_extractable": bool(official_extract),
        "official_correct_no_usable_box": correct and not has_usable_box,
        "has_think": has_think,
        "has_box": raw_box_count > 0,
        "has_usable_box": has_usable_box,
        "ideal_box": ideal_box,
        "fallback_box_missing_think": fallback_box_missing_think,
        "prethink_box": prethink_box,
        "is_scattered": has_usable_box and raw_box_count > max(contiguous_count, len(boxes_full), 1),
        "empty_or_unusable_box": raw_box_count > 0 and not has_usable_box,
        "boxes_before": boxes_before,
        "boxes_after": boxes_after,
        "boxes_full": boxes_full,
        "raw_box_count": raw_box_count,
        "answer_text": (
            _joined_box_answer(boxes_after)
            if boxes_after
            else (_joined_box_answer(boxes_before) if boxes_before else official_extract)
        ),
        "is_multi_answer": len(gold) > 1,
        "is_mcq": bool(options),
    }


def _no_usable_box_reward(tokens: float) -> float:
    if _reward_phase == "closure":
        if tokens <= 1000:
            return -0.15
        if tokens <= 2000:
            return -0.35
        return REWARD_FLOOR
    if tokens <= 1500:
        return -0.15
    if tokens <= 3000:
        return -0.35
    return REWARD_FLOOR


def _base_reward(analysis: dict[str, Any]) -> tuple[float, list[str]]:
    correct = analysis["correct"]
    tokens = analysis["tokens"]

    if analysis["is_scattered"]:
        return (0.20 if correct else -0.20), ["scattered_box", "correct" if correct else "wrong"]

    if analysis["ideal_box"]:
        return (1.00 if correct else 0.05), ["ideal_box", "correct" if correct else "wrong"]

    if analysis["fallback_box_missing_think"]:
        return (0.70 if correct else 0.02), ["fallback_missing_think", "correct" if correct else "wrong"]

    if analysis["prethink_box"]:
        return (0.45 if correct else -0.05), ["prethink_box", "correct" if correct else "wrong"]

    if analysis["official_correct_no_usable_box"]:
        if analysis["has_box"]:
            return (0.20 if tokens <= 3000 else 0.05), ["correct_unusable_box"]
        return (0.30 if tokens <= 3000 else 0.10), ["correct_no_box"]

    if not analysis["has_usable_box"]:
        if analysis["has_box"]:
            return min(-0.30, _no_usable_box_reward(tokens)), ["empty_or_unusable_box"]
        return _no_usable_box_reward(tokens), ["no_usable_box"]

    return -0.10, ["malformed_box"]


def _shape_adjustment(analysis: dict[str, Any], gold: list[str]) -> tuple[float, list[str]]:
    adjustment = 0.0
    tags = []
    answer_text = str(analysis["answer_text"]).strip()

    if analysis["is_multi_answer"]:
        pred_parts = _split_top_level_commas(answer_text)
        if len(pred_parts) == len(gold):
            adjustment += REWARD_MULTI_ANSWER_COUNT_BONUS
            tags.append("multi_count_match")
        else:
            adjustment -= REWARD_MULTI_ANSWER_COUNT_PENALTY
            tags.append("multi_count_mismatch")

    if analysis["is_mcq"]:
        if re.fullmatch(r"[A-Ja-j]", answer_text):
            adjustment += REWARD_MCQ_LETTER_BONUS
            tags.append("mcq_letter")
        else:
            adjustment -= REWARD_MCQ_SHAPE_PENALTY
            tags.append("mcq_bad_shape")

    return adjustment, tags


def _batch_relative_length_penalty(analysis: dict[str, Any], group: list[dict[str, Any]]) -> float:
    if analysis["correct"] and analysis["ideal_box"]:
        return 0.0

    if not analysis["has_box"] and not analysis["official_correct_no_usable_box"]:
        return 0.0

    correct_lengths = [
        item["tokens"]
        for item in group
        if item["correct"] and item["has_usable_box"]
    ]

    if correct_lengths:
        if _reward_phase == "closure":
            allowance = max(1000.0, 1.15 * (sum(correct_lengths) / len(correct_lengths)))
        else:
            allowance = max(1500.0, 1.25 * (sum(correct_lengths) / len(correct_lengths)))
    elif _reward_phase == "closure":
        allowance = 2500.0 if analysis["correct"] else 1800.0
    else:
        allowance = 5000.0 if analysis["correct"] else 3000.0

    if analysis["tokens"] <= allowance:
        return 0.0

    excess_ratio = (analysis["tokens"] - allowance) / allowance
    if analysis["correct"]:
        return min(0.10 if _reward_phase == "closure" else 0.15, 0.08 * excess_ratio)
    return min(0.35 if _reward_phase == "closure" else 0.40, (0.25 if _reward_phase == "closure" else 0.20) * excess_ratio)


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
        "wrong",
        "ideal_box",
        "fallback_missing_think",
        "prethink_box",
        "correct_no_box",
        "correct_unusable_box",
        "no_usable_box",
        "empty_or_unusable_box",
        "malformed_box",
        "scattered_box",
        "multi_count_match",
        "multi_count_mismatch",
        "mcq_letter",
        "mcq_bad_shape",
        "length_penalized",
    ]
    counts = " ".join(f"{key}={_reward_stats[key]}" for key in key_order if _reward_stats[key])
    print(f"[reward] n={_reward_total} avg={avg_reward:.4f} {counts}", flush=True)


def compute_reward(analysis: dict[str, Any], group: list[dict[str, Any]], gold: list[str]) -> float:
    reward, tags = _base_reward(analysis)
    adjustment, shape_tags = _shape_adjustment(analysis, gold)
    length_penalty = _batch_relative_length_penalty(analysis, group)
    if length_penalty:
        tags.append("length_penalized")

    reward = reward + adjustment - length_penalty
    reward = min(REWARD_CAP, max(REWARD_FLOOR, float(reward)))
    _record_reward_stats(tags + shape_tags, reward)
    return reward


def reward_fn(
    prompts: list[str],
    completions: list[str],
    gold_answer: list[str],
    options: list[str],
    **kwargs: Any,
) -> list[float]:
    analyses = []
    golds = []
    group_map: dict[tuple[str, str, str], list[dict[str, Any]]] = collections.defaultdict(list)

    for prompt, completion, gold_json, opts_json in zip(prompts, completions, gold_answer, options):
        gold = json.loads(gold_json)
        opts = json.loads(opts_json)
        analysis = _analyze_completion(completion, gold, opts)
        analyses.append(analysis)
        golds.append(gold)
        group_map[(str(prompt), gold_json, opts_json)].append(analysis)

    rewards = []
    for prompt, analysis, gold_json, opts_json, gold in zip(prompts, analyses, gold_answer, options, golds):
        group = group_map[(str(prompt), gold_json, opts_json)]
        rewards.append(compute_reward(analysis, group, gold))
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

    def __init__(
        self,
        gdrive_remote: str,
        local_dir: Path,
        use_drive_path: bool = False,
        drive_checkpoint_dir: Path = DRIVE_CHECKPOINT_DIR,
    ) -> None:
        self.gdrive_remote = gdrive_remote
        self.local_dir = local_dir
        self.use_drive_path = use_drive_path
        self.drive_checkpoint_dir = drive_checkpoint_dir

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
            target_path = self.drive_checkpoint_dir / f"checkpoint-{step}"
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
    global _reward_phase
    _reward_phase = args.reward_phase
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
            "reward_phase": args.reward_phase,
            "min_difficulty_level": args.min_difficulty_level,
            "public_warmup_steps": args.public_warmup_steps,
            "warmup_public_mix_ratio": args.warmup_public_mix_ratio,
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
    effective_min_difficulty = args.min_difficulty_level
    if effective_min_difficulty is None and args.reward_phase == "closure":
        effective_min_difficulty = 4
    problems = filter_by_min_difficulty(problems, effective_min_difficulty)
    public_problems = load_public_grpo_train(args.public_train_path)

    # One prompt per step; GRPO samples num_generations completions for that prompt.
    batch_per_step = 1
    dataset = build_mixed_curriculum_dataset(
        problems,
        public_problems,
        args.public_mix_ratio,
        args.warmup_public_mix_ratio,
        args.public_warmup_steps,
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
                drive_checkpoint_dir=args.drive_checkpoint_dir,
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
        "--reward-phase",
        choices=["closure", "long"],
        default="long",
        help=(
            "Reward length shaping mode. closure is stricter and defaults to "
            "difficulty_level >= 4; long is looser for hard reasoning."
        ),
    )
    parser.add_argument(
        "--min-difficulty-level",
        type=int,
        default=None,
        help=(
            "Keep only curriculum rows with difficulty_level >= this value. "
            "Levels are inverted: 5 is easiest, 1 is hardest. Defaults to 4 "
            "for --reward-phase closure and no filter for long."
        ),
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
        "--public-warmup-steps",
        type=int,
        default=0,
        help="Use --warmup-public-mix-ratio for this many initial steps.",
    )
    parser.add_argument(
        "--warmup-public-mix-ratio",
        type=float,
        default=0.0,
        help="Public mix ratio during --public-warmup-steps.",
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
    parser.add_argument(
        "--drive-checkpoint-dir",
        type=Path,
        default=DRIVE_CHECKPOINT_DIR,
        help="Mounted Drive checkpoint directory used with --use-drive-path.",
    )
    args = parser.parse_args()
    main(args)
