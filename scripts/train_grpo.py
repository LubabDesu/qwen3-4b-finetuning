"""
GRPO training for math reasoning with soft curriculum sampling.

Usage:
    python scripts/train_grpo.py --model Qwen/Qwen3-4B-Thinking-2507
    python scripts/train_grpo.py --model checkpoints/grpo/checkpoint-1900 --run-name grpo-ckpt1900

Reads artifacts/grpo/filtered_problems.jsonl produced by filter_grpo_data.py.
Saves checkpoints to checkpoints/grpo/ every 100 steps.
Backs up to Google Drive via rclone after each checkpoint.
"""

import argparse
import collections
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from datasets import Dataset
from transformers import AutoTokenizer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from trl import GRPOConfig, GRPOTrainer

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from judger import Judger  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────

FILTERED_DATA_PATH = ROOT / "artifacts" / "grpo" / "filtered_problems.jsonl"
CHECKPOINT_DIR = ROOT / "checkpoints" / "grpo"
EVAL_DATA_PATH = ROOT / "heldout_eval_set.jsonl"

SYSTEM_PROMPT = (
    "You are a helpful assistant. Think step by step, "
    "then give your final answer inside \\boxed{}."
)

TOTAL_STEPS = 1000
EVAL_STEPS = 50
SAVE_STEPS = 100
LR = 3e-7
TEMPERATURE = 1.0
KL_COEF = 0.01
GROUP_SIZE = 8
MAX_NEW_TOKENS = 6144
REWARD_LENGTH_THRESHOLD = 4096
REWARD_LENGTH_PENALTY_SCALE = 0.2
STOP_FLAT_STEPS = 200       # stop if reward/mean flat for this many consecutive steps
STOP_KL_MAX = 10.0          # stop if kl_divergence exceeds this
STOP_LENGTH_MAX = 7000      # stop if mean response length exceeds this (tokens)
STOP_STD_MIN = 0.05         # stop if reward/std drops below this

_strict_judger = Judger(strict_extract=True)
_loose_judger = Judger(strict_extract=False)


# ── Prompt formatting ────────────────────────────────────────────────────────

def build_prompt(question: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ── Curriculum dataset construction ─────────────────────────────────────────

def load_filtered_problems() -> list[dict]:
    problems = []
    with open(FILTERED_DATA_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
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
        rows.append({
            "prompt": build_prompt(p["question"]),
            "gold_answer": json.dumps(p["gold_answer"]),
            "options": json.dumps(p.get("options", [])),
        })

    return Dataset.from_list(rows)


# ── Reward function ──────────────────────────────────────────────────────────

def _judge_strict(completion: str, gold: list[str], options: list[str]) -> bool:
    try:
        return _strict_judger.auto_judge(pred=completion, gold=gold, options=options)
    except Exception:
        return False


def _judge_loose(completion: str, gold: list[str], options: list[str]) -> bool:
    try:
        return _loose_judger.auto_judge(pred=completion, gold=gold, options=options)
    except Exception:
        return False


def _length_penalty(n_chars: int) -> float:
    # Approximate char-to-token ratio ~4:1 for math text
    n_tokens_approx = n_chars / 4.0
    excess = max(0.0, (n_tokens_approx - REWARD_LENGTH_THRESHOLD) / REWARD_LENGTH_THRESHOLD)
    return min(excess * REWARD_LENGTH_PENALTY_SCALE, REWARD_LENGTH_PENALTY_SCALE)


def compute_reward(completion: str, gold: list[str], options: list[str]) -> float:
    has_think = "</think>" in completion
    has_boxed = "\\boxed{" in completion

    correct_strict = _judge_strict(completion, gold, options)
    correct_loose = correct_strict or _judge_loose(completion, gold, options)

    if correct_strict:
        reward = 1.0 - _length_penalty(len(completion))
    elif correct_loose:
        # Capability present but answer not in \boxed{}
        reward = 0.6
    elif has_boxed:
        # Has format but wrong answer
        reward = 0.1
    else:
        reward = 0.0

    if not has_think:
        reward -= 0.05

    return float(reward)


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


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_on_heldout(
    model: Any,
    tokenizer: Any,
    n_questions: int = 100,
    max_new_tokens: int = 4096,
) -> float:
    """Run greedy eval on heldout set, return strict accuracy."""
    if not EVAL_DATA_PATH.exists():
        return 0.0

    eval_items = []
    with open(EVAL_DATA_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                eval_items.append(json.loads(line))
    eval_items = eval_items[:n_questions]

    model.eval()
    correct = 0
    for item in eval_items:
        question = item.get("question", item.get("problem", ""))
        gold = item.get("answer", item.get("gold_answer", []))
        if isinstance(gold, str):
            gold = [gold]
        options = item.get("options", [])

        prompt = build_prompt(question)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        if _judge_strict(completion, gold, options):
            correct += 1

    model.train()
    return correct / len(eval_items)


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
    - reward/std < STOP_STD_MIN
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
            reason = f"reward/std={reward_std:.4f} < {STOP_STD_MIN}"

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

    def __init__(self, gdrive_remote: str, local_dir: Path) -> None:
        self.gdrive_remote = gdrive_remote
        self.local_dir = local_dir

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        step = state.global_step
        checkpoint_path = self.local_dir / f"checkpoint-{step}"
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


# ── WandB eval callback ──────────────────────────────────────────────────────

class EvalAccuracyCallback(TrainerCallback):
    """Runs heldout evaluation every eval_steps and logs strict_accuracy to WandB."""

    def __init__(self, eval_steps: int) -> None:
        self.eval_steps = eval_steps
        self._model = None
        self._tokenizer = None

    def on_init_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Any = None,
        tokenizer: Any = None,
        **kwargs: Any,
    ) -> None:
        # Capture references set up by the trainer at init time
        self._model = model
        self._tokenizer = tokenizer

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Any = None,
        tokenizer: Any = None,
        **kwargs: Any,
    ) -> None:
        if state.global_step % self.eval_steps != 0 or state.global_step == 0:
            return
        # Prefer kwargs model over stored reference (trainer may pass it directly)
        _model = model or self._model
        _tokenizer = tokenizer or self._tokenizer
        if _model is None or _tokenizer is None:
            return
        accuracy = evaluate_on_heldout(_model, _tokenizer)
        print(f"\n[Eval] Step {state.global_step}: strict_accuracy = {accuracy:.4f}")
        if wandb.run is not None:
            wandb.log({"strict_accuracy": accuracy}, step=state.global_step)


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
            "total_steps": TOTAL_STEPS,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
    )

    # Load problems and build curriculum dataset
    problems = load_filtered_problems()
    print(f"Loaded {len(problems)} filtered problems")

    # Effective prompts per step: per_device * n_devices (approximate with 1 device here)
    batch_per_step = args.per_device_batch_size
    dataset = build_curriculum_dataset(problems, TOTAL_STEPS, batch_per_step)
    print(f"Built curriculum dataset: {len(dataset)} rows")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # GRPOConfig — TRL 0.12 parameter names
    grpo_config = GRPOConfig(
        output_dir=str(CHECKPOINT_DIR),
        run_name=args.run_name,
        learning_rate=LR,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.group_size,    # rollouts per prompt
        temperature=TEMPERATURE,
        beta=KL_COEF,                       # KL penalty coefficient
        max_new_tokens=MAX_NEW_TOKENS,
        max_steps=TOTAL_STEPS,
        save_steps=SAVE_STEPS,
        logging_steps=EVAL_STEPS,
        report_to="wandb",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        dataloader_num_workers=0,
        remove_unused_columns=False,        # keep gold_answer, options columns
        seed=42,
    )

    # Callbacks
    callbacks = [
        StopConditionCallback(eval_steps=EVAL_STEPS),
        EvalAccuracyCallback(eval_steps=EVAL_STEPS),
    ]
    if args.gdrive_remote:
        callbacks.append(RcloneBackupCallback(args.gdrive_remote, CHECKPOINT_DIR))

    # Trainer
    trainer = GRPOTrainer(
        model=args.model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    print(f"Starting GRPO training: {args.model}")
    print(f"  group_size={args.group_size}, lr={LR}, kl_coef={KL_COEF}")
    print(f"  max_new_tokens={MAX_NEW_TOKENS}, total_steps={TOTAL_STEPS}")
    print(f"  eval every {EVAL_STEPS} steps, save every {SAVE_STEPS} steps")

    trainer.train()
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
        "--group-size",
        type=int,
        default=GROUP_SIZE,
        help="Number of rollouts per prompt (reduce to 4 if OOM)",
    )
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=1,
        help="Prompts per device per step",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--gdrive-remote",
        default="gdrive:151B",
        help="rclone remote:path for checkpoint backup (empty to disable)",
    )
    args = parser.parse_args()
    main(args)
