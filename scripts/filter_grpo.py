"""
Filter training problems by base model pass rate using vLLM offline inference.

Loads public.jsonl + DeepMath-103K (difficulty>=5, free-form only) + MMLU-Pro math.
DeepMath is kept without inference and mapped directly from source difficulty to
levels 1-5. public.jsonl and MMLU-Pro generate 6 responses per problem, are
scored with the strict Judger, and keep only problems where 1-5/6 are correct
(difficulty_level = pass_count).

Outputs:
- artifacts/grpo/deepmath_filtered_problems.jsonl
- artifacts/grpo/inference_filtered_problems_partial.jsonl
- artifacts/grpo/inference_filtered_problems.jsonl
- artifacts/grpo/filtered_problems.jsonl
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from vllm import LLM, SamplingParams

# Add project root to path so judger.py is importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from judger import Judger  # noqa: E402

STRICT_JUDGER = Judger(strict_extract=True)

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
NUM_RESPONSES = 6
MAX_NEW_TOKENS = 3000
TEMPERATURE = 1.0
DEEPMATH_DATASET = "zwhe99/DeepMath-103K"
DEEPMATH_MIN_DIFFICULTY = 5
SAVE_EVERY_BATCHES = 5

# Qwen3 thinking chat template
SYSTEM_PROMPT = """You are an expert mathematician. Solve the problem step by step and put your final answer within \\boxed{}.
For multi-part questions with [ANS] blanks: put all answers comma-separated in one \\boxed{}.
For MCQ: identify the correct option and put only the letter in \\boxed{}.
Never round intermediate calculations.
Give answers in full precision unless the problem explicitly requests rounding."""


def load_mmlu_pro_math() -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    problems = []
    for i, item in enumerate(ds):
        if item.get("category") != "math":
            continue
        options = item.get("options", [])
        answer = item.get("answer", "")
        if not answer or not options:
            continue
        problems.append({
            "id": f"mmlu_pro_{i}",
            "question": item["question"],
            "gold_answer": [str(answer)],
            "options": options,
            "source": "mmlu_pro",
        })
    print(f"  MMLU-Pro math: {len(problems)} MCQ problems")
    return problems


def build_prompt(question: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n"
    )


def _load_progress(progress_path: Path, total_seen: int) -> dict:
    if not progress_path.exists():
        return {
            "batch_index": 0,
            "stats": {
                "total_seen": total_seen,
                "kept": 0,
                "dropped_too_hard": 0,
                "dropped_too_easy": 0,
                "level_distribution": {i: 0 for i in range(1, NUM_RESPONSES)},
            },
        }

    with open(progress_path, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)

    checkpoint_stats = checkpoint.get("stats", {})
    if checkpoint_stats.get("total_seen") != total_seen:
        print(
            "Warning: progress checkpoint total_seen does not match current problem set. "
            "Restarting from batch 0."
        )
        return {
            "batch_index": 0,
            "stats": {
                "total_seen": total_seen,
                "kept": 0,
                "dropped_too_hard": 0,
                "dropped_too_easy": 0,
                "level_distribution": {i: 0 for i in range(1, NUM_RESPONSES)},
            },
        }

    return {
        "batch_index": checkpoint.get("batch_index", 0),
        "stats": checkpoint_stats,
    }


def _save_progress(progress_path: Path, batch_index: int, stats: dict) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = progress_path.with_suffix(progress_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump({"batch_index": batch_index, "stats": stats}, f)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(progress_path)


def _append_batch_results(partial_file, kept_batch: list[dict]) -> None:
    for item in kept_batch:
        partial_file.write(json.dumps(item, ensure_ascii=False) + "\n")
    partial_file.flush()
    os.fsync(partial_file.fileno())


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _merge_jsonl_files(output_path: Path, input_paths: list[Path]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_f:
        for input_path in input_paths:
            if not input_path.exists():
                continue
            with open(input_path, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    out_f.write(line)
        out_f.flush()
        os.fsync(out_f.fileno())


def is_mcq_answer(answer: str) -> bool:
    """Heuristic: single letter A-J suggests MCQ."""
    return bool(re.fullmatch(r"[A-Ja-j]", answer.strip()))


def score_response(response: str, gold: list[str], options: list[str]) -> bool:
    """Strict judge: answer must be in \\boxed{} and correct."""
    try:
        return STRICT_JUDGER.auto_judge(
            pred=response,
            gold=gold,
            options=options if options else [],
        )
    except Exception:
        return False


def load_public_jsonl(path: Path) -> list[dict]:
    problems = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            gold = item.get("answer", [])
            if isinstance(gold, str):
                gold = [gold]
            options = item.get("options", [])
            problems.append({
                "id": f"public_{item['id']}",
                "question": item["question"],
                "gold_answer": gold,
                "options": options,
                "source": "public",
            })
    return problems


def load_deepmath(min_difficulty: int) -> list[dict]:
    from datasets import load_dataset

    print(f"Loading DeepMath-103K (difficulty>={min_difficulty})...")
    ds = load_dataset(DEEPMATH_DATASET, split="train")

    problems = []
    skipped_mcq = 0
    skipped_difficulty = 0
    skipped_unmapped = 0

    for i, item in enumerate(ds):
        # Filter difficulty
        diff = item.get("difficulty", 0)
        if isinstance(diff, str):
            try:
                diff = float(diff)
            except ValueError:
                diff = 0
        if diff < min_difficulty:
            skipped_difficulty += 1
            continue

        # Get answer — try final_answer first, fall back to answer
        final_answer = item.get("final_answer") or item.get("answer", "")
        if not final_answer:
            continue

        # Filter free-form: skip single-letter MCQ answers
        if is_mcq_answer(str(final_answer)):
            skipped_mcq += 1
            continue

        question = item.get("problem") or item.get("question", "")
        if not question:
            continue

        level = map_deepmath_difficulty_to_level(diff)
        if level is None:
            skipped_unmapped += 1
            continue

        problems.append({
            "id": f"deepmath_{i}",
            "question": question,
            "gold_answer": [str(final_answer)],
            "options": [],
            "pass_count": None,
            "difficulty_level": level,
            "difficulty": diff,
            "source": "deepmath",
        })

    print(
        f"  DeepMath: {len(problems)} kept, "
        f"{skipped_difficulty} below difficulty, "
        f"{skipped_mcq} MCQ skipped, "
        f"{skipped_unmapped} unmapped difficulty skipped"
    )
    return problems


def map_deepmath_difficulty_to_level(difficulty: float) -> int | None:
    if 9 <= difficulty <= 10:
        return 1
    if 7 <= difficulty <= 8:
        return 2
    if difficulty == 6:
        return 3
    if difficulty == 5.5:
        return 4
    if difficulty == 5:
        return 5
    return None


def run_vllm_inference(
    llm: LLM,
    prompts: list[str],
    n: int,
    temperature: float,
    max_new_tokens: int,
) -> list[list[str]]:
    """Returns list[n responses] for each prompt."""
    sampling_params = SamplingParams(
        n=n,
        temperature=temperature,
        max_tokens=max_new_tokens,
        stop=["<|im_end|>"],
    )
    outputs = llm.generate(prompts, sampling_params)
    return [
        [output.outputs[j].text for j in range(n)]
        for output in outputs
    ]


def filter_problems(
    problems: list[dict],
    llm: LLM,
    batch_size: int = 32,
    partial_path: Path = None,
    progress_path: Path = None,
    initial_stats: dict | None = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> dict:
    prompts = [build_prompt(p["question"]) for p in problems]
    total_batches = (len(prompts) + batch_size - 1) // batch_size
    expected_total_seen = initial_stats["total_seen"] if initial_stats is not None else len(problems)
    progress = _load_progress(progress_path, expected_total_seen) if progress_path else {
        "batch_index": 0,
        "stats": {
            "total_seen": expected_total_seen,
            "kept": 0,
            "dropped_too_hard": 0,
            "dropped_too_easy": 0,
            "level_distribution": {i: 0 for i in range(1, NUM_RESPONSES)},
        },
    }
    if initial_stats is not None and progress["batch_index"] == 0:
        progress["stats"] = initial_stats

    start_batch = progress["batch_index"]
    stats = progress["stats"]

    if start_batch >= total_batches:
        print("Resume checkpoint indicates all batches were already processed.")
        return stats

    if partial_path is not None:
        partial_path.parent.mkdir(parents=True, exist_ok=True)

    overall_start_time = time.time()
    pending_kept: list[dict] = []
    with open(partial_path, "a", encoding="utf-8") as partial_file:
        for batch_index in range(start_batch, total_batches):
            batch_start_time = time.time()
            start = batch_index * batch_size
            end = min(start + batch_size, len(prompts))
            batch = prompts[start:end]
            batch_problems = problems[start:end]
            print(
                f"  Inference batch {batch_index + 1}/{total_batches} "
                f"(problems {start + 1}-{end})"
            )
            responses = run_vllm_inference(
                llm, batch, n=NUM_RESPONSES,
                temperature=TEMPERATURE, max_new_tokens=max_new_tokens,
            )

            kept_batch = []
            for problem, response_list in zip(batch_problems, responses):
                gold = problem["gold_answer"]
                options = problem["options"]
                pass_count = sum(score_response(r, gold, options) for r in response_list)

                if pass_count == 0:
                    stats["dropped_too_hard"] += 1
                elif pass_count == NUM_RESPONSES:
                    stats["dropped_too_easy"] += 1
                else:
                    level = pass_count
                    stats["level_distribution"][level] = stats["level_distribution"].get(level, 0) + 1
                    kept_item = {
                        "id": problem["id"],
                        "question": problem["question"],
                        "gold_answer": gold,
                        "options": options,
                        "pass_count": pass_count,
                        "difficulty_level": level,
                        "source": problem["source"],
                    }
                    kept_batch.append(kept_item)
                    stats["kept"] += 1

            if kept_batch:
                pending_kept.extend(kept_batch)

            should_save = (
                (batch_index + 1) % SAVE_EVERY_BATCHES == 0
                or batch_index + 1 == total_batches
            )
            if should_save and pending_kept:
                _append_batch_results(partial_file, pending_kept)
                pending_kept = []

            if should_save and progress_path is not None:
                _save_progress(progress_path, batch_index + 1, stats)

            batch_elapsed = time.time() - batch_start_time
            batches_done = batch_index + 1 - start_batch
            avg_batch_time = (time.time() - overall_start_time) / max(batches_done, 1)
            remaining_batches = total_batches - (batch_index + 1)
            eta_seconds = int(round(avg_batch_time * remaining_batches))
            print(
                f"    Batch time: {batch_elapsed:.1f}s | "
                f"ETA: {eta_seconds // 60}m {eta_seconds % 60}s"
            )

    return stats


def main(args: argparse.Namespace) -> None:
    # Load data sources
    public_problems = load_public_jsonl(ROOT / "data" / "public.jsonl")
    print(f"Loaded {len(public_problems)} problems from public.jsonl")

    deepmath_problems = load_deepmath(args.min_difficulty)

    mmlu_problems = load_mmlu_pro_math()
    inference_problems = public_problems + mmlu_problems
    print(f"DeepMath kept without inference: {len(deepmath_problems)}")
    print(f"Problems to score with inference: {len(inference_problems)}")

    print("Filtering problems...")
    artifacts_dir = ROOT / "artifacts" / "grpo"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    deepmath_path = artifacts_dir / "deepmath_filtered_problems.jsonl"
    inference_partial_path = artifacts_dir / "inference_filtered_problems_partial.jsonl"
    inference_progress_path = artifacts_dir / "inference_filter_progress.json"
    inference_final_path = artifacts_dir / "inference_filtered_problems.jsonl"
    final_path = artifacts_dir / "filtered_problems.jsonl"

    _write_jsonl(deepmath_path, deepmath_problems)
    print(f"Wrote {len(deepmath_problems)} DeepMath problems to {deepmath_path}")

    if inference_progress_path.exists() and not inference_partial_path.exists():
        print("Warning: inference progress file exists but inference partial output is missing. Restarting inference from batch 0.")
        inference_progress_path.unlink()
        if inference_final_path.exists():
            inference_final_path.unlink()

    initial_stats = {
        "total_seen": len(deepmath_problems) + len(inference_problems),
        "kept": len(deepmath_problems),
        "dropped_too_hard": 0,
        "dropped_too_easy": 0,
        "level_distribution": {i: 0 for i in range(1, NUM_RESPONSES)},
    }
    for item in deepmath_problems:
        level = item["difficulty_level"]
        initial_stats["level_distribution"][level] += 1

    # Initialize vLLM only for public.jsonl + MMLU-Pro.
    print(f"Loading model {args.model}...")
    llm_kwargs = dict(
        model=args.model,
        dtype="float16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=42,
    )
    llm = LLM(**llm_kwargs)

    stats = filter_problems(
        inference_problems,
        llm,
        batch_size=args.batch_size,
        partial_path=inference_partial_path,
        progress_path=inference_progress_path,
        initial_stats=initial_stats,
        max_new_tokens=args.max_new_tokens,
    )

    if inference_partial_path.exists():
        inference_partial_path.replace(inference_final_path)
    else:
        inference_final_path.parent.mkdir(parents=True, exist_ok=True)
        with open(inference_final_path, "w", encoding="utf-8"):
            pass

    _merge_jsonl_files(final_path, [deepmath_path, inference_final_path])

    print(f"\nSaved {stats['kept']} problems to {final_path}")
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Total seen:        {stats['total_seen']}")
    print(f"Kept:              {stats['kept']}")
    print(f"Dropped too hard:  {stats['dropped_too_hard']}  (0/{NUM_RESPONSES} correct)")
    print(f"Dropped too easy:  {stats['dropped_too_easy']}  ({NUM_RESPONSES}/{NUM_RESPONSES} correct)")
    print(f"\nDifficulty distribution:")
    for level in sorted(stats["level_distribution"]):
        count = stats["level_distribution"][level]
        label = "hardest" if level == 1 else "easiest" if level == NUM_RESPONSES - 1 else ""
        print(f"  Level {level} {label}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter GRPO training data by base model pass rate")
    parser.add_argument("--model", default=MODEL_ID, help="Model ID or path for vLLM inference")
    parser.add_argument("--min-difficulty", type=int, default=DEEPMATH_MIN_DIFFICULTY, help="Min DeepMath difficulty")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size (prompts)")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS, help="Max new tokens per response")
    parser.add_argument("--max-model-len", type=int, default=8192, help="vLLM max model length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="vLLM GPU memory utilization")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size")
    args = parser.parse_args()
    main(args)
