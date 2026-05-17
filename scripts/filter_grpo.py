"""
Filter training problems by base model pass rate using vLLM offline inference.

Loads public.jsonl + DeepMath-103K (difficulty>=5, free-form only),
generates 6 responses per problem, scores with strict Judger, keeps
problems where 1-5/6 correct (difficulty_level = pass_count).

Output: artifacts/grpo/filtered_problems.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

from vllm import LLM, SamplingParams

# Add project root to path so judger.py is importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from judger import Judger  # noqa: E402

STRICT_JUDGER = Judger(strict_extract=True)

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
NUM_RESPONSES = 6
MAX_NEW_TOKENS = 4096
TEMPERATURE = 1.0
DEEPMATH_DATASET = "zwhe99/DeepMath-103K"
DEEPMATH_MIN_DIFFICULTY = 5

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

        problems.append({
            "id": f"deepmath_{i}",
            "question": question,
            "gold_answer": [str(final_answer)],
            "options": [],
            "source": "deepmath",
        })

    print(
        f"  DeepMath: {len(problems)} kept, "
        f"{skipped_difficulty} below difficulty, "
        f"{skipped_mcq} MCQ skipped"
    )
    return problems


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
) -> tuple[list[dict], dict]:
    prompts = [build_prompt(p["question"]) for p in problems]

    # Run inference in batches to avoid OOM
    all_responses: list[list[str]] = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        print(f"  Inference batch {i // batch_size + 1}/{(len(prompts) + batch_size - 1) // batch_size}")
        responses = run_vllm_inference(
            llm, batch, n=NUM_RESPONSES,
            temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS,
        )
        all_responses.extend(responses)

    kept = []
    dropped_too_easy = 0
    dropped_too_hard = 0
    level_counts = {i: 0 for i in range(1, NUM_RESPONSES)}

    for problem, responses in zip(problems, all_responses):
        gold = problem["gold_answer"]
        options = problem["options"]
        pass_count = sum(score_response(r, gold, options) for r in responses)

        if pass_count == 0:
            dropped_too_hard += 1
        elif pass_count == NUM_RESPONSES:
            dropped_too_easy += 1
        else:
            level = pass_count  # 1=hardest, 5=easiest
            level_counts[level] = level_counts.get(level, 0) + 1
            kept.append({
                "id": problem["id"],
                "question": problem["question"],
                "gold_answer": gold,
                "options": options,
                "pass_count": pass_count,
                "difficulty_level": level,
                "source": problem["source"],
            })

    stats = {
        "total_seen": len(problems),
        "kept": len(kept),
        "dropped_too_hard": dropped_too_hard,
        "dropped_too_easy": dropped_too_easy,
        "level_distribution": level_counts,
    }
    return kept, stats


def main(args: argparse.Namespace) -> None:
    # Load data sources
    public_problems = load_public_jsonl(ROOT / "data" / "public.jsonl")
    print(f"Loaded {len(public_problems)} problems from public.jsonl")

    deepmath_problems = load_deepmath(args.min_difficulty)

    mmlu_problems = load_mmlu_pro_math()
    all_problems = public_problems + deepmath_problems + mmlu_problems
    print(f"Total problems to evaluate: {len(all_problems)}")

    # Initialize vLLM
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

    # Filter
    print("Filtering problems...")
    kept, stats = filter_problems(all_problems, llm, batch_size=args.batch_size)

    # Save
    out_path = ROOT / "artifacts" / "grpo" / "filtered_problems.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for item in kept:
            f.write(json.dumps(item) + "\n")

    print(f"\nSaved {len(kept)} problems to {out_path}")
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
        print(f"  Level {level} ({level}/{NUM_RESPONSES} correct) {label}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter GRPO training data by base model pass rate")
    parser.add_argument("--model", default=MODEL_ID, help="Model ID or path for vLLM inference")
    parser.add_argument("--min-difficulty", type=int, default=DEEPMATH_MIN_DIFFICULTY, help="Min DeepMath difficulty")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size (prompts)")
    parser.add_argument("--max-model-len", type=int, default=8192, help="vLLM max model length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="vLLM GPU memory utilization")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size")
    args = parser.parse_args()
    main(args)
