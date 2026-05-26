#!/usr/bin/env python3
"""
Run vLLM finalizer on the 212 unboxed, non-rescued rows from submission_draft_3.csv.
"""

import csv
import json
import re
import os
from pathlib import Path
from tqdm import tqdm

DRAFT3 = "artifacts/private_reasoning_paths/submission_draft_3.csv"
INJECT = "artifacts/private_reasoning_paths/rs_sft_public526_110_still_zero_box_inject80_n4_8000.jsonl"
FINALIZER_OUT = "artifacts/private_reasoning_paths/finalizer_outputs.jsonl"

MODEL_PATH = "checkpoints/merged_unified_ckpt500_waitle5_boxed"

def has_boxed(text: str) -> bool:
    return bool(re.search(r"\\boxed\{", text))

def main():
    # 1. Identify rescued rows to skip
    rescued = set()
    if os.path.exists(INJECT):
        with open(INJECT) as f:
            for line in f:
                row = json.loads(line)
                qid = row["question_id"]
                if any(has_boxed(g) for g in row["generations"]):
                    rescued.add(qid)
    print(f"Loaded {len(rescued)} rescued question IDs from injection.")

    # 2. Load unboxed rows from DRAFT3 that are not rescued
    to_finalize = []
    with open(DRAFT3, "r") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            qid = int(row["id"])
            resp = row["response"]
            if not has_boxed(resp) and qid not in rescued:
                to_finalize.append({
                    "id": qid,
                    "response": resp
                })

    print(f"Found {len(to_finalize)} rows to run finalizer on.")
    if not to_finalize:
        print("No rows to finalize!")
        return

    # 3. Setup finalizer prompts
    prompts = []
    for item in to_finalize:
        trace = item["response"]
        prompt_text = (
            "Here is the prior reasoning from another attempt.\n\n"
            f"{trace}\n\n"
            "Extract the final answer only. Do not solve again. Do not explain. "
            "Output exactly one final answer in \\boxed{}."
        )
        prompts.append({
            "id": item["id"],
            "prompt": prompt_text
        })

    # 4. Initialize vLLM
    print("Initializing vLLM...")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        enforce_eager=False,
    )

    sampling = SamplingParams(
        temperature=0.0,  # Greedy for extraction
        max_tokens=512,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    # 5. Generate completions
    print("Generating finalizer outputs...")
    input_prompts = [p["prompt"] for p in prompts]
    outputs = llm.generate(input_prompts, sampling)

    # 6. Save results
    os.makedirs(os.path.dirname(FINALIZER_OUT), exist_ok=True)
    with open(FINALIZER_OUT, "w") as f:
        for p, out in zip(prompts, outputs):
            gen_text = out.outputs[0].text.strip()
            f.write(json.dumps({
                "question_id": p["id"],
                "finalizer_output": gen_text
            }) + "\n")

    print(f"Finalizer outputs written to {FINALIZER_OUT}")

if __name__ == "__main__":
    main()
