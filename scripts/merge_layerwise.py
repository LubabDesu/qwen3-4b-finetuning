#!/usr/bin/env python3
"""Layer-wise interpolation between base and SFT merged model.

Lower layers (early indices) get higher lambda (keep more SFT).
Upper layers (late indices) get lower lambda (revert toward base).
This surgically preserves MCQ gains from lower layers while recovering
multi-answer ability that lives in upper layers.

Usage:
    python scripts/merge_layerwise.py \
        --sft-path checkpoints/eval_merged_stage1_2/merged_checkpoint-1900 \
        --output checkpoints/merged_layerwise_07_02 \
        [--base-model Qwen/Qwen3-4B-Thinking-2507] \
        [--lambda-high 0.7] [--lambda-low 0.2]
"""

from __future__ import annotations

import argparse
import gc
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layer-wise model interpolation.")
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3-4B-Thinking-2507",
        help="HuggingFace model ID or local path for the base model.",
    )
    parser.add_argument(
        "--sft-path",
        required=True,
        type=Path,
        help="Path to the merged (full-weight, fp16/bf16) SFT model.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Where to save the interpolated model.",
    )
    parser.add_argument(
        "--lambda-high",
        type=float,
        default=0.7,
        help="Lambda for layer 0 (how much SFT delta to add). Default 0.7.",
    )
    parser.add_argument(
        "--lambda-low",
        type=float,
        default=0.2,
        help="Lambda for the last layer. Default 0.2.",
    )
    parser.add_argument(
        "--lambda-other",
        type=float,
        default=0.5,
        help="Lambda for embedding and other non-layer params. Default 0.5.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Dtype to load models in. Default bfloat16.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print(f"[merge_layerwise] loading base model: {args.base_model}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        device_map="cpu",
        trust_remote_code=True,
    )

    print(f"[merge_layerwise] loading SFT model: {args.sft_path}", flush=True)
    sft = AutoModelForCausalLM.from_pretrained(
        str(args.sft_path),
        torch_dtype=torch_dtype,
        device_map="cpu",
        trust_remote_code=True,
    )

    num_layers = base.config.num_hidden_layers
    print(f"[merge_layerwise] num_hidden_layers={num_layers}, lambda {args.lambda_high:.2f} → {args.lambda_low:.2f}", flush=True)

    sft_sd = sft.state_dict()
    layer_re = re.compile(r"\.layers\.(\d+)\.")

    modified = 0
    skipped_missing = 0
    for name, param in base.named_parameters():
        if name not in sft_sd:
            skipped_missing += 1
            continue

        sft_param = sft_sd[name]
        delta = sft_param - param.data

        m = layer_re.search(name)
        if m:
            layer_idx = int(m.group(1))
            # linear interpolation: layer 0 → lambda_high, last layer → lambda_low
            t = layer_idx / max(num_layers - 1, 1)
            lam = args.lambda_high + t * (args.lambda_low - args.lambda_high)
        else:
            lam = args.lambda_other

        param.data.add_(lam * delta)
        modified += 1

    print(f"[merge_layerwise] interpolated {modified} params, skipped {skipped_missing} missing", flush=True)

    # Free SFT model memory before saving
    del sft
    del sft_sd
    gc.collect()

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"[merge_layerwise] saving to {args.output.resolve()}", flush=True)
    base.save_pretrained(str(args.output))

    print("[merge_layerwise] saving tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.sft_path), trust_remote_code=True)
    tokenizer.save_pretrained(str(args.output))

    print("[merge_layerwise] done.", flush=True)


if __name__ == "__main__":
    main()
