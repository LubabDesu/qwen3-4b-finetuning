#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into a full base model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a full model.")
    parser.add_argument("--base-model", required=True, type=Path, help="Full base model path.")
    parser.add_argument("--adapter", required=True, type=Path, help="LoRA adapter path.")
    parser.add_argument("--output", required=True, type=Path, help="Output full model path.")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model load dtype.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    print(f"[merge_lora] loading base: {args.base_model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.base_model),
        torch_dtype=torch_dtype,
        device_map="cpu",
        trust_remote_code=True,
    )

    print(f"[merge_lora] loading adapter: {args.adapter}", flush=True)
    model = PeftModel.from_pretrained(model, str(args.adapter))

    print("[merge_lora] merging adapter into base", flush=True)
    merged = model.merge_and_unload()

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"[merge_lora] saving model: {args.output}", flush=True)
    merged.save_pretrained(str(args.output), safe_serialization=True)

    print("[merge_lora] saving tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.adapter), trust_remote_code=True)
    tokenizer.save_pretrained(str(args.output))

    print("[merge_lora] done", flush=True)


if __name__ == "__main__":
    main()
