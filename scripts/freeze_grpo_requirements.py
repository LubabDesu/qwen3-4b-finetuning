#!/usr/bin/env python3
"""Freeze the important packages from the dedicated GRPO virtualenv."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VENV = ROOT / ".venv_grpo"
DEFAULT_CURATED = ROOT / "requirements.grpo.txt"
DEFAULT_FULL = ROOT / "requirements.grpo.full-freeze.txt"

IMPORTANT_PACKAGES = {
    "accelerate",
    "datasets",
    "einops",
    "fastapi",
    "gguf",
    "huggingface-hub",
    "lm-format-enforcer",
    "mistral-common",
    "numpy",
    "openai",
    "outlines",
    "pandas",
    "peft",
    "protobuf",
    "pyarrow",
    "ray",
    "safetensors",
    "sentencepiece",
    "sympy",
    "tiktoken",
    "tokenizers",
    "torch",
    "torchvision",
    "transformers",
    "triton",
    "trl",
    "uvicorn",
    "vllm",
    "wandb",
    "xformers",
}

IMPORTANT_PREFIXES = (
    "nvidia-",
)


def package_name(requirement: str) -> str:
    name = requirement
    for separator in ("===", "==", ">=", "<=", "~=", "!=", ">", "<"):
        if separator in name:
            name = name.split(separator, 1)[0]
            break
    return name.strip().replace("_", "-").lower()


def run_pip(python_bin: Path, *args: str) -> list[str]:
    result = subprocess.run(
        [str(python_bin), "-m", "pip", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and not line.startswith("#")
    ]


def is_important(requirement: str) -> bool:
    name = package_name(requirement)
    return name in IMPORTANT_PACKAGES or any(
        name.startswith(prefix) for prefix in IMPORTANT_PREFIXES
    )


def render_curated(python_version: str, requirements: list[str]) -> str:
    lines = [
        "# Curated GRPO environment pins.",
        "# Generated from .venv_grpo by scripts/freeze_grpo_requirements.py.",
        f"# Python: {python_version}",
        "# Recreate with:",
        "#   python3.10 -m venv .venv_grpo",
        "#   .venv_grpo/bin/python -m pip install --upgrade pip",
        "#   .venv_grpo/bin/python -m pip install -r requirements.grpo.txt",
        "# Torch CUDA wheels need the PyTorch CUDA 12.1 index.",
        "--extra-index-url https://download.pytorch.org/whl/cu121",
        "",
    ]
    lines.extend(requirements)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze curated and full requirements for the GRPO virtualenv."
    )
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV)
    parser.add_argument("--curated-out", type=Path, default=DEFAULT_CURATED)
    parser.add_argument("--full-out", type=Path, default=DEFAULT_FULL)
    args = parser.parse_args()

    python_bin = args.venv / "bin" / "python"
    if not python_bin.exists():
        raise SystemExit(f"Missing virtualenv Python: {python_bin}")

    full_freeze = run_pip(python_bin, "freeze")
    curated = sorted(
        (line for line in full_freeze if is_important(line)),
        key=lambda line: package_name(line),
    )
    python_version = subprocess.run(
        [str(python_bin), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    args.full_out.write_text("\n".join(full_freeze) + "\n")
    args.curated_out.write_text(render_curated(python_version, curated))

    print(f"Wrote {args.curated_out} ({len(curated)} curated pins)")
    print(f"Wrote {args.full_out} ({len(full_freeze)} full pins)")


if __name__ == "__main__":
    main()
