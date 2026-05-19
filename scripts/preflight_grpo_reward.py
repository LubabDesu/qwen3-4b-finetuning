#!/usr/bin/env python3
"""Preflight checks for train_grpo reward logic.

Run before a long GRPO job:
    python scripts/preflight_grpo_reward.py

This intentionally avoids launching a model. It checks reward tiers, the empty
box case, batch-relative length shaping, and background-thread judging.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib
import signal
import sys
import threading
import time
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT))


def _install_heavy_import_stubs() -> None:
    """Let this script import train_grpo on machines without torch/trl."""
    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_bf16_supported=lambda: False)
        sys.modules["torch"] = torch

    if "wandb" not in sys.modules:
        wandb = types.ModuleType("wandb")
        wandb.init = lambda *args, **kwargs: None
        wandb.finish = lambda *args, **kwargs: None
        sys.modules["wandb"] = wandb

    if "datasets" not in sys.modules:
        datasets = types.ModuleType("datasets")

        class Dataset(list):
            @classmethod
            def from_list(cls, rows):
                return cls(rows)

        datasets.Dataset = Dataset
        sys.modules["datasets"] = datasets

    if "peft" not in sys.modules:
        peft = types.ModuleType("peft")
        peft.LoraConfig = lambda *args, **kwargs: None
        sys.modules["peft"] = peft

    if "transformers" not in sys.modules:
        transformers = types.ModuleType("transformers")

        class _Dummy:
            def __init__(self, *args, **kwargs):
                pass

        transformers.AutoTokenizer = _Dummy
        transformers.TrainerCallback = _Dummy
        transformers.TrainerControl = _Dummy
        transformers.TrainerState = _Dummy
        transformers.TrainingArguments = _Dummy
        sys.modules["transformers"] = transformers

    if "trl" not in sys.modules:
        trl = types.ModuleType("trl")

        class _Dummy:
            def __init__(self, *args, **kwargs):
                pass

        trl.GRPOConfig = _Dummy
        trl.GRPOTrainer = _Dummy
        sys.modules["trl"] = trl


def _import_train_grpo(force_stubs: bool):
    if force_stubs:
        _install_heavy_import_stubs()
    try:
        return importlib.import_module("train_grpo")
    except ModuleNotFoundError as exc:
        if exc.name in {"torch", "wandb", "datasets", "peft", "transformers", "trl"}:
            print(f"[warn] missing {exc.name}; using lightweight import stubs")
            _install_heavy_import_stubs()
            return importlib.import_module("train_grpo")
        if exc.name in {"sympy", "antlr4", "numpy"}:
            raise SystemExit(
                f"[fail] missing {exc.name}; run this inside the training/eval env "
                "where scripts/judger.py dependencies are installed"
            ) from exc
        raise


def _assert(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        suffix = f": {detail}" if detail else ""
        raise AssertionError(f"{name} failed{suffix}")
    print(f"[ok] {name}")


def _reward(train_grpo, completion: str, gold: list[str], options: list[str] | None = None):
    analysis = train_grpo._analyze_completion(completion, gold, options or [])
    reward = train_grpo.compute_reward(analysis, [analysis], gold)
    return reward, analysis


def check_reward_tiers(train_grpo) -> None:
    ideal = "<think>2+2=4</think>\n\\boxed{4}"
    reward, analysis = _reward(train_grpo, ideal, ["4"])
    _assert("ideal correct reward", reward == 1.0, f"reward={reward}, analysis={analysis}")

    missing_think = "2+2=4, so \\boxed{4}"
    reward, analysis = _reward(train_grpo, missing_think, ["4"])
    _assert("missing think boxed partial", 0.49 <= reward <= 0.51, f"reward={reward}, analysis={analysis}")

    prethink = "<think>2+2=4 so \\boxed{4}</think>\nFinal answer is 4."
    reward, analysis = _reward(train_grpo, prethink, ["4"])
    _assert("prethink boxed partial", 0.49 <= reward <= 0.51, f"reward={reward}, analysis={analysis}")

    no_box = "Reasoning gives the answer 4."
    reward, analysis = _reward(train_grpo, no_box, ["4"])
    _assert(
        "official fallback no-box partial",
        reward > 0,
        f"reward={reward}, correct={analysis['correct']}, extract={analysis['official_extract']}",
    )

    empty_box = "Reasoning gives the answer 4.\n\\boxed{}"
    reward, analysis = _reward(train_grpo, empty_box, ["4"])
    _assert(
        "empty box punished",
        reward <= -0.30 and not analysis["correct"],
        f"reward={reward}, correct={analysis['correct']}, extract={analysis['official_extract']}",
    )

    wrong_no_box = "I cannot solve this. " + ("ramble " * 1600)
    reward, analysis = _reward(train_grpo, wrong_no_box, ["4"])
    _assert("wrong no-box punished", reward <= -0.19, f"reward={reward}, tokens={analysis['tokens']}")

    multi = "<think>Compute separately.</think>\n\\boxed{2}\\boxed{4}"
    reward, analysis = _reward(train_grpo, multi, ["2", "4"])
    _assert("contiguous multi boxes accepted for shape", "2, 4" == analysis["answer_text"])
    _assert("multi count bonus applies", reward >= 1.0, f"reward={reward}, answer={analysis['answer_text']}")


def check_flat_clipped_penalty(train_grpo) -> None:
    old_max_new_tokens = train_grpo._max_new_tokens
    train_grpo._max_new_tokens = 100
    clipped = "<think>" + ("x" * 380) + "</think>\n\\boxed{4}"
    reward, analysis = _reward(train_grpo, clipped, ["4"])
    _assert(
        "flat clipped penalty",
        0.79 <= reward <= 0.81,
        f"reward={reward}, tokens={analysis['tokens']}, max_new_tokens={train_grpo._max_new_tokens}",
    )
    train_grpo._max_new_tokens = old_max_new_tokens

    long_good = "<think>" + ("needed " * 4000) + "</think>\n\\boxed{4}"
    long_good_a = train_grpo._analyze_completion(long_good, ["4"], [])
    _assert("long ideal still parses", long_good_a["ideal_box"], f"analysis={long_good_a}")


def check_signal_thread_safety(train_grpo, workers: int) -> None:
    completion = "<think>2+2=4</think>\n\\boxed{4}"
    old_signal = train_grpo.judger_module.signal.signal
    old_alarm = train_grpo.judger_module.signal.alarm

    def judge_once() -> bool:
        return train_grpo._judge_official(completion, ["4"], [])

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda _: judge_once(), range(workers * 4)))

    _assert("threaded official judging", all(results), f"results={results}")
    _assert("signal.signal restored", train_grpo.judger_module.signal.signal is old_signal)
    _assert("signal.alarm restored", train_grpo.judger_module.signal.alarm is old_alarm)

    barrier = threading.Barrier(workers)

    def judger_id() -> int:
        barrier.wait(timeout=5)
        time.sleep(0.05)
        return id(train_grpo._get_official_judger())

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        ids = list(pool.map(lambda _: judger_id(), range(workers)))

    _assert("thread-local judgers", len(set(ids)) == workers, f"ids={ids}")

    try:
        direct = train_grpo.Judger(strict_extract=False)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(direct.auto_judge, completion, ["4"], []).result(timeout=5)
    except ValueError as exc:
        if "signal only works in main thread" in str(exc):
            print("[ok] raw Judger still exposes main-thread signal failure; wrapper covers it")
            return
        raise
    except Exception as exc:
        print(f"[info] raw Judger raised {type(exc).__name__}: {exc}")
        return
    print("[info] raw Judger did not fail in this runtime; wrapper still passed threaded check")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stub-heavy-imports", action="store_true")
    args = parser.parse_args()

    train_grpo = _import_train_grpo(args.stub_heavy_imports)
    check_reward_tiers(train_grpo)
    check_flat_clipped_penalty(train_grpo)
    check_signal_thread_safety(train_grpo, args.workers)
    _assert("global signal module intact", signal.signal is train_grpo.judger_module.signal.signal)
    print("[ok] GRPO reward preflight passed")


if __name__ == "__main__":
    main()
