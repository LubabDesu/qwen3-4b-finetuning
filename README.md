# CSE151B Kaggle Math Inference

This repository contains a reproducible inference pipeline for the CSE151B math competition submission format.

## Model

Default Hugging Face model:

```text
LubabDesu2/qwen3-4b-thinking-2507-waitle5
```

The script also accepts a local checkpoint or another Hugging Face model through `--model_paths`.

## Input and Output

Input is a JSONL file with one problem per line:

```json
{"id": 0, "question": "...", "options": ["..."]}
```

Only `id` and `question` are required. `options` is optional.

Output is a quoted CSV:

```text
id,response
```

The response preserves reasoning. Physical newlines are escaped as literal `\n` so each row is one CSV line.

## Run

Recommended full run on an A100 80GB:

```bash
nohup env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
python run_inference.py \
  --input competition-data/private.jsonl \
  --output submission_drafts/submission.csv \
  --model_paths LubabDesu2/qwen3-4b-thinking-2507-waitle5 \
> logs/run_inference.log 2>&1 & echo $! > logs/run_inference.pid
```

Small smoke test:

```bash
nohup env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
python run_inference.py \
  --input competition-data/private.jsonl \
  --output submission_drafts/smoke.csv \
  --model_paths LubabDesu2/qwen3-4b-thinking-2507-waitle5 \
  --limit 8 \
  --standard-max-tokens 768 \
  --hard-max-tokens 1536 \
  --safety-max-tokens 1536 \
  --standard-batch-size 2 \
  --hard-batch-size 1 \
  --hard-max-model-len 8192 \
  --hard-rope-scaling-json '' \
> logs/smoke.log 2>&1 & echo $! > logs/smoke.pid
```

Check progress:

```bash
tail -f logs/run_inference.log
```

## Pipeline Summary

`run_inference.py` implements `run_inference(input_jsonl, output_csv, model_paths, seed=42)` and a CLI.

Stages:

1. Deterministic route by question text into `STANDARD` or `HARD`.
2. Generate standard rows with normal context and 15k token budget.
3. Generate hard rows with YaRN long context and 40k token budget.
4. Canonicalize every response to exactly one final `\boxed{...}` using `scripts/judger.py`.
5. Apply only guarded deterministic formatting repairs:
   - collapse `A,C,F` to `ACF` for single-blank multi-select answers when all components are option letters.
   - collapse a consecutive option-letter run only when it makes a multi-blank answer have exactly the required component count.
6. Safety-regenerate only structurally bad rows: truncated reasoning, empty/no valid boxed answer, prose boxed answer, or underfilled multi-blank answer.
7. Run a non-solving extractor only for rows still structurally invalid after safety regen. The extractor is forced with a `\boxed{` assistant prefill and a `}` stop token.
8. Final validation asserts unique ids, no null/empty responses, one physical CSV line per row, and non-empty `judger.extract_ans` after unescaping.

The pipeline intentionally does not use hard-coded row ids and does not use risky deterministic math overrides such as modulo auto-computation or MCQ value-to-letter conversion.

## Important Files

Commit these for inference:

```text
run_inference.py
scripts/judger.py
scripts/build_draft12_safe.py
requirements.txt
requirements.full-freeze.txt
README.md
.gitignore
```

Do not commit checkpoints, logs, submissions, private data, or model weight files.
