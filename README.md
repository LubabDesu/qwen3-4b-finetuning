# Post-Training Qwen3-4B for Competition Math

Took **Qwen3-4B-Thinking from a 58.3% baseline to 0.706** on a Kaggle-style math
benchmark, covering free-form, MCQ, and multi-answer problems. The final result
came from 9 post-training experiments plus an inference-time recovery pipeline.

This repo contains the full reproducible inference pipeline (`run_inference.py`)
and the key scripts used to build, train, evaluate, and package the final
submission. Generated checkpoints, logs, draft submissions, and intermediate
datasets are intentionally kept out of git.

---

## TL;DR

| Stage | Score | Key insight |
|---|---:|---|
| Base Qwen3-4B-Thinking | 58.3% | Strong native math priors |
| SFT variants | 47-55% | Naive SFT hurt multi-answer performance |
| GRPO | 58% | No standalone gain, but useful groundwork |
| GRPO -> RS-SFT (`waitle5`) | 61.0% | Best standalone checkpoint after strict data curation |
| + inference pipeline | **0.706** | Truncation recovery plus topic routing |

Final competition result: **0.706**. The most interesting part of the project was
the process: many experiments failed, and each failure narrowed the path to the
pipeline that worked.

---

## What I Learned

- **Do not fine-tune away strong priors.** Qwen3-4B-Thinking already reasoned
  well. Naive SFT on distilled traces collapsed multi-answer accuracy across
  every variant tried.
- **Overthinking, not capability, was the bottleneck.** Failure-mode analysis
  showed many wrong answers came from formatting, truncation, or verification
  loops rather than from missing math ability.
- **GRPO + RS-SFT was the recipe, not RS-SFT alone.** GRPO alone did not improve
  final accuracy, but it helped diagnose overthinking and produced the checkpoint
  used for RS-SFT rejection sampling.
- **Data curation beat method complexity.** The strongest lever was filtering
  RS-SFT traces to keep concise, single-terminal-box reasoning.
- **Trust the model's own conclusion over a second-pass solver.** When a
  response truncated, regenerating at higher token budget and accepting only
  naturally concluded answers worked better than using an extractor to infer
  values from unfinished reasoning.

---

## The Inference Pipeline

The standalone model scored about 61%. The remaining gain came from an
inference-time pipeline that recovers structurally broken generations without
retraining.

```text
1. ROUTE      Deterministic topic routing:
              number theory / olympiad / sequences -> HARD
              everything else                       -> STANDARD

2. GENERATE   STANDARD rows: normal context, 15k token budget
              HARD rows:     YaRN long context, 40k token budget

3. REROUTE    Response-based audit:
              standard rows that truncate or fail to produce a valid box
              are pulled into the repair path.

4. REPAIR     Guarded deterministic formatting fixes only:
              - canonicalize to exactly one final \boxed{}
              - collapse option-letter runs only when structurally valid

5. REGEN      Safety regeneration for structurally bad rows only:
              truncated / empty / prose-in-box / underfilled multi-blank

6. EXTRACT    Last-mile, non-solving extractor for rows still invalid:
              forced \boxed{ prefill plus } stop token

7. VALIDATE   Assert unique ids, no nulls, one CSV line per row, and
              non-empty judger extraction after unescaping.
```

Stages 1 and 4 are fully deterministic. Stages 2 and 5 are stochastic. The
pipeline does not use hard-coded row ids; every bucket is derived from
question/response structure, so it generalizes to new test splits.

---

## Reproduce

### Environment

Use a Python virtual environment and install the pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If the environment already exists:

```bash
source .venv/bin/activate
```

### Model

Default Hugging Face model:

```text
LubabDesu2/qwen3-4b-thinking-2507-waitle5
```

The script also accepts a local checkpoint or another Hugging Face model through
`--model_paths`.

### Input / Output

Input is JSONL, one problem per line. Only `id` and `question` are required;
`options` is optional.

```json
{"id": 0, "question": "...", "options": ["..."]}
```

Output is a quoted CSV with columns:

```text
id,response
```

Reasoning is preserved. Physical newlines are escaped as literal `\n` so each
row is one CSV line.

### Full Run

```bash
python run_inference.py \
  --input competition-data/private.jsonl \
  --output submission_drafts/submission.csv \
  --model_paths LubabDesu2/qwen3-4b-thinking-2507-waitle5
```

### Smoke Test

```bash
python run_inference.py \
  --input competition-data/private.jsonl \
  --output submission_drafts/smoke.csv \
  --model_paths LubabDesu2/qwen3-4b-thinking-2507-waitle5 \
  --limit 8 --standard-max-tokens 768 --hard-max-tokens 1536 \
  --safety-max-tokens 1536 --standard-batch-size 2 --hard-batch-size 1 \
  --hard-max-model-len 8192 --hard-rope-scaling-json ''
```

### Production Run With Logging

```bash
mkdir -p logs submission_drafts

nohup env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
python run_inference.py \
  --input competition-data/private.jsonl \
  --output submission_drafts/submission.csv \
  --model_paths LubabDesu2/qwen3-4b-thinking-2507-waitle5 \
> logs/run_inference.log 2>&1 & echo $! > logs/run_inference.pid

tail -f logs/run_inference.log
```

### Hardware / Runtime

- GPU: A100 80GB, `tensor_parallel_size=1`
- STANDARD pass is fast; HARD pass with YaRN and 40k tokens is the slow part
- Full private-set inference takes several hours

### Entry Point

`run_inference.py` exposes:

```python
run_inference(input_jsonl, output_csv, model_paths, seed=42)
```

It also provides a CLI. One call reproduces the submission CSV end-to-end.

---

## Repo Layout

```text
run_inference.py                         # single entry point, full inference pipeline
competition-data/                        # public data and sample submission
artifacts/post_training_curriculum/      # small checked-in notes/specs
scripts/build_stage1_2_r2.py             # cleaned Stage 1.2 dataset builder
scripts/sanity_check_stage1_2.py         # dataset validation
scripts/train_stage1_2.py                # LoRA training entry point
scripts/evaluate_stage1_2_checkpoints.py # checkpoint merge/eval harness
scripts/judger.py                        # competition-style judging utilities
scripts/build_draft12_safe.py            # extraction/canonicalization helpers
requirements.txt                         # pinned inference deps
requirements.full-freeze.txt             # exact frozen environment
```

The rest of `scripts/` contains experiment utilities from the post-training
search: GRPO, RS-SFT data construction, failure analysis, draft merging, and
submission post-processing. They are kept for transparency, but the shortest
path to reproduce the final CSV starts at `run_inference.py`.

Not committed: checkpoints, logs, draft submissions, private data, model weight
files, local virtual environments, scratch outputs, and generated intermediate
datasets.

---

## Notes on Reproducibility

- Fixed per-row seeds; `scripts/judger.py` is committed.
- Stochastic stages can produce small run-to-run variation on the hardest rows.
- `max_model_len` and batch sizes affect throughput and memory usage. They can
  be tuned to the available GPU.
