# Project Status

## Current State

- Stage 1_2 dataset builder has been patched for stricter cleaning:
  - rejects Chinese characters in questions/reasoning/targets
  - rejects abrupt reasoning tails
  - normalizes MCQ answers like `(B)` to `B`
  - prints visible build progress with `--log-every`
- Latest Stage 1_2 dataset sanity check passed:
  - `8480` rows total
  - `6480` OpenR1
  - `1000` MathInstruct aqua_rat
  - `1000` Stage 0 format anchors
- Stage 1_2 training completed for one full epoch.
- Local retained Trainer checkpoints:
  - `checkpoints/trainer_stage1_2/checkpoint-400`
  - `checkpoints/trainer_stage1_2/checkpoint-500`
  - `checkpoints/trainer_stage1_2/checkpoint-530`
- Google Drive has checkpoints:
  - `checkpoint-100`
  - `checkpoint-200`
  - `checkpoint-300`
  - `checkpoint-400`
  - `checkpoint-500`
  - `checkpoint-530`
  - Drive path: `gdrive:151B_SP26_Competition/checkpoints/stage1_2/trainer_stage1_2`
- Final LoRA and merged model are local:
  - `checkpoints/lora_stage1_2`
  - `checkpoints/merged_stage1_2`

## Important Scripts

- `scripts/build_stage1_2_r2.py`
  - Builds the cleaned Stage 1_2 dataset.
- `scripts/sanity_check_stage1_2.py`
  - Validates Stage 1_2 JSONL before training.
- `scripts/train_stage1_2.py`
  - Terminal/nohup training entrypoint.
  - Saves Trainer checkpoints every 100 steps by default.
  - Can sync checkpoints/final outputs to Drive with `--drive-target`.
- `scripts/evaluate_stage1_2_checkpoints.py`
  - Downloads missing checkpoints from Drive.
  - Merges LoRA checkpoints.
  - Evaluates merged checkpoints with vLLM.
  - Supports public-only eval with `--public-only`.
- `scripts/judger.py` and `scripts/utils.py`
  - Competition-style judging utilities for public eval.

## Eval Status

- Uploaded public eval file:
  - `artifacts/post_training_curriculum/eval/public.jsonl`
  - `1126` rows total
  - `375` option/MCQ rows
- First public eval attempt used `EVAL_MAX_NEW_TOKENS=1024`.
- That was too short:
  - responses averaged around `638` words
  - many responses truncated before `\boxed{}`
  - partial checkpoint-100 public eval showed low boxed compliance
- `scripts/evaluate_stage1_2_checkpoints.py` now supports:
  - `--max-new-tokens 4096`

## Recommended Next Eval Command

Stop any old eval first if it is still running:

```bash
ps -ef | grep evaluate_stage1_2_checkpoints
```

Then run public 200-row eval with a larger generation cap:

```bash
mkdir -p logs

nohup env PYTHONUNBUFFERED=1 .venv/bin/python scripts/evaluate_stage1_2_checkpoints.py \
  --public-only \
  --public-path artifacts/post_training_curriculum/eval/public.jsonl \
  --public-sample-size 200 \
  --public-seed 42 \
  --steps 100 200 300 400 500 530 \
  --drive-source "gdrive:151B_SP26_Competition/checkpoints/stage1_2/trainer_stage1_2" \
  --batch-size 5 \
  --max-new-tokens 4096 \
  > logs/eval_public_200_4096.log 2>&1 &

echo $! > logs/eval_public_200_4096.pid
```

Watch progress:

```bash
tail -f logs/eval_public_200_4096.log
```

## Git Guidance

Commit code/config only:

```bash
git add AGENTS.md requirements.txt requirements.full-freeze.txt \
  cse151b_post_training_curriculum.ipynb \
  scripts/build_stage1_2_r2.py \
  scripts/train_stage1_2.py \
  scripts/sanity_check_stage1_2.py \
  scripts/evaluate_stage1_2_checkpoints.py \
  scripts/judger.py \
  scripts/utils.py
```

Do not commit:

- `checkpoints/`
- `logs/`
- generated `artifacts/post_training_curriculum/datasets/`
- generated `artifacts/post_training_curriculum/eval/`

