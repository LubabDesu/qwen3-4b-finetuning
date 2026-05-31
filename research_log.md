# Research Log

This log records the main experiments, failures, infrastructure lessons, and
reward designs behind the final CSE151B competition submission. It is written as
a retrospective: some experiments were partial, broken, or superseded, but they
explain why the final pipeline looks the way it does.

## Key Numbers Reference

| Result | Score | Notes |
|---|---:|---|
| Base Qwen3-4B-Thinking | 58.3% | Strong starting point; later work mostly preserved or recovered its native reasoning. |
| Best standalone GRPO -> RS-SFT checkpoint (`waitle5`) | 61.0% | Best model-only result after strict rejection-sampled curation. |
| Draft 10 / ultrasafe inference baseline | 0.674 | Strong deterministic formatting and recovery baseline before later natural-conclusion reruns. |
| NT self-consistency | 0.681 | Small gain from voting on number-theory-like rows. |
| Rerun 16k natural conclusion | 0.692 | Key jump: regenerate structurally broken rows at longer budget and trust natural final boxes. |
| Draft 16 adaptive regen 30k/40k | 0.703 | Adaptive hard-row regeneration with long context and YaRN. |
| Stats regeneration | 0.703 | Flat versus draft 16; did not add measurable gain. |
| Final safe formatting | 0.706 | Final lift from guarded option-letter collapse and conservative formatting fixes. |

## Infrastructure Reference

- GCP was not usable for the main runs because A100 quota was effectively zero.
- Colab was usable but fragile: `torchao`, `transformers`, `vllm`, `peft`, CUDA,
  and bitsandbytes versions had to be kept compatible.
- Working setups ended up being environment-specific. L4 and Colab dependency
  stacks both needed pinning; treating the environment as disposable caused too
  much lost time.
- vLLM seeds did not guarantee bitwise-identical output across different batch
  compositions. Reproducibility needed fixed sampling seeds plus stable batching
  where practical, and final selection could not assume exact replay.
- YaRN `rope_scaling` had to be verified explicitly. Silent failure risk from
  `hf_overrides` meant long-context runs needed audit logs confirming the model
  actually received the intended rope configuration.

## Training Experiments

### Exp 0: Baseline

Qwen3-4B-Thinking started at 58.3%. The important conclusion was that the base
model already had useful math priors. Later fine-tuning had to avoid damaging
multi-answer and MCQ behavior.

### Exp 1-4: Early SFT Variants

Naive SFT variants underperformed. The repeated failure pattern was that
distilled or over-curated traces made the model less robust on answer format,
especially multi-answer rows.

### Exp 5: Broken Eval Lesson

The eval setup was flawed because MCQ options were not included in the training
prompt. The model often produced the mathematical value instead of the required
letter, and roughly 38% of eval errors came from this mismatch. The lesson was
sharp: eval infrastructure must be validated before training, not after.

This also means the broken GRPO eval should not be compared directly to the
later strict scores. The low GRPO eval number reflected infrastructure damage,
not only model capability.

### Exp 6: Phase 1 GRPO

GRPO did not produce a standalone score jump, but it exposed useful failure
modes and gave groundwork for later RS-SFT. The reward emphasized correctness,
usable boxing, MCQ shape, multi-answer count, and clipped-generation penalties.

### Exp 7-8: RS-SFT and Waitle Ablations

The best standalone checkpoint came from GRPO -> RS-SFT with strict data
curation. The key was not more data; it was keeping concise traces with one
terminal usable box and avoiding examples that taught long verification loops.

### Exp 9: Final Model Selection

The final model-only result was still not enough. The last gain came from
inference-time routing, regeneration, extraction fallback, and strict CSV
validation rather than more training.

## Inference Pipeline Experiments

### Draft 10 / Ultrasafe Baseline: 0.674

Draft 10 and the ultrasafe variant established the first strong post-training
inference baseline. The pipeline focused on conservative answer cleanup:
canonicalize to one final `\boxed{}`, preserve the model's reasoning, avoid
solving in post-processing, and reject structurally risky repairs.

This was the base for the final pipeline because it showed that much of the
remaining loss was format and truncation recovery, not missing math capability.

### NT Self-Consistency: 0.681

Number-theory-like rows were routed into a small self-consistency path. Multiple
samples were generated and voted by extracted final answer. This helped because
many number theory and combinatorics rows have compact answer spaces, so correct
solutions repeat more often than wrong branches.

The gain was real but modest. Voting only worked when the invalid-vote rate was
low enough and the extractor could reliably identify the terminal answer.

### Majority Voting Failure: 0.51

A broader n=6 majority-voting attempt failed badly, scoring about 0.51. Around
31% of generations had no usable box, so null votes dominated or corrupted the
aggregation. The conclusion was that voting is not a free accuracy multiplier:
it only helps after boxed-answer compliance is already high.

### Rerun 16k Natural Conclusion: 0.692

The key insight was to regenerate structurally broken rows with a longer budget
and accept natural model conclusions instead of asking a second-stage solver or
extractor to infer the answer from unfinished reasoning.

This changed the failure profile. Many rows were not mathematically impossible;
they were truncated before the final `\boxed{}` or stuck in verification loops.
Giving them enough room to conclude naturally recovered more than aggressive
post-hoc extraction.

### Adaptive Regen 30k/40k: 0.703

Draft 16 extended the rerun idea into adaptive regeneration. Rows were bucketed
by structural risk and topic. Hard rows, especially number theory, olympiad-like,
sequence, digit/base, and recurrence problems, used long-context settings.

The long-context path used YaRN with a larger `max_model_len`, high token
budgets, and smaller batch sizes to stay inside memory. This produced the major
late-stage lift from 0.692 to 0.703.

### Stats Regeneration: 0.703 Flat

An additional statistics-focused regeneration pass did not move the score. The
likely reason was that the remaining stats errors were not dominated by
truncation or answer-box structure, so the same natural-conclusion strategy did
not add new signal.

### Final Safe Formatting: 0.706

The final lift came from guarded formatting changes, especially safe
option-letter collapse. The rule only collapsed letter runs when the structure
made it clear that the model had produced option letters in a recoverable shape.

This mattered because MCQ rows are easy to damage with over-broad cleanup. The
final version added a small gain without repeating the wholesale-replacement
failure described below.

### The 0.667 Disaster

One attempted cleanup caused a drop to 0.667. The problem was wholesale
replacement: too many responses were overwritten by brittle extracted answers,
and MCQ behavior was destroyed. This was the strongest evidence that final
post-processing had to be conservative and structure-aware.

## Extractor Failure Modes

The two-stage extractor was meant to recover rows that had good reasoning but no
clean final answer. It failed when it grabbed intermediate values instead of the
actual terminal answer.

Concrete failures:

- id 929: extracted `5.860` only, missing the real answer context.
- id 491: grabbed an `x-bar` intermediate value.
- id 857: grabbed prose rather than a clean final answer.

This is why the final pipeline favors natural-conclusion regeneration over
second-pass solving or aggressive extraction. Extraction is useful only as a
last-mile structural fallback, not as the main recovery mechanism.

## Failure Mode Analysis

The main failure classes were:

- Capability errors: the model solved the math incorrectly.
- Truncation errors: the model was on a plausible path but ran out of tokens.
- Format errors: the answer existed but was not in a valid terminal box.
- MCQ shape errors: the model produced a value or malformed option instead of a
  single required letter.
- Multi-answer count errors: the model produced too few, too many, or repeated
  components for `[ANS]` blanks.

The strict-vs-capability gap was large. At one checkpoint, capability was about
57.3% while strict score was about 43.7%, a 13.6 point formatting loss. This
justified treating answer structure as a first-class objective.

## Eval Reliability

- 100-question public samples had roughly +/-5% variance, which made checkpoint
  selection noisy.
- Capability scoring and strict scoring had to be tracked separately. Strict
  score punished format failures that capability scoring could reveal as
  recoverable.
- The GRPO eval was fundamentally broken because of the MCQ prompt mismatch and
  should not be compared directly against later clean evals.
- Public-only evals were useful for iteration, but final decisions needed
  failure inspection, not just headline score.

## GRPO Reward Design

### Phase 1 GRPO Reward (Exp 6)

```text
correct + ideal box:        +1.00
correct + scattered box:    +0.70
correct + usable bad format:+0.50
correct + no usable box:    +0.30
wrong + ideal box:          +0.10
wrong + scattered box:      -0.30
wrong + usable bad format:  +0.00
wrong + no usable box:      -0.20
multi-answer count match:   +0.10
multi-answer count mismatch:-0.15
MCQ single letter:          +0.03
MCQ bad shape:              -0.05
clipped generation:         -0.25
floor/cap: [-0.50, +1.00]
```

This reward tried to preserve math correctness while nudging the model toward
usable final boxes. In practice, the eval infrastructure problems limited how
much could be learned from the headline score.

### Phase 3 Planned Reward

This reward was designed but never fully run.

```text
correct math:               +0.60
wrong math:                 +0.00
exact one final box:        +0.35
usable but bad box:         +0.15
no usable final box:        -0.15
right count in one box:     +0.15
split multiple boxes:       -0.15
wrong answer count:         -0.15
box inside <think>:         -0.15
text after final box:       -0.10
wait-loop after token 2000: up to -0.15 capped
MCQ single A-J:             +0.05
not single A-J:             -0.08
floor/cap: [-0.50, +1.00]
wrong-answer cap:           +0.40
```

The planned design separated math correctness from answer-structure reward more
cleanly. It also capped wrong-answer reward so a wrong but well-formatted output
could not outrank a correct solution.

## ORPO Failure Post-Mortem

The ORPO run used only about 40 preference pairs. Chosen log-probabilities stayed
below rejected log-probabilities throughout training, margins remained negative,
and MCQ performance dropped by about 6.2 percentage points.

Likely root causes:

- Too few pairs to learn stable preferences.
- Truncated chosen examples.
- No explicit MCQ guardrails.
- QLoRA noise relative to the small dataset size.

The conclusion was not that ORPO is useless, but that this run was too small and
too noisy to be a fair test.

## Key Learnings

- Validate eval infrastructure before training. The MCQ prompt mismatch wasted
  signal and made early comparisons misleading.
- Preserve strong base-model priors. Naive SFT can erase useful reasoning
  behavior faster than it adds format discipline.
- Treat final-answer structure as a first-class metric. A model can be capable
  but still lose heavily on strict judging.
- Voting only helps when invalid outputs are rare. Otherwise null votes and
  extractor errors overwhelm the benefit.
- Natural-conclusion regeneration is safer than extracting answers from
  unfinished reasoning.
- Post-processing must be guarded and local. Wholesale replacement can destroy
  MCQ rows and erase correct model conclusions.

## What I Would Do Differently

- Validate eval infrastructure first, including MCQ option prompts, strict
  extraction, truncation accounting, and public/private schema assumptions.
- Run RS-SFT from the base model as an ablation, skipping GRPO, to isolate how
  much of the gain came from rejection-sampled curation alone.
- Try n=16+ self-consistency only on targeted number theory and combinatorics
  rows, where the answer space is small and voting is most likely to help.
- Fully execute the structural GRPO reward with zone-based penalties for no-box,
  bad-box, false-negative, and post-box text failures.
- Retry ORPO only with 500+ balanced, untruncated preference pairs and explicit
  MCQ/multi-answer guardrails.
- Add automated eval-infra tests before every training run, including boxed
  extraction fixtures and MCQ letter-vs-value checks.

## Status Tracker

- Final submission score: 0.706.
- Final model path: `LubabDesu2/qwen3-4b-thinking-2507-waitle5`.
- Final inference entry point: `run_inference.py`.
- Compatibility entry point: `main.py`.
- Checkpoints, draft submissions, logs, generated datasets, and private data are
  intentionally excluded from git.
