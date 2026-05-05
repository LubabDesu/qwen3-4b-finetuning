# Stage 0 Rerun Failure Analysis

Source files:

- `artifacts/post_training_curriculum/eval/stage0_eval_results.jsonl`
- `artifacts/post_training_curriculum/eval/stage0_eval_summary.json`
- `artifacts/post_training_curriculum/eval/heldout_eval_set.jsonl`

This report analyzes the new Stage 0 rerun after these fixes:

- Rebuilt heldout set with 200 unique IDs.
- Fixed nested `\boxed{...}` answer extraction.
- Fixed malformed MATH gold answers.
- Fixed Python prompt escaping so `\boxed` is not emitted as a backspace control character.
- Added concise eval prompt.
- Increased eval max generation budget to `EVAL_MAX_NEW_TOKENS = 3072`.

Important: the rerun improved overall accuracy, but MCQ and multi-answer behavior remain major bottlenecks.

## Headline Metrics

| Metric | Old Stage 0 | New Stage 0 Rerun | Change |
|---|---:|---:|---:|
| Overall accuracy | 36/200 = 18.0% | 48/200 = 24.0% | +6.0 pts |
| Boxed format compliance | 38/200 = 19.0% | 54/200 = 27.0% | +8.0 pts |
| MCQ accuracy | 2/20 = 10.0% | 1/20 = 5.0% | -5.0 pts |
| Multi-answer accuracy | 0/15 = 0.0% | 0/15 = 0.0% | no change |
| Average response words | 1063.8 | 1476.7 | worse |

The new parser and prompt fixes helped some free-form MATH problems, but the increased token budget also allowed the model to produce longer unfinished reasoning. Format compliance is still far below the target.

## Current Rerun Summary

From `stage0_eval_summary.json`:

```json
{
  "n": 200,
  "accuracy": 0.24,
  "correct": 48,
  "mcq_accuracy": 0.05,
  "mcq_total": 20,
  "multi_answer_accuracy": 0.0,
  "multi_answer_total": 15,
  "avg_response_words": 1476.71,
  "boxed_compliance_rate": 0.27,
  "inference_backend": "vllm"
}
```

## Breakdown By Slice

| Slice | Count | Correct | Accuracy | Format OK | Format Rate | Avg Words |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 200 | 48 | 24.0% | 54 | 27.0% | 1476.7 |
| Public | 50 | 6 | 12.0% | 8 | 16.0% | 1531.8 |
| MATH test | 150 | 42 | 28.0% | 46 | 30.7% | 1458.4 |
| MCQ | 20 | 1 | 5.0% | 3 | 15.0% | 1711.7 |
| Multi-answer | 15 | 0 | 0.0% | 0 | 0.0% | 1494.3 |
| Single free-form | 165 | 47 | 28.5% | 51 | 30.9% | 1446.6 |

The model is mainly improving on single-answer free-form MATH rows. Public-format tasks are still poor, especially MCQ and multi-blank questions.

## Correctness And Format Cross-Tab

| Correct? | Format OK? | Count | Interpretation |
|---|---|---:|---|
| True | True | 38 | clean wins |
| True | False | 10 | model solved but failed output contract |
| False | True | 16 | model followed format but answer was wrong |
| False | False | 136 | both reasoning/answer and format failed |

There are still `10` recoverable cases where the model likely had the right answer but failed final formatting.

## Response Shape Failure

| Response shape | Count |
|---|---:|
| Missing `</think>` | 132 |
| Format OK | 54 |
| Has `</think>` but no valid box after it | 10 |
| Multiple boxes after `</think>` | 2 |
| Other format failure | 2 |

Dominant failure remains generation that never reaches the final answer. The model still spends too much of the budget in reasoning.

## Word Count Effect

| Word Count Bucket | Rows | Format OK | Correct |
|---|---:|---:|---:|
| 0-599 | 11 | 9 | 9 |
| 600-999 | 29 | 25 | 21 |
| 1000-1499 | 51 | 18 | 10 |
| 1500-2199 | 98 | 2 | 8 |
| 2200+ | 11 | 0 | 0 |

Shorter outputs are much more likely to be formatted and correct. Longer outputs are usually unfinished or confused.

Conclusion: raising `max_new_tokens` alone is not the fix. It gives the model more room to ramble. The model needs training/inference pressure to finish early and produce exactly one final box.

## MCQ Failure Analysis

MCQ is the largest visible problem.

Current MCQ result:

- `20` MCQ rows.
- `1` correct.
- `3` format OK.
- Average response length: `1711.7` words.

Main MCQ failure modes:

1. The model solves or partially solves, but never emits final boxed letter.
2. The model mentions the correct option in reasoning, then keeps going until cutoff.
3. The model boxes a wrong guessed option after a vague comparison.
4. The model tries to reason through too many options.
5. The model lacks training on the course’s 9-10 option MCQ format.

### MCQ Example: Row 1, ID `228`

Gold: `I`

Question: smallest positive root interval for `x^3 - 3x + 1 = 0`.

Observed behavior:

- Model finds that the root lies between `0` and `0.5`.
- Model explicitly says `Option I: [0, 0.5]` contains the root.
- Response does not close `</think>`.
- Response does not emit `\boxed{I}`.
- Marked incorrect and format bad.

Interpretation: this is not a math failure. It is a completion/format failure.

### MCQ Example: Row 3, ID `563`

Gold: `C`

Observed boxed answer: `I`

Behavior:

- The model reaches a final box, so format passes.
- It chooses the wrong option.
- The reasoning becomes generic and option-matching is unreliable.

Interpretation: this is a real MCQ reasoning/selection failure.

### MCQ Example: Row 48, ID `592`

Gold: `G`

Observed behavior:

- `correct=True`, but `format_ok=False`.
- The answer appears in the tail as option `G`.
- No clean final boxed answer after `</think>`.

Interpretation: another recoverable formatting failure.

## Multi-Answer Failure Analysis

Multi-answer remains completely broken.

Current multi-answer result:

- `15` multi-answer rows.
- `0` correct.
- `0` format OK.
- Average response length: `1494.3` words.

Main multi-answer failure modes:

1. Gives answers in prose but not in final comma-separated box.
2. Boxes only the first answer.
3. Uses labels inside final answer, or repeats `[ANS]` placeholders.
4. Does not include all blanks.
5. Gets overwhelmed by table-style or many-part prompts.
6. Fails to count `[ANS]` blanks reliably.

### Multi Example: Row 6, ID `285`

Gold:

```text
9, 16, 27, 26
```

Observed behavior:

- Model gives the correct values in prose:
  - `9`
  - `16`
  - `27`
  - `26`
- It never emits a valid final `\boxed{9, 16, 27, 26}`.
- Marked incorrect and format bad.

Interpretation: output contract failure, not math failure.

### Multi Example: Row 32, ID `198`

Gold has 5 answers.

Observed behavior:

- Model boxes only one answer:

```text
\boxed{m = k \cdot r}
```

- It misses the remaining four required answers.

Interpretation: the model does not understand that final box must contain one answer per `[ANS]` blank.

### Multi Example: Row 41, ID `161`

Gold has 2 answers.

Observed behavior:

- Model boxes only:

```text
\boxed{decreasing}
```

- It omits the numeric second answer.

Interpretation: partial-answer finalization failure.

## Course Public Dataset Shape

The course-provided public dataset has `1126` rows.

| Property | Count |
|---|---:|
| Total rows | 1126 |
| Rows with options | 375 |
| Rows without options | 751 |
| Multi-blank rows | 415 |
| List-answer rows | 741 |
| String-answer rows | 385 |

Option count distribution:

| Number of options | Rows |
|---:|---:|
| 10 | 336 |
| 9 | 14 |
| 8 | 7 |
| 7 | 10 |
| 6 | 4 |
| 5 | 4 |
| 0 | 751 |

Blank count distribution:

| Number of `[ANS]` blanks | Rows |
|---:|---:|
| 0 | 386 |
| 1 | 325 |
| 2 | 172 |
| 3 | 88 |
| 4 | 60 |
| 5 | 31 |
| 6 | 21 |
| 7+ | 44 |

Large multi-answer examples exist, including rows with `24` and `42` blanks.

## Rough Topic Coverage In Public Dataset

These are approximate keyword buckets; categories overlap.

| Topic / Format Bucket | Rows | MCQ Rows | Multi Rows |
|---|---:|---:|---:|
| Geometry / trig | 397 | 104 | 180 |
| Algebra / functions | 244 | 80 | 112 |
| Statistics / regression / data | 205 | 10 | 147 |
| Finance / units / word problems | 157 | 10 | 85 |
| Calculus / integral / derivative | 112 | 87 | 16 |
| Probability / combinatorics | 99 | 31 | 53 |
| Linear algebra / vectors | 30 | 23 | 7 |
| Tables / many blanks | 40 | 1 | 31 |

This explains why MATH-only or mostly-MATH training does not cover the course distribution well. The course data has many structured, applied, table, unit, and multi-blank outputs.

## Existing Stage 0 Training Data

Current `stage0_records.jsonl`:

| Property | Count |
|---|---:|
| Total records | 608 |
| `public_vanilla_correct` | 500 |
| `synthetic_multi_answer_stage0` | 108 |
| MCQ records | 186 |
| Non-MCQ records | 422 |

Answer-count distribution in Stage 0 training:

| `n_ans` | Records |
|---:|---:|
| 0 | 186 |
| 1 | 163 |
| 2 | 108 |
| 3 | 105 |
| 4 | 25 |
| 5 | 5 |
| 6 | 6 |
| 7+ | 9 |

Stage 0 does not have enough high-quality examples for:

- 10-option MCQ selection.
- 5+ answer final boxes.
- table completion.
- stats/data tasks with many numeric answers.
- unit/category answers.
- concise finalization after reasoning.

## What This Means

The new run says:

1. Parser and gold fixes helped.
2. More generation budget helped a little but also increased rambling.
3. The model still does not reliably obey the final-output contract.
4. MCQ and multi-answer are separate failure families.
5. Stage 0 is not done. It needs a Stage 0b focused on format and course-specific output shapes.

The model’s free-form single-answer behavior is noticeably better than MCQ/multi-answer behavior. The next training should not be generic math training; it should target the course’s answer formats.

## Recommended Stage 0b Dataset

Build a new formatting-focused dataset with roughly `1500-2500` examples.

Suggested mix:

| Data Type | Count | Purpose |
|---|---:|---|
| 10-option MCQ, course-style | 600-800 | teach boxed letter selection |
| Multi-answer `[ANS]` rows | 600-800 | teach comma-separated final boxes |
| Table / many-blank rows | 250-400 | teach long answer lists |
| Single-answer clean rows | 200-300 | preserve basic formatting |
| Optional `math_mcqa` rows | 500-1000 | reinforce MCQ habit, but only 4-option |

Do not rely only on `math_mcqa`. It has 4 options, while the course public dataset mostly uses 10 options.

## Distillation Strategy

Use teacher models for clean traces, but give the teacher the known answer. Do not ask the teacher to freely solve if the answer is already known.

Recommended teacher prompt:

```text
You are writing a supervised fine-tuning trace.

Question:
{question}

Known correct final answer:
{answer}

Write a concise solution that leads to the known answer.

Required output:
<think>
Brief reasoning. Do not repeat the problem. Do not discuss every MCQ option unless necessary.
</think>

\boxed{{answer}}

Rules:
- Do not change the known final answer.
- For MCQ, final box contains only the letter.
- For [ANS] blanks, final box contains answers only, comma-separated, in order.
- No labels inside the box.
- No text after the final box.
```

For multi-answer examples, use:

```text
Known correct final answer:
answer1, answer2, answer3, ...
```

The final target should be:

```text
\boxed{answer1, answer2, answer3, ...}
```

Bad final:

```text
\boxed{a. answer1; b. answer2}
```

Good final:

```text
\boxed{answer1, answer2}
```

## Avoiding Memorization

Direct distillation on `public.jsonl` can teach memorized public answers. Whether that is acceptable depends on course rules.

Safer approach:

1. Exclude heldout public eval IDs.
2. Use remaining public rows as style/format training.
3. Generate synthetic variants with changed numbers and recomputed answers.
4. Use external datasets like `math_mcqa` for additional MCQ shape.
5. Filter exact question overlap with `heldout_eval_set.jsonl`.

Recommended split:

- `40%` direct public-style known-answer distill, excluding heldout.
- `40%` synthetic public-style variants.
- `20%` external MCQ/math data, filtered.

## Training Recommendation

Do a small Stage 0b first, not a full long run.

Suggested first attempt:

- Dataset: `1500-2500` examples.
- Steps: `200-400`.
- Learning rate: no higher than current Stage 0 unless you are intentionally doing format-only correction.
- Keep examples short and clean.
- Ensure every assistant target ends with exactly one `\boxed{...}`.

Evaluate immediately after Stage 0b.

Targets for next eval:

| Metric | Minimum Useful Target |
|---|---:|
| Boxed format compliance | >80% |
| MCQ accuracy | >40% |
| Multi-answer accuracy | >50% |
| Average response words | <900 |

If format compliance stays low, continue format training. If format compliance rises but accuracy remains low, shift to skill/math data.

## Immediate Next Actions

1. Do not increase token budget again yet.
2. Keep `EVAL_MAX_NEW_TOKENS = 3072` as safety net.
3. Build Stage 0b targeted dataset.
4. Include many 10-option MCQ examples.
5. Include many multi-answer examples, especially 3-8 answers.
6. Include some table/many-blank examples.
7. Distill from a strong teacher using known answers.
8. Filter heldout public IDs and exact MATH overlap.
9. Train small Stage 0b.
10. Re-evaluate and compare by slice.

