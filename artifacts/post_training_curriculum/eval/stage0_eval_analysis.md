# Stage 0 Eval Analysis

Source files:
- `stage0_eval_results.jsonl`
- `stage0_eval_summary.json`

## Headline

Stage 0 currently scores **36/200 = 18.0% accuracy** with **38/200 = 19.0% boxed-format compliance**.

The biggest issue is not just mathematical ability. Most generations are too long, do not close the thinking block, and never emit the required final `\boxed{...}` answer. The eval is therefore measuring a mixture of:

1. Real math errors.
2. Formatting/completion failures.
3. Evaluation-data issues, especially malformed extracted MATH gold answers.
4. Interpretability issues caused by duplicate IDs for all MATH rows.

## Metrics

| Slice | Count | Correct | Accuracy | Format OK | Format Rate | Avg Words |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 200 | 36 | 18.0% | 38 | 19.0% | 1063.8 |
| Public | 50 | 6 | 12.0% | 6 | 12.0% | - |
| MATH test | 150 | 30 | 20.0% | 32 | 21.3% | - |
| MCQ | 20 | 2 | 10.0% | 3 | 15.0% | 1176.4 |
| Free-form | 180 | 34 | 18.9% | 35 | 19.4% | 1051.2 |
| Multi-answer blanks | 15 | 0 | 0.0% | 0 | 0.0% | 1043.1 |
| Single free-form | 165 | 34 | 20.6% | 35 | 21.2% | 1052.0 |

Correctness/format cross-tab:

| Category | Count |
|---|---:|
| Correct and format OK | 29 |
| Correct but format bad | 7 |
| Format OK but wrong | 9 |
| Wrong and format bad | 155 |

Response-shape counts:

| Response shape | Count |
|---|---:|
| Missing `</think>` | 149 |
| Has `</think>` but no valid box after it | 11 |
| Multiple boxes after `</think>` | 2 |
| Format OK | 38 |

Length is strongly correlated with format failure:

| Word-count bucket | Rows | Format OK | Correct |
|---|---:|---:|---:|
| 0-299 | 3 | 1 | 1 |
| 300-599 | 11 | 8 | 7 |
| 600-899 | 33 | 18 | 13 |
| 900-1099 | 56 | 6 | 8 |
| 1100+ | 97 | 5 | 7 |

## Current Failure Modes

### 1. Runs out of budget before final answer

This is the dominant failure mode. `149/200` responses never close `</think>`, so the required final answer never appears. The average response is over 1000 words, and `97/200` are at least 1100 words.

Example: row 1, id `228`, gold `I`, marked incorrect and format bad.

The model actually reaches the right option in the body: it says the smallest positive root is in `(0, 0.5)` and identifies "Option I: [0, 0.5]". But the response ends while still verifying `f(0.5) = -0.375`; there is no `</think>` and no final `\boxed{I}`. The fallback MCQ extractor then cannot reliably recover the intended answer.

### 2. The prompt encourages too much MCQ work

The system prompt says: "MCQ: explicitly verify EVERY option. State why wrong options fail." For 10-option math questions, this produces huge answers and increases truncation risk. MCQ accuracy is only `2/20 = 10%`, and MCQ format compliance is only `3/20 = 15%`.

Example: row 3, id `563`, gold `C`, marked incorrect and format bad.

The model gets bogged down deriving implicit derivatives and repeatedly corrects itself. It never reaches a final option.

### 3. Multi-answer public questions are not being finalized in the required shape

All `15/15` multi-answer rows fail, and all `15/15` also fail format compliance. The model often solves some or all blanks, but it does not emit exactly one comma-separated boxed answer with the right number of parts.

Example: row 6, id `285`, gold `['9', '16', '27', '26']`, marked incorrect and format bad.

The model states the right values in the reasoning:

`1a: 9`, `1b: 16`, `2a: 27`, `2b: 26`

But the final marker is malformed as a backspace/control-character version of `boxed`, not a valid `\boxed{9, 16, 27, 26}`.

### 4. Some MATH gold answers are malformed

There are `8/150` MATH rows where the extracted gold answer has unbalanced braces, usually because `extract_last_boxed()` does not robustly parse nested LaTeX. These all count as incorrect.

Examples:

| Row | Question | Extracted Gold | Model final |
|---:|---|---|---|
| 51 | `Compute \tan 210^\circ` | `\frac{\sqrt{3` | `\boxed{\dfrac{\sqrt{3}}{3}}` |
| 74 | `Find \sin \frac{4 \pi}{3}` | `-\frac{\sqrt{3` | `\boxed{-\dfrac{\sqrt{3}}{2}}` |
| 90 | `Find \csc(-120^\circ)` | `-\frac{2 \sqrt{3` | `\boxed{-\frac{2\sqrt{3}}{3}}` |

These are evaluation extraction errors. Row 51 and row 90 look mathematically correct from the model side but are marked wrong because the gold target is broken.

### 5. MATH row IDs are all duplicated

All 150 MATH test rows have the same ID: `math_test_Precalculus_50`.

This comes from the eval-set builder using:

`'id': f"math_test_{ex.get('type','')}_{len(eval_rows)}"`

while iterating before the MATH rows are appended to `eval_rows`, so `len(eval_rows)` stays at 50 for every selected MATH example. This makes row-level analysis much harder. Use JSONL line number for now.

### 6. The model sometimes repeats scaffold/prompt text

Some public multi-answer outputs include text like:

`Self-check: does your answer satisfy ALL parts of the question?`

and

`After </think>, write exactly ONE ...`

after the first `</think>`, sometimes followed by a second `</think>`. That suggests the model has learned to echo formatting instructions instead of simply following them.

### 7. Real math/reasoning errors still exist

Not everything is formatting. Some formatted outputs are simply wrong.

Example: row 42, id `600`, gold `G`, marked incorrect but format OK.

The model boxes `J` for an algorithmic sequence MCQ while the gold is `G`. The explanation is hand-wavy near the end: it says manual computation is complex and guesses the intended option.

Example: row 104, MATH, gold `4`, marked incorrect but format OK.

The model concludes the equation is not well-posed and boxes `0`, but the target answer is `4`.

## Interpretation

The reported 18% score is a pessimistic and noisy estimate of model ability. The main bottleneck is output discipline:

- Too many responses are verbose and unfinished.
- The model often delays the final answer until after a long chain of checks.
- The evaluator requires exactly one boxed expression after `</think>`, but most responses never satisfy that contract.

There is also a real evaluator hygiene issue:

- MATH answer extraction breaks on nested boxed LaTeX.
- MATH IDs are not unique.

Before treating Stage 0 as "bad at math", fix the eval plumbing and force shorter, earlier finalization. A cleaner re-run would probably raise the measured score even without additional training.

## Recommended Next Fixes

1. Fix `extract_all_boxed()` to handle nested braces, then rebuild `heldout_eval_set.jsonl`.
2. Generate unique MATH IDs, for example using the source index or a running MATH counter.
3. For eval, remove the "verify EVERY option" instruction or make it conditional/brief.
4. Add a hard answer-first format reminder near generation: after reasoning, exactly `</think>\n\n\boxed{...}` and nothing else.
5. Consider lowering verbosity during eval, or train on shorter traces where the final answer appears reliably before the token budget.
6. For public multi-answer rows, add targeted examples with exactly one comma-separated box and no labels inside the final box.

## Follow-Up Fixes Applied

The notebook has now been updated so future evals use a concise `EVAL_SYSTEM_PROMPT`, `EVAL_MAX_NEW_TOKENS = 3072`, robust nested-brace `\boxed{...}` extraction, and unique MATH IDs.

Important: `stage0_eval_results.jsonl` is still the old completed run. Re-run `evaluate_model(...)` to produce a fresh results file under the fixed prompt and rebuilt heldout set.
