# Stage 1-2 Public Eval Summary

This report summarizes the 300-question public eval runs for the base model and
the Stage 1-2 no-anchor checkpoints. Scores are strict automated eval scores
before any lenient postprocessing.

## Overall Ranking

| Rank | Model | Strict Accuracy | Correct / 300 | MCQ Accuracy | Multi-Answer Accuracy | Boxed Compliance |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Base model | 58.33% | 175 | 59.29% | 51.52% | 49.67% |
| 2 | checkpoint-1900 | 47.67% | 143 | 68.14% | 19.19% | 57.33% |
| 3 | checkpoint-1800 | 47.00% | 141 | 61.95% | 20.20% | 57.00% |
| 4 | checkpoint-1600 | 46.00% | 138 | 61.06% | 21.21% | 53.00% |
| 5 | checkpoint-1100 | 45.00% | 135 | 64.60% | 21.21% | 55.67% |
| 6 | checkpoint-1200 | 44.33% | 133 | 64.60% | 17.17% | 56.33% |
| 7 | checkpoint-1500 | 42.33% | 127 | 61.06% | 17.17% | 55.67% |
| 8 | checkpoint-1300 | 41.33% | 124 | 58.41% | 16.16% | 54.33% |

## Main Takeaways

- The base model is the strongest overall on strict accuracy: 58.33%.
- The best fine-tuned checkpoint is checkpoint-1900 at 47.67%.
- checkpoint-1900 has the best MCQ accuracy at 68.14%, beating the base model on MCQ.
- Fine-tuning appears to have hurt non-MCQ and multi-answer performance. The base model has 51.52% multi-answer accuracy, while the fine-tuned checkpoints are around 16-21%.
- The fine-tuned checkpoints have better boxed compliance than the base model, but that formatting improvement did not translate into better overall strict accuracy.

## Current Interpretation

Stage 1-2 training improved answer formatting and MCQ behavior, especially by
checkpoint-1900, but it likely over-specialized or damaged broader free-form math
reasoning. For the next stage, the priority should be improving non-MCQ reasoning
and multi-answer formatting without losing the base model's general capability.

