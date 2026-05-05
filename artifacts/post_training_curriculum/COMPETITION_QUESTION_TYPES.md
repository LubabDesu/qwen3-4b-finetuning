# Competition Question Types

This competition uses math questions with either an `options` field for MCQ or free-form `[ANS]` placeholders. The preferred model output format is:

```text
<think>
reasoning
</think>

\boxed{answer}
```

For multiple `[ANS]` blanks, put one answer per blank in order inside a single final box:

```text
\boxed{ans1, ans2, ans3}
```

## Answer-Format Types

- **Standard MCQ**: `options` exists. Output only the option letter, such as `\boxed{C}`.
- **Single `[ANS]` fill-in**: one blank, usually numeric or algebraic.
- **Multi-part `[ANS]` fill-in**: several blanks; answer in order with comma separation.
- **Check-all-that-apply / multi-select**: usually embedded in multi-part rows; output letters only, comma-separated.
- **Long symbolic MCQ**: 10-option calculus, algebra, recurrence, linear algebra, or expression-matching questions.

## Core Topic Families

- **Algebra and arithmetic**: fractions, percentages, ratios, equations, slopes/intercepts, sequences, arithmetic means, unit conversions.
- **Precalculus and functions**: graph translations, roots, logarithms, exponentials, polynomial behavior, trigonometry, radians/degrees.
- **Geometry and measurement**: angles of elevation, triangle/circle geometry, area/volume/perimeter, units, applied measurement.
- **Statistics and probability**: confidence intervals, hypothesis tests, Type I/II errors, sampling, regression/table interpretation, conditional probability.
- **Calculus**: derivatives, integrals, Taylor series, implicit differentiation, extrema/inflection, root-finding bounds.
- **Linear algebra**: vectors, projections, matrices, determinants, eigenvalues, rank, linear maps, projective-space-style MCQ.
- **Discrete and algorithmic math**: recurrences, asymptotic bounds, integer/prime/modular reasoning, combinatorics, sequences.

## Representative Examples

- MCQ calculus/root-finding: choose an interval satisfying derivative and second-derivative bounds.
- MCQ implicit differentiation: choose the expression for a mixed partial derivative.
- Multi-part linear equation: y-intercept, slope, and increasing/decreasing choice.
- Multi-part arithmetic: quotient/remainder word problems with several blanks.
- Single numeric fill: convert radians to degrees or compute a confidence interval.
- Statistics table/regression: multiple numeric outputs plus a final yes/no or option letter.
- Measurement units: choose units like millimeters, meters, kilometers, and centimeters for several blanks.

## Training Implications

- Preserve exact output discipline above all: one final box after `</think>`.
- Include enough multi-part `[ANS]` training so the model learns ordered comma-separated final boxes.
- Include standard MCQ, but avoid exhaustive option-by-option explanations.
- Strip internal `\boxed{...}` wrappers from solution text before wrapping the final answer.
- Avoid tutorial meta-language such as "This question examines", "Key points", "difficulty", and "In summary".
- Keep Stage 1 reasoning concise enough that the model does not drift into long classroom-solution mode.
