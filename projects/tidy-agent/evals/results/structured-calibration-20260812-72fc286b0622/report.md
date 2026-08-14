# E0/E1/E2/E3 abstention calibration — structured-calibration-20260812-72fc286b0622

> **Development-calibration evidence, not Holdout validation.** The consumed
> 41-file Holdout was not used to produce, tune, or select any result in this
> document. This document describes what was observed on a 47-file
> post-holdout Development fixture; it is not an external-validation claim.

- Commit: `72fc286b06225f5d6978f8515a188ef1a485c65f`
- Model: `ollama_chat/qwen3.5:4b` digest `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd`
- Repetitions: 5 per condition

## Methodology note — 47 unique files, not 235 independent samples

The calibration fixture contains **47 unique Development files**. Each of
E0/E1/E2/E3 was run against the identical fixture **5 times at
temperature=0**, and every repetition reproduced the exact same per-file
decision (verified: zero files changed category, or classify/review status,
across repetitions, for any condition). The primary evaluation unit below is
therefore the **47 unique files**, not the 235 (47 × 5) repeated scored
decisions recorded per condition. The 235-decision figure remains meaningful
for request counts, token accounting, latency aggregation, and as
determinism/stability evidence — it must not be read as 235 independent test
examples or as a larger effective sample size than the fixture provides.

### Unique-file-level outcomes (n = 47: 30 real-category, 17 ground-truth `_ToReview`)

| Condition | Correct automatic | Incorrect automatic | Review | Coverage |
|---|---:|---:|---:|---:|
| E0 | 26 | 14 | 7 | 85.1% |
| E1 | 22 | 4 | 21 | 55.3% |
| E2 | 23 | 4 | 20 | 57.4% |
| E3 | 21 | 0 | 26 | 44.7% |

For E3 specifically, of the 47 unique files:
- **17/17** ground-truth `_ToReview` files were reviewed (0 incorrectly automated).
- **21/30** real-category files were correctly automated.
- **9/30** real-category files were unnecessarily reviewed (safe, but a coverage cost).
- **0/47** files received an incorrect automatic classification.

## Main results (aggregated over the 235 repeated, deterministic decisions per condition)

| Condition | Accuracy | Unsafe automation | Automation coverage | Review rate | Accuracy decided | Review recall | Review precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| E0 — two-pass metadata control (production baseline) | 66.0% | 29.8% | 85.1% | 14.9% | 65.0% | 29.4% | 71.4% |
| E1 — order-perturbed disagreement abstention | 74.5% | 8.5% | 55.3% | 44.7% | 84.6% | 76.5% | 61.9% |
| E2 — explicit structured abstention (decision + nullable category) | 80.9% | 8.5% | 57.4% | 42.6% | 85.2% | 88.2% | 75.0% |
| E3 — explicit abstention + order-perturbed agreement gate | 80.9% | 0.0% | 44.7% | 55.3% | 100.0% | 100.0% | 65.4% |

E1's second pass and E3's second pass are each an **order-perturbed second
pass**: the identical model, weights, prompt wording, and schema as pass 1,
with only the source list order reversed. They are not an "independent
second opinion" — the two passes are correlated, not statistically
independent, because nothing about the model or task differs between them
except input order.

## E3 mechanism: an abstention-consistency gate, not an ensemble

E3's safety benefit did **not** come from resolving category-vs-category
disagreement. Across all 235 repeated decisions, the two order-perturbed
passes never independently proposed two different valid categories for the
same file:

| Pass 1 / Pass 2 outcome | Count (of 235) |
|---|---:|
| classify/classify, same category (agree) | 105 |
| classify/classify, different category (disagree) | **0** |
| classify / review | 30 |
| review / classify | 10 |
| review / review | 80 |
| invalid on either side | 10 |

Because the fixture and model were run deterministically (temperature=0),
these 235 repeated counts reflect only 5 identical copies of the same
47-file outcome, not 235 independent trials — read the underlying pattern as
47 unique per-file mechanics, repeated identically 5 times.

E3's entire measured safety benefit therefore came from the
**classify-vs-review decision being inconsistent under order perturbation**
(30 + 10 = 40 of 235 pass-pairs), not from the model disagreeing with itself
about *which* category applies once it decides to classify. E3 is more
accurately described as a **deterministic abstention-consistency gate**
layered on top of explicit abstention (E2) than as an independent
two-model-opinion ensemble. Production code must still handle a
category-vs-category disagreement safely (routing it to `_ToReview`), since
a 0-count in one 47-file development fixture is not a proof that this case
cannot occur.

## Cost scenarios (lower is better — these are costs to minimize; correct automatic = 0, unnecessary review = 1, incorrect automatic = 3/5/10)

| Condition | safety_heavy (10/1/0) | balanced (5/1/0) | coverage_heavy (3/1/0) |
|---|---:|---:|---:|
| E0 | 735 | 385 | 245 |
| E1 | 305 | 205 | 165 |
| E2 | 300 | 200 | 160 |
| E3 | 130 | 130 | 130 |

E3 has the lowest (best) cost under all three weighting scenarios in this
experiment, including coverage-heavy, because its cost consists entirely of
review-only penalties (130 unnecessary/necessary reviews × weight 1) with no
incorrect-automatic penalty at all.

## Stability across repetitions

All four conditions were fully deterministic: 0 files changed category, and
0 files changed classify/review status, across the 5 repetitions of any
condition.

- E0: 26 consistently correct / 14 consistently wrong / 7 consistently reviewed.
- E1: 22 consistently correct / 4 consistently wrong / 21 consistently reviewed.
- E2: 23 consistently correct / 4 consistently wrong / 20 consistently reviewed.
- E3: 21 consistently correct / 0 consistently wrong / 26 consistently reviewed.

## Candidate conclusion (Development-only; selection criteria were fixed before this run)

Priority order used for selection: (1) unsafe automation, (2) accuracy on
decided files, (3) review recall, (4) useful coverage, (5) stability, (6)
simplicity, (7) cost.

- **E0** — too aggressive for this project's safety-first objective: 29.8%
  unsafe automation is not acceptable for a file organizer that moves real
  user files.
- **E1** — a substantial improvement over E0, but dominated by E2 in this
  experiment (same unsafe-automation rate, worse review recall, twice the
  request cost).
- **E2** — a strong single-call pragmatic candidate: same unsafe-automation
  rate as E1 at roughly half the model-call cost, with better review recall.
- **E3** — the selected safety-first candidate for the next validation
  stage. On this 47-file Development calibration fixture, **E3 produced no
  observed incorrect automatic classifications** while automatically
  classifying 21/47 files. This is Development evidence from one 47-file
  fixture, not an external-validation guarantee, not a proof of 0% future
  unsafe-automation, and not evidence about content-enabled classification
  (this experiment was metadata-only throughout).

No external (Holdout) validation of any candidate has occurred. The
41-file Holdout consumed in an earlier task was not used to produce, check,
or influence any number in this report.
