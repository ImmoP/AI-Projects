# Evaluation

This document is the detailed evaluation history behind the compact
[Evaluation](../README.md#evaluation) section in the README. It exists so the
README can stay short while every number here stays traceable to a committed
artifact in `evals/results/`.

## Evaluation philosophy

Automation coverage (the share of files an agent decides on its own) is not
the target this project optimizes first. Tidy Agent is a filesystem tool, so
an *incorrect automatic classification* — a file silently moved to the wrong
place — is treated as materially worse than a file routed to `_ToReview/` for
a human to place. `_ToReview/` is an intentional abstention mechanism, not a
failure mode: every candidate design in this project is compared first on how
often it is automatically *wrong*, then on how much it automates, not the
other way around.

## Candidate evolution

```text
E0  two-pass metadata control (no abstention gate)
 -> E1/E2  abstention variants
 -> E3  two-pass order-perturbed agreement gate           [production default]
 -> E4-current  deterministic ambiguity/conflict veto on top of E3
 -> E4-refined  precision-tuned veto variant (not advanced)
 -> E5  classifier + verifier role-separation (not advanced)
```

`E3` is the conservative two-pass agreement classifier: a file automates only
when two structurally identical classification passes, with source order
reversed on the second pass, independently return the same category. It is
the only candidate wired into production (`StructuredClassifier.classify_with_agreement_gate`
in `src/tidy/classification.py`, invoked from `build_combined_plan`).

`E4-current` starts from `E3`'s automatic decisions only and layers a small,
dataset-independent, deterministic filename-cue veto that can still redirect
an agreed decision to `_ToReview/` if it detects a cross-category conflict
signal. It adds no additional model request. `E4-current` was selected over
`E4-refined` and `E5` as the sole **Development** candidate for the final,
independent Holdout — see "Independent final Holdout v4" below. **Selecting
`E4-current` for the Holdout did not make it the production default.**
Production classification is unmodified `E3`; `E4-current` exists as an
evaluation candidate layered in `evals/post_holdout_candidates.py`, not in
`src/tidy/classification.py`.

`E5` (role-separated classifier + verifier) was measured once against
Development fixtures and was dominated by `E3`/`E4-current` on every
top-priority criterion (unsafe automation, then accuracy on decided, then
review recall). It was not advanced to any Holdout.

## Development evidence

All numbers below are **Development** — used to choose between candidate
designs, never combined with a Holdout result, and reproducible from the
linked report in `evals/results/`.

| Cycle | Candidates | Fixture(s) | Headline result | Report |
|---|---|---|---|---|
| A/B/C/D | metadata / grouping / content axes | 76-file fixture, 5 warm reps | A (metadata-only) stays the simplest default; content axis gain not established (peeked files were empty) | [`structured-abcd-20260811-25b9ba1e89a1`](../evals/results/structured-abcd-20260811-25b9ba1e89a1/report.md) |
| A/E/C | one-pass vs. two-pass control vs. content | 76-file fixture, 5 warm reps | Two-pass control (`E`) +4.7 pts over one-pass; `E` preselected for Holdout on safety grounds | [`structured-aec-20260811-fc9d5fe9654d`](../evals/results/structured-aec-20260811-fc9d5fe9654d/report.md) |
| E0–E3 abstention calibration | `E0`, `E1`, `E2`, `E3` | 47-file `evals/calibration` | `E3`: 0/47 incorrect automatic classifications (vs. 14/47 for `E0`) at 21/47 automated; `E3` selected as production default | [`structured-calibration-20260812-72fc286b0622`](../evals/results/structured-calibration-20260812-72fc286b0622/report.md) |
| E3/E4/E5 post-Holdout-v2 | `E3`, `E4`, `E5` | `calibration` + `boundary_calibration` (113 files) | `E4`: 0.0% unsafe automation, 29.2% coverage, 100% accuracy on decided; recommended over `E3`, but on thin evidence (1 true-positive veto vs. 3 false positives) | [`post-holdout-development-20260812-aa84fa7e1e0a`](../evals/results/post-holdout-development-20260812-aa84fa7e1e0a/report.md) |
| E4 precision refinement | `E3`, `E4-current`, `E4-refined` | 4 fixtures combined (257 files) | 4 unique `E3` automatic errors observed (meets the frozen minimum of 3); `E4-current` beat `E4-refined` on the frozen priority order and was selected for Holdout v3/v4 | [`e4-precision-development-20260812-b6095d49ee1d`](../evals/results/e4-precision-development-20260812-b6095d49ee1d/report.md) |

Two earlier one-time Holdouts also exist in the repository's history and are
documented in full in the README's evaluation narrative: a 41-file Holdout
against candidate `E` (`structured-e-holdout-20260811-fc9d5fe9654d`, strict
accuracy 65.9%, motivating the `E0`–`E3` abstention cycle above), and a
120-file Holdout v3 against `E3`+`E4-current` that was interrupted and
inconclusive before this project moved to the final Holdout v4 protocol
below. Neither is combined with any Development number, and neither is
rerun.

## Independent final Holdout v4

Holdout v4 is the final, one-time evaluation for this project, governed by
[`evals/final_portfolio_holdout_protocol.md`](../evals/final_portfolio_holdout_protocol.md):

- **150 frozen cases**, authored clean-room with no access to any prior
  Holdout or Development fixture at the case level
  (`evals/holdout_v4/AUTHORING_FROZEN.json`).
- Accepted under a frozen rule — zero exact historical filename overlap —
  documented in the protocol; near-lexical similarity is reported as
  diagnostic only and never gates acceptance.
- **Frozen candidate**: `E3 -> E4-current`, selected before authoring, with no
  post-Holdout tuning permitted (`evals/holdout_v4/candidate_selection.json`).
- **Consumed exactly once.** `evals/holdout_v4/CONSUMED.json` records the run
  id, pipeline, fixture hash, and timestamp of the single permitted
  execution. There is no Holdout v5 and no rerun of v4.

### Result

```text
evaluation_status = partial_inconclusive
```

From the authoritative
[`evaluation_status.json`](../evals/results/holdout-v4-e4-20260813-39423af/evaluation_status.json):

| Field | Value |
|---|---|
| `provider_errors` | 0 |
| `parse_failures` | 0 |
| `schema_failures` | 1 |
| `all_required_provider_responses_received` | `true` |
| `E3_gate_completed` | `true` |
| `E4_current_completed` | `true` |
| `result_persistence_completed` | `true` |
| `evaluation_valid` | **`false`** |
| `evaluation_status` | **`partial_inconclusive`** |

### Root cause

Both provider requests completed (`dispatch.attempted = 2`,
`dispatch.returned = 2`, `dispatch.failed = 0`), and there were zero provider
errors and zero JSON parse failures. However, one of the two responses failed
structured-output schema validation. The associated telemetry in
[`summary.json`](../evals/results/holdout-v4-e4-20260813-39423af/summary.json)
records:

```text
schema_validation_failures = 1
incomplete_responses        = 1
invented_source_responses   = 1
fallback_to_review_count    = 1
```

That is: one response was incomplete and invented a source filename that was
not in the request, which trips the classifier's structured-output
validation and forces a fallback to `_ToReview/` for the affected case(s).
This is a **schema-contract violation on one of two required responses**, not
a filesystem-safety failure, a crash, or a lost result — the run still
persisted successfully and both `E3` and `E4-current` completed. The
precommitted validity rule for this protocol treats any schema-validation
failure on a required response as sufficient to invalidate the evaluation as
a whole, independent of how the rest of the run behaved.

### Why no rerun

The protocol governs "exactly one live evaluation after Git freeze" with "no
post-Holdout tuning." Holdout v4 became permanently consumed at its first
measured model request, before its outcome was known. Rerunning it, editing
its ground truth, retrying only the failed request, or substituting a
different candidate against the same fixture would all violate the one-time
protocol this project explicitly adopted after three earlier Holdout v4
authoring attempts were rejected for unrelated reasons (see the protocol
document, section 1). A materially different candidate would require an
entirely new, independently constructed Holdout — there isn't one, and none
is planned as part of this portfolio's scope.

### What can and cannot be concluded

**Can conclude:**

- The infrastructure reached the provider and completed both required
  requests with zero provider errors and zero parse failures.
- `E3` gate evaluation and `E4-current` veto evaluation both completed.
- Results were persisted successfully and are fully inspectable.
- The evaluation harness's fail-closed validity rule worked as designed: it
  distinguished "the process completed" from "the scientific evaluation is
  valid," and refused to promote diagnostic counts to a benchmark result
  after detecting a schema-contract violation.

**Cannot conclude:**

- A valid final accuracy figure for `E3`+`E4-current`.
- A valid final unsafe-automation rate for `E3`+`E4-current`.
- A valid generalization estimate from this Holdout to unseen data.

### On the diagnostic counts

The run's scoring block contains raw counts — expected_source_count=150, auto_count=44, review_count=106, matches_ground_truth_count=96 — because scoring still executed against the run output. These are diagnostic artifacts of an invalidated evaluation, not valid accuracy or generalization estimates. They are retained only as forensic run artifacts and are not used as Holdout quality metrics.

## Production status

Production classification is, and remains, unmodified `E3`
(`src/tidy/classification.py`). `E4-current` was never integrated into
production and the inconclusive Holdout v4 result does not change that.
Development selection is closed; evaluation is closed; there is no planned
Holdout v5.
