# E4 Precision-Refinement — Live Development Evaluation

Experiment ID: `e4-precision-development-20260812-b6095d49ee1d`
Frozen commit: `b6095d49ee1d7bf63336d413f7c2f18a191c41ce`
Model: `ollama_chat/qwen3.5:4b`, digest `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd`, Q4_K_M, temperature 0, thinking disabled, num_ctx 8192

## Critical methodological note (read first)

The frozen harness's own `evidence_strength_note` / `success_criteria_evaluation` fields in
`summary.json` compute "unique E3 automatic errors" directly from
`primary.raw_counts.incorrect_automatic`. That field is **summed across all 5 repetitions**
(925 combined observations), not deduplicated to unique files. Because every condition/fixture
was fully deterministic (zero unstable files, confirmed below), the raw count is exactly
5× the true unique-file count.

- Harness-reported (repetition-summed): combined E3 `incorrect_automatic` = 5 → harness treats
  this as "5 unique errors," marks `underpowered: false`, and passes criterion 5.
- Actual unique-file count (verified against the deterministic per-file stability breakdown):
  combined E3 unique automatic errors = **1**.

Per the frozen rule in section 21/31 ("if E3 produces fewer than 3 unique automatic errors...",
`MINIMUM_E3_ERRORS_FOR_ROBUST_VETO_ESTIMATE = 3`), 1 < 3, so the veto precision/recall estimate
in this cycle is **underpowered**, contradicting the harness's self-reported flag. This report
uses the corrected, deduplicated unique-file interpretation throughout and flags this discrepancy
explicitly rather than silently repeating the harness's own summary field. The harness code
itself was not modified.

All other harness-reported *rates* (precision, recall, coverage, accuracy-on-decided, etc.) are
ratios of two equally-inflated counts and are therefore numerically correct as reported; only the
*raw counts* (and quantities derived by multiplying a rate by a raw count) needed correction.

## A. Freeze verification

- HEAD: `b6095d49ee1d7bf63336d413f7c2f18a191c41ce` (expected, matched)
- Worktree: clean before and after
- Production files (`src/tidy/classification.py`, `src/tidy/cli.py`, `config/rules.yaml`): no local modifications, before or after
- Fixture manifests: `_verify_dataset_manifest` passed for all three fixtures (harness would have aborted otherwise); fixture file counts independently confirmed: calibration=47, boundary_calibration=66, veto_precision_calibration=72, total=185
- Model digest before: `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd` (verified via `ollama show` and via the configured remote endpoint's `/api/tags`)
- Model digest after: identical (manifest.json `model_identity_check: "unchanged"`)

## B. Execution

- Repetitions: 5, all three fixtures, counterbalanced schedule as frozen in the harness
- Total model calls: 30 (5 reps × 3 fixtures × 2 passes) + 1 discarded warmup = 31, matching the frozen call-accounting note exactly
- E4-current / E4-refined additional model calls: 0 (both derived deterministically from the shared E3 result)
- All runs status `ok`; zero aborts, zero infrastructure failures
- `complete: true` in summary.json

## C. Main metrics (unique-file primary; total_files_scored / raw_counts corrected ÷5 from harness output)

### calibration (47 files)

| Candidate | correct auto | incorrect auto | review | strict acc. | unsafe auto. | coverage | review rate | acc. on decided | review recall | review precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E3 | 21 | 0 | 26 | 0.809 | 0.000 | 0.447 | 0.553 | 1.000 | 1.000 | 0.654 |
| E4-current | 20 | 0 | 27 | 0.787 | 0.000 | 0.426 | 0.574 | 1.000 | 1.000 | 0.630 |
| E4-refined | 20 | 0 | 27 | 0.787 | 0.000 | 0.426 | 0.574 | 1.000 | 1.000 | 0.630 |

### boundary_calibration (66 files)

| Candidate | correct auto | incorrect auto | review | strict acc. | unsafe auto. | coverage | review rate | acc. on decided | review recall | review precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E3 | 15 | 1 | 50 | 0.606 | 0.0152 | 0.242 | 0.758 | 0.938 | 0.962 | 0.500 |
| E4-current | 13 | 0 | 53 | 0.591 | 0.000 | 0.197 | 0.803 | 1.000 | 1.000 | 0.491 |
| E4-refined | 14 | 0 | 52 | 0.606 | 0.000 | 0.212 | 0.788 | 1.000 | 1.000 | 0.500 |

### veto_precision_calibration (72 files)

| Candidate | correct auto | incorrect auto | review | strict acc. | unsafe auto. | coverage | review rate | acc. on decided | review recall | review precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E3 | 33 | 0 | 39 | 0.792 | 0.000 | 0.458 | 0.542 | 1.000 | 1.000 | 0.615 |
| E4-current | 32 | 0 | 40 | 0.778 | 0.000 | 0.444 | 0.556 | 1.000 | 1.000 | 0.600 |
| E4-refined | 33 | 0 | 39 | 0.792 | 0.000 | 0.458 | 0.542 | 1.000 | 1.000 | 0.615 |

### Combined (185 files)

| Candidate | correct auto | incorrect auto | review | strict acc. | unsafe auto. | coverage | review rate | acc. on decided | review recall | review precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E3 | 69 | 1 | 115 | 0.730 | 0.0054 | 0.378 | 0.622 | 0.986 | 0.985 | 0.574 |
| E4-current | 65 | 0 | 120 | 0.714 | 0.000 | 0.351 | 0.649 | 1.000 | 1.000 | 0.558 |
| E4-refined | 67 | 0 | 118 | 0.724 | 0.000 | 0.362 | 0.638 | 1.000 | 1.000 | 0.568 |

Real-category subset and ground-truth-review subset (combined, unique-file):

| Candidate | correct auto (real-cat) | wrong auto (real-cat) | false review (real-cat) | correctly reviewed (GT-review) | incorrectly automated (GT-review) |
|---|---:|---:|---:|---:|---:|
| E3 | 69 | 0 | 49 | 66 | 1 |
| E4-current | 65 | 0 | 53 | 67 | 0 |
| E4-refined | 67 | 0 | 51 | 67 | 0 |

(Real-category n=118, GT-review n=67, combined across the three fixtures.)

## D. E3 automatic-error evidence

- Unique E3 automatic errors: **1** combined (0 in calibration, 1 in boundary_calibration, 0 in veto_precision_calibration)
- Evidence-strength threshold (≥3 unique errors): **NOT met** — 1 < 3
- Per the frozen rule: **"insufficient correlated-error events to estimate veto precision/recall robustly."**
- classify/classify-same branch (combined, unique): n=70, correct=69, incorrect=1, error rate=1.4%

E3 gate outcomes (combined, unique-file):

| Bucket | Count |
|---|---:|
| classify/classify same | 70 |
| classify/classify different | 15 |
| classify/review | 16 |
| review/classify | 12 |
| review/review | 69 |
| invalid | 3 |

## E. E4-current veto analysis (unique-file, corrected)

| Fixture | E3 candidates | accepted | vetoed | TP veto | FP veto | E3 errors surviving | veto precision | veto recall | FP/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| calibration | 21 | 20 | 1 | 0 | 1 | 0 | 0.000 | n/a (0 errors) | ∞ |
| boundary_calibration | 16 | 13 | 3 | 1 | 2 | 0 | 0.333 | 1.000 | 2.0 |
| veto_precision_calibration | 33 | 32 | 1 | 0 | 1 | 0 | 0.000 | n/a (0 errors) | ∞ |
| **Combined** | **70** | **65** | **5** | **1** | **4** | **0** | **0.200** | **1.000** | **4.0** |

Reason codes (combined, unique, deduped): `MULTI_CATEGORY_STRONG_CUES`=4, `AMBIGUITY_MARKER_WITH_CLAIM`=1, `NO_CONFLICT`=65 (no veto).

## F. E4-refined veto analysis (unique-file, corrected)

| Fixture | E3 candidates | accepted | vetoed | TP veto | FP veto | E3 errors surviving | veto precision | veto recall | FP/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| calibration | 21 | 20 | 1 | 0 | 1 | 0 | 0.000 | n/a (0 errors) | ∞ |
| boundary_calibration | 16 | 14 | 2 | 1 | 1 | 0 | 0.500 | 1.000 | 1.0 |
| veto_precision_calibration | 33 | 33 | 0 | 0 | 0 | 0 | n/a (0 vetoes) | n/a (0 errors) | n/a |
| **Combined** | **70** | **67** | **3** | **1** | **2** | **0** | **0.333** | **1.000** | **2.0** |

Reason codes across all 70 candidates (combined, unique, deduped): `NO_CONFLICT`=64, `MULTI_CUE_SOFT_CONFLICT`=3 (soft signal present but not blocking — all 3 remained accepted, not vetoed), `PREDICTED_CATEGORY_UNSUPPORTED`=2 (hard), `EXPLICIT_CATEGORY_AMBIGUITY`=1 (hard).

Verified directly against per-file detail: all 3 unique vetoes are hard-tier. boundary_calibration: `installation_handbuch_drucker_serie` (FP, `PREDICTED_CATEGORY_UNSUPPORTED`, hard) and `praesentations_folien_oder_bild_serie` (the single combined true-positive veto — E3 predicted `Images`, ground truth `_ToReview` — reason `EXPLICIT_CATEGORY_AMBIGUITY`, hard). calibration: one further `PREDICTED_CATEGORY_UNSUPPORTED` (FP, hard). The 3 `MULTI_CUE_SOFT_CONFLICT` occurrences are all soft-signal-only cases that stayed accepted (e.g. `treiber_versionshistorie_aenderungsprotokoll_dokument` in veto_precision_calibration — see section H), confirming soft signals alone do not trigger a veto in E4-refined, only the hard-conflict families do.

Files with no signal: 64 (of 70 candidates). Files with soft signal only (accepted, not vetoed): 3. Files with hard veto: 3 (all 3 unique vetoed cases combined).

## G. Direct E4-current vs E4-refined comparison

| Metric | calibration | boundary_calibration | veto_precision_calibration | Combined |
|---|---|---|---|---|
| TP vetoes (cur / ref) | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| FP vetoes (cur / ref) | 1 / 1 | 2 / 1 | 1 / 0 | 4 / 2 |
| Veto precision (cur / ref) | 0.0 / 0.0 | 0.333 / 0.500 | 0.0 / n/a | 0.200 / 0.333 |
| Veto recall (cur / ref) | n/a / n/a | 1.0 / 1.0 | n/a / n/a | 1.0 / 1.0 |
| FP/TP (cur / ref) | ∞ / ∞ | 2.0 / 1.0 | ∞ / n/a | 4.0 / 2.0 |
| Unsafe automation (cur / ref) | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| Automation coverage (cur / ref) | 0.426 / 0.426 | 0.197 / 0.212 | 0.444 / 0.458 | 0.351 / 0.362 |
| Accuracy on decided (cur / ref) | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 |
| Review recall (cur / ref) | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 |

E4-refined strictly dominates or ties E4-current on every fixture and every column above (fewer or equal false-positive vetoes, equal-or-higher coverage, equal-or-better veto precision), but the entire comparison rests on **1 unique true-positive-veto event** combined (the single E3 error, in boundary_calibration). See section D.

## H. Hard-negative analysis (veto_precision_calibration's 24 hard negatives)

These are real-category cases with legitimately mixed lexical cues; a precision-oriented veto should preserve all of them. E3 itself abstained on 9/24 of these (not a veto artifact); of the **15/24 that E3 got right and that were therefore veto-eligible**:

| Candidate | preserved (of 15 eligible) | falsely vetoed (of 15 eligible) | false-veto rate (eligible) | falsely vetoed (of all 24) |
|---|---:|---:|---:|---:|
| E4-current | 14 | 1 | 6.7% | 1 |
| E4-refined | 15 | 0 | 0.0% | 0 |

The single false veto (E4-current only) is `treiber_versionshistorie_aenderungsprotokoll_dokument` (ground truth Documents, E3 correct): E4-current's hard `conflict_detected` logic flagged the co-occurring "treiber"/"dokument" cue families and vetoed it (`MULTI_CATEGORY_STRONG_CUES`); E4-refined classified the same signal as a non-blocking `MULTI_CUE_SOFT_CONFLICT` (soft tier) and preserved the correct automatic decision. This is exactly the failure mode `veto_precision_calibration`'s README states the fixture was built to measure, and E4-refined resolved it in this run.

This n=15 comparison is much better powered than the n=1 combined TP-veto comparison in section D, though it is still Development (not Holdout) evidence and confined to one fixture.

## I. True-ambiguity-family analysis (veto_precision_calibration's 24 review cases)

| Family | n | E4-current correctly reviewed | E4-current incorrectly automated | E4-refined correctly reviewed | E4-refined incorrectly automated |
|---|---:|---:|---:|---:|---:|
| Explicit category-boundary ambiguity | 12 | 12 | 0 | 12 | 0 |
| Generic insufficient metadata | 6 | 6 | 0 | 6 | 0 |
| Container/content uncertainty | 6 | 6 | 0 | 6 | 0 |

All 24 cases were already routed to review by E3 itself (E3 final = `_ToReview` for every one), before either veto could act — both veto variants trivially preserve E3's own abstention here (`veto_applicable=False` for these files in both conditions). E4-refined's precision gain (section H) is not paid for by missed genuine ambiguity in this fixture, but this family also does not exercise the veto logic directly — E3's own two-pass agreement gate already handled every one of these 24 cases correctly.

## J. Stability (5 repetitions)

Zero unstable files in every condition × fixture combination (185/185 unique files fully deterministic across all 5 repetitions, all three conditions). This is runtime/protocol reproducibility evidence only, not independent semantic evidence (section 14). No category instability, no automatic/review instability observed.

| Fixture | E3 unstable | E4-current unstable | E4-refined unstable |
|---|---:|---:|---:|
| calibration | 0/47 | 0/47 | 0/47 |
| boundary_calibration | 0/66 | 0/66 | 0/66 |
| veto_precision_calibration | 0/72 | 0/72 | 0/72 |
| Combined | 0/185 | 0/185 | 0/185 |

## K. Cost scenarios (unique-file totals; lower is better)

| Fixture | Scenario | E3 | E4-current | E4-refined | Winner |
|---|---|---:|---:|---:|---|
| calibration | safety-heavy | 26 | 27 | 27 | E3 |
| calibration | balanced | 26 | 27 | 27 | E3 |
| calibration | coverage-heavy | 26 | 27 | 27 | E3 |
| boundary_calibration | safety-heavy | 60 | 53 | 52 | E4-refined |
| boundary_calibration | balanced | 55 | 53 | 52 | E4-refined |
| boundary_calibration | coverage-heavy | 53 | 53 | 52 | E4-refined |
| veto_precision_calibration | safety-heavy | 39 | 40 | 39 | E3 = E4-refined (tie) |
| veto_precision_calibration | balanced | 39 | 40 | 39 | E3 = E4-refined (tie) |
| veto_precision_calibration | coverage-heavy | 39 | 40 | 39 | E3 = E4-refined (tie) |
| **Combined** | **safety-heavy** | **125** | **120** | **118** | **E4-refined** |
| **Combined** | **balanced** | **120** | **120** | **118** | **E4-refined** |
| **Combined** | **coverage-heavy** | **118** | **120** | **118** | **E3 = E4-refined (tie)** |

E4-refined wins or ties the combined result in all three scenarios and is never worse than E4-current in any scenario/fixture. E3 wins outright only on the two fixtures with zero real E3 errors (calibration, veto_precision_calibration), where any veto can only add false-positive-review cost with nothing to catch.

## L. Protocol reliability

Across all 30 model calls (15 repetition-fixture E3 calls, each running pass1+pass2):

- Provider errors: 0
- Parse failures: 0
- Duplicate source responses: 0
- Invented category responses: 0
- Invalid decision enums: captured as the `invalid` gate bucket (3 unique files, all in calibration — see section D)

Schema-validation / fallback events (deterministic — identical every repetition, so unique-file counts below; repeated-observation totals in parentheses are 5× these):

| Fixture | schema_validation_failures | incomplete_responses | invented_source_responses | safe fallback→review |
|---|---:|---:|---:|---:|
| calibration | 2 (10) | 1 (5) | 2 (10) | 2 (10) |
| boundary_calibration | 0 | 0 | 0 | 0 |
| veto_precision_calibration | 1 (5) | 1 (5) | 1 (5) | 1 (5) |
| Combined unique | 3 | 2 | 3 | 3 |

All observed failures were routed safely to `_ToReview` via the fallback mechanism (`fallback_to_review_count`), never to an incorrect automatic decision — consistent with zero unsafe automation across every condition.

## M. Operational cost

- Total model calls: 30 (+1 discarded warmup)
- Calls per fixture: 10 each (5 reps × 2 passes)
- Calls per repetition (all 3 fixtures): 6
- E4-current additional model calls: 0
- E4-refined additional model calls: 0
- Total latency (sum of all repetition wall-times): 1123.0s (~18.7 min); telemetry-measured model latency sum: 1039.0s
- Latency per fixture: calibration 236.6s, boundary_calibration 401.7s, veto_precision_calibration 484.7s
- Input tokens: 23,040 total; completion tokens: 36,770 total; total tokens: 59,810

## N. Frozen success criteria (corrected for the unique-file evidence count)

| # | Criterion | Verdict | Supporting metric |
|---|---|---|---|
| 1 | Unsafe automation no worse than E4-current | **PASS** | Both 0.000% combined |
| 2 | Accuracy on decided no worse than E4-current | **PASS** | Both 1.000 combined |
| 3 | Review recall not worse by more than 1 unique case | **PASS** | Both 1.000 combined; delta = 0 cases |
| 4 | Coverage not more than 3pp below E4-current | **PASS** | E4-refined coverage (36.2%) is *higher* than E4-current (35.1%), delta +1.1pp |
| 5 | Veto precision materially higher (needs ≥3 unique E3 errors) | **UNDERPOWERED** | Only 1 unique E3 automatic error combined (harness's own field reported 5, which is the repetition-summed count, not unique — see the methodological note above). Nominal precision moved 0.200→0.333, but this rests on a single TP-catch event. |
| 6 | Improvement not confined to one fixture | **PARTIAL / effectively one fixture** | Precision delta is 0 in calibration (no TP or FP difference beyond identical single FP each), undefined in veto_precision_calibration (0 vetoes for E4-refined there), and the only real TP/FP movement is in boundary_calibration. The FP-reduction on hard negatives (section H) is also confined to veto_precision_calibration. |
| 7 | No catastrophic false-veto pattern | **PASS** | Max unique FP vetoes in any one fixture is 2 (E4-current, boundary_calibration); E4-refined never exceeds 1 FP in any fixture |

The harness's own `success_criteria_evaluation` field marks criterion 5 as passed and `underpowered: false`; that field is superseded by this report's analysis per the methodological note above, since it uses the repetition-inflated raw count. Criterion 6 is also weaker under a unique-file reading than the harness's raw per-fixture deltas suggest, because two of the three fixtures have zero true-positive-veto events to differentiate on.

## O. Candidate recommendation

Applying the frozen priority order (unsafe automation → accuracy on decided → review recall → coverage → false-review burden → cross-fixture performance → stability → simplicity → cost):

1. Unsafe automation: E4-current and E4-refined tie (both 0.0%, beating E3's 0.54%)
2. Accuracy on decided: E4-current and E4-refined tie (both 1.000)
3. Review recall: tie (both 1.000)
4. Coverage: E4-refined is higher (36.2% vs 35.1%) — E4-refined ahead
5. False-review burden: E4-refined has fewer combined false-positive vetoes (2 vs 4) and a better hard-negative false-veto rate (0% vs 6.7% on eligible hard negatives) — E4-refined ahead
6. Cross-fixture performance: E4-refined is never worse than E4-current on any fixture/metric observed, but the differentiating evidence is concentrated in one fixture and one E3-error event — weak positive for E4-refined, heavily caveated
7. Stability: tie (both fully deterministic)
8. Simplicity: E4-current is simpler (fewer rule families) — slight edge to E4-current
9. Latency/token cost: identical (zero additional model calls for either)

**On priorities 1–5, E4-refined is equal-or-better than E4-current everywhere measured in this cycle, with zero regressions.** The evidence is thin (n=1 true-positive-veto event, n=15 for the better-powered hard-negative comparison), but every directional signal favors E4-refined and none favor E4-current.

## P. Holdout-v3 readiness

**Corrected 2026-08-12**: this section previously selected "A, with an explicit underpowered caveat" immediately after stating "D is the technically correct default" — an internal contradiction. The frozen protocol (§27/§38) requires ALL seven success criteria to hold before treating E4-refined as clearly promising under the decision rule; criterion 5 is UNDERPOWERED (not PASS) and criterion 6 is only PARTIAL, so not all seven hold. The corrected, internally consistent selection is:

**Selected: D — more Development work required.**

**E4-refined remains the leading Development candidate.** It matched E4-current on unsafe automation (0.0%), accuracy on decided (1.000), and review recall (1.000), while having higher combined automation coverage and fewer combined false-positive vetoes, and it was never worse than E4-current on any measured Development metric. However, only 1 unique E3 automatic error occurred across the 185 Development files — below the frozen minimum of 3 — so veto precision/recall remains underpowered (criterion 5), and the improvement is not clearly demonstrated across all three fixtures (criterion 6: PARTIAL). This is not a safety guarantee and not a Holdout-ready pass: a further Development cycle that increases the frozen-fixture E3-error count (e.g., a fourth stress fixture targeting cases likely to trip E3 itself) is required before a new Holdout v3 is spent.

To be explicit about what this does and does not say: `best current Development candidate = E4-refined` and `ready for Holdout v3 = no` are two separate conclusions, not one — the second does not follow automatically from the first.

## P.1 Raw vs. corrected artifact provenance

This experiment produced two summary-level artifacts, which must not be conflated:

- **`summary.json`** is the untouched, byte-for-byte original output of the frozen harness (`evals/run_e4_precision_development.py`). It is preserved unmodified for reproducibility. It contains a known methodological defect: its `evidence_strength_note` / `success_criteria_evaluation` fields compute "unique E3 automatic errors" directly from `primary.raw_counts.incorrect_automatic`, which is summed across all 5 deterministic repetitions (925 combined observations) rather than deduplicated to unique files. This makes `summary.json` self-report `combined_e3_errors: 5`, `underpowered: false`, and criterion 5 as passed — all incorrect readings of the underlying unique-file evidence.
- **`summary_unique_files.json`** is a new, separately-persisted derived artifact that corrects only this semantic interpretation. It performs no new measurement and invokes no model inference: it divides `summary.json`'s repetition-summed raw counts by 5 (valid because every condition/fixture was verified fully deterministic — zero unstable files, see `summary.json`'s `stability` blocks) and carries all rate/ratio fields through unchanged (a ratio of two counts scaled equally by 5 is unaffected by the scaling). It supersedes `summary.json`'s evidence-strength note and success-criteria verdicts — and only those — not the underlying raw result.
- Every raw *rate* metric already reported in this document (precision, recall, coverage, accuracy-on-decided, review rate, etc.) was numerically valid in the original report and remains so: numerator and denominator were both multiplied by 5, so the ratio is unaffected by the counting defect.
- Raw *counts* reported directly from `summary.json` (e.g. "5 automatic errors," "925 total files scored") must never be read as unique-file counts. The counting defect itself will be fixed in the harness source in a separate future task; this result-correction pass deliberately leaves `evals/run_e4_precision_development.py` untouched so the exact code that produced this completed experiment is preserved alongside its output.

## Q. Safety/privacy confirmation

- Metadata-only throughout: confirmed via `SECURITY_ZERO_FIELDS` check built into the harness (`peek_requests_authorized=0`, `content_unavailable=0` for every repetition — harness raises `RuntimeError` otherwise, and none was raised)
- No `--read-contents`, `--allow-remote-content`, peek tool, or content parser was constructed (confirmed by reading the harness source; not invoked from the CLI)
- No plan execution; no filesystem mutation of fixtures (fixtures were staged to a temporary directory via `_stage_fixture`, never modified in place)
- No Holdout case-level access: `evals/holdout/` and `evals/holdout_v2/` were never opened during this cycle (confirmed via git diff — no changes, and no read tool calls were issued against those paths in this session)
- Python validated all source/category outputs via the frozen scoring functions; no arbitrary destination paths were introduced (no plan was ever executed)

## R. Git state

Generated files (all under a new, previously-nonexistent results directory):

```
evals/results/e4-precision-development-20260812-b6095d49ee1d/manifest.json
evals/results/e4-precision-development-20260812-b6095d49ee1d/summary.json
evals/results/e4-precision-development-20260812-b6095d49ee1d/summary_unique_files.json
evals/results/e4-precision-development-20260812-b6095d49ee1d/raw_runs-calibration.jsonl
evals/results/e4-precision-development-20260812-b6095d49ee1d/raw_runs-boundary_calibration.jsonl
evals/results/e4-precision-development-20260812-b6095d49ee1d/raw_runs-veto_precision_calibration.jsonl
evals/results/e4-precision-development-20260812-b6095d49ee1d/per_file_evidence.json
evals/results/e4-precision-development-20260812-b6095d49ee1d/report.md
```

`summary_unique_files.json` was added in a follow-up result-correction pass (see section P.1); all other files were generated by the original harness run and are unchanged.

Post-run integrity checks confirmed: `git status --short` shows only the new untracked directory above; `git diff --check` clean; `git diff --name-only` empty (no tracked file was modified); greps for `src/tidy/classification.py`, `src/tidy/cli.py`, `config/rules.yaml`, `evals/holdout/`, `evals/holdout_v2/` against `git diff --name-only` all returned no output.

Recommended manual git add command (not executed):

```
git add evals/results/e4-precision-development-20260812-b6095d49ee1d/
```

Recommended commit message (not executed):

```
tidy: record E4 precision-refinement development evaluation
```
