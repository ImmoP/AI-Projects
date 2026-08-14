# E4 Precision — LIVE Four-Fixture Development Evaluation

Experiment ID: `e3-error-four-fixture-development-20260812-739d150`
Frozen commit: `739d150488a6815f55ef00c3718ad0b4af610b9a`
Model: `ollama_chat/qwen3.5:4b`, digest `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd`, Q4_K_M, temperature 0, thinking disabled, num_ctx 8192

## Headline result (read first)

Adding `e3_error_calibration` raised the combined unique E3 automatic-error count from 1 (prior cycle) to **4** — the frozen ≥3 evidence threshold is **met for the first time**. With real statistical power behind it, the result reverses the prior (underpowered) cycles' tentative lean toward E4-refined: **E4-current now outperforms E4-refined** on unsafe automation, accuracy on decided, and veto precision, combined. The regression is concentrated entirely in `e3_error_calibration` — E4-refined's hard-negative advantage on `veto_precision_calibration` is fully preserved (0% vs E4-current's 6.7% false-veto rate on eligible hard negatives, unchanged from the prior cycle), but on the new fixture E4-refined caught **zero** of the 3 E3 automatic errors while E4-current caught 1, and E4-refined added more false-positive vetoes there (2 vs 1). This is a legitimate, adequately-powered Development finding, not a data artifact — see section D for full evidence-strength accounting and section R for the mechanical criteria evaluation.

## A. Freeze verification

- HEAD before run: `739d150488a6815f55ef00c3718ad0b4af610b9a` (expected, matched); worktree clean
- Production files (`src/tidy/classification.py`, `src/tidy/cli.py`, `config/rules.yaml`), candidate file (`evals/post_holdout_candidates.py`), harness (`evals/run_e4_precision_development.py`): no local modifications, before or after
- Fixture manifests verified via `_verify_dataset_manifest` before inference (harness would have aborted otherwise): calibration=47, boundary_calibration=66, veto_precision_calibration=72, e3_error_calibration=72, combined=257 — all matched
- Model digest before: `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd` (verified via the configured remote endpoint's `/api/tags`)
- Model digest after: identical (manifest.json `model_identity_check: "unchanged"`)

## B. Execution

- Repetitions: 5, all four fixtures, counterbalanced schedule as frozen in the harness
- Total measured model calls: 40 (4 fixtures × 5 reps × 2 E3 passes) + 1 discarded warmup = 41, exactly matching the expected accounting
- E4-current / E4-refined additional model calls: 0 (both derived deterministically from the shared E3 result per repetition)
- All runs status `ok`; zero aborts, zero infrastructure failures, `complete: true`

## C. Main unique-file metrics

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

### e3_error_calibration (72 files) — new fixture

| Candidate | correct auto | incorrect auto | review | strict acc. | unsafe auto. | coverage | review rate | acc. on decided | review recall | review precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E3 | 15 | 3 | 54 | 0.528 | 0.0417 | 0.250 | 0.750 | 0.833 | 0.958 | 0.426 |
| E4-current | 14 | 2 | 56 | 0.514 | 0.0278 | 0.222 | 0.778 | 0.875 | 0.958 | 0.411 |
| E4-refined | 13 | 3 | 56 | 0.500 | 0.0417 | 0.222 | 0.778 | 0.813 | 0.958 | 0.411 |

E4-refined shows **no improvement over raw E3** on this fixture (same unsafe automation rate, 4.17%, and *lower* accuracy-on-decided than E3 itself); E4-current is the only candidate that improves on E3 here.

### Combined (257 files)

| Candidate | correct auto | incorrect auto | review | strict acc. | unsafe auto. | coverage | review rate | acc. on decided | review recall | review precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E3 | 84 | 4 | 169 | 0.673 | 0.0156 | 0.342 | 0.658 | 0.955 | 0.978 | 0.527 |
| E4-current | 79 | 2 | 176 | 0.658 | 0.0078 | 0.315 | 0.685 | 0.975 | 0.989 | 0.511 |
| E4-refined | 80 | 3 | 174 | 0.661 | 0.0117 | 0.323 | 0.677 | 0.964 | 0.989 | 0.517 |

**E4-current has the lowest unsafe automation and the highest accuracy-on-decided of all three candidates, combined.** E4-refined sits between E3 and E4-current on both, having only partially fixed E3's errors this cycle.

## D. E3 evidence threshold

| Fixture | Unique E3 automatic errors |
|---|---:|
| calibration | 0 |
| boundary_calibration | 1 |
| veto_precision_calibration | 0 |
| e3_error_calibration | 3 |
| **Combined** | **4** |

Threshold: `MINIMUM_E3_ERRORS_FOR_ROBUST_VETO_ESTIMATE = 3`. **4 ≥ 3 → evidence threshold MET** for the first time in this project's history. `evidence_strength.underpowered = false`, `evidence_strength_note = null` (harness-computed, correctly using `unique_file_metrics`, not the repetition-summed block).

All 4 unique errors occurred in the `classify_classify_same` gate branch (E3's two order-perturbed passes agreed on the same wrong category) — combined `classify_classify_same_branch`: n=88, correct=84, incorrect=4, error rate=4.55%. Since E3's agreement gate only ever automates from this branch (every other gate outcome falls back to review by construction — confirmed: automatic decisions=88 exactly equals `classify_classify_same` n=88), **100% of this run's E3 automatic errors are the correlated-error failure mode** the frozen methodological conclusion describes ("same-category agreement across E3's two order-perturbed passes can preserve correlated semantic errors").

## E. E3 error density (unique errors / unique automatic decisions)

| Fixture | Errors | Automatic decisions | Density |
|---|---:|---:|---:|
| calibration | 0 | 21 | 0.000 |
| boundary_calibration | 1 | 16 | 0.0625 |
| veto_precision_calibration | 0 | 33 | 0.000 |
| e3_error_calibration | 3 | 18 | 0.167 |
| **Combined** | **4** | **88** | **0.0455** |

`e3_error_calibration`'s density (16.7%) is roughly 2.7× `boundary_calibration`'s (6.25%) and infinitely higher than the two zero-error fixtures — confirming the new fixture achieved its design goal of concentrating difficult E3 decisions, distinct from `unsafe_automation_rate` (which divides by all 257 files, not just the 88 automated ones).

## F. E3 correlated-error branch (classify/classify-same)

| Fixture | n | correct | incorrect | error rate |
|---|---:|---:|---:|---:|
| calibration | 21 | 21 | 0 | 0.0% |
| boundary_calibration | 16 | 15 | 1 | 6.25% |
| veto_precision_calibration | 33 | 33 | 0 | 0.0% |
| e3_error_calibration | 18 | 15 | 3 | 16.7% |
| **Combined** | **88** | **84** | **4** | **4.55%** |

This branch is the entirety of E3's automatic-decision population (see section D) — there is no separate "classify/classify-different produces a wrong automatic decision" pathway in this design.

## G. e3_error_calibration stress-family analysis

| Family | files | E3 automatic | E3 review | correct auto | incorrect auto | error density | acc. on decided |
|---|---:|---:|---:|---:|---:|---:|---:|
| subject_vs_artifact | 16 | 5 | 11 | 4 | 1 | 20.0% | 0.800 |
| tool_vs_output | 10 | 5 | 5 | 4 | 1 | 20.0% | 0.800 |
| container_lexical_trap | 8 | 2 | 6 | 2 | 0 | 0.0% | 1.000 |
| installer_driver_trap | 8 | 5 | 3 | 5 | 0 | 0.0% | 1.000 |
| media_document_trap | 6 | 0 | 6 | 0 | 0 | n/a | n/a |
| latent_dual_role | 10 | 1 | 9 | 0 | 1 | 100%* | 0.000 |
| latent_container_content | 6 | 0 | 6 | 0 | 0 | n/a | n/a |
| dominant_cue_ambiguity | 8 | 0 | 8 | 0 | 0 | n/a | n/a |

\* `latent_dual_role`'s single automatic decision was E3 confidently choosing one of the two plausible artifact types for a genuinely `_ToReview` file — this is a **failed-abstention** error (see section H), not a wrong-category classification.

**Only 3 of 8 stress families produced an E3 error this cycle: `subject_vs_artifact`, `tool_vs_output`, and `latent_dual_role`.** The other five families (`container_lexical_trap`, `installer_driver_trap`, `media_document_trap`, `latent_container_content`, `dominant_cue_ambiguity`) produced zero E3 errors — either because E3 correctly automated them, or (more often) because E3 itself abstained. This is useful evidence for a future fixture iteration: the subject/artifact and tool/output relational traps are the families actually defeating E3 in this run; the lexical-trap and latent-ambiguity families were not.

## H. Review escape vs. wrong real-category automation

Two distinct failure modes, per the frozen distinction:

**Review escape** (ground-truth-`_ToReview` file that E3 confidently automated instead of abstaining) — `e3_error_calibration`, n=24 GT-review files:

| | E3 correctly reviewed | E3 incorrectly automated | E4-current rescued | E4-refined rescued |
|---|---:|---:|---:|---:|
| | 23 | 1 | 0 | 0 |

The single review-escape case (`latent_dual_role` family) was **not rescued by either veto** — neither E4-current nor E4-refined flagged it as `veto_applicable` (E3's confident single-category output for that file carried no detectable cue conflict), so it survives as an unsafe automatic error under both veto conditions.

**Wrong real-category automation** (real-category ground truth, E3 automated the wrong category) — `e3_error_calibration` real-category subset, n=48:

| | E3 correct auto | E3 wrong-category auto | E3 review | E4-current catches | E4-refined catches |
|---|---:|---:|---:|---:|---:|
| | 15 | 2 | 31 | 1 of 2 | 0 of 2 |

The 2 wrong-category errors are the `subject_vs_artifact` and `tool_vs_output` family cases from section G. E4-current caught the `subject_vs_artifact` one; E4-refined caught neither.

## I. E4-current veto analysis (unique-file)

| Fixture | E3 candidates | accepted | vetoed | TP | FP | errors surviving | precision | recall | FP/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| calibration | 21 | 20 | 1 | 0 | 1 | 0 | 0.000 | n/a (0 errors) | ∞ |
| boundary_calibration | 16 | 13 | 3 | 1 | 2 | 0 | 0.333 | 1.000 | 2.0 |
| veto_precision_calibration | 33 | 32 | 1 | 0 | 1 | 0 | 0.000 | n/a (0 errors) | ∞ |
| e3_error_calibration | 18 | 16 | 2 | 1 | 1 | 2 | 0.500 | 0.333 | 1.0 |
| **Combined** | **88** | **81** | **7** | **2** | **5** | **2** | **0.286** | **0.500** | **2.5** |

Reason codes (combined, unique): `MULTI_CATEGORY_STRONG_CUES`=6, `AMBIGUITY_MARKER_WITH_CLAIM`=1, `NO_CONFLICT`=81 (no veto).

## J. E4-refined veto analysis (unique-file)

| Fixture | E3 candidates | accepted | vetoed | TP | FP | errors surviving | precision | recall | FP/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| calibration | 21 | 20 | 1 | 0 | 1 | 0 | 0.000 | n/a (0 errors) | ∞ |
| boundary_calibration | 16 | 14 | 2 | 1 | 1 | 0 | 0.500 | 1.000 | 1.0 |
| veto_precision_calibration | 33 | 33 | 0 | 0 | 0 | 0 | n/a (0 vetoes) | n/a (0 errors) | n/a |
| e3_error_calibration | 18 | 16 | 2 | 0 | 2 | 3 | 0.000 | 0.000 | ∞ |
| **Combined** | **88** | **83** | **5** | **1** | **4** | **3** | **0.200** | **0.250** | **4.0** |

Reason codes (combined, unique): hard — `EXPLICIT_CATEGORY_AMBIGUITY`=1, `PREDICTED_CATEGORY_UNSUPPORTED`=4 (sum=5=vetoed, all vetoes this run are hard-tier); soft-signal-only (accepted, not vetoed) — `CONTAINER_WORD_SOFT_SIGNAL`=1, `MULTI_CUE_SOFT_CONFLICT`=4; no-signal — `NO_CONFLICT`=78.

Files with no signal: 78/88. Soft-signal-only (accepted): 5/88. Hard-veto: 5/88 (all vetoed cases are hard-tier this run — no soft-tier veto occurred).

**On `e3_error_calibration` specifically, E4-refined's veto recall is 0.000 — it caught none of the 3 E3 automatic errors there**, while adding 2 unique false-positive vetoes (both hard-tier: `PREDICTED_CATEGORY_UNSUPPORTED` on a `tool_vs_output` file, and on an `installer_driver_trap` file that E3 got right).

## K. Error-family capture matrix (e3_error_calibration)

| Stress family | E3 automatic errors | E4-current caught | E4-refined caught | E4-current FP | E4-refined FP |
|---|---:|---:|---:|---:|---:|
| subject_vs_artifact | 1 | 1 | 0 | 1 | 0 |
| tool_vs_output | 1 | 0 | 0 | 0 | 1 |
| latent_dual_role | 1 | 0 | 0 | 0 | 0 |
| container_lexical_trap | 0 | — | — | 0 | 0 |
| installer_driver_trap | 0 | — | — | 0 | 1 |
| media_document_trap | 0 | — | — | 0 | 0 |
| latent_container_content | 0 | — | — | 0 | 0 |
| dominant_cue_ambiguity | 0 | — | — | 0 | 0 |

Neither veto catches the `tool_vs_output` or `latent_dual_role` errors. E4-current catches the single `subject_vs_artifact` error (at the cost of one *different* `subject_vs_artifact` false positive); E4-refined catches none of the three, and its two false positives land on families (`tool_vs_output`, `installer_driver_trap`) that produced zero or only one real error — i.e., E4-refined's soft/hard-tier calibration, tuned against `veto_precision_calibration`'s specific cue patterns, does not transfer cleanly to this fixture's relational (subject-vs-artifact, tool-vs-output) traps. **Do not infer safety from aggregate precision alone** — this table is the reason why: E4-refined's combined precision (0.2) looks comparable to E4-current's in isolation, but the capture matrix shows it is driven by catching zero new-fixture errors rather than a proportionally-scaled share of them.

## L. Direct E4-current vs E4-refined comparison

| Metric | calibration | boundary_calibration | veto_precision_calibration | e3_error_calibration | Combined |
|---|---|---|---|---|---|
| TP vetoes (cur / ref) | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 0 | 2 / 1 |
| FP vetoes (cur / ref) | 1 / 1 | 2 / 1 | 1 / 0 | 1 / 2 | 5 / 4 |
| Veto precision (cur / ref) | 0.0 / 0.0 | 0.333 / 0.500 | 0.0 / n/a | 0.500 / 0.000 | **0.286 / 0.200** |
| Veto recall (cur / ref) | n/a / n/a | 1.0 / 1.0 | n/a / n/a | 0.333 / 0.000 | **0.500 / 0.250** |
| FP/TP (cur / ref) | ∞ / ∞ | 2.0 / 1.0 | ∞ / n/a | 1.0 / ∞ | 2.5 / 4.0 |
| Unsafe automation (cur / ref) | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0278 / 0.0417 | **0.0078 / 0.0117** |
| Automation coverage (cur / ref) | 0.426 / 0.426 | 0.197 / 0.212 | 0.444 / 0.458 | 0.222 / 0.222 | 0.315 / 0.323 |
| Accuracy on decided (cur / ref) | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 0.875 / 0.813 | **0.975 / 0.964** |
| Review recall (cur / ref) | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 0.958 / 0.958 | 0.989 / 0.989 |

**E4-refined strictly dominates or ties on `boundary_calibration` and `veto_precision_calibration` (the two fixtures it was previously tuned/tested against), but E4-current strictly dominates on `e3_error_calibration` (higher TP, lower FP, higher recall, lower unsafe automation, higher accuracy-on-decided) — and that fixture's larger error count (3 vs 1 combined elsewhere) tips every bolded combined metric in E4-current's favor.** `calibration` is a tie for both (no true errors there for either to differentiate on).

## M. veto_precision_calibration hard-negative regression diagnostic

Re-run against this cycle's fresh predictions (not reused from the prior cycle) on the same 24 hard-negative cases, of which E3 itself correctly automated 15/24 (identical to the prior cycle — E3 is unmodified and fully deterministic):

| Candidate | preserved (of 15 eligible) | falsely vetoed (of 15 eligible) | false-veto rate |
|---|---:|---:|---:|
| E4-current | 14 | 1 | 6.7% |
| E4-refined | 15 | 0 | 0.0% |

**Identical to the prior cycle.** E4-refined's core design goal — avoiding false vetoes on legitimate cue-mixing filenames — is fully preserved and unaffected by the new fixture's results. The regression documented in sections I–L is confined entirely to `e3_error_calibration`'s different failure families, not a general degradation of E4-refined's precision-oriented design.

## N. Stability

Zero unstable files across every condition × fixture combination (257/257 unique files fully deterministic across all 5 repetitions, all three conditions — `unstable_files: []` and `unstable_unique_files: []` throughout `summary.json`'s `stability` blocks). This is runtime/protocol reproducibility evidence only, not independent semantic evidence — repeated identical outcomes across 5 repetitions do not add additional unique-file evidence beyond what a single repetition would show.

| Fixture | E3 unstable | E4-current unstable | E4-refined unstable |
|---|---:|---:|---:|
| calibration | 0/47 | 0/47 | 0/47 |
| boundary_calibration | 0/66 | 0/66 | 0/66 |
| veto_precision_calibration | 0/72 | 0/72 | 0/72 |
| e3_error_calibration | 0/72 | 0/72 | 0/72 |
| Combined | 0/257 | 0/257 | 0/257 |

## O. Protocol reliability

Across all 40 measured model calls (20 repetition-fixture E3 calls, each running pass1+pass2): 0 provider errors, 0 parse failures, 0 duplicate source responses, 0 invented category responses across every fixture.

Schema-validation / fallback events (deterministic — identical every repetition, so unique-file counts below; repeated-observation totals in parentheses are 5× these):

| Fixture | schema_validation_failures | incomplete_responses | invented_source_responses | safe fallback→review |
|---|---:|---:|---:|---:|
| calibration | 2 (10) | 1 (5) | 2 (10) | 2 (10) |
| boundary_calibration | 0 | 0 | 0 | 0 |
| veto_precision_calibration | 1 (5) | 1 (5) | 1 (5) | 1 (5) |
| e3_error_calibration | 0 | 0 | 0 | 0 |
| Combined unique | 3 | 2 | 3 | 3 |

Every observed failure routed safely to `_ToReview` via the fallback mechanism, never to an incorrect automatic decision. The new `e3_error_calibration` fixture triggered zero protocol-level failures — all 4 unique E3 automatic errors there are genuine semantic misclassifications, not artifacts of a parsing/schema failure.

## P. Operational cost

- Total measured model calls: 40 (+1 discarded warmup)
- Calls per fixture: 10 each (5 reps × 2 passes); calls per repetition (all 4 fixtures): 8
- E4-current additional model calls: 0; E4-refined additional model calls: 0
- Total telemetry-measured model latency: 1211.6s (~20.2 min); latency per fixture: calibration 176.4s, boundary_calibration 325.1s, veto_precision_calibration 362.1s, e3_error_calibration 348.0s
- Input tokens: 32,210 total; completion tokens: 51,575 total; total tokens: 83,785

## Q. Cost scenarios (unique-file totals; lower is better)

| Fixture | Scenario | E3 | E4-current | E4-refined | Winner |
|---|---|---:|---:|---:|---|
| calibration | all three | 26 | 27 | 27 | E3 |
| boundary_calibration | safety-heavy | 60 | 53 | 52 | E4-refined |
| boundary_calibration | balanced | 55 | 53 | 52 | E4-refined |
| boundary_calibration | coverage-heavy | 53 | 53 | 52 | E4-refined |
| veto_precision_calibration | all three | 39 | 40 | 39 | E3 = E4-refined (tie) |
| e3_error_calibration | safety-heavy | 84 | **76** | 86 | **E4-current** |
| e3_error_calibration | balanced | 69 | **66** | 71 | **E4-current** |
| e3_error_calibration | coverage-heavy | 63 | **62** | 65 | **E4-current** |
| **Combined** | **safety-heavy** | 209 | **196** | 204 | **E4-current** |
| **Combined** | **balanced** | 189 | **186** | 189 | **E4-current** |
| **Combined** | **coverage-heavy** | **181** | 182 | 183 | **E3** |

**E4-refined does not win a single combined cost scenario this cycle** (it wins only `boundary_calibration` outright, and ties `veto_precision_calibration`); on `e3_error_calibration` it is the *most expensive* candidate in every scenario, more expensive even than doing nothing (raw E3), because it adds review burden (2 extra vetoes) without catching any additional errors there.

## R. Frozen success criteria

| # | Criterion | Verdict | Supporting metric |
|---|---|---|---|
| 1 | Unsafe automation no worse than E4-current | **FAIL** | E4-refined 1.17% > E4-current 0.78% combined |
| 2 | Accuracy on decided no worse than E4-current | **FAIL** | E4-refined 0.964 < E4-current 0.975 combined |
| 3 | Review recall not worse by more than 1 unique case | **PASS** | Both 0.989 combined; `review_recall_delta_unique_cases = 0` |
| 4 | Coverage not more than 3pp below E4-current | **PASS** | E4-refined coverage (32.3%) is *higher* than E4-current (31.5%), delta −0.8pp |
| 5 | Veto precision materially higher (needs ≥3 unique E3 errors) | **FAIL** | Evidence threshold now MET (4 ≥ 3, `underpowered: false`); E4-refined precision 0.200 < E4-current 0.286 |
| 6 | Improvement not confined to one fixture | **Mixed, net unfavorable** | Per-fixture precision deltas (current−refined): boundary_calibration −0.167 (refined ahead), calibration 0.0 (tie), e3_error_calibration **+0.5** (current strongly ahead), veto_precision_calibration undefined (0 refined vetoes). Improvement is not literally confined to one fixture, but it is not net-positive either — the one large-magnitude signal (e3_error_calibration) favors E4-current. |
| 7 | No catastrophic false-veto pattern | **PASS** | Max unique FP vetoes in any one fixture is 2 (E4-current/boundary_calibration and E4-refined/e3_error_calibration); no fixture shows a clearly catastrophic pattern by itself |

**Criteria 1, 2, and 5 all FAIL.** Per the frozen rule (section 40 of the brief: "Do not invent easier criteria because the new fixture is adversarial"), E4-refined does **not** satisfy the frozen success-criteria set this cycle — this is a mechanical, adequately-powered result, not an artifact of underpowering.

## S. Candidate recommendation

Applying the frozen priority order (unsafe automation → accuracy on decided → review recall → coverage → false-review burden → cross-fixture performance → stability → simplicity → cost):

1. **Unsafe automation: E4-current wins outright** (0.78% vs E4-refined 1.17% vs E3 1.56%). This is the first, highest-priority criterion, and it already resolves the comparison in E4-current's favor over both other candidates.
2. Accuracy on decided: E4-current wins again (0.975 > E4-refined 0.964 > E3 0.955) — reinforces the same direction.
3. Review recall: E4-current and E4-refined tie (0.989), both beat E3 (0.978).
4. Coverage: E4-refined is marginally higher than E4-current (32.3% vs 31.5%), but this does not overturn criteria 1–2, which already differentiate.
5. False-review burden: E4-refined has fewer combined false-positive vetoes (4 vs 5) — a point in its favor — but combined with far fewer true-positive catches (1 vs 2), i.e. it is achieving fewer FPs partly by catching less overall, not by being more precise where it does act.
6. Cross-fixture performance: E4-current is the only candidate that improves on raw E3 on `e3_error_calibration`; E4-refined does not improve on E3 there at all (same unsafe rate, worse accuracy-on-decided than E3 itself).
7. Stability: tie (both fully deterministic).
8. Simplicity: E4-current is simpler (fewer rule families) — a secondary point in its favor, consistent with priority 1–2 already deciding this.
9. Latency/token cost: identical (zero additional model calls for either).

**On priorities 1 and 2 — the two highest-priority criteria — E4-current strictly beats E4-refined this cycle, with adequate evidence power behind the comparison (4 unique E3 errors, threshold met).** This reverses the tentative, underpowered lean toward E4-refined from all prior cycles. E4-current now also strictly beats raw E3 on priorities 1–2 (its usual role: reduce unsafe automation at some coverage cost).

**Recommended candidate this cycle: E4-current.**

## T. Holdout-v3 readiness

Per the frozen decision rule: `unique_e3_automatic_errors = 4 ≥ 3`, so the evidence threshold is met and criterion 5 is a genuine FAIL (not UNDERPOWERED). E4-refined does **not** satisfy the frozen success-criteria set (criteria 1, 2, 5 all fail) — so selection A (freeze E4-refined for Holdout v3) is **not** available under the frozen decision rule, regardless of how promising it looked in prior underpowered cycles.

**Selected: B — E4-current remains preferable.**

E4-current strictly beats both E3 and E4-refined on the two highest-priority criteria (unsafe automation, accuracy on decided) combined, with adequately-powered evidence behind the comparison for the first time. This is a reversal of the working hypothesis from the prior two (underpowered) Development cycles, driven specifically by `e3_error_calibration`'s three E3 automatic errors — the fixture built for exactly this purpose. If E4-current were to be advanced toward a future Holdout v3, that would be **candidate readiness for one-time testing on a NEW Holdout v3**, not validated production readiness — that is a separate, later decision this task does not make. No Holdout v3 was created, accessed, or referenced beyond this description.

E4-refined's continued advantage on `veto_precision_calibration`'s hard-negative pattern (section M) is real and unaffected — a future refinement of E4-refined's soft/hard-tier cue calibration to also cover the `subject_vs_artifact`/`tool_vs_output` relational traps this fixture exposed could plausibly reconcile both strengths, but that is candidate tuning and explicitly out of scope for this task.

## U. Safety/privacy confirmation

- Metadata-only throughout: confirmed via the harness's built-in `SECURITY_ZERO_FIELDS` check (`peek_requests_authorized=0`, `content_unavailable=0` for every repetition across all four fixtures — the harness raises `RuntimeError` otherwise, and none was raised)
- No `--read-contents`, `--allow-remote-content`, peek tool, or content parser was constructed or invoked
- No plan execution; no filesystem mutation of any fixture (fixtures were staged to a temporary directory, never modified in place)
- No candidate mutation: `evals/post_holdout_candidates.py` untouched (E3, E4-current, E4-refined ran exactly as frozen)
- No production mutation: `src/tidy/classification.py`, `src/tidy/cli.py`, `config/rules.yaml` untouched
- No Holdout access: `evals/holdout/` and `evals/holdout_v2/` were never opened during this cycle
- No Holdout v3 was created
- Python validated all source/category outputs via the frozen scoring functions; no arbitrary destination paths were introduced (no plan was ever executed)

## V. Git state

Generated files (all under a new, previously-nonexistent results directory):

```
evals/results/e3-error-four-fixture-development-20260812-739d150/manifest.json
evals/results/e3-error-four-fixture-development-20260812-739d150/summary.json
evals/results/e3-error-four-fixture-development-20260812-739d150/raw_runs-calibration.jsonl
evals/results/e3-error-four-fixture-development-20260812-739d150/raw_runs-boundary_calibration.jsonl
evals/results/e3-error-four-fixture-development-20260812-739d150/raw_runs-veto_precision_calibration.jsonl
evals/results/e3-error-four-fixture-development-20260812-739d150/raw_runs-e3_error_calibration.jsonl
evals/results/e3-error-four-fixture-development-20260812-739d150/per_file_evidence.json
evals/results/e3-error-four-fixture-development-20260812-739d150/report.md
```

Post-run integrity checks confirmed: `git status --short` shows only the new untracked directory above; `git diff --check` clean; `git diff --name-only` empty (no tracked file was modified); greps for `src/tidy/classification.py`, `src/tidy/cli.py`, `config/rules.yaml`, `evals/post_holdout_candidates.py`, `evals/run_e4_precision_development.py`, `evals/calibration/`, `evals/boundary_calibration/`, `evals/veto_precision_calibration/`, `evals/e3_error_calibration/`, `evals/holdout/`, `evals/holdout_v2/` against `git diff --name-only` all returned no output.

Recommended manual git add command (from the AI repository root, not executed):

```
git add \
  projects/Agents/Tidy_Agent/evals/results/e3-error-four-fixture-development-20260812-739d150/
```

Recommended commit message (not executed):

```
tidy: record four-fixture E3 error development evaluation
```
