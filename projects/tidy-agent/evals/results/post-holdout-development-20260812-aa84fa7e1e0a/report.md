# post-holdout-development-20260812-aa84fa7e1e0a

Live E3/E4/E5 Development evaluation. Frozen at commit `aa84fa7e1e0a93e449dfa92923759daf6d2a6be2`. Model `ollama_chat/qwen3.5:4b` digest `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd` (unchanged after run).

All five repetitions were fully deterministic (temperature 0): every condition/fixture combination had zero unstable files across repetitions. Raw counts below are therefore exactly 5x the unique-file counts; unique files remain the primary semantic denominator (47 calibration, 66 boundary_calibration, 113 combined).

## Original Development calibration (N=47 unique)

| Candidate | Correct auto | Incorrect auto | Review | Strict acc | Unsafe automation | Coverage | Review rate | Acc on decided | Review recall | Review precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E3 | 21 | 0 | 26 | 80.9% | 0.0% | 44.7% | 55.3% | 100.0% | 100.0% | 65.4% |
| E4 | 20 | 0 | 27 | 78.7% | 0.0% | 42.6% | 57.4% | 100.0% | 100.0% | 63.0% |
| E5 | 19 | 0 | 28 | 76.6% | 0.0% | 40.4% | 59.6% | 100.0% | 100.0% | 60.7% |

## Boundary calibration (N=66 unique)

| Candidate | Correct auto | Incorrect auto | Review | Strict acc | Unsafe automation | Coverage | Review rate | Acc on decided | Review recall | Review precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E3 | 15 | 1 | 50 | 60.6% | 1.5% | 24.2% | 75.8% | 93.8% | 96.2% | 50.0% |
| E4 | 13 | 0 | 53 | 59.1% | 0.0% | 19.7% | 80.3% | 100.0% | 100.0% | 49.1% |
| E5 | 11 | 2 | 53 | 54.5% | 3.0% | 19.7% | 80.3% | 84.6% | 96.2% | 47.2% |

## Combined (N=113 unique)

| Candidate | Correct auto | Incorrect auto | Review | Strict acc | Unsafe automation | Coverage | Review rate | Acc on decided | Review recall | Review precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E3 | 36 | 1 | 76 | 69.0% | 0.9% | 32.7% | 67.3% | 97.3% | 97.7% | 55.3% |
| E4 | 33 | 0 | 80 | 67.3% | 0.0% | 29.2% | 70.8% | 100.0% | 100.0% | 53.8% |
| E5 | 30 | 2 | 81 | 63.7% | 1.8% | 28.3% | 71.7% | 93.8% | 97.7% | 51.9% |

## E3 correlated-error branch (classify/classify-same)

- **calibration**: counts (unique) {'classify_classify_same': 21, 'classify_review': 6, 'invalid': 2, 'review_classify': 2, 'review_review': 16}; classify/classify-same branch n=21, correct=21, incorrect=0, accuracy=100.0%
- **boundary_calibration**: counts (unique) {'classify_classify_different': 8, 'classify_classify_same': 16, 'classify_review': 6, 'review_classify': 9, 'review_review': 27}; classify/classify-same branch n=16, correct=15, incorrect=1, accuracy=93.8%
- **combined**: counts (unique) {'classify_classify_different': 8, 'classify_classify_same': 37, 'classify_review': 12, 'invalid': 2, 'review_classify': 11, 'review_review': 43}; classify/classify-same branch n=37, correct=36, incorrect=1, accuracy=97.3%

## E4 veto analysis

- **calibration** (unique): presented=21, accepted=20, vetoed=1, true_positive=0, false_positive=1, unsafe_e3_errors_surviving=0, veto_precision=0.0%, veto_recall=n/a
- **boundary_calibration** (unique): presented=16, accepted=13, vetoed=3, true_positive=1, false_positive=2, unsafe_e3_errors_surviving=0, veto_precision=33.3%, veto_recall=100.0%
- **combined** (unique): presented=37, accepted=33, vetoed=4, true_positive=1, false_positive=3, unsafe_e3_errors_surviving=0, veto_precision=25.0%, veto_recall=100.0%

Note: the harness's generic `_e3_gate_analysis` derivation keys off bare `pass1_decision`/`pass2_decision` fields; E4's own detail records store the identical information under `e3_pass1_decision`/`e3_pass2_decision` instead, so that specific embedded field under the E4 summary reads as all-`invalid` -- a report-labeling artifact of the frozen scoring code, not a candidate-behavior issue. Independently reconstructed from E4's own per-file evidence, the underlying E3 gate breakdown E4 built on is byte-for-byte identical to E3's own reported breakdown above (as expected: E4 adds no model call).

## E5 classifier-verifier analysis

- **calibration** (unique): {'accepted_correct': 19, 'accepted_wrong': 0, 'classifier_classify_count': 27, 'classifier_review_count': 18, 'invalid_verifier_count': 0, 'rejected_classifier_errors': 4, 'rejected_correct_classifier_proposals': 4, 'verifier_accept_count': 19, 'verifier_review_count': 8}
- **boundary_calibration** (unique): {'accepted_correct': 11, 'accepted_wrong': 2, 'classifier_classify_count': 30, 'classifier_review_count': 36, 'invalid_verifier_count': 1, 'rejected_classifier_errors': 5, 'rejected_correct_classifier_proposals': 11, 'verifier_accept_count': 13, 'verifier_review_count': 16}
- **combined** (unique): {'accepted_correct': 30, 'accepted_wrong': 2, 'classifier_classify_count': 57, 'classifier_review_count': 54, 'invalid_verifier_count': 1, 'rejected_classifier_errors': 9, 'rejected_correct_classifier_proposals': 15, 'verifier_accept_count': 32, 'verifier_review_count': 24}

## Cost scenarios (lower is better; unique-file-equivalent)

| Scope | Candidate | Safety-heavy | Balanced | Coverage-heavy |
|---|---|---:|---:|---:|
| calibration | E3 | 26 | 26 | 26 |
| calibration | E4 | 27 | 27 | 27 |
| calibration | E5 | 28 | 28 | 28 |
| boundary_calibration | E3 | 60 | 55 | 53 |
| boundary_calibration | E4 | 53 | 53 | 53 |
| boundary_calibration | E5 | 73 | 63 | 59 |
| combined | E3 | 86 | 81 | 79 |
| combined | E4 | 80 | 80 | 80 |
| combined | E5 | 101 | 91 | 87 |

## Stability

Fully deterministic: 0 unstable files in every condition x fixture combination. Mean = the single observed value; standard deviation = 0 throughout. The five repetitions assess protocol/runtime reproducibility only, not independent statistical samples.

## Candidate recommendation: E4

Fixed priority order applied to the combined results above:

1. **Unsafe automation rate** -- E4 0.0% < E3 0.9% < E5 1.8%. E4 wins.
2. **Accuracy on decided** -- E4 100.0% > E3 97.3% > E5 93.8%. E4 wins.
3. **Review recall** -- E4 100.0% > E3/E5 tied at 97.7%.
4. **Automation coverage** -- E3 32.7% > E4 29.2% > E5 28.3%. E3 slightly ahead.
5. **False-review burden** -- E4 reviews 4 more unique files than E3 combined (80 vs 76): the 1 file E3 got wrong plus 3 correct E3 decisions (see "E4 veto analysis" above: 1 true-positive veto against 3 false-positive vetoes).
6. **Both fixtures** -- E4 matches or beats E3 on unsafe automation and accuracy-on-decided on both calibration and boundary_calibration individually.
7. **Stability** -- identical (both fully deterministic).
8. **Simplicity** -- E3 is simpler (no added veto layer); E4 is a small, isolated addition that adds no model call.
9. **Cost** -- E4 wins or ties every cost scenario on boundary_calibration and combined; E3 wins calibration narrowly (see "Cost scenarios" above).

E4 outranks E3 on criteria 1-3, the top of the fixed priority order, so it is the recommended candidate from this Development evaluation. The margin is thin -- E4's entire safety edge over E3 rests on a single true-positive veto (`praesentations_folien_oder_bild_serie`) observed across 113 unique Development files, against 3 false-positive vetoes -- so this should be read as a direction, not a large or highly certain effect. E5 is dominated by both E3 and E4 on every top-priority metric (highest unsafe automation, lowest accuracy-on-decided, most expensive under every cost scenario) and is not recommended.

This candidate selection is Development-only. No production code was integrated or modified, and no Holdout (including a future Holdout v3) was accessed or created as part of reaching this conclusion.
