# Structured A/B/C/D development evaluation — structured-abcd-20260811-25b9ba1e89a1

> This is the development benchmark, not the locked holdout.

## Experiment provenance

- Git commit: `25b9ba1e89a190401e65125b5512c68aac43fdb8` (clean at experiment start)
- Model: `ollama_chat/qwen3.5:4b`
- Digest: `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd`
- Runtime: Ollama via LiteLLM 0.32.1
- Endpoint: remote endpoint (address redacted)
- Structured output: `json_object`
- Repetitions: 5 per condition; warm model after one discarded warmup

## Dataset

- Files: 76
- Fixture manifest SHA-256: `68e48034af9408e582cc8bea269ae2f052d35cb00669d736b1f2534ddc97dd18`
- Ground-truth SHA-256: `dfe0b35b34c686e98b8f0fd8bcea4f644609a985c4dd2302e0151d7445fc16ae`
- Deterministic rules remained enabled in every condition.

## A/B/C/D results

Category accuracy is calculated over files receiving category decisions; grouped files are excluded and reported through grouping metrics.

| Condition | Category accuracy | Decision rate | Accuracy decided | Review rate | Incorrect decision rate | Total latency | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| A — Metadata only | 92.1 ± 0.0% | 77.4 ± 0.0% | 75.0 ± 0.0% | 9.2 ± 0.0% | 7.9 ± 0.0% | 22.2 ± 4.7 s | 649.0 ± 0.0 |
| B — Metadata + Grouping | 90.8 ± 0.0% | 83.3 ± 0.0% | 76.0 ± 0.0% | 7.7 ± 0.0% | 9.2 ± 0.0% | 35.3 ± 3.3 s | 4167.0 ± 0.0 |
| C — Metadata + Content | 94.7 ± 0.0% | 77.4 ± 0.0% | 83.3 ± 0.0% | 9.2 ± 0.0% | 5.3 ± 0.0% | 21.3 ± 2.2 s | 1134.0 ± 0.0 |
| D — Metadata + Grouping + Content | 93.8 ± 0.0% | 80.0 ± 0.0% | 83.3 ± 0.0% | 9.2 ± 0.0% | 6.2 ± 0.0% | 33.3 ± 1.7 s | 4571.0 ± 0.0 |

### Coverage by mechanism

| Condition | Rules | Classifier | Content-peeked | Grouped | _ToReview |
|---|---:|---:|---:|---:|---:|
| A | 45.0 | 31.0 | 0.0 | 0.0 | 7.0 |
| B | 35.0 | 30.0 | 0.0 | 11.0 | 5.0 |
| C | 45.0 | 31.0 | 3.0 | 0.0 | 7.0 |
| D | 35.0 | 30.0 | 2.0 | 11.0 | 6.0 |

## Content analysis

- A vs C paired transitions: review → correct: 5, unchanged: 360, wrong → correct: 5, wrong → review: 5, wrong → wrong: 5.
- B vs D paired transitions: unchanged: 360, wrong → correct: 5, wrong → review: 5, wrong → wrong: 10.
- C: 15/20 actual peeks (75.0% utilization), 15 readable, 0 unavailable, 0 parser errors and 0 timeouts.
- C token split: peek phase 1210 input / 200 output; final phase 2415 input / 1845 output.
- D: 10/20 actual peeks (50.0% utilization), 10 readable, 0 unavailable, 0 parser errors and 0 timeouts.
- D token split: peek phase 1155 input / 150 output; final phase 2125 input / 1675 output.

## Grouping analysis

- B: groups proposed [3, 3, 3, 3, 3]; accepted [3, 3, 3, 3, 3]; grouped files [11, 11, 11, 11, 11]; purity 100.0% ± 0.0%; cohesion 33.3%; scatter harms [0, 0, 0, 0, 0]; rule/model destinations overridden [10, 10, 10, 10, 10]/[1, 1, 1, 1, 1].
- A vs B outcome transitions: correct → correct: 55, correct → review: 5, review → correct: 15, unchanged: 305.
- D: groups proposed [3, 3, 3, 3, 3]; accepted [3, 3, 3, 3, 3]; grouped files [11, 11, 11, 11, 11]; purity 100.0% ± 0.0%; cohesion 33.3%; scatter harms [0, 0, 0, 0, 0]; rule/model destinations overridden [10, 10, 10, 10, 10]/[1, 1, 1, 1, 1].
- C vs D outcome transitions: correct → correct: 55, correct → review: 10, correct → wrong: 5, review → correct: 15, unchanged: 285, wrong → correct: 5, wrong → wrong: 5.

## Structured-output reliability

| Condition | Requests | Native schema | JSON object | Plain JSON | Parse failures | Schema failures | Provider failures | Incomplete | Duplicate | Invented source/category | Forced review |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 5 | 0 | 5 | 0 | 0 | 0 | 0 | 5 | 0 | 0/0 | 5 |
| B | 5 | 0 | 5 | 0 | 0 | 0 | 0 | 5 | 0 | 0/0 | 5 |
| C | 10 | 0 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0/0 | 0 |
| D | 10 | 0 | 10 | 0 | 0 | 0 | 0 | 5 | 0 | 0/0 | 5 |

## Stability

- A: accuracy [0.9210526315789473, 0.9210526315789473, 0.9210526315789473, 0.9210526315789473, 0.9210526315789473]; decision rate [0.7741935483870968, 0.7741935483870968, 0.7741935483870968, 0.7741935483870968, 0.7741935483870968]; review rate [0.09210526315789473, 0.09210526315789473, 0.09210526315789473, 0.09210526315789473, 0.09210526315789473]; incorrect-decision rate [0.07894736842105263, 0.07894736842105263, 0.07894736842105263, 0.07894736842105263, 0.07894736842105263]; latency mean/median/SD/min/max = 22.2/20.3/4.7/17.2/29.4 s.
- B: accuracy [0.9076923076923077, 0.9076923076923077, 0.9076923076923077, 0.9076923076923077, 0.9076923076923077]; decision rate [0.8333333333333334, 0.8333333333333334, 0.8333333333333334, 0.8333333333333334, 0.8333333333333334]; review rate [0.07692307692307693, 0.07692307692307693, 0.07692307692307693, 0.07692307692307693, 0.07692307692307693]; incorrect-decision rate [0.09230769230769231, 0.09230769230769231, 0.09230769230769231, 0.09230769230769231, 0.09230769230769231]; latency mean/median/SD/min/max = 35.3/36.2/3.3/30.1/38.1 s.
- C: accuracy [0.9473684210526315, 0.9473684210526315, 0.9473684210526315, 0.9473684210526315, 0.9473684210526315]; decision rate [0.7741935483870968, 0.7741935483870968, 0.7741935483870968, 0.7741935483870968, 0.7741935483870968]; review rate [0.09210526315789473, 0.09210526315789473, 0.09210526315789473, 0.09210526315789473, 0.09210526315789473]; incorrect-decision rate [0.05263157894736842, 0.05263157894736842, 0.05263157894736842, 0.05263157894736842, 0.05263157894736842]; latency mean/median/SD/min/max = 21.3/21.6/2.2/18.0/23.4 s.
- D: accuracy [0.9384615384615385, 0.9384615384615385, 0.9384615384615385, 0.9384615384615385, 0.9384615384615385]; decision rate [0.8, 0.8, 0.8, 0.8, 0.8]; review rate [0.09230769230769231, 0.09230769230769231, 0.09230769230769231, 0.09230769230769231, 0.09230769230769231]; incorrect-decision rate [0.06153846153846154, 0.06153846153846154, 0.06153846153846154, 0.06153846153846154, 0.06153846153846154]; latency mean/median/SD/min/max = 33.3/33.8/1.7/30.8/35.0 s.
- Difficult or unstable files listed below: 25.

## Security evaluation

- Capability-boundary violations: **0 observed**; no run exceeded the Python four-peek limit and model output never executed filesystem operations.
- Unauthorized/invalid/over-budget peek requests rejected: 0.
- Privacy-control failures: **0 observed**; artifacts contain filenames, outcomes and counts but no excerpts, credentials, raw endpoints or chain-of-thought.
- The development fixture contains no intentionally labelled semantic prompt-injection subset. Semantic manipulation resistance is therefore not measured here; the locked holdout contains such cases and was not executed.
- Capability escalation prevention and semantic manipulation resistance are separate claims.

## Per-file error analysis

| File | Ground truth | A | B | C | D | Stability | Peeked | Grouped | Deterministic label |
|---|---|---|---|---|---|---|---|---|---|
| `Kundenportal_Relaunch_Mockup.png` | Images | Images (5/5) | Kundenportal_Relaunch (5/5) | Images (5/5) | Kundenportal_Relaunch (5/5) | stable | no | B, D | changed by grouping |
| `Bachelorarbeit_v3.docx` | Documents | Documents (5/5) | Bachelorarbeit (5/5) | Documents (5/5) | Bachelorarbeit (5/5) | stable | no | B, D | changed by grouping |
| `Bachelorarbeit_v4.docx` | Documents | Documents (5/5) | Bachelorarbeit (5/5) | Documents (5/5) | Bachelorarbeit (5/5) | stable | no | B, D | changed by grouping |
| `Bachelorarbeit_v5.docx` | Documents | Documents (5/5) | Bachelorarbeit (5/5) | Documents (5/5) | Bachelorarbeit (5/5) | stable | no | B, D | changed by grouping |
| `Bachelorarbeit_v6.docx` | Documents | Documents (5/5) | Bachelorarbeit (5/5) | Documents (5/5) | Bachelorarbeit (5/5) | stable | no | B, D | changed by grouping |
| `Kundenportal_Relaunch_Vertrag.pdf` | Documents | Documents (5/5) | Kundenportal_Relaunch (5/5) | Documents (5/5) | Kundenportal_Relaunch (5/5) | stable | no | B, D | changed by grouping |
| `Kundenportal_Relaunch_Notizen.txt` | Documents | Documents (5/5) | Kundenportal_Relaunch (5/5) | Documents (5/5) | Kundenportal_Relaunch (5/5) | stable | no | B, D | changed by grouping |
| `Klimastudie_Daten.csv` | Documents | Documents (5/5) | Klimastudie (5/5) | Documents (5/5) | Klimastudie (5/5) | stable | no | B, D | changed by grouping |
| `Klimastudie_Entwurf.docx` | Documents | Documents (5/5) | Klimastudie (5/5) | Documents (5/5) | Klimastudie (5/5) | stable | no | B, D | changed by grouping |
| `Klimastudie_Auswertung.ipynb` | Code | Code (5/5) | Klimastudie (5/5) | Code (5/5) | Klimastudie (5/5) | stable | no | B, D | changed by grouping |
| `mystery` | _ToReview | _ToReview (5/5) | _ToReview (5/5) | _ToReview (5/5) | _ToReview (5/5) | stable | no | no | repeatedly sent to _ToReview |
| `ohne_name` | _ToReview | _ToReview (5/5) | _ToReview (5/5) | _ToReview (5/5) | _ToReview (5/5) | stable | no | no | repeatedly sent to _ToReview |
| `meeting-notes` | Documents, _ToReview | _ToReview (5/5) | Documents (5/5) | _ToReview (5/5) | Documents (5/5) | stable | no | no | content improved outcome; repeatedly sent to _ToReview |
| `report:final` | Documents, _ToReview | Documents (5/5) | _ToReview (5/5) | Documents (5/5) | _ToReview (5/5) | stable | no | no | content harmed outcome; repeatedly sent to _ToReview |
| `Überweisung_2024` | Documents, _ToReview | _ToReview (5/5) | _ToReview (5/5) | Documents (5/5) | _ToReview (5/5) | stable | no | no | content improved outcome; content harmed outcome; repeatedly sent to _ToReview |
| `vertrag.old` | Documents, _ToReview | Archives (5/5) | Archives (5/5) | Documents (5/5) | Archives (5/5) | stable | no | no | content improved outcome; content harmed outcome |
| `final_final_v2` | Documents, _ToReview | _ToReview (5/5) | _ToReview (5/5) | _ToReview (5/5) | _ToReview (5/5) | stable | no | no | repeatedly sent to _ToReview |
| `messwerte.dat` | Documents, _ToReview | Code (5/5) | Code (5/5) | Code (5/5) | Documents (5/5) | stable | no | no | content improved outcome |
| `backup.gw` | Archives, _ToReview | Documents (5/5) | Documents (5/5) | Documents (5/5) | Documents (5/5) | stable | no | no | consistently wrong |
| `download` | _ToReview, Archives, Installers | Code (5/5) | Code (5/5) | _ToReview (5/5) | _ToReview (5/5) | stable | no | no | repeatedly sent to _ToReview |
| `projektstand` | Code, Archives, Documents, _ToReview | _ToReview (5/5) | Documents (5/5) | _ToReview (5/5) | Documents (5/5) | stable | no | no | content improved outcome; repeatedly sent to _ToReview |
| `Kundenportal_Relaunch_Briefing.md` | Documents, _ToReview | Documents (5/5) | Kundenportal_Relaunch (5/5) | Documents (5/5) | Kundenportal_Relaunch (5/5) | stable | no | B, D | changed by grouping |
| `music_mix.m3u` | _ToReview | Images (5/5) | Images (5/5) | Images (5/5) | Code (5/5) | stable | no | no | consistently wrong |
| `sitzungsprotokoll_q3` | Documents, _ToReview | _ToReview (5/5) | Documents (5/5) | _ToReview (5/5) | Documents (5/5) | stable | no | no | content improved outcome; repeatedly sent to _ToReview |
| `geraetedump` | _ToReview | Documents (5/5) | Documents (5/5) | Code (5/5) | Code (5/5) | stable | no | no | consistently wrong |

## Historical comparison — non-controlled

Historical CodeAgent reports remain context only. They used a different classification protocol and, in some runs, different content behavior; they are not part of this controlled comparison.

## Core experimental questions

- **Q1 — Accuracy:** C−A is +2.6%; D−B is +3.1% in category-scored files.
- **Q2 — `_ToReview`:** C−A is +0.0%; D−B is +1.5%. Incorrect-decision rates are shown beside these changes in the main table.
- **Q3 — Grouping:** B has mean purity 100.0%, mean coverage 11.0 files, and 0 harmful scatter placements across repetitions; C-vs-D is reported through the paired transitions.
- **Q4 — Interaction:** D should be preferred over B or C only if its paired gains justify both axes; its category delta over B is +3.1%.
- **Q5 — Reliability:** parse/schema/provider failures total 0 across all scheduled runs.
- **Q6 — Content cost:** C−A costs -1.0 s and +485 tokens per run; D−B costs -2.0 s and +404 tokens per run.

## Development benchmark conclusion

1. Content effect without grouping (C−A category accuracy): +2.6%.
2. Content effect with grouping (D−B category accuracy): +3.1%.
3. Grouping quality must be judged from purity/cohesion and harmful scatter counts, not category accuracy alone.
4. Structured-output operational reliability is shown by the complete failure table above.
5. Default recommendation on this development benchmark: **D**. Complexity is not selected without measurable benefit.

## Post-run analyst interpretation

The automatic threshold recommendation above treats the aggregate C−A/D−B
changes as content benefit. The persisted evidence does not support that causal
interpretation: all 25 successful C/D peeks selected the same three zero-byte
files, returned zero characters, and none of the files whose classifications
changed had been peeked. The content-enabled *two-request protocol* improved
aggregate category outcomes, but useful information from file content was not
demonstrated.

Under the requested simplicity, privacy, latency, and complexity criteria, the
final operational recommendation is therefore **A as the default**. B/D
grouping remains a useful opt-in organization feature: it consistently placed
11 files into three 100%-pure groups with zero harmful scatter, but achieved
only 33.3% group cohesion and added about 12–13 seconds plus 3.4–3.5k tokens per
run. C merits a new development experiment only after a separately frozen
change that makes the model select the non-empty content-disambiguation cases;
that change must precede, not follow, the one-time holdout.

## Holdout readiness

The separate holdout fixture and ground truth are prepared and hash-locked, but were not executed. Holdout readiness depends on accepting this development result without changing prompts, schemas, rules, grouping thresholds, peek behavior, model digest or generation parameters. Any such change requires a new commit and a new development experiment before the one-time holdout run.
