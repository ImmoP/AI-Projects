# A/E/C content diagnostic — structured-aec-20260811-fc9d5fe9654d

> Development benchmark only. The locked holdout was not executed.

## Provenance

- Commit: `fc9d5fe9654dc127d83d77668be51a03d1cc3d49` (clean at experiment start)
- Model: `ollama_chat/qwen3.5:4b`
- Digest: `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd`
- Structured output: `json_object`
- Repetitions: 5 per condition; one discarded warmup
- Development files: 76

## Main results

| Condition | Accuracy | Decision rate | Accuracy decided | Review | Incorrect | Latency | Tokens | Authorized peeks | Non-empty peeks | Characters |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A — One-pass metadata | 91.1 ± 0.6% | 82.6 ± 2.9% | 73.5 ± 0.9% | 7.1 ± 1.2% | 8.9 ± 0.6% | 16.4 ± 1.6 s | 645.8 ± 1.8 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| E — Two-pass metadata control | 95.8 ± 0.6% | 67.1 ± 1.4% | 84.6 ± 2.6% | 13.4 ± 0.6% | 4.2 ± 0.6% | 18.1 ± 1.2 s | 1032.2 ± 0.4 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| C — Two-pass metadata + content | 93.4 ± 0.0% | 74.2 ± 0.0% | 78.3 ± 0.0% | 10.5 ± 0.0% | 6.6 ± 0.0% | 17.9 ± 1.3 s | 1227.0 ± 0.0 | 3.0 ± 0.0 | 2.0 ± 0.0 | 467.0 ± 0.0 |

## Candidate selection

Phase 1 received relative source, normalized extension, byte size, and reader type only. Zero-byte, pre-known unsupported, and over-limit files were excluded in Python before model selection.
- E: candidates 15/155 across runs; empty filtered 140; requested 15; authorized 0; non-empty 0; informative-peek rate 0.0%; characters delivered 0.
- E requested sources: {'geraetedump': 5, 'sitzungsprotokoll_q3': 5, 'steuerbescheid_2024': 5}.
- C: candidates 15/155 across runs; empty filtered 140; requested 15; authorized 15; non-empty 10; informative-peek rate 66.7%; characters delivered 2335.
- C requested sources: {'geraetedump': 5, 'sitzungsprotokoll_q3': 5, 'steuerbescheid_2024': 5}.

## Causal comparisons

- A vs E: correct → review: 10, correct → wrong: 5, review → correct: 4, unchanged: 338, wrong → correct: 5, wrong → review: 18.
- E vs C: correct → review: 4, correct → wrong: 5, review → correct: 10, review → wrong: 5, unchanged: 355, wrong → correct: 1.
- Second-pass/control effect E−A: +4.7%.
- Actual-content effect C−E: -2.4%.
- Non-empty delivered content changed 5 paired file outcomes, improved 5, and harmed 0.
- Demonstrably improved files: ['steuerbescheid_2024'].
- Demonstrably harmed files: none.

## Per-file evidence

| Rep | File | Ground truth | A | E | C | E requested | C requested/authorized | Non-empty | Bytes/chars |
|---:|---|---|---|---|---|---:|---:|---:|---:|
| 1 | `meeting-notes` | Documents, _ToReview | _ToReview | _ToReview | Documents | no | no/no | no | 0/0 |
| 1 | `vertrag.old` | Documents, _ToReview | Archives | Documents | Archives | no | no/no | no | 0/0 |
| 1 | `final_final_v2` | Documents, _ToReview | _ToReview | _ToReview | Code | no | no/no | no | 0/0 |
| 1 | `urlaub_2025` | Images, _ToReview | Images | Documents | Images | no | no/no | no | 0/0 |
| 1 | `setup_latest` | Installers, _ToReview | Installers | _ToReview | _ToReview | no | no/no | no | 0/0 |
| 1 | `download` | _ToReview, Archives, Installers | Code | _ToReview | _ToReview | no | no/no | no | 0/0 |
| 1 | `steuerbescheid_2024` | Documents, _ToReview | Documents | _ToReview | Documents | yes | yes/yes | yes | 260/254 |
| 1 | `sitzungsprotokoll_q3` | Documents, _ToReview | _ToReview | _ToReview | _ToReview | yes | yes/yes | yes | 215/213 |
| 1 | `geraetedump` | _ToReview | Documents | _ToReview | _ToReview | yes | yes/yes | no | 2056/0 |
| 2 | `meeting-notes` | Documents, _ToReview | Code | _ToReview | Documents | no | no/no | no | 0/0 |
| 2 | `report:final` | Documents, _ToReview | _ToReview | Documents | Documents | no | no/no | no | 0/0 |
| 2 | `vertrag.old` | Documents, _ToReview | Archives | Documents | Archives | no | no/no | no | 0/0 |
| 2 | `final_final_v2` | Documents, _ToReview | _ToReview | _ToReview | Code | no | no/no | no | 0/0 |
| 2 | `messwerte.dat` | Documents, _ToReview | Documents | Code | Code | no | no/no | no | 0/0 |
| 2 | `setup_latest` | Installers, _ToReview | Installers | Installers | _ToReview | no | no/no | no | 0/0 |
| 2 | `download` | _ToReview, Archives, Installers | Code | _ToReview | _ToReview | no | no/no | no | 0/0 |
| 2 | `README` | Documents, Code, _ToReview | Code | Documents | Documents | no | no/no | no | 0/0 |
| 2 | `projektstand` | Code, Archives, Documents, _ToReview | Documents | _ToReview | _ToReview | no | no/no | no | 0/0 |
| 2 | `steuerbescheid_2024` | Documents, _ToReview | Documents | _ToReview | Documents | yes | yes/yes | yes | 260/254 |
| 2 | `sitzungsprotokoll_q3` | Documents, _ToReview | Code | _ToReview | _ToReview | yes | yes/yes | yes | 215/213 |
| 2 | `geraetedump` | _ToReview | Documents | _ToReview | _ToReview | yes | yes/yes | no | 2056/0 |
| 3 | `meeting-notes` | Documents, _ToReview | Code | _ToReview | Documents | no | no/no | no | 0/0 |
| 3 | `report:final` | Documents, _ToReview | _ToReview | Documents | Documents | no | no/no | no | 0/0 |
| 3 | `vertrag.old` | Documents, _ToReview | Archives | Documents | Archives | no | no/no | no | 0/0 |
| 3 | `final_final_v2` | Documents, _ToReview | _ToReview | _ToReview | Code | no | no/no | no | 0/0 |
| 3 | `messwerte.dat` | Documents, _ToReview | Documents | Code | Code | no | no/no | no | 0/0 |
| 3 | `setup_latest` | Installers, _ToReview | Installers | Installers | _ToReview | no | no/no | no | 0/0 |
| 3 | `download` | _ToReview, Archives, Installers | Code | _ToReview | _ToReview | no | no/no | no | 0/0 |
| 3 | `README` | Documents, Code, _ToReview | Code | Documents | Documents | no | no/no | no | 0/0 |
| 3 | `projektstand` | Code, Archives, Documents, _ToReview | Documents | _ToReview | _ToReview | no | no/no | no | 0/0 |
| 3 | `steuerbescheid_2024` | Documents, _ToReview | Documents | _ToReview | Documents | yes | yes/yes | yes | 260/254 |
| 3 | `sitzungsprotokoll_q3` | Documents, _ToReview | Code | _ToReview | _ToReview | yes | yes/yes | yes | 215/213 |
| 3 | `geraetedump` | _ToReview | Documents | _ToReview | _ToReview | yes | yes/yes | no | 2056/0 |
| 4 | `meeting-notes` | Documents, _ToReview | Code | _ToReview | Documents | no | no/no | no | 0/0 |
| 4 | `report:final` | Documents, _ToReview | _ToReview | Documents | Documents | no | no/no | no | 0/0 |
| 4 | `vertrag.old` | Documents, _ToReview | Archives | Documents | Archives | no | no/no | no | 0/0 |
| 4 | `final_final_v2` | Documents, _ToReview | _ToReview | _ToReview | Code | no | no/no | no | 0/0 |
| 4 | `messwerte.dat` | Documents, _ToReview | Documents | Code | Code | no | no/no | no | 0/0 |
| 4 | `setup_latest` | Installers, _ToReview | Installers | Installers | _ToReview | no | no/no | no | 0/0 |
| 4 | `download` | _ToReview, Archives, Installers | Code | _ToReview | _ToReview | no | no/no | no | 0/0 |
| 4 | `README` | Documents, Code, _ToReview | Code | Documents | Documents | no | no/no | no | 0/0 |
| 4 | `projektstand` | Code, Archives, Documents, _ToReview | Documents | _ToReview | _ToReview | no | no/no | no | 0/0 |
| 4 | `steuerbescheid_2024` | Documents, _ToReview | Documents | _ToReview | Documents | yes | yes/yes | yes | 260/254 |
| 4 | `sitzungsprotokoll_q3` | Documents, _ToReview | Code | _ToReview | _ToReview | yes | yes/yes | yes | 215/213 |
| 4 | `geraetedump` | _ToReview | Documents | _ToReview | _ToReview | yes | yes/yes | no | 2056/0 |
| 5 | `meeting-notes` | Documents, _ToReview | Code | _ToReview | Documents | no | no/no | no | 0/0 |
| 5 | `report:final` | Documents, _ToReview | _ToReview | Documents | Documents | no | no/no | no | 0/0 |
| 5 | `vertrag.old` | Documents, _ToReview | Archives | Documents | Archives | no | no/no | no | 0/0 |
| 5 | `final_final_v2` | Documents, _ToReview | _ToReview | _ToReview | Code | no | no/no | no | 0/0 |
| 5 | `messwerte.dat` | Documents, _ToReview | Documents | Code | Code | no | no/no | no | 0/0 |
| 5 | `setup_latest` | Installers, _ToReview | Installers | Installers | _ToReview | no | no/no | no | 0/0 |
| 5 | `download` | _ToReview, Archives, Installers | Code | _ToReview | _ToReview | no | no/no | no | 0/0 |
| 5 | `README` | Documents, Code, _ToReview | Code | Documents | Documents | no | no/no | no | 0/0 |
| 5 | `projektstand` | Code, Archives, Documents, _ToReview | Documents | _ToReview | _ToReview | no | no/no | no | 0/0 |
| 5 | `steuerbescheid_2024` | Documents, _ToReview | Documents | _ToReview | Documents | yes | yes/yes | yes | 260/254 |
| 5 | `sitzungsprotokoll_q3` | Documents, _ToReview | Code | _ToReview | _ToReview | yes | yes/yes | yes | 215/213 |
| 5 | `geraetedump` | _ToReview | Documents | _ToReview | _ToReview | yes | yes/yes | no | 2056/0 |

## Recommendation

Keep A as default; content did not outperform the metadata control. Further development is needed before content can be justified.

## Safety and holdout

- E constructed no peek tool, read no file content, and invoked no parser.
- C never exceeded 4 authorized peeks per run.
- Persisted artifacts contain per-file counts but no excerpts, absolute paths, raw endpoints, credentials, or chain-of-thought.
- The 41-file holdout remained hash-locked and was not executed.
- Holdout readiness depends on accepting this result without another prompt/schema/rule/model change; otherwise freeze and rerun development first.

The pre-change root-cause table is in [content_selection_root_cause.md](../../content_selection_root_cause.md).
