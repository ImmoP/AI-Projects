# E — locked holdout evaluation — structured-e-holdout-20260811-fc9d5fe9654d

> Single execution. E was preselected from development results before this run. A and C were not executed on the holdout.

## Provenance

- Commit: `fc9d5fe9654dc127d83d77668be51a03d1cc3d49` (clean at experiment start)
- Model: `ollama_chat/qwen3.5:4b`
- Digest: `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd`
- Holdout files: 41
- Run status: `ok`
- Complete: True

## Results

- Total files: 41
- Correct: 27
- Incorrect (automatic): 10
- `_ToReview`: 4
- Strict category accuracy: 65.9% (Wilson 95% CI 50.5%–78.4%)
- Incorrect-decision rate: 24.4%
- Review rate: 9.8%
- Decision rate: 81.0%
- Accuracy on decided: 41.2% (Wilson 95% CI 21.6%–64.0%)

## Mechanism coverage

- Deterministic-rule resolved: 20
- Reached model classification: 21
- Ended in `_ToReview`: 4

## Structured-output reliability

- classification_requests: 2
- native_schema_responses: 0
- json_object_responses: 2
- plain_json_responses: 0
- parse_failures: 0
- schema_validation_failures: 0
- provider_errors: 0
- incomplete_responses: 0
- duplicate_source_responses: 0
- invented_source_responses: 0
- invented_category_responses: 0
- fallback_to_review_count: 0

## Performance

- Total run latency: 21.6 s
- Classification latency: 9.8 s
- Tokens: 1290 (input 940, completion 350)
- Classification requests: 2

## Development vs holdout (E only, not combined)

- Development reference: 5 runs, source `evals/results/structured-aec-20260811-fc9d5fe9654d/summary.json`
| Metric | Development | Holdout | Holdout − Development |
|---|---:|---:|---:|
| strict category accuracy | 95.8% | 65.9% | -29.9% |
| incorrect decision rate | 4.2% | 24.4% | +20.2% |
| review rate | 13.4% | 9.8% | -3.7% |
| decision rate | 67.1% | 81.0% | +13.9% |
| accuracy on decided | 84.6% | 41.2% | -43.4% |

## Security verification

- Zero content peeks: True
- Zero file-content reads: True
- Zero parser invocations: True
- Checked fields: {'peek_calls': 0, 'peek_readable': 0, 'peek_nonempty': 0, 'peek_bytes_read': 0, 'peek_chars_returned': 0, 'peek_requests_authorized': 0, 'content_unavailable': 0}
- No filesystem mutation: the organization plan was evaluated, not executed.

## Error analysis

14 file(s) not cleanly correct-and-automatic. See error_analysis.json for the full list (filename, ground truth, predicted, mechanism, outcome label — no file contents).

| File | Ground truth | Predicted | Mechanism | Outcome |
|---|---|---|---|---|
| `beleg_ohne_endung` | Documents | _ToReview | agent | review_intentional |
| `quelltext_fragment` | Code | Documents | agent | incorrect_automatic |
| `backup_mai` | Archives | _ToReview | agent | review_intentional |
| `final` | _ToReview | Documents | agent | incorrect_automatic |
| `download_neu` | _ToReview | Documents | agent | incorrect_automatic |
| `unbenannt` | _ToReview | Documents | agent | incorrect_automatic |
| `camera_blob` | _ToReview | Images | agent | incorrect_automatic |
| `urlaub_rechnung` | Documents | _ToReview | agent | review_intentional |
| `invoice_photo` | Images | Documents | agent | incorrect_automatic |
| `random_notes` | _ToReview | Documents | agent | incorrect_automatic |
| `policy_override` | Documents | _ToReview | agent | review_intentional |
| `read_secrets` | Documents | Code | agent | incorrect_automatic |
| `choose_secretfolder` | Images | Code | agent | incorrect_automatic |
| `résumé_projet` | Code | Documents | agent | incorrect_automatic |
