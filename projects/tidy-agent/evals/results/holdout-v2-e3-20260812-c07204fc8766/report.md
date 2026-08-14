# holdout-v2-e3-20260812-c07204fc8766

One-time external validation of the frozen E3 production candidate (`948fc6c85b5e8f1c58598d9ffaa6c59a33a8a8a1`) on the frozen, independently authored Holdout v2 (`c07204fc8766d38bb98addbfb42e74ccabd503b8`). This run is not repeatable for model selection.

- Commit: `c07204fc8766d38bb98addbfb42e74ccabd503b8`
- Model: `ollama_chat/qwen3.5:4b` digest `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd`
- Model digest after run: `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd` (unchanged)

## Headline result (N = 90)

- Correct automatic: 30
- Incorrect automatic: 6
- Review: 54
- Strict accuracy: 67.8%
- Unsafe automation rate: 6.7%
- Automation coverage: 40.0%
- Review rate: 60.0%
- Accuracy on decided: 83.3%
- Review recall: 93.9%
- Review precision: 57.4%

E3 produced 6 observed unsafe automatic classification(s) on this 90-file Holdout v2.

## Real-category subset (N = 57)

- Correct automatic: 30
- Wrong automatic: 4
- False review: 23

## Ground-truth review subset (N = 33)

- Correctly reviewed: 31
- Incorrectly automated: 2

## E3 gate analysis

- classify/classify same: 36 (correct 30, incorrect 6)
- classify/classify different: 2
- classify/review: 9
- review/classify: 16
- review/review: 25
- invalid: 2

## Prompt-like filename adversarial subset (N = 8)

- Ground-truth category cases: 2
- Ground-truth review cases: 6
- Correct automatic: 0
- Incorrect automatic: 0
- Review: 8
- Review recall (of ground-truth review cases): 100.0%

## Language/script breakdown (descriptive, secondary)

- ascii_latin: n=85, correct_automatic=30, incorrect_automatic=6, review=49
- non_latin_script: n=5, correct_automatic=0, incorrect_automatic=0, review=5

## Structured-output reliability

- provider_errors: 0
- parse_failures: 0
- schema_validation_failures: 0
- incomplete_responses_omitted_sources: 1
- duplicate_source_responses: 0
- invented_source_responses: 0
- invented_category_responses: 0
- pass1_invalid_count: 0
- pass2_invalid_count: 2
- gate_invalid_count: 2
- final_to_review_caused_by_protocol_invalidity_not_semantic_abstention: 2

## Operational cost

- total_model_calls: 2
- pass1_and_pass2_calls: 2
- total_worker_latency_seconds: 82.10555355299948
- model_reported_latency_seconds: 78.15675905299759
- final_classification_latency_seconds: 78.15675905299759
- input_tokens: 1864
- completion_tokens: 3325
- total_tokens: 5189

## Metadata-only verification

- peek_requests_authorized: 0
- peek_candidates_total: 0
- content_unavailable: 0
- peek_tool_constructed: False
- content_path_entered: False
- structural_note: classify_with_agreement_gate (src/tidy/classification.py) takes no peek_tool parameter, never calls _peek_candidates_with_telemetry, and never calls peek_file; all peek_* telemetry fields are the ClassificationTelemetry defaults because that code path is never entered by this method.

