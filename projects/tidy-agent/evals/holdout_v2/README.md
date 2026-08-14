# Holdout v2 — locked, not yet evaluated

## Purpose

Independent validation of the frozen E3 production candidate (explicit
structured abstention plus a deterministic two-pass agreement gate) on
genuinely unseen unresolved-file metadata. This is external validation
relative to the Development calibration cycle that selected E3 — it is not
another development/tuning round, and it must not be used as one.

## Freeze candidate

`948fc6c85b5e8f1c58598d9ffaa6c59a33a8a8a1` — "tidy: integrate explicit
abstention agreement gate".

## Size

90 unique cases, all extensionless and 0 bytes (metadata-only; content is
never read by the future evaluation).

## Ground-truth distribution

| Category | Count |
|---|---:|
| Images | 11 |
| Documents | 12 |
| Archives | 11 |
| Code | 11 |
| Installers | 12 |
| **Real-category total** | **57 (63.3%)** |
| `_ToReview` | 33 (36.7%) |
| **Total** | **90** |

The `_ToReview` set includes 6 newly authored "prompt-like adversarial
filename" cases (instruction-like or authority-impersonating text embedded
in the filename, with no legitimate topical content behind it) and 27
non-adversarial ambiguous/insufficient cases (generic version/backup/draft
language, multilingual ambiguity across German, English, French, Spanish,
Italian, Russian, Chinese, Japanese, and Korean, and filenames carrying two
independently plausible categories). Two further prompt-like-styled cases
carry a **real** category, because their legitimate topical content
(a quarterly report; a driver setup) is independently sufficient once the
instruction-like wrapper is disregarded — not every adversarial-styled name
is `_ToReview` by construction, matching the requirement that this subset
tests semantic influence on the classification label, not filesystem
capability: no classification outcome here can read, move, or delete
anything without a separate human approving the resulting plan.

## Construction protocol

- Cases were independently authored for this task without opening
  `evals/holdout/build_fixture.py`, `evals/holdout/expected.yaml`, or any
  other old-Holdout case-level file, and without inspecting old-Holdout
  predictions, per-file evidence, or reports for individual case content.
- `evals/calibration/fixture/` and `evals/calibration/expected.yaml`
  case-level content was not inspected or reused for case generation.
- Ground truth was assigned before any model inference, using a single
  test: given only the metadata a production metadata-only classifier is
  permitted to see, can a human safely assign exactly one permitted
  category? If yes, that category; if no, `_ToReview`. No case was labelled
  `_ToReview` merely because a model might struggle with it, and no case was
  given a real category merely because a plausible keyword appears.
- No model inference of any kind (no Ollama, no LiteLLM, no hosted
  provider) occurred during construction. `model_inference_performed: false`
  is recorded directly in `fixture_manifest.json`.
- The fixture is metadata-only: every file is 0 bytes, and every case is
  extensionless by construction, so none can be resolved by
  `config/rules.yaml` (extension-keyed) or excluded by its glob patterns —
  all 90 cases are guaranteed, by construction and verified by an offline
  test, to reach unresolved-file (E3) classification.
- This is a process claim of independence, not a mathematical proof: a
  single author's fixture still carries that author's stylistic and
  linguistic biases, and no case-by-case lexical comparison against the old
  Holdout or Development fixture was performed (doing so would itself
  require inspecting the prohibited case-level data).
- The next phase is exactly one frozen E3 live evaluation, after the user
  manually reviews, stages, commits, and provides the freeze commit for this
  fixture. No prediction of any kind exists yet.

## Limitations

- Manually constructed by a single author; author bias in phrasing,
  language selection, and topic choice is possible and not independently
  audited.
- 90 cases is large enough for a meaningfully more precise point estimate
  than the original 41-file Holdout, but still small enough that individual
  wide-margin cases can move aggregate rates noticeably; treat any future
  single-evaluation result with the same caution as before (raw counts and
  per-file evidence over a single headline percentage).
- This fixture tests the metadata-only path only; it says nothing about
  `--read-contents` behaviour, which E3 has not touched or been validated
  against.
