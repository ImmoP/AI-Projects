# Holdout v5 — authoring boundary (READ BEFORE AUTHORING)

This directory will hold the **future, independent Holdout v5** for the
revised `E4-batched` candidate. **It currently contains no fixture and no
ground truth on purpose.** Only infrastructure lives here right now
(`blind_audit.py`). The runner `evals/run_holdout_v5_e4.py` refuses to run
until every frozen artifact below exists and validates.

## Why a new Holdout

Holdout v4 was consumed exactly once and was invalidated by its own
precommitted validity rule (one required structured-output response failed
schema/cardinality validation — an incomplete long-list response that also
invented a source). It remains historically `PARTIAL_INCONCLUSIVE` and is
**never** rerun, edited, or reinterpreted. Holdout v5 is **not** a second
attempt on the same frozen system: it evaluates a *revised* system version
(`E4-batched`) whose transport/schema/batching reliability changed after v4.
No classification-policy tuning was performed against v4 labels.

## Clean-room authoring boundary (mandatory)

The actual v5 cases **must be authored in a separate task/process** by an
author that is explicitly instructed **NOT** to read any of the following:

- `evals/fixture/`, `evals/dev/` (development fixtures and `expected.yaml`)
- `evals/calibration/`, `evals/boundary_calibration/`,
  `evals/e3_error_calibration/`, `evals/veto_precision_calibration/` case
  contents
- `evals/holdout/`, `evals/holdout_v2/`, `evals/holdout_v3/`,
  `evals/holdout_v4/` case contents
- any prior `ground_truth.json` / `expected.yaml`
- any prior per-file prediction, error example, or Holdout report

The coding agent that built the batching/validation reliability fix **had**
access to historical evaluation material, which is exactly why it must **not**
author the v5 cases.

The clean-room author may receive **only**:

- the category definitions (`config/rules.yaml` category names),
- the allowed artifact/file-type conventions (extensionless, zero-byte,
  NFC-normalized, Windows-safe, deterministic-rule-unresolvable filenames),
- the desired sample count and class-balance methodology,
- the protocol requirements (`evals/holdout_v5_protocol.md`),
- the output schema for `ground_truth.json`.

## Required artifacts before the runner will run

Created by the clean-room authoring + freeze step (not by this task):

- `fixture/` — the authored zero-byte, extensionless case files.
- `ground_truth.json` — per-case labels, frozen **before** any inference.
- `AUTHORING_FROZEN.json` — authoring-freeze marker with dataset/ground-truth
  hashes and `model_inference_performed: false`.
- `candidate_selection.json` — pins `E4-batched`, `selection_closed: true`,
  `holdout_v5_inference_performed: false`.
- `code_pins.json` — sha256 of every source/doc file the run depends on.
- `fixture_manifest.json` — dataset hash, composition, and the blind-audit
  result (exact historical overlap **must be 0**).

Created automatically at run time (never pre-created, never committed):

- `CONSUMED.json` — local, gitignored, written atomically immediately before
  the first measured request. Its existence blocks any further run.
- `consumption_record.json` — the permanent, committed record written only
  **after** a real execution.

Do **not** create fake/placeholder cases here. Do **not** run inference.
