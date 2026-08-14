# Holdout v3 -- E4-current -- INTERRUPTED, NO USABLE EVIDENCE

Consumed: true
Run complete: false
First measured request timestamp: 2026-08-12T22:37:44Z
Measured model calls confirmed: 0

## What happened

The frozen runner (`evals/run_holdout_v3_e4.py`) was invoked to perform the
one-time live Holdout v3 evaluation. It completed, in order:

1. `is_consumed()` check (false)
2. `verify_code_pins()` (passed)
3. `verify_fixture_hash()` (passed)
4. `verify_model_identity()` (passed -- digest
   `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd`,
   `Q4_K_M`, `temperature=0`, thinking disabled, `num_ctx=8192`)
5. `create_result_directory()` -- created this directory
6. Wrote `evals/holdout_v3/CONSUMED.json` synchronously, immediately before
   issuing the first measured request (this is the frozen design: the
   marker is written *before* the request so an ambiguous crash during the
   request is always treated as consumed)

The invoking process was then interrupted (the interactive tool call that
launched it was rejected/terminated) before `evals.post_holdout_candidates.run_e4`
returned. Nothing after step 6 executed: no per-file classification, no
scoring, no diagnostics.

## Evidence that no model request actually landed

`~/.ollama/logs/server.log` around the consumption timestamp
(`2026-08-13 00:37:44` local time, matching `2026-08-12T22:37:44Z`) shows
only three `POST /api/show` calls (1-3 ms each -- litellm's provider
capability probe), and no `/api/chat` or `/api/generate` call at all in
this window or afterward:

```text
[GIN] 2026/08/13 - 00:37:44 | 200 |    2.777044ms | 127.0.0.1 | POST "/api/show"
[GIN] 2026/08/13 - 00:37:44 | 200 |    1.331853ms | 127.0.0.1 | POST "/api/show"
[GIN] 2026/08/13 - 00:37:44 | 200 |    1.358996ms | 127.0.0.1 | POST "/api/show"
```

`ollama ps` immediately afterward showed no loaded model. This strongly
suggests the actual classification request (which would carry the 120
filenames) never reached the model -- the process died during Python-side
setup or the litellm capability check, before the HTTP request for pass 1
was issued.

## Why this is still reported as consumed

Per the frozen runner's specification, the consumption marker is written
synchronously *before* the first measured request is issued, specifically
so this exact kind of ambiguous crash is conservatively treated as
consumption. This is a deliberate, frozen design choice, not a bug in this
report. Holdout v3 is therefore **permanently consumed** even though no
usable per-file evidence exists.

## What must not happen next

* Holdout v3 must not be rerun, restarted, or selectively re-evaluated.
* `evals/holdout_v3/CONSUMED.json` must not be deleted or reset.
* No candidate substitution or fixture edit may be used to "recover" this
  Holdout.
* A materially different future candidate requires a new, independently
  constructed Holdout (v4), not a repeat of v3.

## Decision

**D -- INCONCLUSIVE.** This is a genuine protocol/infrastructure failure
(an interrupted process, not a model or candidate result): the consumed
Holdout produced zero per-file predictions and therefore no usable
evidence about `E4-current`'s generalization. No PASS/CONDITIONAL/FAIL
determination about `E4-current` can be drawn from this consumption event.
