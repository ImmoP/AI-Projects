# Synthetic one-time-runner smoke fixture

**Not evaluation evidence.** This fixture exists only to give
`evals/run_one_time_smoke.py` a realistic-sized (120-file) batch to exercise
the real request/response/persistence lifecycle against, before a future
independent Holdout is constructed. It is rerunnable and may be regenerated
freely with:

```bash
python3 evals/synthetic_one_time_smoke/build_fixture.py
```

## Why this exists

Holdout v3's live run was interrupted between writing its consumption
marker and the first measured provider request. The result directory it
left behind was completely empty: nothing distinguished "marker written,
dispatch never started" from "provider request in flight" from "provider
response returned". This fixture, together with
`evals/one_time_eval_runtime.py` (a new, generic lifecycle/journaling
module) and `evals/run_one_time_smoke.py`, exists to prove that a hardened
persistence design survives that same failure mode before it is used for
anything that matters (a new Holdout).

## Composition

120 extensionless, zero-byte, NFC-normalized, cross-platform-safe, unique
files generated from an obvious `synthetic_<family>_case_<NNN>_<descriptor>`
template:

* 20 `synthetic_document_case_*`
* 20 `synthetic_code_case_*`
* 20 `synthetic_image_case_*`
* 20 `synthetic_archive_case_*`
* 20 `synthetic_installer_case_*`
* 20 `synthetic_review_case_*` (intentionally generic/ambiguous)

`fixture_manifest.json` records an `intended_category` per file for sanity
diagnostics only. **Classification accuracy on this fixture is not
scientific evidence of anything** -- it is not tuned against, not used for
candidate selection, and never will be. Its only job is to look enough like
a real batch (same file count, same real/review split shape) to load-test
the runtime.
