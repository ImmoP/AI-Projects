# tidy-agent evaluation — repeated runs

- Runs: **2**
- Run mode: **fixed categories only** (`--no-group`, content reading `False`)
- Metric family: **category accuracy only**
- No grouping ran, so no clustering metric is defined in this mode; clustering rows are omitted rather than reported as 0 or N/A.
- Comparable only with other reports whose run mode line is identical.
- Model: `ollama_chat/qwen3.5:4b`
- Think: `False`
- Standard run timeout: 1800.0 s
- Ground truth: `expected.yaml sha256:c97e0d99d26b`
- Endpoint: `http://127.0.0.1:11434`
- Model digest: `2a654d98e6fba55d`
- Quantization: `Q4_K_M`
- Parameter size: `4.7B`
- Context length: `8192`
- Model placement: `CPU only (size_vram=0)`
- Evaluated: 2026-08-11T12:23:49.636694+00:00
- Statuses: run 1: ok, run 2: ok
- Repetition isolation: model unloaded before every run (`keep_alive=0`), so each repetition reloads the weights

**How far this range reaches.** At temperature 0 repetitions against one warm process reproduce each other exactly, so a spread of zero there is a property of the setup, not a measurement of stability. These runs reload the model between repetitions, which removes the warm-process part of that objection but still shares one machine, one server build, and one sitting.

**Measured between-session delta.** The same configuration (`qwen3.5:4b`, `--no-think`, category mode) scored 75.0% (21/28) unresolved accuracy on 2026-08-10 at 17:44 UTC and 82.1% (23/28) at 20:52 UTC — 7.1 points apart with identical flags and code. That difference is itself a measurement and outranks any within-session range printed below.

A range that overlaps another mode's range is not evidence of a difference between the modes.

**Every repetition in this batch produced identical values.** Reloading the model between runs did not reproduce the between-session delta either, so whatever varies is not the warm process alone — most likely load-time conditions this harness does not control. The range stays a lower bound.

| Metric | run 1 | run 2 | Range (cold-start runs) |
|---|---:|---:|---:|
| Category accuracy, unresolved files (`_ToReview` accepted) | 82.1% (23/28) | 82.1% (23/28) | identical |
| Decision rate, unresolved files | 89.3% (25/28) | 89.3% (25/28) | identical |
| Accuracy on decided files only | 80.0% (20/25) | 80.0% (20/25) | identical |
| Files omitted by the model | 0/28 | 0/28 | identical |
| Classification latency | 682.8 s | 854.4 s | 682.797 – 854.442 |

Individual reports:

- [`placement-cpu-local-run1.md`](placement-cpu-local-run1.md)
- [`placement-cpu-local-run2.md`](placement-cpu-local-run2.md)
