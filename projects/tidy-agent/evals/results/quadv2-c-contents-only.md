# tidy-agent evaluation — repeated runs

- Runs: **3**
- Run mode: **fixed categories only** (`--no-group`, content reading `True`)
- Metric family: **category accuracy only**
- No grouping ran, so no clustering metric is defined in this mode; clustering rows are omitted rather than reported as 0 or N/A.
- Comparable only with other reports whose run mode line is identical.
- Model: `ollama_chat/qwen3.5:4b`
- Think: `False`
- Standard run timeout: 900.0 s
- Ground truth: `expected.yaml sha256:dfe0b35b34c6`
- Endpoint: `remote host on a private network (address redacted)`
- Model digest: `2a654d98e6fba55d`
- Quantization: `Q4_K_M`
- Parameter size: `4.7B`
- Context length: `8192`
- Model placement: `GPU/VRAM 3341088193 bytes`
- Evaluated: 2026-08-11T13:07:44.517867+00:00
- Statuses: run 1: ok, run 2: ok, run 3: ok
- Repetition isolation: model unloaded before every run (`keep_alive=0`), so each repetition reloads the weights

**How far this range reaches.** At temperature 0 repetitions against one warm process reproduce each other exactly, so a spread of zero there is a property of the setup, not a measurement of stability. These runs reload the model between repetitions, which removes the warm-process part of that objection but still shares one machine, one server build, and one sitting.

**Measured between-session delta.** The same configuration (`qwen3.5:4b`, `--no-think`, category mode) scored 75.0% (21/28) unresolved accuracy on 2026-08-10 at 17:44 UTC and 82.1% (23/28) at 20:52 UTC — 7.1 points apart with identical flags and code. That difference is itself a measurement and outranks any within-session range printed below.

A range that overlaps another mode's range is not evidence of a difference between the modes.

**Every repetition in this batch produced identical values.** Reloading the model between runs did not reproduce the between-session delta either, so whatever varies is not the warm process alone — most likely load-time conditions this harness does not control. The range stays a lower bound.

| Metric | run 1 | run 2 | run 3 | Range (cold-start runs) |
|---|---:|---:|---:|---:|
| Category accuracy, unresolved files (`_ToReview` accepted) | 90.3% (28/31) | 90.3% (28/31) | 90.3% (28/31) | identical |
| Decision rate, unresolved files | 54.8% (17/31) | 54.8% (17/31) | 54.8% (17/31) | identical |
| Accuracy on decided files only | 82.4% (14/17) | 82.4% (14/17) | 82.4% (14/17) | identical |
| Files omitted by the model | 1/31 | 1/31 | 1/31 | identical |
| Classification latency | 45.5 s | 45.6 s | 45.7 s | 45.4555 – 45.7197 |

Individual reports:

- [`quadv2-c-contents-only-run1.md`](quadv2-c-contents-only-run1.md)
- [`quadv2-c-contents-only-run2.md`](quadv2-c-contents-only-run2.md)
- [`quadv2-c-contents-only-run3.md`](quadv2-c-contents-only-run3.md)
