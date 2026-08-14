# tidy-agent evaluation — repeated runs

- Runs: **3**
- Run mode: **fixed categories only** (`--no-group`, content reading `False`)
- Metric family: **category accuracy only**
- No grouping ran, so no clustering metric is defined in this mode; clustering rows are omitted rather than reported as 0 or N/A.
- Comparable only with other reports whose run mode line is identical.
- Model: `ollama_chat/qwen3.5:4b`
- Think: `False`
- Standard run timeout: 300.0 s
- Evaluated: 2026-08-10T22:16:52.127541+00:00
- Statuses: run 1: ok, run 2: ok, run 3: ok
- Repetition isolation: model unloaded before every run (`keep_alive=0`), so each repetition reloads the weights

**How far this range reaches.** At temperature 0 repetitions against one warm process reproduce each other exactly, so a spread of zero there is a property of the setup, not a measurement of stability. These runs reload the model between repetitions, which removes the warm-process part of that objection but still shares one machine, one server build, and one sitting.

**Measured between-session delta.** The same configuration (`qwen3.5:4b`, `--no-think`, category mode) scored 75.0% (21/28) unresolved accuracy on 2026-08-10 at 17:44 UTC and 82.1% (23/28) at 20:52 UTC — 7.1 points apart with identical flags and code. That difference is itself a measurement and outranks any within-session range printed below.

A range that overlaps another mode's range is not evidence of a difference between the modes.

**Every repetition in this batch produced identical values.** Reloading the model between runs did not reproduce the between-session delta either, so whatever varies is not the warm process alone — most likely load-time conditions this harness does not control. The range stays a lower bound.

| Metric | run 1 | run 2 | run 3 | Range (cold-start runs) |
|---|---:|---:|---:|---:|
| Category accuracy, unresolved files (`_ToReview` accepted) | 82.1% (23/28) | 82.1% (23/28) | 82.1% (23/28) | identical |
| Decision rate, unresolved files | 78.6% (22/28) | 78.6% (22/28) | 78.6% (22/28) | identical |
| Accuracy on decided files only | 77.3% (17/22) | 77.3% (17/22) | 77.3% (17/22) | identical |
| Files omitted by the model | 1/28 | 1/28 | 1/28 | identical |
| Classification latency | 32.2 s | 32.4 s | 32.5 s | 32.2467 – 32.5078 |

Individual reports:

- [`mode-categories-no-think-qwen3.5-4b-x3-run1.md`](mode-categories-no-think-qwen3.5-4b-x3-run1.md)
- [`mode-categories-no-think-qwen3.5-4b-x3-run2.md`](mode-categories-no-think-qwen3.5-4b-x3-run2.md)
- [`mode-categories-no-think-qwen3.5-4b-x3-run3.md`](mode-categories-no-think-qwen3.5-4b-x3-run3.md)
