# tidy-agent evaluation — repeated runs

- Runs: **3**
- Run mode: **grouping** (`--group`, content reading `False`)
- Metric family: **clustering metrics + category accuracy for ungrouped files**
- Comparable only with other reports whose run mode line is identical.
- Model: `ollama_chat/qwen3.5:4b`
- Think: `False`
- Standard run timeout: 300.0 s
- Clustering run timeout: 900.0 s
- Evaluated: 2026-08-10T22:13:48.991353+00:00
- Statuses: run 1: ok, run 2: ok, run 3: ok
- Repetition isolation: model unloaded before every run (`keep_alive=0`), so each repetition reloads the weights

**How far this range reaches.** At temperature 0 repetitions against one warm process reproduce each other exactly, so a spread of zero there is a property of the setup, not a measurement of stability. These runs reload the model between repetitions, which removes the warm-process part of that objection but still shares one machine, one server build, and one sitting.

**Measured between-session delta.** The same configuration (`qwen3.5:4b`, `--no-think`, category mode) scored 75.0% (21/28) unresolved accuracy on 2026-08-10 at 17:44 UTC and 82.1% (23/28) at 20:52 UTC — 7.1 points apart with identical flags and code. That difference is itself a measurement and outranks any within-session range printed below.

A range that overlaps another mode's range is not evidence of a difference between the modes.

**Every repetition in this batch produced identical values.** Reloading the model between runs did not reproduce the between-session delta either, so whatever varies is not the warm process alone — most likely load-time conditions this harness does not control. The range stays a lower bound.

| Metric | run 1 | run 2 | run 3 | Range (cold-start runs) |
|---|---:|---:|---:|---:|
| Clustering purity, files in group folders | 100.0% (15/15) | 100.0% (15/15) | 100.0% (15/15) | identical |
| Fully co-located expected groups | 100.0% (3/3) | 100.0% (3/3) | 100.0% (3/3) | identical |
| Ground-truth files placed in a group folder | 15/21 | 15/21 | 15/21 | identical |
| Scatter files in an accepted group folder | 0/6 | 0/6 | 0/6 | identical |
| Scatter files in a proposed cluster | 0/6 | 0/6 | 0/6 | identical |
| Category accuracy, unresolved files (`_ToReview` accepted) | 91.7% (22/24) | 91.7% (22/24) | 91.7% (22/24) | identical |
| Decision rate, unresolved files | 62.5% (15/24) | 62.5% (15/24) | 62.5% (15/24) | identical |
| Accuracy on decided files only | 86.7% (13/15) | 86.7% (13/15) | 86.7% (13/15) | identical |
| Files omitted by the model | 1/24 | 1/24 | 1/24 | identical |
| Classification latency | 26.8 s | 26.9 s | 27.0 s | 26.7856 – 26.9656 |

Individual reports:

- [`mode-grouping-no-think-qwen3.5-4b-x3-run1.md`](mode-grouping-no-think-qwen3.5-4b-x3-run1.md)
- [`mode-grouping-no-think-qwen3.5-4b-x3-run2.md`](mode-grouping-no-think-qwen3.5-4b-x3-run2.md)
- [`mode-grouping-no-think-qwen3.5-4b-x3-run3.md`](mode-grouping-no-think-qwen3.5-4b-x3-run3.md)
