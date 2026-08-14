# tidy-agent evaluation — repeated runs

- Runs: **3**
- Run mode: **grouping** (`--group`, content reading `False`)
- Metric family: **clustering metrics + category accuracy for ungrouped files**
- Comparable only with other reports whose run mode line is identical.
- Model: `ollama_chat/qwen3.5:4b`
- Think: `False`
- Standard run timeout: 900.0 s
- Clustering run timeout: 1200.0 s
- Ground truth: `expected.yaml sha256:dfe0b35b34c6`
- Endpoint: `remote host on a private network (address redacted)`
- Model digest: `2a654d98e6fba55d`
- Quantization: `Q4_K_M`
- Parameter size: `4.7B`
- Context length: `8192`
- Model placement: `GPU/VRAM 3341088193 bytes`
- Evaluated: 2026-08-11T13:04:26.542488+00:00
- Statuses: run 1: ok, run 2: ok, run 3: ok
- Repetition isolation: model unloaded before every run (`keep_alive=0`), so each repetition reloads the weights

**How far this range reaches.** At temperature 0 repetitions against one warm process reproduce each other exactly, so a spread of zero there is a property of the setup, not a measurement of stability. These runs reload the model between repetitions, which removes the warm-process part of that objection but still shares one machine, one server build, and one sitting.

**Measured between-session delta.** The same configuration (`qwen3.5:4b`, `--no-think`, category mode) scored 75.0% (21/28) unresolved accuracy on 2026-08-10 at 17:44 UTC and 82.1% (23/28) at 20:52 UTC — 7.1 points apart with identical flags and code. That difference is itself a measurement and outranks any within-session range printed below.

A range that overlaps another mode's range is not evidence of a difference between the modes.

**Every repetition in this batch produced identical values.** Reloading the model between runs did not reproduce the between-session delta either, so whatever varies is not the warm process alone — most likely load-time conditions this harness does not control. The range stays a lower bound.

| Metric | run 1 | run 2 | run 3 | Range (cold-start runs) |
|---|---:|---:|---:|---:|
| Clustering purity, files in group folders | 100.0% (11/11) | 100.0% (11/11) | 100.0% (11/11) | identical |
| Fully co-located expected groups | 33.3% (1/3) | 33.3% (1/3) | 33.3% (1/3) | identical |
| Ground-truth files placed in a group folder | 11/21 | 11/21 | 11/21 | identical |
| Scatter files in an accepted group folder | 0/6 | 0/6 | 0/6 | identical |
| Scatter files in a proposed cluster | 0/6 | 0/6 | 0/6 | identical |
| Category accuracy, unresolved files (`_ToReview` accepted) | 93.3% (28/30) | 93.3% (28/30) | 93.3% (28/30) | identical |
| Decision rate, unresolved files | 60.0% (18/30) | 60.0% (18/30) | 60.0% (18/30) | identical |
| Accuracy on decided files only | 88.9% (16/18) | 88.9% (16/18) | 88.9% (16/18) | identical |
| Files omitted by the model | 1/30 | 1/30 | 1/30 | identical |
| Classification latency | 27.3 s | 27.2 s | 27.1 s | 27.0938 – 27.3073 |

Individual reports:

- [`quadv2-b-grouping-only-run1.md`](quadv2-b-grouping-only-run1.md)
- [`quadv2-b-grouping-only-run2.md`](quadv2-b-grouping-only-run2.md)
- [`quadv2-b-grouping-only-run3.md`](quadv2-b-grouping-only-run3.md)
