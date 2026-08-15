# AI / Machine Learning Portfolio


Contact: [@ImmoP](https://github.com/ImmoP) on GitHub, or
[Immo.Primus@gmx.de](mailto:Immo.Primus@gmx.de).

[![CI](https://github.com/ImmoP/AI-Projects/actions/workflows/ci.yml/badge.svg)](https://github.com/ImmoP/AI-Projects/actions/workflows/ci.yml)
[![Python 3.10 | 3.12 | 3.13](https://img.shields.io/badge/python-3.10%20%7C%203.12%20%7C%203.13-blue)](.github/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

## Quickstart

```bash
git clone https://github.com/ImmoP/AI-Projects.git
cd AI-Projects
python scripts/smoke.py
```

**Prerequisites:** Python 3.12+ on `PATH` (the strictest per-project
requirement). [`uv`](https://docs.astral.sh/uv/) is used automatically if
it's on `PATH` (faster installs); otherwise the script falls back to
`python -m venv` + `pip`. Nothing else needs to be installed first — the
script sets up each project's own isolated environment for you.

**First run:** installs all five projects' dependencies into isolated,
gitignored environments under `.cache/smoke-envs/` — two projects install
`torch`, one also installs `tensorflow` — which takes **about 9 minutes**.
Later runs reuse those environments and finish in well under a minute. Pass
`--no-install` to skip straight to testing once the environments already
exist, or run `python scripts/smoke.py <project-name>` to check a single
project. Nothing is installed into your own Python interpreter, and none of
the smoke tests themselves make network calls.

## Projects

The projects below are ordered by engineering depth first, then by ML
substance, then by applied LLM/agent work, with the explicitly educational
project listed last.

| Project | What it demonstrates | Core stack | Result / Evidence | Maturity |
| --- | --- | --- | --- | --- |
| [tidy-agent](./projects/tidy-agent/) | Safe LLM-assisted filesystem automation with deterministic execution, human approval and fail-closed controls. | Python, smolagents/LiteLLM, Ollama, structured outputs, atomic filesystem operations. | Development calibration: 0/47 incorrect automatic classifications (vs. 14/47 baseline), 21/47 automated. Independent Holdout v4 (n=150) ran once and was invalidated by a schema-contract failure on one response — `evaluation_valid = false`, no accuracy number is reported. Holdout v5, against a revised candidate, is planned but not yet run. | Applied / portfolio-complete implementation. |
| [email-spam-detector](./projects/email-spam-detector/) | Transformer fine-tuning combined with temporal sender/domain reputation and email-security feature engineering. | PyTorch, GPT-2, SPF/DKIM/DMARC features, logistic regression. | Public test: 98.93% accuracy, 98.57% spam precision, 99.13% spam recall (n=20,304). | Applied research / active development. |
| [rag-system](./projects/rag-system/) | Four-way RAG benchmark comparing dense retrieval, hybrid BM25+dense RRF, cross-encoder reranking, and a ReAct agentic retrieval loop on Open RAGBench. | Python, SentenceTransformers/BGE, BM25/RRF, cross-encoder reranking, Ollama, ReAct. | Full-corpus benchmark (n=100 queries / 1,000 documents, vs. Naive-dense baseline): Hybrid+Reranker reached 99% document Hit@1 (near-ceiling — Hit@5 = 100% at this n) and 81% section Hit@1; Hybrid gave the strongest quality/latency trade-off. Single-machine consistency across variants is not documented. | Applied / completed comparative benchmark. |
| [food-finder](./projects/food-finder/) | Tool-using LLM agent over an external search API. | smolagents, LiteLLM, Google Places API, MCP, Gradio. | Demo — working CLI, MCP tool and Gradio interface; no benchmarked result. | Applied demo. |
| [llm-from-scratch](./projects/llm-from-scratch/) | Transformer and GPT fundamentals implemented explicitly for learning. | PyTorch, tokenization, transformer architecture, GPT-2 weight loading. | Learning project — tokenization → GPT-style model → GPT-2 weight loading → classifier fine-tuning; no benchmarked result. | Educational implementation. |

## What it looks like

`tidy-agent`'s dry-run output — every run previews a plan before anything
touches disk, and nothing here has moved yet:

```text
[_ToReview]
ORIGIN  SOURCE        DESTINATION             REASON
------  ------------  ----------------------  ------------------------------------------
rule    mystery_file  _ToReview/mystery_file  No matching extension rule; agent disabled

[Documents]
ORIGIN  SOURCE       DESTINATION            REASON
------  -----------  ---------------------  -------------------------------------
rule    invoice.pdf  Documents/invoice.pdf  Matched extension rule for Documents/

Dry-run complete: planned=2
```

Full walkthrough, including the approval step and undo, in
[tidy-agent's README](./projects/tidy-agent/README.md#demo).

## Testing

Each project has its own layered test suite (**smoke** / **unit** / **eval**
— see each project's README for the split). `python scripts/smoke.py` (see
[Quickstart](#quickstart)) is the one cross-project command that checks all
five pipelines run at all, straight after a fresh clone.

CI (`.github/workflows/ci.yml`) runs the fuller `pytest -m "not slow"` per
project directly (no `scripts/smoke.py` involved there), scoped with path
filtering so a change in one project doesn't retrigger every other
project's suite. `tidy-agent`'s suite additionally runs across a 3 OS ×
3 Python-version matrix (Ubuntu, macOS, Windows; Python 3.10, 3.12, 3.13);
the other four projects run once each on Ubuntu with Python 3.12, plus one
repository-wide hygiene job — 14 jobs in total.

## Licensing and Attribution

Original code in this repository is MIT licensed — see [LICENSE](./LICENSE).
Third-party material — Sebastian Raschka's book and companion code,
pretrained GPT-2 weights, `smolagents`, and Ollama — is used under its own
license and is not relicensed by this repository; see
[THIRD_PARTY.md](./THIRD_PARTY.md) for the full list, and each project's own
README/notices for anything project-specific.

## Engineering Notes

[docs/engineering-notes.md](./docs/engineering-notes.md) documents two
cross-platform issues found while getting CI green on Windows (an NTFS
Alternate Data Streams checkout failure, and a `pathlib` absoluteness
difference that made one test platform-dependent), each as symptom /
diagnosis / solution.

## Limitations

- **No claim of model quality is made by the smoke-test layer.** `python
  scripts/smoke.py` and CI verify that each pipeline *runs* — imports
  resolve, a forward pass completes, a plan validates — not that any model
  is accurate. Accuracy and quality numbers, where reported, live in each
  project's own README and are scoped to that project's own evaluation
  setup, not to this repository as a whole.
- **The repository as a whole is not guaranteed reproducible from a fresh
  clone.** Some `email-spam-detector` workflows require private mailbox
  data and locally trained model artifacts that are intentionally excluded
  from Git; see that project's README for what is and isn't reproducible.
- **`tidy-agent` cannot mutate the filesystem on Windows.** A pre-move
  identity check the executor relies on was found to be unreliable on NTFS,
  so mutation is refused there rather than run with an unverified safety
  guarantee; dry-run is unaffected. See that project's README and
  [docs/engineering-notes.md](./docs/engineering-notes.md).
- **Projects vary in maturity.** This is an evolving portfolio: some are
  applied/active work, one is explicitly a learning project following a
  published book. Each README states its own status; none of these projects
  should be assumed production-ready.
- **Private data is never committed.** Raw mailbox exports, private
  evaluation outputs, and trained model checkpoints are excluded from Git
  throughout; see the affected projects' own READMEs for their specific
  data-handling notes.
