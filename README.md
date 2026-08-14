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

 Engineering depth first, then ML
substance, then applied LLM/agent work, then the explicitly educational
project last.

| Project | What it demonstrates | Core stack | Result / Evidence | Maturity |
| --- | --- | --- | --- | --- |
| [tidy-agent](./projects/tidy-agent/) | Safe LLM-assisted filesystem automation with deterministic execution, human approval and fail-closed controls. | Python, smolagents/LiteLLM, Ollama, structured outputs, atomic filesystem operations. | Cross-platform CI on Ubuntu, macOS and Windows across Python 3.10, 3.12 and 3.13; fresh classifier holdout pending. | Applied / portfolio-complete implementation. |
| [email-spam-detector](./projects/email-spam-detector/) | Transformer fine-tuning combined with temporal sender/domain reputation and email-security feature engineering. | PyTorch, GPT-2, SPF/DKIM/DMARC features, logistic regression. | Public test: 98.93% accuracy, 98.57% spam precision, 99.13% spam recall (n=20,304). | Applied research / active development. |
| [rag-system](./projects/rag-system/) | Classical, hybrid/reranked and agentic RAG with a shared evaluation framework. | Python, sentence-transformers/BGE, BM25, cross-encoder reranking, Ollama, ReAct. | Comparative benchmark framework implemented; final evaluation in progress. | Applied / learning project. |
| [food-finder](./projects/food-finder/) | Tool-using LLM agent over an external search API. | smolagents, LiteLLM, Google Places API, MCP, Gradio. | Working CLI, MCP tool and Gradio interface. | Applied demo. |
| [llm-from-scratch](./projects/llm-from-scratch/) | Transformer and GPT fundamentals implemented explicitly for learning. | PyTorch, tokenization, transformer architecture, GPT-2 weight loading. | Tokenization → GPT-style model → GPT-2 weight loading → classifier fine-tuning. | Educational implementation. |

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
- **Projects vary in maturity.** Some are applied/active work, one is
  explicitly a learning project following a published book. Each README
  states its own status; none of these projects should be assumed
  production-ready.
- **Private data is never committed.** Raw mailbox exports, private
  evaluation outputs, and trained model checkpoints are excluded from Git
  throughout; see the affected projects' own READMEs for their specific
  data-handling notes.

## Status

This is an evolving portfolio. Projects range from applied/active work to
explicitly educational exercises and should not be assumed production-ready.
