# AI / Machine Learning Portfolio


Contact: [@ImmoP](https://github.com/ImmoP) on GitHub, or
[Immo.Primus@gmx.de](mailto:Immo.Primus@gmx.de).

[![CI](https://github.com/ImmoP/AI/actions/workflows/ci.yml/badge.svg)](https://github.com/ImmoP/AI/actions/workflows/ci.yml)
[![Python 3.10 | 3.12 | 3.13](https://img.shields.io/badge/python-3.10%20%7C%203.12%20%7C%203.13-blue)](.github/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

## Quickstart

```bash
git clone https://github.com/ImmoP/AI.git
cd AI
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

Sorted to serve both roles I'm open to: engineering depth first, then ML
substance, then applied LLM/agent work, then the explicitly educational
project last.

| Project | Goal | Technologies | Status | Data availability | Docs |
| --- | --- | --- | --- | --- | --- |
| [tidy-agent](./projects/tidy-agent/) | Human-approved directory organizer: an LLM proposes a file-move plan, a separate deterministic executor validates and applies it. | Python, smolagents/LiteLLM (local Ollama model), structured classification, atomic no-clobber filesystem operations | Implementation complete for portfolio scope; evaluation closed | Self-contained — synthetic dev/eval fixtures are committed, no private data | [README](./projects/tidy-agent/README.md) |
| [email-spam-detector](./projects/email-spam-detector/) | Classify email spam/phishing by combining a fine-tuned language model with sender/domain reputation and email-authentication signals. | PyTorch, fine-tuned GPT-2, SPF/DKIM/DMARC feature extraction, logistic-regression fusion | Applied research / active development | Private mailbox data not distributed; public HF/UCI evaluation data referenced | [README](./projects/email-spam-detector/README.md) |
| [rag-system](./projects/rag-system/) | Retrieval-augmented generation built from scratch — classical and agentic (ReAct) retrieval over a PDF corpus, compared side by side. | Manual chunking/embedding/retrieval, BM25 + dense hybrid search, cross-encoder reranking, Ollama generation | Applied / learning project, benchmark framework in progress | 19 arXiv papers included; Open RAGBench benchmark data fetched separately | [README](./projects/rag-system/README.md) |
| [food-finder](./projects/food-finder/) | Natural-language restaurant search agent over the Google Places API, exposed as both a CLI and an MCP tool. | smolagents `CodeAgent`, LiteLLM (Ollama/OpenRouter/Groq), Google Places API, Gradio UI | Applied / demo | No private data; requires a user-supplied Google Places API key | [README](./projects/food-finder/README.md) |
| [llm-from-scratch](./projects/llm-from-scratch/) | **Learning project**, implemented while working through Sebastian Raschka's *Build a Large Language Model (From Scratch)*: a GPT-style model built up from tokenization through pretraining and fine-tuning. | PyTorch, tokenization, transformer architecture, GPT-2 weight loading | Educational — not an original-research contribution | Small learning corpus included; pretrained GPT-2 weights and UCI dataset downloaded at runtime | [README](./projects/llm-from-scratch/README.md) |

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
