# LLM from Scratch

## Overview

This is an educational implementation project for studying GPT-style language models in PyTorch, from tokenization and transformer components through language-model training and task-specific fine-tuning.

Substantial portions of the implementation are adapted from or closely follow Sebastian Raschka's official [`LLMs-from-scratch`](https://github.com/rasbt/LLMs-from-scratch) repository, pinned for provenance to revision [`33f5b246766464910accf1c70e668811cfc4bf08`](https://github.com/rasbt/LLMs-from-scratch/tree/33f5b246766464910accf1c70e668811cfc4bf08). Local modifications have been made. The software license does not cover associated book prose or figures; those materials are not redistributed here.

The notebook Markdown rewrite is complete. The project has been fresh-clone smoke tested on Windows with Python 3.12.10. This verifies the scoped setup and runtime paths described below; it does not claim that full training results were reproduced.

## Scope

The notebook explores:

- tokenization and input-target dataset construction;
- token and positional embeddings;
- self-attention, causal attention, and multi-head attention;
- layer normalization, GELU activations, feed-forward layers, and residual connections;
- transformer blocks and a GPT-style architecture;
- language-model training, evaluation, and text generation;
- loading pretrained GPT-2 weights;
- classification fine-tuning; and
- instruction fine-tuning experiments.

## Setup and reproducibility

The fresh-clone smoke test passed on Windows with Python 3.12.10. [`requirements.txt`](./requirements.txt) pins the direct dependency versions used for that test (plus `pytest`, added later for the automated test suite below); it is intentionally not a full transitive lock file. CPU execution was sufficient for the smoke-test operations, and CUDA was not required or validated.

### Tested environment

| Component | Tested version |
|---|---:|
| Python | 3.12.10 |
| torch | 2.13.0 |
| tiktoken | 0.13.0 |
| numpy | 2.5.1 |
| pandas | 3.0.5 |
| matplotlib | 3.11.1 |
| tqdm | 4.70.0 |
| tensorflow | 2.21.0 |
| psutil | 7.2.2 |
| jupyter | 1.1.1 |
| ipykernel | 7.3.0 |

From the project root, create a virtual environment:

```shell
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activate it on macOS or Linux:

```shell
source .venv/bin/activate
```

Install the tested direct dependencies and register a dedicated Jupyter kernel:

```shell
python -m pip install -r requirements.txt
python -m ipykernel install --user --name llm-from-scratch --display-name "Python 3.12 (llm-from-scratch)"
```

Launch Jupyter from this project root and open [`notebooks/llm-from-scratch.ipynb`](./notebooks/llm-from-scratch.ipynb):

```shell
jupyter notebook notebooks/llm-from-scratch.ipynb
```

Select the `Python 3.12 (llm-from-scratch)` kernel in Jupyter. The notebook also resolves paths when launched from the repository root or notebook directory, but the project root is the canonical launch location.

### Smoke-test scope

**Verified:** dependency installation and imports; project path handling and runtime isolation; corpus access and tokenization; architecture forward paths; pretrained GPT-2 124M checkpoint loading; UCI dataset preparation and a classification batch forward path; and the instruction dataset, collator, and model forward path.

**Not verified by this smoke test:** full language-model training, full classification fine-tuning, full instruction fine-tuning, final training metrics, CUDA-specific behavior, Apple MPS behavior, or enabled Ollama evaluation.

Full training and fine-tuning were intentionally excluded. Ollama evaluation remains optional and disabled by default.

## Tests

The notebook's core building blocks (tokenizer wrapper, sliding-window dataset, GPT model, training-step loss) live in [`src/llm_from_scratch/`](./src/llm_from_scratch/) so the notebook and an automated test suite share one definition instead of two copies drifting apart.

- **Smoke** (`pytest -m smoke`, or `python scripts/smoke.py` from the repo root): does the pipeline run at all? Tokenizer encode/decode roundtrip, a forward pass through a tiny (16-dim, 2-layer) randomly initialized `GPTModel` with the expected output shape, and two AdamW training steps on that tiny model checked for a finite (non-NaN) loss. Seed fixed, CPU-only, no downloads, runs in well under 30 seconds. These checks say nothing about model quality; they only confirm that the code paths execute correctly.
- **Unit**: covered by the same smoke-marked tests above. This project's test layer doesn't currently separate the two tiers further.
- **Eval** (not in CI): actual pretraining, fine-tuning, and generation quality live in the notebook itself and need the real GPT-2 124M checkpoint, the full corpus, and non-trivial compute. See "Smoke-test scope" above for what the notebook's own fresh-clone run covers.

Run the automated tests from this project root:

```shell
pytest -m smoke   # same as the full suite here, but explicit
pytest             # the full suite
```

`tests/fixtures/mini_corpus.txt` (~18 KB) is an original short story written specifically for this test suite, not sourced from Raschka's book or any other text. It exists only to give the tokenizer and dataloader tests real, varied prose to chunk and encode.

## Runtime paths and data

The tracked corpus remains at [`data/the-verdict.txt`](./data/the-verdict.txt). All downloads, derived datasets, checkpoints, plots, and generated responses are routed below the project-local `.runtime/` directory, which is ignored by Git:

```text
.runtime/
|-- downloads/
|-- gpt2/
|-- datasets/
|-- checkpoints/
|-- plots/
`-- generated/
```

Runtime artifacts include the pinned GPT-2 helper, original OpenAI GPT-2 files, the UCI SMS Spam Collection and derived CSV splits, the pinned instruction dataset, model checkpoints, PDF plots, and generated instruction responses. They are downloaded or generated as the relevant cells run and must not be committed.

The pretrained workflow selects GPT-2 `124M`. TensorFlow is required only to read the original TensorFlow checkpoint format before parameters are mapped into the PyTorch model. The GPT-2 download is a substantial external artifact.

Classification uses the UCI SMS Spam Collection. Instruction data is downloaded at runtime from the pinned Raschka revision. The notebook records SHA-256 expectations for the pinned helper and instruction dataset so cached copies are identified before use.

## Compute and optional sections

PyTorch falls back to CPU when CUDA is unavailable; CUDA is optional. Small demonstrations are comparatively lightweight, but full language-model training, GPT-2 download/loading, classification fine-tuning, instruction fine-tuning, and full response generation are expensive sections. Training or fine-tuning the 124M model can be impractically slow on CPU, and this project makes no runtime guarantee for full training there.

Automated response evaluation is optional post-processing. It defaults to disabled through `RUN_EXTERNAL_EVALUATION = False`, so the core notebook does not require Ollama. Enabling it requires a separately installed and running local Ollama service plus the configured evaluator model; the notebook neither installs Ollama nor pulls a model automatically. Ollama requests remain local to `http://localhost:11434`.

## Project structure

```text
llm-from-scratch/
|-- .gitignore
|-- README.md
|-- pyproject.toml
|-- requirements.txt
|-- THIRD_PARTY_NOTICES.md
|-- data/
|   |-- README.md
|   `-- the-verdict.txt
|-- licenses/
|   `-- RASCHKA-LLMS-FROM-SCRATCH-APACHE-2.0.txt
|-- notebooks/
|   `-- llm-from-scratch.ipynb
|-- src/
|   `-- llm_from_scratch/
|       |-- config.py
|       |-- data.py
|       |-- model.py
|       |-- tokenizer.py
|       `-- training.py
`-- tests/
    |-- conftest.py
    |-- fixtures/
    |   `-- mini_corpus.txt
    |-- test_model_forward.py
    |-- test_tokenizer.py
    `-- test_training_step.py
```

## Notebook and report status

The notebook is the main implementation artifact. Embedded images and stored execution outputs were removed for public-release hygiene, and its explanatory Markdown has been rewritten in original, project-focused language. Its installation and non-training runtime paths are fresh-clone smoke tested within the Windows/Python 3.12.10 scope documented above; expensive training and external evaluation remain outside that verified scope.

The previous PDF chapter summary was removed from the public tree. A future report may replace it only with an original implementation report focused on this project's design, experiments, and independently created results.

## Attribution and licensing

Detailed provenance, pinned revisions, dataset attribution, and runtime-use boundaries are documented in [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md).

The scoped copy of the license applying to Raschka-derived software is stored at [`licenses/RASCHKA-LLMS-FROM-SCRATCH-APACHE-2.0.txt`](./licenses/RASCHKA-LLMS-FROM-SCRATCH-APACHE-2.0.txt). It applies to the relevant third-party software, not automatically to the rest of this portfolio repository.

No blanket project license is declared here.
