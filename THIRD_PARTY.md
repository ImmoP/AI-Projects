# Third-Party Notices

[LICENSE](./LICENSE) (MIT) covers this repository's original code. It does not
extend to the third-party material listed below, which is used under its own
license or terms. Referencing a component here is not a claim of ownership
over it.

## Sebastian Raschka — *Build a Large Language Model (From Scratch)* and `LLMs-from-scratch`

- Used in: [`llm-from-scratch`](./projects/llm-from-scratch/)
- The `llm-from-scratch` project was developed while studying Raschka's book
  (Manning) and adapts portions of its companion repository,
  [`rasbt/LLMs-from-scratch`](https://github.com/rasbt/LLMs-from-scratch),
  pinned to a specific revision for provenance.
- The companion repository's code is Apache-2.0 licensed; a scoped copy of
  that license is kept at
  [`projects/llm-from-scratch/licenses/RASCHKA-LLMS-FROM-SCRATCH-APACHE-2.0.txt`](./projects/llm-from-scratch/licenses/RASCHKA-LLMS-FROM-SCRATCH-APACHE-2.0.txt).
- The Apache-2.0 grant covers the companion repository's software only. Book
  prose and book figures are not covered by it and are not redistributed
  here.
- Full per-file provenance (including the adapted `gpt_download.py` helper
  and the pinned training-corpus excerpt) is in
  [`projects/llm-from-scratch/THIRD_PARTY_NOTICES.md`](./projects/llm-from-scratch/THIRD_PARTY_NOTICES.md).

## OpenAI GPT-2 (pretrained weights)

- Used in: [`llm-from-scratch`](./projects/llm-from-scratch/) and
  [`email-spam-detector`](./projects/email-spam-detector/)
- Source: [`openai/gpt-2`](https://github.com/openai/gpt-2)
- License: [Modified MIT](https://github.com/openai/gpt-2/blob/master/LICENSE)
- Both projects download the pretrained GPT-2 124M weights at runtime from
  the official source; the weights are not committed to this repository.

## smolagents

- Used in: [`tidy-agent`](./projects/tidy-agent/) and
  [`food-finder`](./projects/food-finder/)
- Source: [`huggingface/smolagents`](https://github.com/huggingface/smolagents)
- License: Apache-2.0
- Used as a library dependency (agent loop and tool-calling), unmodified.

## Ollama

- Used by (as an external, separately installed runtime, not vendored): 
  [`tidy-agent`](./projects/tidy-agent/), [`food-finder`](./projects/food-finder/),
  and [`rag-system`](./projects/rag-system/), for local model inference.
- Source: [`ollama/ollama`](https://github.com/ollama/ollama)
- License: MIT
- Not included in this repository; installed and run separately by whoever
  runs these projects.

## Other project-level notices

Additional attributions specific to one project (for example the UCI SMS
Spam Collection dataset and the public-domain source of the `llm-from-scratch`
training excerpt) are documented where they apply, not duplicated here:

- [`projects/llm-from-scratch/THIRD_PARTY_NOTICES.md`](./projects/llm-from-scratch/THIRD_PARTY_NOTICES.md)
