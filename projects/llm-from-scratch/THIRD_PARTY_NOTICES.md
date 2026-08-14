# Third-Party Notices

This file records the provenance and use of third-party components associated with the educational `llm-from-scratch` project. A license listed for one component does not automatically apply to other project content.

## Sebastian Raschka / LLMs-from-scratch

- Author and copyright holder: Sebastian Raschka
- Source: <https://github.com/rasbt/LLMs-from-scratch>
- Pinned revision: [`33f5b246766464910accf1c70e668811cfc4bf08`](https://github.com/rasbt/LLMs-from-scratch/tree/33f5b246766464910accf1c70e668811cfc4bf08)
- Applicable terms: the upstream repository's Apache-licensed software terms, preserved in [`licenses/RASCHKA-LLMS-FROM-SCRATCH-APACHE-2.0.txt`](./licenses/RASCHKA-LLMS-FROM-SCRATCH-APACHE-2.0.txt)
- Use here: portions of the notebook implementation are adapted from or closely follow the upstream chapter implementations. Local modifications exist.

The software license is cited only for covered repository software. It is not asserted as permission to redistribute the associated book prose or book figures. Embedded figures have been removed, and the notebook prose has been rewritten in original, project-focused language. Fresh-clone smoke testing has passed on Windows with Python 3.12.10 within the documented non-training scope.

## gpt_download.py

- Copyright: Sebastian Raschka
- Upstream path: [`ch05/01_main-chapter-code/gpt_download.py`](https://github.com/rasbt/LLMs-from-scratch/blob/33f5b246766464910accf1c70e668811cfc4bf08/ch05/01_main-chapter-code/gpt_download.py)
- Pinned revision: `33f5b246766464910accf1c70e668811cfc4bf08`
- Applicable terms: the upstream repository's Apache-licensed software terms
- Use here: downloaded at runtime from the pinned source; no helper copy is intentionally redistributed in this project

## OpenAI GPT-2

- Maintainer: OpenAI
- Official repository: <https://github.com/openai/gpt-2>
- Pinned repository revision: [`9b63575ef42771a015060c964af2c3da4cf7c8ab`](https://github.com/openai/gpt-2/tree/9b63575ef42771a015060c964af2c3da4cf7c8ab)
- Repository software license: [Modified MIT](https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/LICENSE)
- Model information: [official GPT-2 model card](https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/model_card.md)
- Use here: GPT-2 model artifacts are downloaded at runtime from the official source and are not committed to this repository

This notice does not claim a separate weight-redistribution license beyond what the official upstream materials explicitly establish.

## UCI SMS Spam Collection

- Creators: Tiago Almeida and Jos Hidalgo
- Work: *SMS Spam Collection*
- Repository: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- DOI: [10.24432/C5CC84](https://doi.org/10.24432/C5CC84)
- License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) (CC BY 4.0)
- Use here: retrieved at runtime for classification exercises; the raw dataset and derived row-level split files are not intended to be committed

Suggested citation from UCI: Almeida, T. & Hidalgo, J. (2011). *SMS Spam Collection* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5CC84>.

## The Verdict corpus

### Literary work

- Author: Edith Wharton
- Story: "The Verdict" (1908)
- Use here: a small educational text corpus for tokenization, dataloading, and language-model training experiments

### Public-domain reference

- Containing work: *The Early Short Fiction of Edith Wharton — Part 2*
- Reference: [Project Gutenberg eBook No. 306](https://www.gutenberg.org/ebooks/306)
- Gutenberg credits: produced by John Hamm and David Widger

Project Gutenberg identifies its edition as public domain in the USA; this notice does not claim worldwide public-domain status.

### Immediate transcription provenance

- Source repository: Sebastian Raschka's [`LLMs-from-scratch`](https://github.com/rasbt/LLMs-from-scratch) companion repository
- Pinned revision: `33f5b246766464910accf1c70e668811cfc4bf08`
- Upstream path: [`ch02/01_main-chapter-code/the-verdict.txt`](https://github.com/rasbt/LLMs-from-scratch/blob/33f5b246766464910accf1c70e668811cfc4bf08/ch02/01_main-chapter-code/the-verdict.txt)

This repository's local transcription has the same token sequence as that pinned companion-repository file. It is not documented as a direct Project Gutenberg download or as byte-for-byte or normalization-identical to the current Gutenberg transcription. Formatting or transcription details may differ between the local file and the current Gutenberg edition.

## Instruction dataset

- Repository: Sebastian Raschka's `LLMs-from-scratch`
- Upstream path: [`ch07/01_main-chapter-code/instruction-data.json`](https://github.com/rasbt/LLMs-from-scratch/blob/33f5b246766464910accf1c70e668811cfc4bf08/ch07/01_main-chapter-code/instruction-data.json)
- Pinned revision: `33f5b246766464910accf1c70e668811cfc4bf08`
- Use here: retrieved at runtime; no copy is redistributed in this project

The file is present in the upstream repository, but its independent artifact-level origin and licensing are not stated clearly enough to claim separately verified redistribution permission. Runtime retrieval avoids committing a local copy while preserving that caveat.

## Book acknowledgement

This educational project was developed while studying Sebastian Raschka's *Build a Large Language Model (From Scratch)*, published by Manning.

The acknowledgement is not a redistribution license. No book prose or book figures are intended to be redistributed in the remediated public version. The notebook's explanatory prose has completed its original project-focused rewrite; fresh-clone smoke testing has passed on Windows with Python 3.12.10 within the documented non-training scope.
