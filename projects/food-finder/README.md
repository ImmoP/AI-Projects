# Food Finder

Food Finder is a small [smolagents](https://github.com/huggingface/smolagents)
agent that turns a natural-language restaurant request into a Google Places API
(New) Text Search. It filters weakly reviewed results, sorts the remaining
restaurants by rating and review count, and asks the model to present a useful
shortlist without inventing venue data.

## Architecture

```text
User task
    |
    v
CodeAgent  --->  configured LiteLLM model (Ollama, OpenRouter, Groq, ...)
    |
    v
search_restaurants tool
    |
    v
Google Places API (New) Text Search
```

`build_agent(model)` accepts any smolagents-compatible model. Provider setup is
therefore kept at the application boundary instead of being hard-coded into the
agent.

## Setup in three steps

1. Create an environment and install the dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Create your local configuration:

   ```bash
   cp .env.example .env
   ```

   Add a Google Places API key to `.env`. The supplied model defaults target
   Ollama at `http://localhost:11434`; both `MODEL_ID` and `API_BASE` can be
   changed. LiteLLM also reads provider credentials such as
   `OPENROUTER_API_KEY` or `GROQ_API_KEY` from the environment.

3. Make sure the selected model is available, then run a request:

   ```bash
   ollama pull qwen3.5:4b
   python app.py "Find the five best Italian restaurants in Frankfurt am Main."
   ```

## Example

```console
$ python app.py "Find highly rated Japanese restaurants in Berlin with at least 100 reviews."
1. Restaurant Example — 4.7 (1,240 reviews)
   Example Street 1, Berlin · PRICE_LEVEL_MODERATE
   https://maps.google.com/...

Selected because it combines a high rating with a large, reliable review base.
```

The exact output changes with Google Places data and the configured model.

Launch the optional browser interface with:

```bash
python app.py --gradio
```

## MCP server

The same `search_restaurants` implementation is exposed as a Model Context
Protocol tool over stdio. Start it directly with:

```bash
python mcp_server.py
```

For Claude Desktop, add the following entry to `claude_desktop_config.json` and
replace `/absolute/path/to/Food_Finder` plus the API-key placeholder:

```json
{
  "mcpServers": {
    "food-finder": {
      "command": "/absolute/path/to/Food_Finder/.venv/bin/python",
      "args": [
        "/absolute/path/to/Food_Finder/mcp_server.py"
      ],
      "env": {
        "GOOGLE_PLACES_API_KEY": "your_google_places_api_key"
      }
    }
  }
}
```

The server contains only a protocol adapter; restaurant lookup, filtering,
sorting, logging, and error handling remain in `src/agent/tools.py`.

## Demo

<!-- Replace this placeholder with docs/food-finder-demo.gif. -->

_Demo GIF coming soon._

## Tests

- **Smoke** (`pytest -m smoke`, or `python scripts/smoke.py` from the repo
  root): does the pipeline run? The offline scripted-agent case
  (`test_evals.py`) and the MCP adapter delegation check
  (`test_mcp_server.py`) -- the two closest things this project has to a
  full request-to-response run, both without a real Google Places call or
  a real model.
- **Unit** (`pytest`, the full 8-test suite): is the logic correct?
  `test_tools.py` mocks `requests.post` (no real Google Places API call),
  `test_evals.py` uses a scripted fake `smolagents` `Model` (no real LLM),
  and `test_mcp_server.py` mocks the shared tool -- the whole suite runs
  offline, no API key needed.
- **Eval** (not in CI, needs a real model/API): how good are the tool
  calls? See [Evaluation](#evaluation) below.

```bash
pytest -m smoke   # fast end-to-end paths only
pytest             # the full suite (still fully offline)
```

## Evaluation

`evals/cases.yaml` contains 20 German and English requests, including normal
city/cuisine searches and edge cases such as an unknown city, expected zero
results, an ambiguous request, and a request without a cuisine. Each case
declares whether a tool call is needed plus the expected tool and arguments.

Run the suite against the model configured in `.env` without spending Google
Places quota:

```bash
python evals/run_evals.py --offline
```

Use another LiteLLM model or endpoint explicitly:

```bash
python evals/run_evals.py \
  --offline \
  --model-id openrouter/openai/gpt-4.1-mini \
  --api-base https://openrouter.ai/api/v1
```

Reports are written as Markdown tables under `evals/results/`. The runner reads
the generated Python actions from `agent.memory`, extracts literal
`search_restaurants(...)` calls with Python's AST, and compares their arguments
deterministically. It does not use an LLM as a judge. `--offline` replaces
`requests.post` with a local fixture for the duration of each case; model calls
still use the configured model.

Illustrative comparison (replace with results from your own environment):

| Model | Tool-call rate | Argument accuracy | Avg. steps | Avg. latency | Error rate |
|---|---:|---:|---:|---:|---:|
| `ollama_chat/qwen3.5:4b` | 85.0% | 81.3% | 2.10 | 3.42 s | 0.0% |
| `openrouter/openai/gpt-4.1-mini` | 80.0% | 93.8% | 1.45 | 1.18 s | 0.0% |

These values are examples only, not benchmark claims.

## Limitations

- Google Places coverage, ratings, and prices can be incomplete or change over time.
- Text Search returns at most 20 places in the current implementation.
- The model still controls tool arguments and presentation quality.
- A Google Places API key and a reachable local or hosted model are required for live use.
