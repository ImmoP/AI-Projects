# Portfolio audit — open items

Generated during the 2026-08-15 documentation audit (root README, tidy-agent,
rag-system, email-spam-detector). These are the things that need a human
decision or actual new work (a run, a screenshot, an authored fixture) —
nothing here was resolved by editing text.

## rag-system

- **No execution-environment record for the full-corpus benchmark.**
  `benchmark/results/summary.json` and the per-query JSONL files contain no
  device, OS, or Ollama-version field, so the latency comparisons in the
  README (Naive 5.29s vs. Reranker 8.87s vs. Agentic 8.44s) can't be
  confirmed to have run under identical hardware. Proposed fix for the next
  benchmark run: write a small `benchmark/results/environment.json` at
  invocation time, modeled on the fields tidy-agent already tracks in
  `evals/results/*/lifecycle.json` / `manifest.json`:

  ```json
  {
    "run_id": "...",
    "timestamp": "...",
    "os": "...",
    "python_version": "...",
    "device": "cpu | cuda:<name> | mps",
    "ollama_version": "...",
    "generator_model": "qwen2.5:3b",
    "generator_model_digest": "...",
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "same_machine_all_systems": true
  }
  ```

  Until this exists, the README's hardware caveat (added in this audit)
  should stay in place rather than be removed.

- **`benchmark/results/summary.json` is now trackable, but not staged.**
  `.gitignore:7` was narrowed from `benchmark/results/` to
  `benchmark/results/*.jsonl` and `benchmark/results/cache/`, so
  `summary.json` will show as untracked in `git status` — see the status
  output at the end of this session. You still need to `git add` it
  yourself; nothing was staged by this audit. The raw per-query JSONL files
  and the embedding cache remain ignored (they're large and reproducible
  from `--summarize-existing`); if you'd rather have the JSONLs in Git too
  for full auditability, that's a separate decision — say so and it's a
  one-line `.gitignore` change.

- **No screenshot or terminal capture exists for rag-system** (or for
  food-finder, email-spam-detector, llm-from-scratch). Only tidy-agent now
  has a text-based demo excerpt, in the root README. A real screenshot of a
  benchmark run or a CLI answer is still open — flagged, not fabricated.

- **Exact HF source dataset isn't named.** email-spam-detector's public test
  split is labeled generically "Public/Hugging Face test"
  (`data/hf_test_prepared.parquet`, built by
  `src/spam_detector/data_processing/prepare_hf_data.py` from a `test.parquet`
  it doesn't itself name). The numbers check out against
  `results/evaluation_results.json` exactly, so nothing was changed — but if
  you want the dataset citable, you'll need to name the actual HF dataset
  repo/version yourself; it isn't recorded anywhere in the code I found.

## tidy-agent

- **Holdout v5 has no fixture or ground truth yet.** `evals/holdout_v5/`
  currently holds only infrastructure (`README.md`, `blind_audit.py`); the
  runner refuses to execute until a clean-room author — someone with no
  access to any prior Development/Holdout material — produces `fixture/`,
  `ground_truth.json`, and the other frozen artifacts listed in that
  README. This is explicitly not something I can do (the authoring boundary
  requires a person/process that hasn't seen the historical eval data,
  which I have, from this session and prior ones).

## Open editorial call I did not make unilaterally

- None outstanding — the one open call from Phase 1 (whether to commit
  `summary.json`) was resolved by you mid-session; see above.
