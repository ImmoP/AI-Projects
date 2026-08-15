"""
Regression tests proving that benchmark CLI tests write *only* to a pytest
temporary directory and never corrupt the real ``benchmark/results``.

Root cause of the original corruption
======================================
Some tests called ``evaluate_rag.main(...)`` while mocking ``save_jsonl``
and ``summarize`` but did **not** supply a temporary ``--out-dir``.
``evaluate_rag.main()`` writes ``summary.json`` independently via::

    with open(out_dir / "summary.json", "w", ...) as fh:
        json.dump(...)

Therefore mocking ``save_jsonl()`` was NOT sufficient to isolate filesystem
writes.  The default ``--out-dir`` is ``config.RESULTS_DIR``, so a test run
overwrote the real ``benchmark/results/summary.json`` with a bogus
test-generated summary (``startup_time_seconds: 0.0006``, empty systems).

These tests prove the fix: every ``main()`` call that uses ``--out-dir``
writes ``summary.json`` *only* inside the specified directory.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from benchmark import config, evaluate_rag


def _examples(n: int = 2) -> list:
    return [
        SimpleNamespace(
            query_id=f"q{i}", question=f"question number {i}",
            gold_doc_id=f"d{i}", gold_section_id=i,
            reference_answer=f"answer {i}", query_type="text", source="text",
        )
        for i in range(n)
    ]


def _patch_benchmark_mocks(captured: dict):
    """Return a context manager that mocks all expensive I/O in main().

    ``save_jsonl`` is mocked to capture writes in *captured* rather than
    touching disk, but ``summary.json`` is still written by the direct
    ``open()`` call inside ``main()`` -- that is exactly the write we want
    to isolate.
    """
    fake_chunk = SimpleNamespace(chunk_id=0)
    fake_cache = Path("/fake/corpus_test.pkl")

    def fake_save_jsonl(records, path):
        captured[path.name] = records

    stack = ExitStack()
    stack.enter_context(patch.object(evaluate_rag, "load_examples",
                                     return_value=_examples(2)))
    stack.enter_context(patch.object(evaluate_rag, "load_embedding_model",
                                     return_value=MagicMock()))
    stack.enter_context(patch.object(evaluate_rag, "get_all_doc_ids",
                                     return_value=["d0"]))
    stack.enter_context(patch.object(evaluate_rag, "select_corpus_doc_ids",
                                     return_value=["d0"]))
    stack.enter_context(patch.object(evaluate_rag, "get_embedded_corpus",
                                     return_value=([fake_chunk], fake_cache)))
    stack.enter_context(patch.object(evaluate_rag, "BM25Index"))
    stack.enter_context(patch.object(evaluate_rag, "Reranker"))
    stack.enter_context(patch.object(evaluate_rag, "_warmup"))
    stack.enter_context(patch.object(evaluate_rag, "evaluate_naive",
                                     return_value={"status": "ok"}))
    stack.enter_context(patch.object(evaluate_rag, "evaluate_hybrid",
                                     return_value={"status": "ok"}))
    stack.enter_context(patch.object(evaluate_rag, "evaluate_reranker",
                                     return_value={"status": "ok"}))
    stack.enter_context(patch.object(evaluate_rag, "evaluate_agentic",
                                     return_value={"status": "ok"}))
    stack.enter_context(patch.object(evaluate_rag, "summarize",
                                     return_value={}))
    stack.enter_context(patch.object(evaluate_rag, "print_summary"))
    stack.enter_context(patch.object(evaluate_rag, "print_comparison"))
    stack.enter_context(patch.object(evaluate_rag, "save_jsonl",
                                     side_effect=fake_save_jsonl))
    return stack


# ---------------------------------------------------------------------------
# A+B.  --out-dir redirects ALL writes (including the direct summary.json
#       open()) away from config.RESULTS_DIR.
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_main_with_out_dir_writes_summary_only_to_tmp_path(tmp_path,
                                                            monkeypatch):
    """A mocked ``--system all`` run with ``--out-dir tmp_path`` must write
    ``summary.json`` **only** inside ``tmp_path`` and must **not** touch
    ``config.RESULTS_DIR``.

    This is the regression test for the original bug: even though
    ``save_jsonl`` is mocked, the direct ``open(out_dir / "summary.json")``
    in ``main()`` used to write to the default ``config.RESULTS_DIR``.
    """
    fake_results = tmp_path / "fake_results_dir"
    fake_results.mkdir()
    sentinel = fake_results / "sentinel.json"
    sentinel_data = {"marker": "do_not_overwrite", "ts": 12345}
    sentinel.write_text(json.dumps(sentinel_data))

    monkeypatch.setattr(config, "RESULTS_DIR", fake_results)

    output_dir = tmp_path / "test_output"
    captured: dict = {}

    with _patch_benchmark_mocks(captured):
        evaluate_rag.main(
            ["--system", "all", "--limit", "2",
             "--out-dir", str(output_dir)])

    # 1. The sentinel in the fake RESULTS_DIR must be untouched.
    assert json.loads(sentinel.read_text()) == sentinel_data

    # 2. No summary.json was written to the fake RESULTS_DIR.
    assert not (fake_results / "summary.json").exists(), (
        "summary.json must NOT be written to config.RESULTS_DIR when "
        "--out-dir is supplied")

    # 3. summary.json WAS written to the --out-dir (tmp_path).
    assert (output_dir / "summary.json").exists(), (
        "summary.json must be written to --out-dir")

    # 4. The on-disk summary.json has the expected top-level structure.
    summary = json.loads((output_dir / "summary.json").read_text())
    assert "systems" in summary


# ---------------------------------------------------------------------------
# Proof of the original bug: WITHOUT --out-dir, the default IS
# config.RESULTS_DIR and summary.json lands there.
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_main_without_out_dir_writes_to_results_dir_default(tmp_path,
                                                             monkeypatch):
    """Proves *why* the fix is necessary: without ``--out-dir``, the default
    is ``config.RESULTS_DIR`` and ``summary.json`` is written there.

    This test patches ``config.RESULTS_DIR`` to a temp dir so it does not
    touch the real benchmark results.
    """
    fake_results = tmp_path / "fake_results_dir"
    fake_results.mkdir()
    monkeypatch.setattr(config, "RESULTS_DIR", fake_results)

    captured: dict = {}

    with _patch_benchmark_mocks(captured):
        evaluate_rag.main(["--system", "all", "--limit", "2"])

    # Without --out-dir, summary.json lands in config.RESULTS_DIR (the
    # default).  This demonstrates the original corruption path.
    assert (fake_results / "summary.json").exists(), (
        "Without --out-dir, summary.json is written to the default "
        "config.RESULTS_DIR -- this is the original bug being guarded "
        "against by the --out-dir tmp_path fix.")
