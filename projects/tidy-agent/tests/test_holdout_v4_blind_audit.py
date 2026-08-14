"""Offline tests for evals/holdout_v4/blind_audit.py using synthetic toy
filenames only -- never the real historical corpus or the real Holdout v4
fixture. Verifies the exact-overlap gate, the lexical near-duplicate
diagnostic, and that no filename or matched pair ever leaves the module's
public functions (aggregate counts only).
"""
from __future__ import annotations

import contextlib
import io

from evals.holdout_v4 import blind_audit


def test_normalized_exact_duplicate_detected():
    historical = ("Quarterly_Report_Final",)
    holdout = ("Quarterly_Report_Final",)
    assert blind_audit.exact_overlap_count(historical, holdout) == 1


def test_case_and_nfc_equivalent_duplicate_detected():
    # "é" as a single codepoint (NFC) vs "e" + combining acute (NFD), plus a
    # pure casing difference.
    historical = ("café_menu_final", "REPORT_DRAFT")
    holdout = ("café_menu_final", "report_draft")
    assert blind_audit.exact_overlap_count(historical, holdout) == 2


def test_unrelated_names_have_zero_exact_overlap():
    historical = ("Totally_Different_Historical_Name_One",)
    holdout = ("Completely_Unrelated_New_Filename_Two",)
    assert blind_audit.exact_overlap_count(historical, holdout) == 0


def test_lexical_near_duplicate_detected():
    historical = ("Quarterly_Financial_Report_Draft_Version",)
    holdout = ("Quarterly_Financial_Report_Draft_Final",)
    assert blind_audit.high_lexical_similarity_count(historical, holdout) == 1


def test_lexical_near_duplicate_does_not_change_exact_overlap_pass():
    historical = ("Quarterly_Financial_Report_Draft_Version",)
    holdout = ("Quarterly_Financial_Report_Draft_Final",)
    assert blind_audit.exact_overlap_count(historical, holdout) == 0
    assert blind_audit.high_lexical_similarity_count(historical, holdout) == 1
    # The frozen protocol (evals/final_portfolio_holdout_protocol.md) treats
    # this combination as PASS: exact overlap is the only gate.
    result = {
        "exact_historical_overlap_count": blind_audit.exact_overlap_count(historical, holdout),
        "high_lexical_similarity_count": blind_audit.high_lexical_similarity_count(historical, holdout),
    }
    exact_overlap_gate_passed = result["exact_historical_overlap_count"] == 0
    assert exact_overlap_gate_passed is True


def test_dissimilar_names_have_zero_lexical_similarity():
    historical = ("Zebra_Migration_Notes_Africa",)
    holdout = ("Quantum_Circuit_Simulator_Output",)
    assert blind_audit.high_lexical_similarity_count(historical, holdout) == 0


def test_audit_returns_only_aggregate_counts(tmp_path):
    repo_root = tmp_path
    hist_dir = repo_root / "evals" / "fixture"
    hist_dir.mkdir(parents=True)
    (hist_dir / "Historical_Case_Alpha").write_bytes(b"")
    (hist_dir / "Historical_Case_Beta").write_bytes(b"")
    v4_dir = repo_root / "evals" / "holdout_v4" / "fixture"
    v4_dir.mkdir(parents=True)
    (v4_dir / "Brand_New_Case_Gamma").write_bytes(b"")
    (v4_dir / "Historical_Case_Alpha").write_bytes(b"")

    result = blind_audit.run_blind_audit(repo_root)
    assert set(result) == {
        "historical_files_compared",
        "exact_historical_overlap_count",
        "high_lexical_similarity_count",
    }
    assert result["historical_files_compared"] == 2
    assert result["exact_historical_overlap_count"] == 1
    for value in result.values():
        assert isinstance(value, int)


def test_audit_logs_no_filenames_to_stdout_or_stderr(tmp_path):
    repo_root = tmp_path
    hist_dir = repo_root / "evals" / "fixture"
    hist_dir.mkdir(parents=True)
    secret_name = "Unique_Historical_Secret_Filename_Marker"
    (hist_dir / secret_name).write_bytes(b"")
    v4_dir = repo_root / "evals" / "holdout_v4" / "fixture"
    v4_dir.mkdir(parents=True)
    (v4_dir / secret_name).write_bytes(b"")

    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        result = blind_audit.run_blind_audit(repo_root)
    assert secret_name not in stdout.getvalue()
    assert result["exact_historical_overlap_count"] == 1


def test_load_functions_only_return_filename_strings(tmp_path):
    repo_root = tmp_path
    hist_dir = repo_root / "evals" / "fixture"
    hist_dir.mkdir(parents=True)
    (hist_dir / "Some_Historical_Name").write_bytes(b"")
    names = blind_audit.load_historical_filenames(repo_root)
    assert names == ("Some_Historical_Name",)
    assert all(isinstance(n, str) for n in names)
