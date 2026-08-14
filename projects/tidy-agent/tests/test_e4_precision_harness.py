"""Offline integration tests for the E3/E4-current/E4-refined precision
harness.

Every model call goes through a ``FakeModel`` (in-process, deterministic,
scripted responses) via a monkeypatched ``build_model``, or through the
pure scoring/aggregation/success-criteria functions directly -- exactly the
same pattern as ``tests/test_post_holdout_harness.py``. Nothing here
touches either consumed Holdout. Importing or testing this module performs
no live inference: only running ``evals/run_e4_precision_development.py``
directly as a script does that, and no test here does so.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import evals.run_e4_precision_development as harness
from evals.run_e4_precision_development import (
    CONDITIONS,
    COUNTERBALANCED_SCHEDULE,
    FIXTURES,
    MINIMUM_E3_ERRORS_FOR_ROBUST_VETO_ESTIMATE,
    SECURITY_ZERO_FIELDS,
    _combined,
    _condition_summary,
    _e4_precision_worker,
    _veto_metrics,
    aggregate,
    case_identity,
    compute_e3_error_calibration_diagnostics,
    compute_evidence_strength,
    deduplicate_to_unique_files,
    e3_error_calibration_family_lookup,
    e3_error_density,
    error_family_capture_matrix,
    evaluate_success_criteria,
    evidence_strength_note,
    parse_args,
    review_escape_rescue,
    score_run,
    stress_family_metrics,
)

REVIEW = "_ToReview"


class FakeModel:
    structured_output_mode = "json_schema"

    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def generate(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        response = self.responses.pop(0)
        return SimpleNamespace(
            content=response,
            token_usage=SimpleNamespace(input_tokens=5, output_tokens=2),
        )


class FakeResultQueue:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def put(self, item: dict) -> None:
        self.items.append(item)


def _classify_pair(*pairs: tuple[str, str]) -> str:
    items = ",".join(
        f'{{"source":"{s}","decision":"classify","category":"{c}"}}' for s, c in pairs
    )
    return f'{{"decisions":[{items}]}}'


# --- Shared-call worker -------------------------------------------------


def test_worker_makes_exactly_two_model_calls_for_all_three_conditions(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "vertrag_dokument_2026").write_bytes(b"")
    (tmp_path / "bild_bearbeitung_batch_prozess_programm").write_bytes(b"")

    responses = _classify_pair(
        ("vertrag_dokument_2026", "Code"), ("bild_bearbeitung_batch_prozess_programm", "Code")
    )
    fake = FakeModel(responses, responses)
    monkeypatch.setattr(harness, "build_model", lambda *a, **k: fake)

    q = FakeResultQueue()
    _e4_precision_worker(q, str(tmp_path), "ollama_chat/qwen3.5:4b", False)

    result = q.items[0]
    assert result["status"] == "ok"
    assert len(fake.calls) == 2  # not 6 -- one shared E3 call, no extra calls per condition
    assert set(result["conditions"]) == {"E3", "E4-current", "E4-refined"}


def test_e4_current_and_e4_refined_diverge_on_the_same_shared_e3_result(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "vertrag_dokument_2026").write_bytes(b"")
    (tmp_path / "bild_bearbeitung_batch_prozess_programm").write_bytes(b"")
    responses = _classify_pair(
        ("vertrag_dokument_2026", "Code"), ("bild_bearbeitung_batch_prozess_programm", "Code")
    )
    fake = FakeModel(responses, responses)
    monkeypatch.setattr(harness, "build_model", lambda *a, **k: fake)

    q = FakeResultQueue()
    _e4_precision_worker(q, str(tmp_path), "ollama_chat/qwen3.5:4b", False)
    conditions = q.items[0]["conditions"]

    # Both start from the identical E3 result...
    assert conditions["E3"]["final"] == {
        "vertrag_dokument_2026": "Code",
        "bild_bearbeitung_batch_prozess_programm": "Code",
    }
    # ...but E4-current vetoes the cue-co-occurrence file (its known
    # false-positive pattern) while missing the unsupported-category file...
    assert conditions["E4-current"]["final"]["bild_bearbeitung_batch_prozess_programm"] == REVIEW
    assert conditions["E4-current"]["final"]["vertrag_dokument_2026"] == "Code"
    # ...while E4-refined does the reverse: preserves the legitimate
    # cue-co-occurrence file and catches the unsupported-category one.
    assert conditions["E4-refined"]["final"]["bild_bearbeitung_batch_prozess_programm"] == "Code"
    assert conditions["E4-refined"]["final"]["vertrag_dokument_2026"] == REVIEW


def test_worker_reports_zero_content_access_fields(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "one").write_bytes(b"")
    responses = _classify_pair(("one", "Documents"))
    fake = FakeModel(responses, responses)
    monkeypatch.setattr(harness, "build_model", lambda *a, **k: fake)

    q = FakeResultQueue()
    _e4_precision_worker(q, str(tmp_path), "ollama_chat/qwen3.5:4b", False)
    telemetry = q.items[0]["telemetry"]
    for field in SECURITY_ZERO_FIELDS:
        assert (telemetry.get(field, 0) or 0) == 0


def test_ground_truth_never_appears_in_any_model_prompt(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "vertrag_dokument_2026").write_bytes(b"")
    responses = _classify_pair(("vertrag_dokument_2026", "Documents"))
    fake = FakeModel(responses, responses)
    monkeypatch.setattr(harness, "build_model", lambda *a, **k: fake)

    q = FakeResultQueue()
    _e4_precision_worker(q, str(tmp_path), "ollama_chat/qwen3.5:4b", False)
    assert q.items[0]["status"] == "ok"
    for call in fake.calls:
        prompt_text = call["messages"][0]["content"][0]["text"]
        assert "ground_truth" not in prompt_text
        assert "expected" not in prompt_text.lower()


# --- Scoring / aggregation ---------------------------------------------


def _raw_run(condition: str, fixture: str, repetition: int, final: dict, detail: list[dict]) -> dict:
    return {
        "experiment_id": "test",
        "fixture": fixture,
        "repetition": repetition,
        "condition": condition,
        "status": "ok",
        "metrics": {"status": "ok", "sources": list(final), "final": final, "detail": detail},
    }


def test_score_run_and_condition_summary_basic() -> None:
    expected = {"a": ["Documents"], "b": [REVIEW]}
    run = _raw_run("E3", "calibration", 1, {"a": "Documents", "b": REVIEW}, [])
    cases = score_run(run["metrics"], expected, review_directory=REVIEW)
    for case in cases:
        case["fixture"] = run["fixture"]
        case["repetition"] = run["repetition"]
    summary = _condition_summary("E3", [{**run, "cases": cases}])
    expected_raw_counts = {"correct_automatic": 1, "incorrect_automatic": 0, "review": 1}
    assert summary["unique_file_metrics"]["primary"]["raw_counts"] == expected_raw_counts
    assert summary["repeated_observation_metrics"]["primary"]["raw_counts"] == expected_raw_counts


def test_veto_metrics_true_and_false_positive() -> None:
    detail = [
        {"filename": "wrong_but_vetoed", "e3_category": "Archives", "veto_applicable": True},
        {"filename": "right_but_vetoed", "e3_category": "Documents", "veto_applicable": True},
        {"filename": "right_and_kept", "e3_category": "Documents", "veto_applicable": True},
    ]
    final = {"wrong_but_vetoed": REVIEW, "right_but_vetoed": REVIEW, "right_and_kept": "Documents"}
    expected = {
        "wrong_but_vetoed": ["Documents"],
        "right_but_vetoed": ["Documents"],
        "right_and_kept": ["Documents"],
    }
    run = _raw_run("E4-refined", "calibration", 1, final, detail)
    cases = score_run(run["metrics"], expected, review_directory=REVIEW)
    metrics = _veto_metrics(cases)
    assert metrics["true_positive_vetoes"] == 1
    assert metrics["false_positive_vetoes"] == 1
    assert metrics["veto_precision"] == 0.5


def test_aggregate_uses_unique_file_denominator_not_repetition_count() -> None:
    """Pins the bugfix: primary (`unique_file_metrics`) semantic sample
    size must stay at the true unique-file count (2) regardless of
    repetition count, while `repeated_observation_metrics` -- diagnostic
    only -- is still allowed to scale with repetitions. The historical bug
    was reading the repetition-multiplied count where the unique count
    belonged; this test fails under that old behavior."""
    expected = {"a": ["Documents"], "b": [REVIEW]}
    runs = [
        _raw_run("E3", "calibration", 1, {"a": "Documents", "b": REVIEW}, []),
        _raw_run("E3", "calibration", 2, {"a": "Documents", "b": REVIEW}, []),
    ]
    result = aggregate(runs, expected, review_directory=REVIEW)
    assert result["unique_file_metrics"]["E3"]["primary"]["total_files_scored"] == 2  # semantic N stays 2
    assert result["repeated_observation_metrics"]["E3"]["primary"]["total_files_scored"] == 4  # 2 reps x 2 files
    assert result["stability"]["E3"]["unique_file_count"] == 2


# --- Success-criteria / evidence-strength rules -----------------------


def test_evidence_strength_note_flags_underpowered_sample() -> None:
    combined_e3 = {"primary": {"raw_counts": {"incorrect_automatic": 1}}}
    note = evidence_strength_note(combined_e3)
    assert note is not None
    assert "insufficient" in note.lower()


def test_evidence_strength_note_is_none_when_powered() -> None:
    combined_e3 = {
        "primary": {
            "raw_counts": {"incorrect_automatic": MINIMUM_E3_ERRORS_FOR_ROBUST_VETO_ESTIMATE}
        }
    }
    assert evidence_strength_note(combined_e3) is None


def _minimal_condition_summary(
    *, unsafe: float, accuracy_decided: float, coverage: float, review_recall: float, review_n: int,
    veto_precision: float | None,
) -> dict:
    return {
        "primary": {
            "unsafe_automation_rate": unsafe,
            "accuracy_on_decided": accuracy_decided,
            "automation_coverage": coverage,
        },
        "review_subset": {"review_recall": review_recall, "n": review_n},
        "veto_analysis": {"veto_precision": veto_precision, "false_positive_vetoes": 0},
    }


def test_success_criteria_underpowered_below_minimum_errors() -> None:
    current = _minimal_condition_summary(
        unsafe=0.02, accuracy_decided=0.9, coverage=0.3, review_recall=0.9, review_n=20, veto_precision=0.25
    )
    refined = _minimal_condition_summary(
        unsafe=0.0, accuracy_decided=1.0, coverage=0.3, review_recall=0.9, review_n=20, veto_precision=0.5
    )
    per_fixture = {"calibration": current}
    per_fixture_ref = {"calibration": refined}
    result = evaluate_success_criteria(
        unique_e3_automatic_errors=1,
        e4_current_combined=current,
        e4_refined_combined=refined,
        per_fixture_current=per_fixture,
        per_fixture_refined=per_fixture_ref,
    )
    assert result["underpowered"] is True
    assert result["unique_e3_automatic_errors"] == 1
    assert result["criteria"]["5_veto_precision_materially_higher"] == "UNDERPOWERED"


def test_success_criteria_all_pass_when_refined_strictly_better() -> None:
    current = _minimal_condition_summary(
        unsafe=0.02, accuracy_decided=0.9, coverage=0.30, review_recall=0.90, review_n=20, veto_precision=0.25
    )
    refined = _minimal_condition_summary(
        unsafe=0.0, accuracy_decided=1.0, coverage=0.29, review_recall=0.90, review_n=20, veto_precision=0.60
    )
    per_fixture = {"calibration": current, "boundary_calibration": current}
    per_fixture_ref = {"calibration": refined, "boundary_calibration": refined}
    result = evaluate_success_criteria(
        unique_e3_automatic_errors=5,
        e4_current_combined=current,
        e4_refined_combined=refined,
        per_fixture_current=per_fixture,
        per_fixture_refined=per_fixture_ref,
    )
    assert result["underpowered"] is False
    c = result["criteria"]
    assert c["1_unsafe_automation_no_worse"] is True
    assert c["2_accuracy_on_decided_no_worse"] is True
    assert c["3_review_recall_not_worse_by_more_than_one_case"] is True
    assert c["4_coverage_not_more_than_3pp_below"] is True
    assert c["5_veto_precision_materially_higher"] == "PASS"


def test_success_criteria_fails_when_coverage_drops_too_much() -> None:
    current = _minimal_condition_summary(
        unsafe=0.0, accuracy_decided=1.0, coverage=0.30, review_recall=0.90, review_n=20, veto_precision=0.25
    )
    refined = _minimal_condition_summary(
        unsafe=0.0, accuracy_decided=1.0, coverage=0.20, review_recall=0.90, review_n=20, veto_precision=0.60
    )
    result = evaluate_success_criteria(
        unique_e3_automatic_errors=5,
        e4_current_combined=current,
        e4_refined_combined=refined,
        per_fixture_current={"calibration": current},
        per_fixture_refined={"calibration": refined},
    )
    assert result["criteria"]["4_coverage_not_more_than_3pp_below"] is False


# --- Unique-file vs. repeated-observation aggregation bugfix regressions ---
#
# The completed 2026-08-12 Development run
# (evals/results/e4-precision-development-20260812-b6095d49ee1d/, preserved
# unmodified) had 185 unique files and 5 repetitions with exactly 1 unique
# E3 automatic error, but the harness at the time reported
# combined_e3_errors=5 (the repetition-summed, not deduplicated, count) and
# incorrectly concluded underpowered=false. These tests pin the fixed
# behavior and would fail under that old logic.


def test_case_identity_distinguishes_same_basename_across_fixtures() -> None:
    case_a = {"fixture": "calibration", "filename": "shared_name", "predicted": "Documents"}
    case_b = {"fixture": "boundary_calibration", "filename": "shared_name", "predicted": "Archives"}
    assert case_identity(case_a) != case_identity(case_b)

    unique_cases, unstable = deduplicate_to_unique_files([case_a, case_b])
    assert len(unique_cases) == 2  # not merged into one
    assert unstable == []  # each identity individually saw only one, self-consistent prediction


def test_regression_same_basename_in_two_fixtures_counts_as_two_unique_cases() -> None:
    expected = {"calibration::shared_name": ["Documents"], "boundary_calibration::shared_name": ["Archives"]}
    runs = [
        _raw_run("E3", "calibration", 1, {"calibration::shared_name": "Documents"}, []),
        _raw_run("E3", "boundary_calibration", 1, {"boundary_calibration::shared_name": "Archives"}, []),
    ]
    result = aggregate(runs, expected, review_directory=REVIEW)
    assert result["stability"]["E3"]["unique_file_count"] == 2
    assert result["unique_file_metrics"]["E3"]["primary"]["total_files_scored"] == 2
    assert result["unique_file_metrics"]["E3"]["primary"]["raw_counts"]["correct_automatic"] == 2


def test_regression_bug_shape_one_unique_error_times_five_repetitions() -> None:
    """Exact shape of the historical bug: 1 fixture, 5 repetitions, the same
    single wrong E3 decision repeated identically every repetition."""
    expected = {"good1": ["Documents"], "bad1": ["Archives"]}
    final = {"good1": "Documents", "bad1": "Documents"}  # bad1 wrong every repetition
    runs = [_raw_run("E3", "calibration", r, dict(final), []) for r in range(1, 6)]
    result = aggregate(runs, expected, review_directory=REVIEW)

    repeated_wrong = result["repeated_observation_metrics"]["E3"]["primary"]["raw_counts"]["incorrect_automatic"]
    unique_wrong = result["unique_file_metrics"]["E3"]["primary"]["raw_counts"]["incorrect_automatic"]
    assert repeated_wrong == 5
    assert unique_wrong == 1

    evidence = compute_evidence_strength(result["unique_file_metrics"]["E3"])
    assert evidence["unique_e3_automatic_errors"] == 1
    assert evidence["underpowered"] is True

    minimal_veto = {
        "primary": {"unsafe_automation_rate": 0.0, "accuracy_on_decided": 1.0, "automation_coverage": 0.5},
        "review_subset": {"review_recall": 1.0, "n": 10},
        "veto_analysis": {"veto_precision": 1.0, "false_positive_vetoes": 0},
    }
    criteria = evaluate_success_criteria(
        unique_e3_automatic_errors=evidence["unique_e3_automatic_errors"],
        e4_current_combined=minimal_veto,
        e4_refined_combined=minimal_veto,
        per_fixture_current={"calibration": minimal_veto},
        per_fixture_refined={"calibration": minimal_veto},
    )
    assert criteria["underpowered"] is True
    assert criteria["criteria"]["5_veto_precision_materially_higher"] == "UNDERPOWERED"

    # The old, buggy behavior must NOT reproduce: unique errors == 5 and
    # underpowered == False are exactly what the pre-fix harness computed here.
    assert unique_wrong != 5
    assert evidence["underpowered"] is not False


def test_regression_three_unique_errors_times_five_repetitions_pins_threshold_boundary() -> None:
    expected = {"bad1": ["Archives"], "bad2": ["Archives"], "bad3": ["Archives"], "good1": ["Documents"]}
    final = {"bad1": "Documents", "bad2": "Documents", "bad3": "Documents", "good1": "Documents"}
    runs = [_raw_run("E3", "calibration", r, dict(final), []) for r in range(1, 6)]
    result = aggregate(runs, expected, review_directory=REVIEW)

    assert result["repeated_observation_metrics"]["E3"]["primary"]["raw_counts"]["incorrect_automatic"] == 15
    assert result["unique_file_metrics"]["E3"]["primary"]["raw_counts"]["incorrect_automatic"] == 3

    evidence = compute_evidence_strength(result["unique_file_metrics"]["E3"])
    assert evidence["unique_e3_automatic_errors"] == 3
    assert evidence["underpowered"] is False


def test_regression_zero_unique_errors_underpowered_and_safe_denominators() -> None:
    expected = {"good1": ["Documents"], "good2": ["Archives"]}
    final = {"good1": "Documents", "good2": "Archives"}
    runs = [_raw_run("E3", "calibration", r, dict(final), []) for r in range(1, 6)]
    result = aggregate(runs, expected, review_directory=REVIEW)

    assert result["unique_file_metrics"]["E3"]["primary"]["raw_counts"]["incorrect_automatic"] == 0
    evidence = compute_evidence_strength(result["unique_file_metrics"]["E3"])
    assert evidence["unique_e3_automatic_errors"] == 0
    assert evidence["underpowered"] is True

    # Veto recall's denominator (unique E3 automatic errors presented) is
    # zero here -- must not raise ZeroDivisionError and must not claim
    # evidence that doesn't exist (None, not 0.0 or 1.0).
    detail = [{"filename": "good1", "e3_category": "Documents", "veto_applicable": True}]
    e4_final = {"good1": "Documents"}
    e4_runs = [_raw_run("E4-refined", "calibration", r, dict(e4_final), list(detail)) for r in range(1, 6)]
    e4_result = aggregate(e4_runs, {"good1": ["Documents"]}, review_directory=REVIEW)
    veto = e4_result["unique_file_metrics"]["E4-refined"]["veto_analysis"]
    assert veto["veto_recall_for_e3_automatic_errors"] is None
    assert veto["true_positive_vetoes"] == 0
    assert veto["vetoed"] == 0
    assert veto["veto_precision"] is None  # no vetoes at all -- also no false claim of evidence


def test_regression_two_unique_errors_times_five_repetitions_underpowered() -> None:
    expected = {"bad1": ["Archives"], "bad2": ["Archives"], "good1": ["Documents"]}
    final = {"bad1": "Documents", "bad2": "Documents", "good1": "Documents"}
    runs = [_raw_run("E3", "calibration", r, dict(final), []) for r in range(1, 6)]
    result = aggregate(runs, expected, review_directory=REVIEW)

    assert result["repeated_observation_metrics"]["E3"]["primary"]["raw_counts"]["incorrect_automatic"] == 10
    assert result["unique_file_metrics"]["E3"]["primary"]["raw_counts"]["incorrect_automatic"] == 2

    evidence = compute_evidence_strength(result["unique_file_metrics"]["E3"])
    assert evidence["unique_e3_automatic_errors"] == 2
    assert evidence["underpowered"] is True


def _build_synthetic_e3_and_e4_refined_runs(fixture: str, repetitions: int) -> tuple[dict, list[dict]]:
    expected = {
        "correct1": ["Documents"],
        "correct2": ["Archives"],
        "wrong1": ["Archives"],  # E3 predicts Documents -> incorrect automatic
        "reviewed1": [REVIEW],
        "vetoed_correct": ["Documents"],  # E3 correct, but E4-refined false-positive vetoes it
        "vetoed_wrong": ["Archives"],  # E3 wrong, and E4-refined true-positive vetoes it
    }
    e3_final = {
        "correct1": "Documents",
        "correct2": "Archives",
        "wrong1": "Documents",
        "reviewed1": REVIEW,
        "vetoed_correct": "Documents",
        "vetoed_wrong": "Documents",
    }
    e4_final = dict(e3_final)
    e4_final["vetoed_correct"] = REVIEW
    e4_final["vetoed_wrong"] = REVIEW
    detail = [
        {"filename": "vetoed_correct", "e3_category": "Documents", "veto_applicable": True},
        {"filename": "vetoed_wrong", "e3_category": "Documents", "veto_applicable": True},
    ]
    e3_runs = [_raw_run("E3", fixture, r, dict(e3_final), []) for r in range(1, repetitions + 1)]
    e4_runs = [_raw_run("E4-refined", fixture, r, dict(e4_final), list(detail)) for r in range(1, repetitions + 1)]
    return expected, e3_runs + e4_runs


def test_regression_repetition_count_invariance_for_unique_file_metrics() -> None:
    """Using identical deterministic semantic outcomes, unique-file primary
    metrics (including veto TP/FP, precision/recall, and the unique E3
    error count) must be identical at 1, 3, and 5 repetitions. Only
    repetition/stability diagnostics (`runs`, repeated-observation totals)
    may differ."""
    unique_snapshots = {}
    for repetitions in (1, 3, 5):
        expected, runs = _build_synthetic_e3_and_e4_refined_runs("calibration", repetitions)
        result = aggregate(runs, expected, review_directory=REVIEW)
        unique_snapshots[repetitions] = result["unique_file_metrics"]

        evidence = compute_evidence_strength(result["unique_file_metrics"]["E3"])
        assert evidence["unique_e3_automatic_errors"] == 2  # "wrong1" and "vetoed_wrong"

    baseline = unique_snapshots[1]
    for repetitions in (3, 5):
        candidate = unique_snapshots[repetitions]
        for condition in ("E3", "E4-refined"):
            base_block = {k: v for k, v in baseline[condition].items() if k != "runs"}
            other_block = {k: v for k, v in candidate[condition].items() if k != "runs"}
            assert base_block == other_block, f"{condition} unique-file metrics changed with repetitions"


def test_regression_cost_invariance_across_repetition_counts() -> None:
    expected = {"correct1": ["Documents"], "wrong1": ["Archives"], "reviewed1": [REVIEW]}
    final = {"correct1": "Documents", "wrong1": "Documents", "reviewed1": REVIEW}
    expected_unique_cost = {"safety_heavy": 11, "balanced": 6, "coverage_heavy": 4}
    for repetitions in (1, 5):
        runs = [_raw_run("E3", "calibration", r, dict(final), []) for r in range(1, repetitions + 1)]
        result = aggregate(runs, expected, review_directory=REVIEW)
        unique_cost = result["unique_file_metrics"]["E3"]["cost_scenarios"]
        repeated_cost = result["repeated_observation_metrics"]["E3"]["cost_scenarios"]
        assert unique_cost == expected_unique_cost  # primary cost does NOT scale with repetitions
        assert repeated_cost["balanced"] == expected_unique_cost["balanced"] * repetitions  # diagnostic view does


def test_regression_veto_tp_fp_deduplicate_to_one_not_five() -> None:
    expected = {"caught_error": ["Archives"], "wrongly_vetoed": ["Documents"]}
    e3_final = {"caught_error": "Documents", "wrongly_vetoed": "Documents"}
    e4_final = {"caught_error": REVIEW, "wrongly_vetoed": REVIEW}
    detail = [
        {"filename": "caught_error", "e3_category": "Documents", "veto_applicable": True},
        {"filename": "wrongly_vetoed", "e3_category": "Documents", "veto_applicable": True},
    ]
    e3_runs = [_raw_run("E3", "calibration", r, dict(e3_final), []) for r in range(1, 6)]
    e4_runs = [_raw_run("E4-refined", "calibration", r, dict(e4_final), list(detail)) for r in range(1, 6)]
    result = aggregate(e3_runs + e4_runs, expected, review_directory=REVIEW)

    unique_veto = result["unique_file_metrics"]["E4-refined"]["veto_analysis"]
    assert unique_veto["true_positive_vetoes"] == 1  # not 5
    assert unique_veto["false_positive_vetoes"] == 1  # not 5

    repeated_veto = result["repeated_observation_metrics"]["E4-refined"]["veto_analysis"]
    assert repeated_veto["true_positive_vetoes"] == 5
    assert repeated_veto["false_positive_vetoes"] == 5


def test_unstable_file_uses_representative_repetition_not_majority_vote(tmp_path: Path, monkeypatch) -> None:
    """A file whose model decision differs across repetitions must be
    flagged unstable, keep all repeated observations, count as ONE unique
    file (not one per repetition), and have its primary semantic metrics
    governed by the documented lowest-repetition-number representative --
    never a silent majority vote. Here a majority vote across 3
    repetitions would pick the wrong category (2 of 3 say "Archives"); the
    frozen policy must still use repetition 1 ("Documents", correct)."""
    (tmp_path / "flaky_file").write_bytes(b"")
    expected = {"flaky_file": ["Documents"]}
    per_repetition_category = {1: "Documents", 2: "Archives", 3: "Archives"}

    runs = []
    for repetition, category in per_repetition_category.items():
        responses = _classify_pair(("flaky_file", category))
        fake = FakeModel(responses, responses)
        monkeypatch.setattr(harness, "build_model", lambda *a, _fake=fake, **k: _fake)
        q = FakeResultQueue()
        _e4_precision_worker(q, str(tmp_path), "ollama_chat/qwen3.5:4b", False)
        shared = q.items[0]
        assert shared["status"] == "ok"
        cond_data = shared["conditions"]["E3"]
        runs.append(
            {
                "experiment_id": "test",
                "fixture": "calibration",
                "repetition": repetition,
                "condition": "E3",
                "status": "ok",
                "metrics": {
                    "status": "ok",
                    "sources": shared["sources"],
                    "final": cond_data["final"],
                    "detail": cond_data["detail"],
                    "telemetry": shared["telemetry"],
                },
            }
        )

    result = aggregate(runs, expected, review_directory=REVIEW)

    # Instability is detected.
    assert "flaky_file" in result["stability"]["E3"]["unstable_files"]
    assert "flaky_file" in result["stability"]["E3"]["unstable_unique_files"]
    assert result["stability"]["E3"]["counts"]["unstable"] == 1

    # All repeated observations are retained, not discarded.
    assert result["repeated_observation_metrics"]["E3"]["primary"]["total_files_scored"] == 3

    # It counts as ONE unique file, not three independent ones.
    assert result["stability"]["E3"]["unique_file_count"] == 1
    assert result["unique_file_metrics"]["E3"]["primary"]["total_files_scored"] == 1

    # Representative-repetition policy (rep 1 = "Documents", correct)
    # governs primary metrics -- not the majority ("Archives", wrong).
    assert result["unique_file_metrics"]["E3"]["primary"]["raw_counts"]["correct_automatic"] == 1
    assert result["unique_file_metrics"]["E3"]["primary"]["raw_counts"]["incorrect_automatic"] == 0


def test_compute_evidence_strength_documents_the_historical_misuse_shape() -> None:
    """Documents *why* the `run_experiment` wiring matters: the function
    itself cannot tell which block it was given, so feeding it the
    repeated-observation block reproduces the exact historical bug shape
    (1 true unique error read as 5, underpowered flipped to False). The
    real fix is at the call site (see
    `test_run_experiment_wires_evidence_strength_and_success_criteria_from_unique_file_metrics`),
    not inside this function."""
    expected = {"good1": ["Documents"], "bad1": ["Archives"]}
    final = {"good1": "Documents", "bad1": "Documents"}
    runs = [_raw_run("E3", "calibration", r, dict(final), []) for r in range(1, 6)]
    result = aggregate(runs, expected, review_directory=REVIEW)

    correct_evidence = compute_evidence_strength(result["unique_file_metrics"]["E3"])
    assert correct_evidence["unique_e3_automatic_errors"] == 1
    assert correct_evidence["underpowered"] is True

    buggy_evidence = compute_evidence_strength(result["repeated_observation_metrics"]["E3"])
    assert buggy_evidence["unique_e3_automatic_errors"] == 5
    assert buggy_evidence["underpowered"] is False


def test_success_criteria_criterion_5_fails_when_precision_not_higher() -> None:
    current = _minimal_condition_summary(
        unsafe=0.0, accuracy_decided=1.0, coverage=0.30, review_recall=0.90, review_n=20, veto_precision=0.5
    )
    refined = _minimal_condition_summary(
        unsafe=0.0, accuracy_decided=1.0, coverage=0.30, review_recall=0.90, review_n=20, veto_precision=0.3
    )
    result = evaluate_success_criteria(
        unique_e3_automatic_errors=5,
        e4_current_combined=current,
        e4_refined_combined=refined,
        per_fixture_current={"calibration": current},
        per_fixture_refined={"calibration": refined},
    )
    assert result["criteria"]["5_veto_precision_materially_higher"] == "FAIL"


def test_success_criterion_3_unique_case_boundary_zero_one_two() -> None:
    """'no worse by more than one unique case' must resolve exactly at the
    0/1/2 unique-case boundary, independent of repetition count (this
    function never sees repetitions -- it consumes already-deduplicated
    review_subset.n, which the unique_file_metrics pipeline guarantees)."""

    def _summary(review_recall: float, review_n: int) -> dict:
        return {
            "primary": {"unsafe_automation_rate": 0.0, "accuracy_on_decided": 1.0, "automation_coverage": 0.5},
            "review_subset": {"review_recall": review_recall, "n": review_n},
            "veto_analysis": {"veto_precision": 0.5, "false_positive_vetoes": 0},
        }

    review_n = 20
    current = _summary(1.0, review_n)

    refined_equal = _summary(1.0, review_n)
    result_equal = evaluate_success_criteria(
        unique_e3_automatic_errors=5,
        e4_current_combined=current,
        e4_refined_combined=refined_equal,
        per_fixture_current={"calibration": current},
        per_fixture_refined={"calibration": refined_equal},
    )
    assert result_equal["review_recall_delta_unique_cases"] == 0
    assert result_equal["criteria"]["3_review_recall_not_worse_by_more_than_one_case"] is True

    refined_one_worse = _summary(1.0 - 1 / review_n, review_n)
    result_one = evaluate_success_criteria(
        unique_e3_automatic_errors=5,
        e4_current_combined=current,
        e4_refined_combined=refined_one_worse,
        per_fixture_current={"calibration": current},
        per_fixture_refined={"calibration": refined_one_worse},
    )
    assert result_one["review_recall_delta_unique_cases"] == 1
    assert result_one["criteria"]["3_review_recall_not_worse_by_more_than_one_case"] is True

    refined_two_worse = _summary(1.0 - 2 / review_n, review_n)
    result_two = evaluate_success_criteria(
        unique_e3_automatic_errors=5,
        e4_current_combined=current,
        e4_refined_combined=refined_two_worse,
        per_fixture_current={"calibration": current},
        per_fixture_refined={"calibration": refined_two_worse},
    )
    assert result_two["review_recall_delta_unique_cases"] == 2
    assert result_two["criteria"]["3_review_recall_not_worse_by_more_than_one_case"] is False


def test_success_criterion_6_distinguishes_multi_fixture_from_single_fixture_improvement() -> None:
    def _fixture_veto(veto_precision: float | None) -> dict:
        return {"veto_analysis": {"veto_precision": veto_precision, "false_positive_vetoes": 0}}

    combined_current = {
        "primary": {"unsafe_automation_rate": 0.0, "accuracy_on_decided": 1.0, "automation_coverage": 0.5},
        "review_subset": {"review_recall": 1.0, "n": 20},
        "veto_analysis": {"veto_precision": 0.2, "false_positive_vetoes": 4},
    }
    combined_refined = {
        "primary": {"unsafe_automation_rate": 0.0, "accuracy_on_decided": 1.0, "automation_coverage": 0.5},
        "review_subset": {"review_recall": 1.0, "n": 20},
        "veto_analysis": {"veto_precision": 0.4, "false_positive_vetoes": 2},
    }

    # Case A: improvement exists on both fixtures.
    per_fixture_current_a = {"calibration": _fixture_veto(0.2), "boundary_calibration": _fixture_veto(0.2)}
    per_fixture_refined_a = {"calibration": _fixture_veto(0.4), "boundary_calibration": _fixture_veto(0.5)}
    result_a = evaluate_success_criteria(
        unique_e3_automatic_errors=5,
        e4_current_combined=combined_current,
        e4_refined_combined=combined_refined,
        per_fixture_current=per_fixture_current_a,
        per_fixture_refined=per_fixture_refined_a,
    )
    deltas_a = result_a["criteria"]["6_improvement_not_confined_to_one_fixture"]

    # Case B: all differentiating improvement is on exactly one fixture.
    per_fixture_current_b = {"calibration": _fixture_veto(0.2), "boundary_calibration": _fixture_veto(0.2)}
    per_fixture_refined_b = {"calibration": _fixture_veto(0.6), "boundary_calibration": _fixture_veto(0.2)}
    result_b = evaluate_success_criteria(
        unique_e3_automatic_errors=5,
        e4_current_combined=combined_current,
        e4_refined_combined=combined_refined,
        per_fixture_current=per_fixture_current_b,
        per_fixture_refined=per_fixture_refined_b,
    )
    deltas_b = result_b["criteria"]["6_improvement_not_confined_to_one_fixture"]

    improving_fixtures_a = [f for f, d in deltas_a.items() if d is not None and d < 0]
    improving_fixtures_b = [f for f, d in deltas_b.items() if d is not None and d < 0]
    assert len(improving_fixtures_a) == 2  # both fixtures show improvement
    assert len(improving_fixtures_b) == 1  # only one fixture shows improvement
    assert deltas_b["boundary_calibration"] == 0  # the other fixture shows zero movement
    assert deltas_a != deltas_b


# --- Structural / defaults --------------------------------------------------


def test_conditions_are_exactly_e3_e4_current_e4_refined() -> None:
    assert set(CONDITIONS) == {"E3", "E4-current", "E4-refined"}


def test_counterbalanced_schedule_covers_all_three_conditions_each_rotation() -> None:
    for rotation in COUNTERBALANCED_SCHEDULE:
        assert set(rotation) == {"E3", "E4-current", "E4-refined"}
        assert len(rotation) == 3


_ALL_FOUR_FIXTURES = {
    "calibration",
    "boundary_calibration",
    "veto_precision_calibration",
    "e3_error_calibration",
}


def test_parse_args_defaults_to_all_four_development_fixtures() -> None:
    args = parse_args([])
    assert args.repetitions == 5
    assert args.think is False
    assert set(args.fixtures) == _ALL_FOUR_FIXTURES
    assert set(FIXTURES) == _ALL_FOUR_FIXTURES
    for paths in FIXTURES.values():
        for key in ("fixture", "expected", "manifest"):
            assert "holdout" not in str(paths[key]).lower()


_HARNESS_SOURCE = Path(__file__).parents[1].joinpath(
    "evals", "run_e4_precision_development.py"
).read_text(encoding="utf-8")


def test_harness_never_reads_either_holdout_fixture_or_content() -> None:
    assert "evals/holdout" not in _HARNESS_SOURCE
    assert "holdout_v2" not in _HARNESS_SOURCE
    assert "read_contents=True" not in _HARNESS_SOURCE
    assert "allow_remote_content=True" not in _HARNESS_SOURCE
    assert "peek_tool" not in _HARNESS_SOURCE


def test_harness_does_not_import_or_touch_e5() -> None:
    assert "run_e5" not in _HARNESS_SOURCE
    assert '"E5"' not in _HARNESS_SOURCE


def test_harness_only_runs_live_inference_as_a_direct_script() -> None:
    assert 'if __name__ == "__main__":' in _HARNESS_SOURCE


def test_harness_is_a_new_file_not_an_edit_of_the_prior_e5_harness() -> None:
    prior_harness = Path(__file__).parents[1].joinpath(
        "evals", "run_post_holdout_development.py"
    )
    assert prior_harness.is_file()  # historical file preserved, untouched
    prior_source = prior_harness.read_text(encoding="utf-8")
    assert '"E5"' in prior_source or "run_e5" in prior_source  # still has E5


def test_run_experiment_wires_evidence_strength_and_success_criteria_from_unique_file_metrics() -> None:
    """Structural regression guard on `run_experiment`'s orchestration code
    (which is not otherwise exercised by these offline tests, since its
    workers run in real subprocesses -- see the worker tests above for the
    in-process equivalent): it must source `compute_evidence_strength` and
    `evaluate_success_criteria`'s inputs from `unique_file_metrics`, never
    from `repeated_observation_metrics` or the old, removed `["summary"]`
    key. That exact wiring mistake -- reading the repetition-multiplied
    block where the deduplicated one belonged -- was the root cause of the
    historical bug."""
    assert 'result["combined"]["summary"]' not in _HARNESS_SOURCE  # old, buggy key: must be fully gone
    assert 'combined_unique = result["combined"]["unique_file_metrics"]' in _HARNESS_SOURCE
    assert "compute_evidence_strength(combined_unique[" in _HARNESS_SOURCE
    assert 'unique_e3_automatic_errors=result["evidence_strength"]["unique_e3_automatic_errors"]' in _HARNESS_SOURCE
    assert 'e4_current_combined=combined_unique["E4-current"]' in _HARNESS_SOURCE
    assert 'e4_refined_combined=combined_unique["E4-refined"]' in _HARNESS_SOURCE
    assert '["per_fixture"][name]["unique_file_metrics"]["E4-current"]' in _HARNESS_SOURCE
    assert '["per_fixture"][name]["unique_file_metrics"]["E4-refined"]' in _HARNESS_SOURCE
    # The old, ambiguous parameter name must be gone from the call site too.
    assert "combined_e3_errors=" not in _HARNESS_SOURCE


# --- e3_error_calibration fixture + harness extension -----------------------


def test_e3_error_calibration_is_registered_and_not_holdout() -> None:
    assert "e3_error_calibration" in FIXTURES
    for key in ("fixture", "expected", "manifest"):
        assert "holdout" not in str(FIXTURES["e3_error_calibration"][key]).lower()


def test_e3_error_calibration_family_lookup_matches_build_fixture_cases() -> None:
    from evals.e3_error_calibration.build_fixture import CASES

    lookup = e3_error_calibration_family_lookup()
    assert len(lookup) == 72
    for name, _category, _rationale, family, _tags in CASES:
        assert lookup[name] == family


def test_e3_error_calibration_module_is_not_imported_at_harness_module_load_time() -> None:
    """The fixture's CASES module (rationale/family strings) must only be
    imported lazily, inside `e3_error_calibration_family_lookup`, never at
    harness module scope -- consistent with item 44 (fixture labels never
    reach a model) and keeping the harness importable even if that
    submodule were unavailable."""
    assert "evals.e3_error_calibration.build_fixture" not in _HARNESS_SOURCE.replace(
        "    from evals.e3_error_calibration.build_fixture import CASES\n", ""
    )
    assert "import evals.e3_error_calibration" not in _HARNESS_SOURCE.split(
        "def e3_error_calibration_family_lookup"
    )[0]


def _e3_error_worker_run(
    tmp_path: Path, monkeypatch, filenames_and_categories: list[tuple[str, str]], repetition: int = 1
) -> tuple[dict, dict]:
    for name, _category in filenames_and_categories:
        (tmp_path / name).write_bytes(b"")
    responses = _classify_pair(*filenames_and_categories)
    fake = FakeModel(responses, responses)
    monkeypatch.setattr(harness, "build_model", lambda *a, **k: fake)
    q = FakeResultQueue()
    _e4_precision_worker(q, str(tmp_path), "ollama_chat/qwen3.5:4b", False)
    return q.items[0], fake.__dict__


def test_worker_makes_exactly_two_model_calls_on_an_e3_error_calibration_style_file(
    tmp_path: Path, monkeypatch
) -> None:
    """Item 30/37: E4-current and E4-refined must still add zero
    additional model calls when run against a file styled like this new
    fixture (same shared-E3-call design, unmodified)."""
    shared, fake_state = _e3_error_worker_run(
        tmp_path, monkeypatch, [("archivkonzept_entwurf", "Archives")]
    )
    assert shared["status"] == "ok"
    assert len(fake_state["calls"]) == 2  # one shared E3 call, no extra calls per condition
    assert set(shared["conditions"]) == {"E3", "E4-current", "E4-refined"}


def test_rationale_and_family_metadata_never_reach_the_model_prompt(tmp_path: Path, monkeypatch) -> None:
    """Item 24/44: rationale text and stress-family tags are evaluator-only
    and must never appear in a model prompt, exactly like ground truth."""
    from evals.e3_error_calibration.build_fixture import CASES

    name, _category, rationale, family, tags = next(
        c for c in CASES if c[0] == "archivkonzept_entwurf"
    )
    shared, fake_state = _e3_error_worker_run(tmp_path, monkeypatch, [(name, "Archives")])
    assert shared["status"] == "ok"
    for call in fake_state["calls"]:
        prompt_text = call["messages"][0]["content"][0]["text"]
        assert rationale not in prompt_text
        assert family not in prompt_text
        assert "subject_vs_artifact" not in prompt_text  # no family value leaks generically
        for tag in ("multilingual", "compound_morphology"):
            assert tag not in prompt_text
        assert "expected" not in prompt_text.lower()
        assert "ground_truth" not in prompt_text


def test_stress_family_metrics_uses_unique_files_and_reports_error_rate() -> None:
    family_by_file = {"a": "subject_vs_artifact", "b": "subject_vs_artifact", "c": "tool_vs_output"}
    cases = [
        {"filename": "a", "is_review": False, "correct": True},
        {"filename": "b", "is_review": False, "correct": False},
        {"filename": "c", "is_review": True, "correct": False},
    ]
    result = stress_family_metrics(cases, family_by_file)
    assert result["subject_vs_artifact"] == {
        "files": 2,
        "automatic_decisions": 2,
        "reviews": 0,
        "correct_automatic": 1,
        "incorrect_automatic": 1,
        "automatic_error_rate": 0.5,
        "accuracy_on_decided": 0.5,
    }
    assert result["tool_vs_output"]["automatic_decisions"] == 0
    assert result["tool_vs_output"]["automatic_error_rate"] is None  # no division by zero


def test_e3_error_density_distinct_from_unsafe_automation_rate() -> None:
    primary = {"raw_counts": {"correct_automatic": 3, "incorrect_automatic": 1, "review": 6}}
    # density = errors / automatic decisions = 1/4, NOT 1/10 (all files)
    assert e3_error_density(primary) == 0.25


def test_e3_error_density_handles_zero_automatic_decisions() -> None:
    primary = {"raw_counts": {"correct_automatic": 0, "incorrect_automatic": 0, "review": 5}}
    assert e3_error_density(primary) is None


def test_error_family_capture_matrix_counts_caught_and_false_positive_per_family() -> None:
    family_by_file = {
        "caught_by_both": "subject_vs_artifact",
        "missed_by_both": "subject_vs_artifact",
        "current_only_fp": "tool_vs_output",
    }
    e3_cases = [
        {"filename": "caught_by_both", "is_review": False, "correct": False},
        {"filename": "missed_by_both", "is_review": False, "correct": False},
        {"filename": "current_only_fp", "is_review": False, "correct": True},
    ]
    e4_current_cases = [
        {"filename": "caught_by_both", "is_review": True, "correct": True},
        {"filename": "missed_by_both", "is_review": False, "correct": False},
        {"filename": "current_only_fp", "is_review": True, "correct": False},
    ]
    e4_refined_cases = [
        {"filename": "caught_by_both", "is_review": True, "correct": True},
        {"filename": "missed_by_both", "is_review": False, "correct": False},
        {"filename": "current_only_fp", "is_review": False, "correct": True},
    ]
    matrix = error_family_capture_matrix(e3_cases, e4_current_cases, e4_refined_cases, family_by_file)
    assert matrix["subject_vs_artifact"]["e3_automatic_errors"] == 2
    assert matrix["subject_vs_artifact"]["e4_current_caught"] == 1
    assert matrix["subject_vs_artifact"]["e4_refined_caught"] == 1
    assert matrix["tool_vs_output"]["e4_current_false_positive"] == 1
    assert matrix["tool_vs_output"]["e4_refined_false_positive"] == 0


def test_review_escape_rescue_counts_e3_abstention_failures_and_e4_rescues() -> None:
    e3_cases = [
        {"filename": "correctly_abstained", "ground_truth_review_only": True, "is_review": True},
        {"filename": "escaped_and_rescued", "ground_truth_review_only": True, "is_review": False},
        {"filename": "escaped_not_rescued", "ground_truth_review_only": True, "is_review": False},
        {"filename": "real_category_file", "ground_truth_review_only": False, "is_review": False},
    ]
    e4_cases = [
        {"filename": "correctly_abstained", "is_review": True},
        {"filename": "escaped_and_rescued", "is_review": True},
        {"filename": "escaped_not_rescued", "is_review": False},
        {"filename": "real_category_file", "is_review": False},
    ]
    result = review_escape_rescue(e3_cases, e4_cases)
    assert result == {
        "n": 3,
        "e3_correctly_reviewed": 1,
        "e3_incorrectly_automated": 2,
        "rescued": 1,
    }


def test_compute_e3_error_calibration_diagnostics_end_to_end_on_synthetic_runs() -> None:
    """Exercises the full compute_e3_error_calibration_diagnostics pipeline
    on synthetic (non-inferred) runs -- no real E3/E4 numbers are produced
    or claimed here, only the aggregation machinery is validated."""
    family_by_file = e3_error_calibration_family_lookup()
    # Two real fixture filenames with distinct primary families, one of
    # which E3 gets wrong.
    subject_file = next(
        name
        for name, family in family_by_file.items()
        if family == "subject_vs_artifact" and name.split("_")[0] not in ("archivkonzept",)
    )
    trap_file = next(name for name, family in family_by_file.items() if family == "tool_vs_output")

    expected = {subject_file: ["Documents"], trap_file: ["Code"]}
    e3_final = {subject_file: "Images", trap_file: "Code"}  # subject_file wrong, trap_file correct
    e4_final = dict(e3_final)
    e4_final[subject_file] = REVIEW  # E4-refined catches the E3 error
    detail = [{"filename": subject_file, "e3_category": "Images", "veto_applicable": True}]

    runs = [
        _raw_run("E3", "e3_error_calibration", 1, dict(e3_final), []),
        _raw_run("E4-refined", "e3_error_calibration", 1, dict(e4_final), list(detail)),
    ]
    diagnostics = compute_e3_error_calibration_diagnostics(runs, expected, review_directory=REVIEW)

    family = family_by_file[subject_file]
    assert diagnostics["stress_family_metrics"][family]["incorrect_automatic"] == 1
    assert diagnostics["e3_error_density"] == 0.5  # 1 wrong of 2 automatic decisions
    assert diagnostics["real_category_subset"]["wrong_automatic"] == 1


def test_combined_unique_n_is_257_across_all_four_fixtures() -> None:
    """Item 31: total unique Development files across calibration (47) +
    boundary_calibration (66) + veto_precision_calibration (72) +
    e3_error_calibration (72) = 257, and every fixture must be reported
    separately as well as combined, never hidden inside a combined-only
    result."""
    expected_sizes = {
        "calibration": 47,
        "boundary_calibration": 66,
        "veto_precision_calibration": 72,
        "e3_error_calibration": 72,
    }
    assert sum(expected_sizes.values()) == 257

    per_fixture: dict[str, dict] = {}
    expected_by_fixture: dict[str, dict[str, list[str]]] = {}
    runs_by_fixture: dict[str, list[dict]] = {}
    for fixture_name, size in expected_sizes.items():
        expected = {f"{fixture_name}_file_{i}": ["Documents"] for i in range(size)}
        final = {name: "Documents" for name in expected}
        runs = [_raw_run("E3", fixture_name, 1, dict(final), [])]
        per_fixture[fixture_name] = aggregate(runs, expected, review_directory=REVIEW)
        expected_by_fixture[fixture_name] = expected
        runs_by_fixture[fixture_name] = runs

    combined = _combined(per_fixture, expected_by_fixture, runs_by_fixture, review_directory=REVIEW)
    assert combined["unique_file_metrics"]["E3"]["primary"]["total_files_scored"] == 257
    assert set(per_fixture) == set(expected_sizes)  # every fixture still reported separately


def test_regression_bug_shape_still_correct_across_four_fixtures() -> None:
    """Item 47: the recently fixed unique-file/repetition-count bug must
    remain fixed once a fourth fixture is added -- 1 unique E3 wrong file,
    5 identical repetitions, still yields unique_e3_automatic_errors == 1
    and underpowered == True, combined across all four fixtures, NOT 5."""
    per_fixture: dict[str, dict] = {}
    expected_by_fixture: dict[str, dict[str, list[str]]] = {}
    runs_by_fixture: dict[str, list[dict]] = {}

    clean_fixtures = {"calibration": 3, "boundary_calibration": 3, "veto_precision_calibration": 3}
    for fixture_name, size in clean_fixtures.items():
        expected = {f"{fixture_name}_file_{i}": ["Documents"] for i in range(size)}
        final = {name: "Documents" for name in expected}
        runs = [_raw_run("E3", fixture_name, r, dict(final), []) for r in range(1, 6)]
        per_fixture[fixture_name] = aggregate(runs, expected, review_directory=REVIEW)
        expected_by_fixture[fixture_name] = expected
        runs_by_fixture[fixture_name] = runs

    error_expected = {"good1": ["Documents"], "bad1": ["Archives"]}
    error_final = {"good1": "Documents", "bad1": "Documents"}  # bad1 wrong every repetition
    error_runs = [_raw_run("E3", "e3_error_calibration", r, dict(error_final), []) for r in range(1, 6)]
    per_fixture["e3_error_calibration"] = aggregate(error_runs, error_expected, review_directory=REVIEW)
    expected_by_fixture["e3_error_calibration"] = error_expected
    runs_by_fixture["e3_error_calibration"] = error_runs

    combined = _combined(per_fixture, expected_by_fixture, runs_by_fixture, review_directory=REVIEW)
    evidence = compute_evidence_strength(combined["unique_file_metrics"]["E3"])
    assert evidence["unique_e3_automatic_errors"] == 1  # not 5
    assert evidence["underpowered"] is True  # not False
