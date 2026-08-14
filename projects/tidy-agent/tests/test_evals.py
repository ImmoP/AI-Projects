from __future__ import annotations

import re
import time
from pathlib import Path
from types import SimpleNamespace

from evals.run_evals import (
    GroupingExpectation,
    _aggregate,
    _category_metrics,
    _content_rows,
    _delayed_result_worker,
    _grouping_metrics,
    endpoint_url,
    main,
    memory_metrics,
    parse_args,
    render_grouping_markdown,
    reset_model_state,
    resolved_endpoint,
    run_evaluation,
    run_in_subprocess,
)


def test_memory_metrics_count_invalid_entries_corrections_steps_and_latency() -> None:
    invalid_feedback = (
        '{"ok": false, "moves": [], "errors": '
        '[{"index": 0}, {"index": 0}, {"index": 2}], '
        '"allowed_categories": ["Images"]}'
    )
    memory = SimpleNamespace(
        steps=[
            SimpleNamespace(
                step_number=1,
                timing=SimpleNamespace(start_time=10.0, end_time=10.4),
                code_action="feedback = propose_plan(moves=moves)",
                tool_calls=[],
                observations=invalid_feedback,
                token_usage=SimpleNamespace(output_tokens=20),
            ),
            SimpleNamespace(
                step_number=2,
                timing=SimpleNamespace(start_time=10.4, end_time=11.0),
                code_action="feedback = propose_plan(moves=corrected)",
                tool_calls=[],
                observations=(
                    '{"ok": true, "moves": [], "errors": [], '
                    '"allowed_categories": ["Images"]}'
                ),
                token_usage=SimpleNamespace(output_tokens=12),
            ),
        ]
    )

    metrics = memory_metrics(memory)

    assert metrics.invalid_plan_entries == 2
    assert metrics.correction_rounds == 1
    assert metrics.steps == 2
    assert metrics.latency_seconds == 1.0
    assert metrics.completion_tokens == 32


def test_case_timeout_returns_without_blocking_following_work() -> None:
    started = time.perf_counter()

    timed_out = run_in_subprocess(
        _delayed_result_worker,
        (0.5,),
        timeout=0.05,
    )
    following = run_in_subprocess(
        _delayed_result_worker,
        (0.0,),
        timeout=10.0,
    )

    assert timed_out["status"] == "timeout"
    assert following["status"] == "ok"
    # Process spawn includes importing smolagents and can take a couple of
    # seconds on slower CI hosts. The important regression is that the 0.5 s
    # worker is terminated at its 0.05 s deadline and does not block the next.
    assert timed_out["latency_seconds"] < 0.5
    assert time.perf_counter() - started < 15.0


def test_aggregate_reports_overall_and_unknown_accuracy_separately() -> None:
    cases = [
        {
            "correct": True,
            "unresolved": False,
            "mode": "rule",
            "predicted": "Images",
            "allowed": ["Images"],
            "status": "ok",
        },
        {
            "correct": False,
            "unresolved": True,
            "mode": "agent",
            "predicted": "_ToReview",
            "allowed": ["Documents", "Archives"],
            "status": "ok",
            "steps": 2,
            "latency_seconds": 1.5,
            "completion_tokens": 40,
        },
    ]

    metrics = _aggregate(cases, totals={"all": 2, "unknown": 1})

    assert metrics["overall_accuracy"] == 0.5
    assert metrics["unknown_accuracy"] == 0.0
    assert metrics["average_steps"] == 2.0
    assert metrics["average_completion_tokens"] == 40.0


def _write_mode_fixture(tmp_path: Path) -> tuple[Path, Path]:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    for filename in ("draft-v1.pdf", "draft-v2.docx", "draft-notes.txt", "photo.jpg"):
        (fixture / filename).touch()
    (fixture / "mystery-name").touch()
    expected = tmp_path / "expected.yaml"
    expected.write_text(
        "files:\n"
        "  draft-v1.pdf: [Documents]\n"
        "  draft-v2.docx: [Documents]\n"
        "  draft-notes.txt: [Documents]\n"
        "  photo.jpg: [Images]\n"
        "  mystery-name: [_ToReview]\n"
        "groups:\n"
        "  draft:\n"
        "    - draft-v1.pdf\n"
        "    - draft-v2.docx\n"
        "    - draft-notes.txt\n"
        "scatter:\n"
        "  - photo.jpg\n",
        encoding="utf-8",
    )
    return fixture, expected


def test_no_group_run_reports_category_metrics_without_clustering_metrics(
    tmp_path: Path,
) -> None:
    """A run without --group forms no groups, so clustering metrics do not apply."""
    fixture, expected = _write_mode_fixture(tmp_path)
    output = tmp_path / "report.md"

    metrics = run_evaluation(
        fixture=fixture,
        expected_path=expected,
        output=output,
        model_id=None,
        think=None,
        timeout=10.0,
        use_agent=False,
    )

    report = output.read_text(encoding="utf-8")
    # Rules place the four extension matches; only the extensionless name falls
    # back to _ToReview, which is also its expected answer.
    assert metrics["overall_accuracy"] == 1.0
    assert metrics["unknown_accuracy"] == 1.0
    assert metrics["unknown_total"] == 1
    assert metrics["omitted_count"] == 0
    assert "Run mode: **fixed categories only**" in report
    assert "Metric family: **category accuracy only**" in report
    assert "Category accuracy, all eligible files" in report
    assert "Category accuracy, all unresolved files (`_ToReview` accepted)" in report
    assert "Decision rate, unresolved files" in report
    assert "Accuracy on decided files only" in report
    assert "Files omitted by the model" in report
    # The decisive regression: no clustering metric may appear in this mode,
    # neither with a value nor as 0/N/A.
    for clustering_row in (
        "Clustering purity",
        "Fully co-located expected groups",
        "Scatter files in an accepted group folder",
        "Invalid proposed folder names",
        "Clustering run timeout",
    ):
        assert clustering_row not in report


def _grouping_payload(**overrides: object) -> dict:
    """Payload of a run that clusters three drafts and leaves the scatter alone."""
    payload = {
        "status": "ok",
        "moves": [
            {"source": "draft-v1.pdf", "destination": "Draft/draft-v1.pdf"},
            {"source": "draft-v2.docx", "destination": "Draft/draft-v2.docx"},
            {"source": "draft-notes.txt", "destination": "Draft/draft-notes.txt"},
            {"source": "photo.jpg", "destination": "Images/photo.jpg"},
        ],
        "grouped_sources": ["draft-notes.txt", "draft-v1.pdf", "draft-v2.docx"],
        "group_folders": ["Draft"],
        "proposed_group_members": [
            "draft-notes.txt",
            "draft-v1.pdf",
            "draft-v2.docx",
        ],
        "discarded_group_members": [],
        "agent_runs": 1,
    }
    payload.update(overrides)
    return payload


def test_grouping_metrics_score_membership_independently_of_categories() -> None:
    """The clustering family is scored only from group membership."""
    expected = GroupingExpectation(
        groups={"draft": ("draft-v1.pdf", "draft-v2.docx", "draft-notes.txt")},
        scatter=("photo.jpg",),
    )

    metrics = _grouping_metrics(expected, _grouping_payload())

    assert metrics["clustering_purity"] == 1.0
    assert metrics["group_cohesion"] == 1.0
    assert metrics["scatter_in_group"] == 0


def test_scatter_in_fixed_category_folder_is_not_a_clustering_error() -> None:
    """Sharing `Documents/` is an extension-rule effect, not a formed cluster.

    The two quantities were previously the same row, which let rule placements
    inflate a clustering metric.
    """
    expected = GroupingExpectation(
        groups={"draft": ("draft-v1.pdf", "draft-v2.docx", "draft-notes.txt")},
        scatter=("recipe.txt", "ticket.pdf", "photo.jpg"),
    )
    payload = _grouping_payload(
        moves=[
            {"source": "draft-v1.pdf", "destination": "Draft/draft-v1.pdf"},
            {"source": "draft-v2.docx", "destination": "Draft/draft-v2.docx"},
            {"source": "draft-notes.txt", "destination": "Draft/draft-notes.txt"},
            {"source": "recipe.txt", "destination": "Documents/recipe.txt"},
            {"source": "ticket.pdf", "destination": "Documents/ticket.pdf"},
            {"source": "photo.jpg", "destination": "Images/photo.jpg"},
        ],
    )

    metrics = _grouping_metrics(expected, payload)

    assert metrics["scatter_in_group"] == 0
    assert metrics["scatter_sharing_category"] == 2
    assert metrics["clustering_purity"] == 1.0


def test_scatter_dropped_by_the_size_filter_is_reported_before_and_after() -> None:
    """A cluster the executor discards is still a clustering decision made."""
    expected = GroupingExpectation(
        groups={"draft": ("draft-v1.pdf", "draft-v2.docx", "draft-notes.txt")},
        scatter=("photo.jpg", "recipe.txt"),
    )
    payload = _grouping_payload(
        moves=[
            {"source": "draft-v1.pdf", "destination": "Draft/draft-v1.pdf"},
            {"source": "draft-v2.docx", "destination": "Draft/draft-v2.docx"},
            {"source": "draft-notes.txt", "destination": "Draft/draft-notes.txt"},
            {"source": "photo.jpg", "destination": "Images/photo.jpg"},
            {"source": "recipe.txt", "destination": "Documents/recipe.txt"},
        ],
        proposed_group_members=[
            "draft-notes.txt",
            "draft-v1.pdf",
            "draft-v2.docx",
            "photo.jpg",
            "recipe.txt",
        ],
        discarded_group_members=["photo.jpg", "recipe.txt"],
    )

    metrics = _grouping_metrics(expected, payload)

    assert metrics["scatter_in_group"] == 0
    assert metrics["scatter_in_proposed_group"] == 2
    assert metrics["scatter_in_discarded_group"] == 2


_PROMPT_METRIC_STUB = {
    "legacy_prompt_chars": 10,
    "compact_prompt_chars": 5,
    "legacy_prompt_tokens": 10,
    "compact_prompt_tokens": 5,
    "prompt_token_reduction": 0.5,
    "prompt_measurement_method": "stub",
}


def _table_rows(report: str, heading: str) -> list[list[str]]:
    """Return the body cells of the Markdown table under *heading*."""
    section = report.split(heading, 1)[1]
    rows = []
    for line in section.splitlines():
        if not line.startswith("|"):
            if rows:
                break
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if set("".join(cells)) <= {"-", ":"} or cells[0] in {"File", "Metric"}:
            continue
        rows.append(cells)
    return rows


def _metric_fraction(report: str, label: str) -> tuple[int, int]:
    """Read the ``(n/m)`` or ``n/m`` value of one metric row."""
    for line in report.splitlines():
        if line.startswith(f"| {label} "):
            match = re.search(r"(\d+)/(\d+)", line)
            assert match, f"no fraction in row: {line}"
            return int(match.group(1)), int(match.group(2))
    raise AssertionError(f"metric row not found: {label}")


def _mixed_run_report() -> str:
    """Render one grouping report that reproduces both reporting conflicts.

    Three scatter files share ``Documents/`` without being clustered, and one
    grouped file carries category ground truth but no clustering ground truth —
    the exact shape in which the report used to contradict its own tables.
    """
    expected = GroupingExpectation(
        groups={"draft": ("draft-v1.pdf", "draft-v2.docx", "draft-notes.txt")},
        scatter=("recipe.txt", "ticket.pdf", "photo.jpg"),
    )
    payload = {
        "status": "ok",
        "moves": [
            {"source": "draft-v1.pdf", "destination": "Draft/draft-v1.pdf"},
            {"source": "draft-v2.docx", "destination": "Draft/draft-v2.docx"},
            {"source": "draft-notes.txt", "destination": "Draft/draft-notes.txt"},
            {"source": "extra.docx", "destination": "Draft/extra.docx"},
            {"source": "recipe.txt", "destination": "Documents/recipe.txt"},
            {"source": "ticket.pdf", "destination": "Documents/ticket.pdf"},
            {"source": "photo.jpg", "destination": "Images/photo.jpg"},
        ],
        "grouped_sources": [
            "draft-notes.txt",
            "draft-v1.pdf",
            "draft-v2.docx",
            "extra.docx",
        ],
        "group_folders": ["Draft"],
        "proposed_group_members": [
            "draft-notes.txt",
            "draft-v1.pdf",
            "draft-v2.docx",
            "extra.docx",
        ],
        "discarded_group_members": [],
        "agent_runs": 2,
    }
    expected_categories = {
        "draft-v1.pdf": ["Documents"],
        "draft-v2.docx": ["Documents"],
        "draft-notes.txt": ["Documents"],
        "extra.docx": ["Documents"],
        "recipe.txt": ["Documents"],
        "ticket.pdf": ["Documents", "_ToReview"],
        "photo.jpg": ["Images"],
    }
    metrics = _grouping_metrics(expected, payload)
    metrics.update(_PROMPT_METRIC_STUB)
    metrics["category"] = _category_metrics(expected_categories, ["ticket.pdf"], payload)
    return render_grouping_markdown(
        metrics,
        model="stub",
        think=False,
        timeout=1.0,
        group_timeout=1.0,
        read_contents=False,
        warmup=None,
    )


def test_scatter_rows_match_the_final_placement_table() -> None:
    """Both scatter counts must be recountable from the report's own table."""
    report = _mixed_run_report()
    placement = _table_rows(report, "## Final placement")
    scatter_rows = [row for row in placement if row[1].startswith("`scatter:")]
    destinations = [row[2] for row in placement]

    in_group = [row for row in scatter_rows if row[3] == "group"]
    sharing_category = [
        row
        for row in scatter_rows
        if row[3] == "category" and destinations.count(row[2]) > 1
    ]

    assert _metric_fraction(report, "Scatter files in an accepted group folder") == (
        len(in_group),
        len(scatter_rows),
    )
    assert _metric_fraction(
        report, "Scatter files sharing a fixed category folder (not a clustering error)"
    ) == (len(sharing_category), len(scatter_rows))
    # The decisive regression: three scatter files land in one `Documents/`
    # folder without any of them having been clustered.
    assert len(sharing_category) == 2
    assert len(in_group) == 0


def test_grouped_file_counts_match_the_report_tables() -> None:
    """The three group-size numbers must agree with the tables that list them."""
    report = _mixed_run_report()
    placement = _table_rows(report, "## Final placement")
    excluded = _table_rows(report, "## Files excluded from category scoring")
    assignments = _table_rows(report, "## Assignments")

    grouped_placement_rows = [row for row in placement if row[3] == "group"]
    excluded_claim = int(
        re.search(
            r"(\d+) scored file\(s\) were placed in a semantic group folder", report
        ).group(1)
    )

    # Claim about excluded files == rows of the table that lists them.
    assert excluded_claim == len(excluded)
    # Clustering denominator == rows of the placement table, not the excluded set.
    assert _metric_fraction(report, "Ground-truth files placed in a group folder") == (
        len(grouped_placement_rows),
        len(placement),
    )
    assert _metric_fraction(report, "Clustering purity, files in group folders")[1] == len(
        grouped_placement_rows
    )
    # The difference between the two counts is stated, not left to be inferred.
    assert (
        f"| Files in a group folder without clustering ground truth | "
        f"{len(excluded) - len(grouped_placement_rows)} |"
    ) in report
    # Excluded files appear in no accuracy table.
    excluded_names = {row[0] for row in excluded}
    assert excluded_names.isdisjoint({row[0] for row in assignments})
    assert len(assignments) == 7 - len(excluded)


def test_decision_rate_separates_deciding_from_being_right() -> None:
    """Abstaining must not score like a correct decision."""
    expected_categories = {
        "decided-right": ["Documents"],
        "decided-wrong": ["Images", "_ToReview"],
        "abstained": ["Documents", "_ToReview"],
    }
    payload = {
        "status": "ok",
        "moves": [
            {"source": "decided-right", "destination": "Documents/decided-right"},
            {"source": "decided-wrong", "destination": "Archives/decided-wrong"},
            {"source": "abstained", "destination": "_ToReview/abstained"},
        ],
        "agent_runs": 1,
    }

    metrics = _category_metrics(
        expected_categories, list(expected_categories), payload
    )

    # The permissive rubric counts the abstention as correct.
    assert metrics["unknown_accuracy"] == 2 / 3
    # The split does not: two of three files were decided, one of them correctly.
    assert metrics["decision_rate"] == 2 / 3
    assert metrics["decided_count"] == 2
    assert metrics["decided_accuracy"] == 0.5
    assert metrics["abstained_count"] == 1
    # Strict subset: only the file with exactly one accepted category.
    assert metrics["strict_total"] == 1
    assert metrics["strict_accuracy"] == 1.0


def test_category_metrics_separate_omitted_files_from_invalid_fallbacks() -> None:
    """Omitted and invalidly assigned files are two different failures."""
    payload = {
        "status": "ok",
        "moves": [
            {"source": "kept", "destination": "Documents/kept"},
            {"source": "lost", "destination": "_ToReview/lost"},
            {"source": "broken", "destination": "_ToReview/broken"},
        ],
        "omitted_sources": ["lost"],
        "invalid_sources": ["broken"],
        "agent_runs": 1,
    }

    metrics = _category_metrics(
        {"kept": ["Documents"], "lost": ["Documents"], "broken": ["Documents"]},
        ["kept", "lost", "broken"],
        payload,
    )

    assert metrics["omitted_count"] == 1
    assert metrics["omitted_files"] == ["lost"]
    assert metrics["invalid_fallback_count"] == 1
    assert metrics["unknown_accuracy"] == 1 / 3


def test_group_timeout_has_independent_default_and_override() -> None:
    assert parse_args([]).timeout == 120.0
    assert parse_args([]).group_timeout == 600.0
    assert parse_args(["--group-timeout", "900"]).group_timeout == 900.0


def test_repeated_runs_report_a_range_instead_of_one_number(tmp_path: Path) -> None:
    """Two runs that differ must be shown as a spread, not averaged away."""
    fixture, expected = _write_mode_fixture(tmp_path)
    output = tmp_path / "summary.md"

    exit_code = main(
        [
            "--no-agent",
            "--repeat",
            "2",
            "--fixture",
            str(fixture),
            "--expected",
            str(expected),
            "--output",
            str(output),
        ]
    )

    summary = output.read_text(encoding="utf-8")
    assert exit_code == 0
    assert "- Runs: **2**" in summary
    assert (tmp_path / "summary-run1.md").is_file()
    assert (tmp_path / "summary-run2.md").is_file()
    assert "| Metric | run 1 | run 2 | Range (within-session, warm) |" in summary
    # The rules-only baseline is deterministic, so every row must say so rather
    # than hide a spread behind a single value.
    assert "identical" in summary
    assert "| Decision rate, unresolved files |" in summary


def test_warm_repetitions_are_labelled_as_a_lower_bound(tmp_path: Path) -> None:
    """A range of zero from one warm process must not read as measured variance."""
    fixture, expected = _write_mode_fixture(tmp_path)
    output = tmp_path / "summary.md"

    main(
        [
            "--no-agent",
            "--repeat",
            "2",
            "--no-reset-between-runs",
            "--fixture",
            str(fixture),
            "--expected",
            str(expected),
            "--output",
            str(output),
        ]
    )

    summary = output.read_text(encoding="utf-8")
    assert "Range (within-session, warm)" in summary
    assert "lower bound of variance" in summary
    assert "Repetition isolation: none" in summary
    # The between-session delta is a measurement and belongs in the report.
    assert "7.1 points apart" in summary


def test_connection_uses_the_real_address_not_the_redacted_label(monkeypatch) -> None:
    """Redaction is for reports only; connecting to the label breaks every run."""
    monkeypatch.setenv("API_BASE", "http://100.104.1.24:11434")

    assert endpoint_url("ollama_chat/x") == "http://100.104.1.24:11434"
    assert resolved_endpoint("ollama_chat/x") != endpoint_url("ollama_chat/x")
    # A reporting label is not a URL, so anything that dials it must not use it.
    assert "://" not in resolved_endpoint("ollama_chat/x")


def test_private_endpoints_are_redacted_in_reports(monkeypatch) -> None:
    """Reports are published; a tailnet or LAN address identifies a machine."""
    for private in (
        "http://100.104.1.24:11434",  # CGNAT range used by tailnets
        "http://192.168.1.9:11434",
        "http://10.0.0.5:11434",
        "http://172.20.0.4:11434",
    ):
        monkeypatch.setenv("API_BASE", private)
        assert resolved_endpoint("ollama_chat/x") == "remote endpoint (address redacted)"

    # Loopback identifies nobody and stays readable; all remote addresses are
    # omitted because reports need locality, not a dialable endpoint.
    monkeypatch.setenv("API_BASE", "http://127.0.0.1:11434")
    assert resolved_endpoint("ollama_chat/x") == "http://127.0.0.1:11434"
    monkeypatch.setenv("API_BASE", "https://api.example.com")
    assert resolved_endpoint("ollama_chat/x") == "remote endpoint (address redacted)"


def test_content_telemetry_rows_never_render_document_excerpts() -> None:
    secret = "PRIVATE DOCUMENT BODY 9f27c"
    metrics = {
        "read_contents": True,
        "unknown_total": 1,
        "peek_eligible": 1,
        "peek_calls": 1,
        "peek_unique_files": 1,
        "peek_source_bytes_considered": len(secret),
        "peek_bytes_read": len(secret),
        "peek_chars_returned": len(secret),
        "peek_readable": 1,
        "endpoint_local": False,
        "private_excerpt": secret,
    }

    rendered = "\n".join(_content_rows(metrics))

    assert secret not in rendered
    assert "Model endpoint locality | remote" in rendered


def test_structured_architecture_telemetry_is_preserved_without_content() -> None:
    payload = {
        "status": "ok",
        "moves": [
            {
                "source": "mystery",
                "destination": "_ToReview/mystery",
                "origin": "agent",
            }
        ],
        "classification_backend": "structured_model",
        "structured_output_mode": "json_schema",
        "classification_requests": 1,
        "peek_phase_requests": 0,
        "final_classification_requests": 1,
        "parse_failures": 0,
        "schema_validation_failures": 0,
        "fallback_to_review_count": 0,
    }

    metrics = _category_metrics(
        {"mystery": ["_ToReview"]},
        ["mystery"],
        payload,
    )

    assert metrics["classification_backend"] == "structured_model"
    assert metrics["structured_output_mode"] == "json_schema"
    assert metrics["classification_requests"] == 1
    assert metrics["final_classification_requests"] == 1


def test_model_reset_is_skipped_for_non_ollama_models() -> None:
    """The reset is provider-specific and must fail loudly rather than silently."""
    result = reset_model_state("gpt-4o-mini")

    assert result["status"] == "skipped"
    assert "gpt-4o-mini" in result["detail"]
