from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest
from tidy.agent import (
    build_classifier,
    build_group_agent,
    build_group_task,
    build_model,
    endpoint_is_local,
    safe_endpoint_display,
)
from tidy.classification import StructuredClassifier


def test_classification_builds_structured_classifier_not_code_agent() -> None:
    classifier = build_classifier(Mock())

    assert isinstance(classifier, StructuredClassifier)
    assert not hasattr(classifier, "python_executor")
    assert not hasattr(classifier, "tools")


@pytest.mark.slow
def test_request_timeout_is_explicit_and_configurable(monkeypatch) -> None:
    """A slow host must not be recorded as a failed agent."""
    monkeypatch.delenv("REQUEST_TIMEOUT", raising=False)
    default_model = build_model("ollama_chat/test-model", think=False)

    monkeypatch.setenv("REQUEST_TIMEOUT", "45")
    configured = build_model("ollama_chat/test-model", think=False)

    # LiteLLM's own default is 600 s, which one CPU-only generation can exceed.
    assert default_model.kwargs["timeout"] == 1800.0
    assert configured.kwargs["timeout"] == 45.0


def test_content_reading_never_reaches_the_clustering_agent() -> None:
    """Grouping stays metadata-only whatever structured classification may read."""
    agent = build_group_agent(Mock(), verbosity_level=0)

    assert "peek_file" not in agent.tools
    assert "Never request or" in agent.instructions


def test_group_agent_has_only_metadata_and_group_proposal_tools() -> None:
    agent = build_group_agent(Mock(), verbosity_level=0)

    domain_tools = set(agent.tools) - {"final_answer"}
    assert domain_tools == {"propose_groups"}
    assert "scan_directory" not in agent.tools


def test_group_task_contains_complete_eligible_set() -> None:
    task = build_group_task(
        [
            {
                "name": "Bachelorarbeit_v4.docx",
                "extension": ".docx",
                "size_bytes": 0,
                "mtime": "2026-08-10T10:20:30.123456+00:00",
            },
            {
                "name": "Kritik_Bachelorarbeit.md",
                "extension": ".md",
                "size_bytes": 0,
                "mtime": "2026-08-10T10:20:30.123456+00:00",
            },
        ]
    )

    assert "Bachelorarbeit_v4.docx" in task
    assert "Kritik_Bachelorarbeit.md" in task
    assert "size_bytes" not in task
    assert "mtime" not in task
    assert "{" not in task
    assert "Every filename not named in a group remains ungrouped" in task


def test_build_classifier_forwards_think_to_litellm_model() -> None:
    classifier = build_classifier("ollama_chat/test-model", think=False)

    assert classifier.backend.model.kwargs["think"] is False


def test_think_can_be_loaded_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("THINK", "false")

    model = build_model("ollama_chat/test-model")

    assert model.kwargs["think"] is False


def test_explicit_think_overrides_environment(monkeypatch) -> None:
    monkeypatch.setenv("THINK", "false")

    model = build_model("ollama_chat/test-model", think=True)

    assert model.kwargs["think"] is True


def test_build_model_logs_effective_configuration(monkeypatch, caplog) -> None:
    monkeypatch.setenv("API_BASE", "http://ollama.example:11434")
    monkeypatch.setenv("THINK", "false")

    with caplog.at_level(logging.INFO, logger="tidy.agent"):
        build_model("ollama_chat/test-model")

    assert "model_id=ollama_chat/test-model" in caplog.text
    assert "api_base=remote endpoint (address redacted)" in caplog.text
    assert "ollama.example" not in caplog.text
    assert "think=False" in caplog.text


def test_endpoint_locality_accepts_only_loopback(monkeypatch) -> None:
    for endpoint in (
        "http://localhost:11434",
        "http://127.0.0.1:11434",
        "http://[::1]:11434",
    ):
        monkeypatch.setenv("API_BASE", endpoint)
        assert endpoint_is_local("ollama_chat/test") is True

    monkeypatch.setenv("API_BASE", "http://192.168.1.10:11434")
    assert endpoint_is_local("ollama_chat/test") is False
    monkeypatch.delenv("API_BASE")
    assert endpoint_is_local("openai/test") is False


def test_safe_endpoint_display_drops_remote_addresses_and_credentials() -> None:
    endpoint = "https://alice:secret@api.example.test/v1?key=private"

    label = safe_endpoint_display("openai/test", api_base=endpoint)

    assert label == "remote endpoint (address redacted)"
    for secret in ("alice", "secret", "api.example.test", "private"):
        assert secret not in label
