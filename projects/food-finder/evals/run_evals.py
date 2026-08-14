"""Run deterministic Food Finder tool-use evaluations against a configured model."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import yaml
from dotenv import load_dotenv
from smolagents import LiteLLMModel, Model

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent import build_agent  # noqa: E402

DEFAULT_CASES_PATH = Path(__file__).with_name("cases.yaml")
TOOL_ARGUMENT_NAMES = ("city", "cuisine", "min_rating", "min_reviews")


@dataclass
class EvalResult:
    case_id: str
    expected_tool: str | None
    observed_tools: list[str]
    observed_arguments: dict[str, Any] | None
    tool_decision_correct: bool
    arguments_correct: bool | None
    steps: int
    latency_seconds: float
    error: str | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--model-id", help="LiteLLM model ID; defaults to MODEL_ID")
    parser.add_argument("--api-base", help="Model endpoint; defaults to API_BASE")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Mock Google Places responses so the evaluation uses no API quota",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Markdown destination; defaults to evals/results/<model>-<timestamp>.md",
    )
    return parser.parse_args(argv)


def load_cases(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    cases = data.get("cases") if isinstance(data, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{path} must contain a non-empty 'cases' list")

    required = {"id", "prompt", "tool_needed", "expected_tool", "expected_arguments"}
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict) or not required.issubset(case):
            missing = required.difference(case if isinstance(case, dict) else {})
            raise ValueError(f"Case {index} is invalid; missing: {sorted(missing)}")
    return cases


def build_model(model_id: str | None, api_base: str | None) -> LiteLLMModel:
    load_dotenv(REPO_ROOT / ".env")
    resolved_model_id = model_id or os.getenv("MODEL_ID", "ollama_chat/qwen3.5:4b")
    configured_base = api_base if api_base is not None else os.getenv("API_BASE", "")
    resolved_base = configured_base.strip() or (
        "http://localhost:11434" if resolved_model_id.startswith("ollama") else None
    )
    model_options: dict[str, Any] = {"temperature": 0.0}
    if resolved_model_id.startswith("ollama"):
        model_options["num_ctx"] = 8192
    return LiteLLMModel(
        model_id=resolved_model_id,
        api_base=resolved_base,
        **model_options,
    )


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return "<dynamic>"


def extract_tool_calls(memory: Any) -> list[dict[str, Any]]:
    """Extract calls to named tools from CodeAgent Python actions in memory."""
    calls: list[dict[str, Any]] = []
    for step in getattr(memory, "steps", []):
        code = getattr(step, "code_action", None)
        if not code:
            continue
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            else:
                continue
            if name != "search_restaurants":
                continue

            arguments = {
                argument_name: _literal(argument)
                for argument_name, argument in zip(TOOL_ARGUMENT_NAMES, node.args)
            }
            arguments.update(
                {
                    keyword.arg: _literal(keyword.value)
                    for keyword in node.keywords
                    if keyword.arg is not None
                }
            )
            calls.append({"name": name, "arguments": arguments})
    return calls


def _normalize(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.casefold().split())
    if isinstance(value, float):
        return round(value, 6)
    return value


def arguments_match(expected: dict[str, Any], observed: dict[str, Any]) -> bool:
    """Compare declared expected arguments; unspecified defaults are ignored."""
    return all(
        key in observed and _normalize(observed[key]) == _normalize(expected_value)
        for key, expected_value in expected.items()
    )


def _offline_response() -> Mock:
    response = Mock()
    response.ok = True
    response.status_code = 200
    response.text = ""
    response.json.return_value = {
        "places": [
            {
                "displayName": {"text": "Offline Test Restaurant"},
                "formattedAddress": "1 Fixture Street",
                "rating": 4.7,
                "userRatingCount": 1500,
                "priceLevel": "PRICE_LEVEL_MODERATE",
                "primaryType": "restaurant",
                "googleMapsUri": "https://maps.example/offline-fixture",
            }
        ]
    }
    return response


def run_case(
    case: dict[str, Any],
    *,
    model: Model,
    max_steps: int,
    offline: bool,
) -> EvalResult:
    agent = build_agent(model, max_steps=max_steps, verbosity_level=0)
    started = time.perf_counter()
    raised_error: str | None = None

    with ExitStack() as stack:
        if offline:
            stack.enter_context(
                patch.dict(os.environ, {"GOOGLE_PLACES_API_KEY": "offline-eval-key"})
            )
            stack.enter_context(
                patch("src.agent.tools.requests.post", return_value=_offline_response())
            )
        try:
            agent.run(case["prompt"])
        except Exception as exc:  # Keep the remaining eval cases running.
            raised_error = f"{type(exc).__name__}: {exc}"

    latency = time.perf_counter() - started
    calls = extract_tool_calls(agent.memory)
    observed_tools = [call["name"] for call in calls]
    expected_tool = case["expected_tool"]
    matching_call = next(
        (call for call in calls if call["name"] == expected_tool),
        None,
    )

    if case["tool_needed"]:
        decision_correct = matching_call is not None
        args_correct = bool(
            matching_call
            and arguments_match(case["expected_arguments"], matching_call["arguments"])
        )
    else:
        decision_correct = not calls
        args_correct = None

    memory_errors = [
        str(step.error)
        for step in getattr(agent.memory, "steps", [])
        if getattr(step, "error", None) is not None
    ]
    error = raised_error or ("; ".join(memory_errors) if memory_errors else None)
    steps = sum(
        1 for step in getattr(agent.memory, "steps", []) if hasattr(step, "step_number")
    )
    return EvalResult(
        case_id=case["id"],
        expected_tool=expected_tool,
        observed_tools=observed_tools,
        observed_arguments=matching_call["arguments"] if matching_call else None,
        tool_decision_correct=decision_correct,
        arguments_correct=args_correct,
        steps=steps,
        latency_seconds=latency,
        error=error,
    )


def _percent(numerator: int, denominator: int) -> str:
    return f"{(100 * numerator / denominator):.1f}%" if denominator else "n/a"


def _cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_report(
    results: list[EvalResult],
    *,
    model_id: str,
    offline: bool,
) -> str:
    tool_calls = sum(bool(result.observed_tools) for result in results)
    tool_decisions = sum(result.tool_decision_correct for result in results)
    argument_results = [
        result.arguments_correct
        for result in results
        if result.arguments_correct is not None
    ]
    errors = sum(result.error is not None for result in results)
    avg_steps = sum(result.steps for result in results) / len(results)
    avg_latency = sum(result.latency_seconds for result in results) / len(results)

    lines = [
        f"# Food Finder evaluation: `{model_id}`",
        "",
        f"- Mode: {'offline (Google Places mocked)' if offline else 'live'}",
        f"- Cases: {len(results)}",
        f"- Tool-call rate: {_percent(tool_calls, len(results))}",
        f"- Tool-decision accuracy: {_percent(tool_decisions, len(results))}",
        "- Argument accuracy: "
        + _percent(
            sum(value is True for value in argument_results),
            len(argument_results),
        ),
        f"- Average steps: {avg_steps:.2f}",
        f"- Average latency: {avg_latency:.2f} s",
        f"- Error rate: {_percent(errors, len(results))}",
        "",
        "| Case | Expected tool | Observed tool(s) | Arguments | Decision correct "
        "| Arguments correct | Steps | Latency (s) | Error |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for result in results:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    result.case_id,
                    result.expected_tool,
                    result.observed_tools,
                    result.observed_arguments,
                    result.tool_decision_correct,
                    result.arguments_correct,
                    result.steps,
                    f"{result.latency_seconds:.2f}",
                    result.error,
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Argument accuracy is an exact, case-insensitive comparison of the declared",
            "expected argument subset against literal `search_restaurants(...)` calls",
            "recorded in `agent.memory`. No judge model is used.",
            "",
        ]
    )
    return "\n".join(lines)


def _default_output_path(model_id: str) -> Path:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", model_id).strip("-") or "model"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / "evals" / "results" / f"{slug}-{timestamp}.md"


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    cases = load_cases(args.cases)
    model = build_model(args.model_id, args.api_base)
    model_id = args.model_id or os.getenv("MODEL_ID", "ollama_chat/qwen3.5:4b")

    results = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['id']}", file=sys.stderr)
        results.append(
            run_case(
                case,
                model=model,
                max_steps=args.max_steps,
                offline=args.offline,
            )
        )

    output = args.output or _default_output_path(model_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_report(results, model_id=model_id, offline=args.offline),
        encoding="utf-8",
    )
    print(output)
    return output


if __name__ == "__main__":
    main()
