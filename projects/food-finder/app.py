"""Command-line and optional Gradio entry points for Food Finder."""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any

from dotenv import load_dotenv
from smolagents import LiteLLMModel
from src.agent import build_agent

DEFAULT_MODEL_ID = "ollama_chat/qwen3.5:4b"
DEFAULT_OLLAMA_API_BASE = "http://localhost:11434"


def build_model_from_environment() -> LiteLLMModel:
    """Create the configured model without coupling the agent to a provider."""
    load_dotenv()
    model_id = os.getenv("MODEL_ID", DEFAULT_MODEL_ID)
    configured_api_base = os.getenv("API_BASE", "").strip()
    api_base = configured_api_base or (
        DEFAULT_OLLAMA_API_BASE if model_id.startswith("ollama") else None
    )

    logging.getLogger(__name__).info(
        "Using model %s%s",
        model_id,
        f" via {api_base}" if api_base else "",
    )
    model_options: dict[str, Any] = {"temperature": 0.1}
    if model_id.startswith("ollama"):
        model_options["num_ctx"] = 8192
    return LiteLLMModel(model_id=model_id, api_base=api_base, **model_options)


def run_task(task: str) -> str:
    """Run one task with an agent configured from the environment."""
    agent = build_agent(build_model_from_environment())
    return str(agent.run(task))


def launch_gradio() -> None:
    """Launch a minimal browser UI; imported lazily for CLI-only users."""
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise SystemExit(
            "Gradio is not installed. Install it with `pip install gradio`."
        ) from exc

    interface = gr.Interface(
        fn=run_task,
        inputs=gr.Textbox(
            label="Restaurant request",
            placeholder="Find the five best Italian restaurants in Frankfurt.",
            lines=3,
        ),
        outputs=gr.Markdown(label="Recommendations"),
        title="Food Finder",
        description="Restaurant recommendations powered by smolagents and Google Places.",
    )
    interface.launch()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find restaurants with an AI agent.")
    parser.add_argument("task", nargs="*", help="Natural-language restaurant request")
    parser.add_argument(
        "--gradio",
        action="store_true",
        help="Launch the optional Gradio web interface",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Any:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.gradio:
        return launch_gradio()
    if not args.task:
        raise SystemExit("Provide a restaurant request or use --gradio.")
    result = run_task(" ".join(args.task))
    print(result)
    return result


if __name__ == "__main__":
    main()
