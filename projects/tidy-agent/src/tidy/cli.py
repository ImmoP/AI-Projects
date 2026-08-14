"""Command-line interface for the proposal/approval/execution workflow."""

from __future__ import annotations

import argparse
import ast
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .agent import (
    ContentAuthorizationError,
    build_classifier,
    build_group_agent,
    build_group_task,
    ensure_content_authorized,
)
from .classification import (
    ClassificationTelemetry,
    ValidatedClassification,
)
from .executor import (
    ExecutionResult,
    GroupingResult,
    JournalError,
    PartialExecutionError,
    PlanExecutor,
    undo,
)
from .rules import RuleSet, classify_directory, load_rules
from .tools import (
    PeekSession,
    metadata_for_names,
    peek_file_for_root,
    propose_groups,
)

LOGGER = logging.getLogger(__name__)
LATEST_RUN = "__latest__"


class AgentRunError(RuntimeError):
    """Content-free wrapper for provider/agent failures."""


def _run_agent_safely(agent: Any, task: str) -> Any:
    """Prevent provider exceptions from echoing prompts or credentials."""
    try:
        return agent.run(task)
    except Exception as exc:
        LOGGER.debug("Agent run failed with %s", type(exc).__name__)
        raise AgentRunError(f"agent run failed ({type(exc).__name__})") from None


@dataclass
class PlanBundle:
    moves: list[dict[str, str]]
    rules: RuleSet
    unresolved_count: int
    classifier: Any | None = None
    group_agent: Any | None = None
    grouping: GroupingResult | None = None
    omitted_sources: tuple[str, ...] = ()
    invalid_sources: tuple[str, ...] = ()
    unproposed_sources: tuple[str, ...] = ()
    peek_metrics: dict[str, Any] = field(default_factory=dict)
    classification_metrics: dict[str, Any] = field(default_factory=dict)
    peeked_sources: tuple[str, ...] = ()
    peek_requested_sources: tuple[str, ...] = ()


def _json_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    text = str(value).strip()
    decoder = json.JSONDecoder()
    for position, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _review_move(name: str, rules: RuleSet, *, origin: str, reason: str) -> dict[str, str]:
    return {
        "source": name,
        "destination": f"{rules.review_directory}/{name}",
        "reason": reason,
        "origin": origin,
    }


def _classification_moves(
    result: ValidatedClassification,
    unresolved: list[str],
    rules: RuleSet,
) -> list[dict[str, str]]:
    """Construct paths only from application-controlled sources and categories."""
    omitted = set(result.omitted_sources)
    invalid = set(result.invalid_sources)
    unproposed = set(result.unproposed_sources)
    merged: list[dict[str, str]] = []
    for name in unresolved:
        category = result.categories.get(name)
        if category is not None:
            merged.append(
                {
                    "source": name,
                    "destination": f"{category}/{name}",
                    "reason": f"Structured classifier selected {category}",
                    "origin": "agent",
                }
            )
            continue
        if name in unproposed:
            fallback, reason = "no-proposal", "Classifier returned no usable response"
        elif name in omitted:
            fallback, reason = "omitted", "Classifier omitted this file"
        elif name in invalid:
            fallback, reason = (
                "invalid",
                "Classifier did not return a valid unique assignment",
            )
        else:  # defensive totality if a custom classifier violates its contract
            fallback, reason = "invalid", "Classifier result was incomplete"
        merged.append(
            {
                **_review_move(name, rules, origin="agent", reason=reason),
                "fallback": fallback,
            }
        )
    return merged


def _validated_agent_groups(
    result: Any,
    *,
    executor: PlanExecutor,
    candidate_files: list[str],
    rules: RuleSet,
) -> GroupingResult:
    """Pass structurally valid group output through the executor boundary."""
    proposed: Any = []
    if isinstance(result, list):
        # Small CodeAgent models sometimes pass their completed list directly to
        # final_answer instead of returning the propose_groups JSON. This remains
        # untrusted input and still crosses both proposal-shape and executor
        # validation below.
        proposed = result
    else:
        payload = _json_object(result)
        if payload and payload.get("ok") is True:
            proposed = payload.get("groups", [])
        elif isinstance(result, str):
            try:
                literal = ast.literal_eval(result.strip())
            except (SyntaxError, ValueError):
                literal = None
            if isinstance(literal, list):
                proposed = literal
    feedback = _json_object(
        propose_groups(groups=proposed if isinstance(proposed, list) else [])
    )
    accepted = feedback.get("groups", []) if feedback and feedback.get("ok") is True else []
    return executor.validate_groups(
        accepted,
        candidate_files=candidate_files,
        existing_categories=[*rules.categories, rules.review_directory],
    )


def build_combined_plan(
    directory: str | Path,
    *,
    use_agent: bool = True,
    model_id: str | None = None,
    think: bool | None = None,
    classifier: Any | None = None,
    group: bool = False,
    group_agent: Any | None = None,
    read_contents: bool = False,
    allow_remote_content: bool = False,
    metadata_control: bool = False,
    classification_mode: str = "agreement_gate",
) -> PlanBundle:
    """Optionally group all eligible files, then apply rules and classification.

    ``read_contents`` affects the structured classifier only. ``metadata_control``
    is an evaluation-only two-request control and never reads content. The
    clustering agent stays metadata-only in either case.

    ``classification_mode`` selects the metadata-only unresolved-file
    mechanism when ``read_contents`` is false and ``metadata_control`` is
    false (content mode and the evaluation-only metadata control each keep
    their own unchanged, unconditional behaviour regardless of this value):

    * ``"agreement_gate"`` (default) -- the production candidate. Two
      order-perturbed explicit classify/review passes; Python accepts a
      category only when both passes independently agree on it, otherwise
      ``_ToReview``. See ``StructuredClassifier.classify_with_agreement_gate``.
    * ``"single_pass"`` -- the original one-request baseline
      (``StructuredClassifier.classify`` with no peek tool and no metadata
      control). Kept for evaluation scripts that pin a specific historical
      baseline condition; not exposed as a CLI flag.
    """
    if classification_mode not in {"agreement_gate", "single_pass"}:
        raise ValueError(f"unknown classification_mode: {classification_mode!r}")
    if read_contents and metadata_control:
        raise ValueError("content reading and metadata control are mutually exclusive")
    # Authorize before scanning, creating a classifier, or issuing a model request.
    # A supplied classifier has no trustworthy provider metadata, so its locality is
    # unknown and fail-closed remote authorization is required.
    if read_contents and use_agent:
        if classifier is None:
            ensure_content_authorized(
                model_id,
                allow_remote_content=allow_remote_content,
            )
        elif not allow_remote_content:
            raise ContentAuthorizationError(
                "The supplied classifier's endpoint locality cannot be verified, so "
                "file excerpts could be transmitted outside this machine. Set "
                "allow_remote_content=True only when that transmission is authorized."
            )
    root = Path(directory).resolve(strict=True)
    rules = load_rules()
    rule_moves, unresolved = classify_directory(root, rules)
    candidate_names = [move["source"] for move in rule_moves] + unresolved
    candidate_metadata = metadata_for_names(root, candidate_names)
    candidate_names = [item["name"] for item in candidate_metadata]
    grouping: GroupingResult | None = None
    active_group_agent: Any | None = None
    grouped_files: frozenset[str] = frozenset()
    combined: list[dict[str, str]] = []

    if group:
        if not use_agent:
            raise ValueError("--group requires the agent; remove --no-agent")
        active_group_agent = group_agent or build_group_agent(model_id, think=think)
        group_result = _run_agent_safely(
            active_group_agent,
            build_group_task(candidate_metadata),
        )
        grouping = _validated_agent_groups(
            group_result,
            executor=PlanExecutor(root),
            candidate_files=candidate_names,
            rules=rules,
        )
        grouped_files = grouping.grouped_files
        combined.extend({**move, "origin": "group"} for move in grouping.moves)
        for entry in grouping.entries:
            if entry.status != "accepted":
                LOGGER.warning(
                    "Group %r %s: %s",
                    entry.folder_name,
                    entry.status,
                    entry.message,
                )

    combined.extend(
        {**move, "origin": "rule"}
        for move in rule_moves
        if move["source"] not in grouped_files
    )
    unresolved = [name for name in unresolved if name not in grouped_files]

    if not unresolved:
        return PlanBundle(
            combined,
            rules,
            0,
            classifier,
            active_group_agent,
            grouping,
        )

    if not use_agent:
        combined.extend(
            _review_move(
                name,
                rules,
                origin="rule",
                reason="No matching extension rule; agent disabled",
            )
            for name in unresolved
        )
        return PlanBundle(
            combined,
            rules,
            len(unresolved),
            None,
            active_group_agent,
            grouping,
        )

    peek_session = PeekSession() if read_contents else None
    bound_peek = (
        peek_file_for_root(root, unresolved, session=peek_session)
        if read_contents
        else None
    )
    active_classifier = classifier or build_classifier(model_id, think=think)
    categories = [*rules.categories, rules.review_directory]
    metadata = metadata_for_names(root, unresolved)
    try:
        if bound_peek is not None:
            # Content mode: unchanged single structured call informed by
            # real, bounded peeked content. classification_mode never
            # applies here -- E3 has not been validated for content-enabled
            # classification.
            result = active_classifier.classify(
                metadata, categories, peek_tool=bound_peek
            )
        elif metadata_control:
            # Evaluation-only two-request control (E0); unchanged.
            result = active_classifier.classify(
                metadata, categories, metadata_control=True
            )
        elif classification_mode == "single_pass":
            # Explicit opt-out to the original one-request baseline.
            result = active_classifier.classify(metadata, categories)
        else:
            # Production default (E3): explicit abstention + agreement gate.
            result = active_classifier.classify_with_agreement_gate(
                metadata, list(rules.categories), review_directory=rules.review_directory
            )
    except Exception as exc:
        LOGGER.debug("Structured classification failed with %s", type(exc).__name__)
        metrics = ClassificationTelemetry(provider_errors=1)
        metrics.fallback_to_review_count = len(unresolved)
        result = ValidatedClassification(
            {}, (), (), tuple(unresolved), (), metrics.snapshot()
        )
    agent_moves = _classification_moves(result, unresolved, rules)
    combined.extend(agent_moves)
    return PlanBundle(
        combined,
        rules,
        len(unresolved),
        active_classifier,
        active_group_agent,
        grouping,
        tuple(move["source"] for move in agent_moves if move.get("fallback") == "omitted"),
        tuple(move["source"] for move in agent_moves if move.get("fallback") == "invalid"),
        tuple(
            move["source"] for move in agent_moves if move.get("fallback") == "no-proposal"
        ),
        peek_session.snapshot() if peek_session is not None else {},
        result.telemetry,
        result.authorized_peek_sources,
        result.requested_peek_sources,
    )


def _category(destination: str) -> str:
    parts = Path(destination).parts
    return parts[0] if parts else "(invalid)"


def print_plan(moves: Sequence[Mapping[str, str]], result: ExecutionResult) -> None:
    """Print the executor-validated preview grouped by destination category."""
    rows: list[tuple[str, str, str, str, str]] = []
    for move, entry in zip(moves, result.entries, strict=True):
        category = _category(entry.destination) if entry.status != "rejected" else "(rejected)"
        rows.append(
            (
                category,
                move.get("origin", "unknown"),
                entry.source,
                entry.destination,
                entry.message or move.get("reason", ""),
            )
        )

    if not rows:
        print("No files need organising.")
        return

    for category in sorted({row[0] for row in rows}, key=str.casefold):
        print(f"\n[{category}]")
        group = [row[1:] for row in rows if row[0] == category]
        headers = ("ORIGIN", "SOURCE", "DESTINATION", "REASON")
        widths = [
            min(48, max(len(headers[index]), *(len(row[index]) for row in group)))
            for index in range(4)
        ]
        print("  ".join(headers[index].ljust(widths[index]) for index in range(4)))
        print("  ".join("-" * width for width in widths))
        for row in group:
            clipped = [
                value if len(value) <= widths[index] else value[: widths[index] - 1] + "…"
                for index, value in enumerate(row)
            ]
            print("  ".join(clipped[index].ljust(widths[index]) for index in range(4)))


def _confirm() -> bool:
    answer = input("\nApply this plan? Type 'yes' to continue: ").strip().casefold()
    return answer == "yes"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tidy",
        description="Propose and safely apply a directory-organising plan.",
    )
    parser.add_argument("--path", default=".", help="directory to organise (default: current)")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--apply", action="store_true", help="apply the approved plan")
    action.add_argument(
        "--undo",
        nargs="?",
        const=LATEST_RUN,
        metavar="RUN_ID",
        help="undo the latest run, or the specified run id",
    )
    parser.add_argument("--model", help="LiteLLM model id (overrides MODEL_ID)")
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument(
        "--think",
        dest="think",
        action="store_true",
        help="enable model reasoning when supported by the provider",
    )
    thinking.add_argument(
        "--no-think",
        dest="think",
        action="store_false",
        help="disable model reasoning when supported by the provider",
    )
    parser.set_defaults(think=None)
    parser.add_argument(
        "--no-agent",
        action="store_true",
        help="use extension rules only; unresolved files go to _ToReview/",
    )
    grouping = parser.add_mutually_exclusive_group()
    grouping.add_argument(
        "--group",
        dest="group",
        action="store_true",
        help="cluster strongly related files before applying extension rules",
    )
    grouping.add_argument(
        "--no-group",
        dest="group",
        action="store_false",
        help="disable semantic clustering (default)",
    )
    parser.set_defaults(group=False)
    reading = parser.add_mutually_exclusive_group()
    reading.add_argument(
        "--read-contents",
        dest="read_contents",
        action="store_true",
        help=(
            "let structured classification read the beginning of unresolved "
            "files; contents are untrusted data and never reach the clustering "
            "agent"
        ),
    )
    reading.add_argument(
        "--no-read-contents",
        dest="read_contents",
        action="store_false",
        help="classify from filename metadata only (default)",
    )
    parser.set_defaults(read_contents=False)
    parser.add_argument(
        "--allow-remote-content",
        action="store_true",
        help=(
            "allow excerpts enabled by --read-contents to be sent to a "
            "non-local model endpoint"
        ),
    )
    parser.add_argument("--yes", action="store_true", help="skip the --apply confirmation prompt")
    parser.add_argument("--verbose", action="store_true", help="enable debug logging")
    return parser.parse_args(argv)


def _print_execution(result: ExecutionResult, *, verb: str) -> None:
    counts: dict[str, int] = {}
    for entry in result.entries:
        counts[entry.status] = counts.get(entry.status, 0) + 1
    summary = ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
    print(f"\n{verb}: {summary or 'no entries'}")
    if result.journal_path:
        print(f"Journal: {result.journal_path}")


def _print_partial_execution(error: PartialExecutionError | JournalError) -> None:
    result = error.result
    if result is None:
        print(f"Error: {error}")
        return
    completed = [entry for entry in result.entries if entry.status == "moved"]
    failed = next(
        (entry for entry in result.entries if entry.status == "failed"), None
    )
    print("Execution partially failed.")
    print(f"\nRun ID: {result.run_id}")
    print(f"Journal: {result.journal_path}")
    print(f"Run state: {result.journal_state}")
    print(f"Completed moves: {len(completed)}")
    for entry in completed:
        print(f"  {entry.source} -> {entry.destination}")
    if failed is not None:
        print(f"Failed move: {failed.source} -> {failed.destination}")
        if failed.message:
            print(f"  {failed.message}")
    if result.run_id and result.journal_state == "partially_failed":
        print(f"\nRecovery:\n  tidy --undo {result.run_id}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.undo is not None:
        run_id = None if args.undo == LATEST_RUN else args.undo
        try:
            result = undo(run_id)
        except (OSError, RuntimeError, ValueError) as exc:
            LOGGER.error("Cannot undo run: %s", exc)
            print(f"Error: {exc}")
            return 2
        complete = result.journal_state == "undone"
        _print_execution(result, verb="Undo complete" if complete else "Undo incomplete")
        return 0 if complete else 2

    try:
        bundle = build_combined_plan(
            args.path,
            use_agent=not args.no_agent,
            model_id=args.model,
            think=args.think,
            group=args.group,
            read_contents=args.read_contents,
            allow_remote_content=args.allow_remote_content,
        )
        executor = PlanExecutor(args.path)
        preview = executor.run(bundle.moves)
    except (OSError, RuntimeError, ValueError) as exc:
        LOGGER.error("Cannot build plan: %s", exc)
        print(f"Error: {exc}")
        return 2

    print_plan(bundle.moves, preview)
    if not args.apply:
        _print_execution(preview, verb="Dry-run complete")
        return 0
    if not args.yes and not _confirm():
        print("Cancelled; no files were moved.")
        return 0

    try:
        if preview.validated_plan is None:  # pragma: no cover - defensive
            raise RuntimeError("dry-run did not produce a validated plan")
        applied = executor.run(preview.validated_plan, apply=True)
    except (PartialExecutionError, JournalError) as exc:
        LOGGER.error("Cannot apply approved plan: %s", exc)
        _print_partial_execution(exc)
        return 2
    except (OSError, RuntimeError, ValueError) as exc:
        LOGGER.error("Cannot apply approved plan: %s", exc)
        print(f"Error: {exc}")
        return 2
    _print_execution(applied, verb="Apply complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
