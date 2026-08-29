"""Minimal command-line entry point for CodeLoop."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .agent import (
    AgentResult,
    AgentRunner,
    MAX_CONFIGURED_STEPS,
    TerminationReason,
    ToolEvent,
)
from .context import (
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_MAX_CONTEXT_MESSAGES,
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_MESSAGES,
    MIN_CONTEXT_CHARS,
    MIN_CONTEXT_MESSAGES,
)
from .llm import OpenAICompatibleClient
from .tools import ToolRegistry
from .workspace import Workspace, WorkspaceError

FAILURE_OBSERVATION_CHARS = 2_000
SUCCESS_STDOUT_CHARS = 500

EXIT_CODES: dict[TerminationReason, int] = {
    "completed": 0,
    "max_steps": 1,
    "repeated_failure": 1,
    "runtime_error": 1,
    "fatal_api_error": 2,
    "user_interrupt": 130,
}


def _show_tool_event(event: ToolEvent) -> None:
    result = event.result
    ok = result.get("ok") is True
    fields = [
        f"name={event.tool_call.name}",
        f"id={event.tool_call.id}",
        f"status={'ok' if ok else 'error'}",
    ]
    error_code = result.get("error_code")
    if isinstance(error_code, str):
        fields.append(f"error_code={error_code}")
    data = result.get("data")
    if isinstance(data, dict) and data.get("exit_code") is not None:
        fields.append(f"exit_code={data['exit_code']}")
    fields.extend(
        (
            f"dispatch_duration_ms={event.dispatch_duration_ms}",
            f"truncated={str(event.truncated).lower()}",
        )
    )
    print(f"[tool] {' '.join(fields)}")

    if not ok:
        _show_excerpt(
            "[observation]",
            json.dumps(result, ensure_ascii=False),
            FAILURE_OBSERVATION_CHARS,
        )
    elif event.tool_call.name == "run_command" and isinstance(data, dict):
        stdout = data.get("stdout")
        if isinstance(stdout, str) and stdout:
            _show_excerpt("[stdout]", stdout, SUCCESS_STDOUT_CHARS)


def _show_excerpt(label: str, text: str, limit: int) -> None:
    marker = "... [display truncated]"
    if len(text) > limit:
        text = text[: limit - len(marker)] + marker
    print(f"{label} {text}")


def _show_agent_result(result: AgentResult) -> None:
    if result.status == "completed":
        print(f"[final] {result.answer}")
    fields = [f"status={result.status}", f"steps={result.steps}"]
    if result.message:
        fields.append(f"message={result.message}")
    output = f"[termination] {' '.join(fields)}"
    print(output, file=sys.stdout if result.status == "completed" else sys.stderr)


def _bounded_max_steps(value: str) -> int:
    return _bounded_integer(
        value,
        name="max steps",
        minimum=1,
        maximum=MAX_CONFIGURED_STEPS,
    )


def _bounded_context_chars(value: str) -> int:
    return _bounded_integer(
        value,
        name="max context chars",
        minimum=MIN_CONTEXT_CHARS,
        maximum=MAX_CONTEXT_CHARS,
    )


def _bounded_context_messages(value: str) -> int:
    return _bounded_integer(
        value,
        name="max context messages",
        minimum=MIN_CONTEXT_MESSAGES,
        maximum=MAX_CONTEXT_MESSAGES,
    )


def _bounded_integer(
    value: str,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be an integer") from exc
    if parsed < minimum or parsed > maximum:
        raise argparse.ArgumentTypeError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the CodeLoop agent.")
    parser.add_argument("task", help="Task for the model to complete")
    parser.add_argument(
        "--workspace",
        default=".",
        help="Workspace directory (default: current directory)",
    )
    parser.add_argument("--max-steps", type=_bounded_max_steps, default=20)
    parser.add_argument(
        "--max-context-chars",
        type=_bounded_context_chars,
        default=DEFAULT_MAX_CONTEXT_CHARS,
        help=(
            "Conversation-history character budget; this is not a token limit "
            f"(default: {DEFAULT_MAX_CONTEXT_CHARS})"
        ),
    )
    parser.add_argument(
        "--max-context-messages",
        type=_bounded_context_messages,
        default=DEFAULT_MAX_CONTEXT_MESSAGES,
        help=(
            "Conversation-history message budget "
            f"(default: {DEFAULT_MAX_CONTEXT_MESSAGES})"
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    api_key = os.environ.get("MODEL_API_KEY", "")
    base_url = os.environ.get("MODEL_BASE_URL", "")
    model = os.environ.get("MODEL_NAME", "")

    if not api_key or not base_url or not model:
        print(
            "Error: MODEL_API_KEY, MODEL_BASE_URL, and MODEL_NAME must be set.",
            file=sys.stderr,
        )
        return 2

    try:
        workspace = Workspace(Path(args.workspace))
        registry = ToolRegistry(workspace, sensitive_values=(api_key,))
        client = OpenAICompatibleClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
        result = AgentRunner(
            client,
            tools=registry,
            max_steps=args.max_steps,
            max_context_chars=args.max_context_chars,
            max_context_messages=args.max_context_messages,
            on_tool_event=_show_tool_event,
        ).run(args.task)
    except WorkspaceError as exc:
        print(f"Error: {exc.message}", file=sys.stderr)
        return 2
    except ValueError:
        print("Error: invalid model or runtime configuration.", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130

    _show_agent_result(result)
    return EXIT_CODES[result.status]


if __name__ == "__main__":
    raise SystemExit(main())
