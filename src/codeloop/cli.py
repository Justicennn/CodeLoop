"""Minimal command-line entry point for CodeLoop."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .agent import AgentRunner, MAX_CONFIGURED_STEPS, TerminationReason
from .llm import OpenAICompatibleClient, ToolCall
from .tools import ToolRegistry, ToolResult
from .workspace import Workspace, WorkspaceError

EXIT_CODES: dict[TerminationReason, int] = {
    "completed": 0,
    "max_steps": 1,
    "repeated_failure": 1,
    "runtime_error": 1,
    "fatal_api_error": 2,
    "user_interrupt": 130,
}


def _show_tool_event(tool_call: ToolCall, result: ToolResult) -> None:
    print(f"[tool] {tool_call.name} (id={tool_call.id})")
    print(f"[tool result] {json.dumps(result, ensure_ascii=False)}")


def _bounded_max_steps(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("max steps must be an integer") from exc
    if parsed < 1 or parsed > MAX_CONFIGURED_STEPS:
        raise argparse.ArgumentTypeError(
            f"max steps must be between 1 and {MAX_CONFIGURED_STEPS}"
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

    if result.status == "completed":
        print(f"[final] {result.answer}")
    else:
        detail = f": {result.message}" if result.message else ""
        print(f"[stopped] {result.status}{detail}", file=sys.stderr)
    return EXIT_CODES[result.status]


if __name__ == "__main__":
    raise SystemExit(main())
