"""Minimal command-line entry point for CodeLoop."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .agent import AgentRunner
from .llm import OpenAICompatibleClient, ToolCall
from .tools import ToolResult


def _show_tool_event(tool_call: ToolCall, result: ToolResult) -> None:
    print(f"[tool] {tool_call.name} (id={tool_call.id})")
    print(f"[tool result] {json.dumps(result, ensure_ascii=False)}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the CodeLoop agent.")
    parser.add_argument("task", help="Task for the model to complete")
    parser.add_argument("--max-steps", type=int, default=20)
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
        client = OpenAICompatibleClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
        result = AgentRunner(
            client,
            max_steps=args.max_steps,
            on_tool_event=_show_tool_event,
        ).run(args.task)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130

    if result.status == "completed":
        print(f"[final] {result.answer}")
        return 0

    print(f"[stopped] max_steps reached after {result.steps} model decisions", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
