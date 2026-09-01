"""Interaction Layer entry point and application composition root."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from ..agent.runner import (
    AgentResult,
    AgentRunner,
    DEFAULT_MAX_STEPS,
    MAX_CONFIGURED_STEPS,
    TerminationReason,
)
from ..agent.context import (
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_MAX_CONTEXT_MESSAGES,
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_MESSAGES,
    MIN_CONTEXT_CHARS,
    MIN_CONTEXT_MESSAGES,
)
from ..execution.tools import ToolRegistry
from ..execution.workspace import Workspace, WorkspaceError
from ..model.client import OpenAICompatibleClient
from .approval import ConsoleCommandApprover
from .console import ConsoleRenderer
from .narration import _NarratingModelClient
from .session import InteractiveSession

EXIT_CODES: dict[TerminationReason, int] = {
    "completed": 0,
    "max_steps": 1,
    "repeated_failure": 1,
    "no_progress": 1,
    "runtime_error": 1,
    "fatal_api_error": 2,
    "user_interrupt": 130,
}


def _show_fallback_result(result: AgentResult) -> None:
    if result.status == "completed":
        if result.verification_status == "verified":
            print("VERIFICATION")
            print("✓ Latest recorded command passed")
        elif result.verification_status == "unverified":
            print("VERIFICATION")
            print("⚠ Managed changes are not verified at the current revision")
        print("DONE")
        print(f"✓ Task completed · {result.steps} steps")
        if result.answer:
            print()
            print(result.answer)
        return
    print("STOPPED", file=sys.stderr)
    print(f"✗ {result.status}", file=sys.stderr)
    if result.message:
        print(result.message, file=sys.stderr)


def _show_plain_narration(text: str) -> None:
    narration = text.strip()
    if narration:
        print(narration)
        print()


def _render_best_effort(
    callback: Callable[..., None],
    *args: object,
) -> bool:
    try:
        callback(*args)
    except Exception:
        return False
    return True


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
    parser.add_argument(
        "task",
        nargs="?",
        help="Task for one-shot mode; omit to start an interactive session",
    )
    parser.add_argument(
        "--workspace",
        default=".",
        help="Workspace directory (default: current directory)",
    )
    parser.add_argument(
        "--max-steps",
        type=_bounded_max_steps,
        default=DEFAULT_MAX_STEPS,
        help=f"Maximum model decisions (default: {DEFAULT_MAX_STEPS})",
    )
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


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    api_key = os.environ.get("MODEL_API_KEY", "")
    base_url = os.environ.get("MODEL_BASE_URL", "")
    model = os.environ.get("MODEL_NAME", "")

    try:
        supports_image_input = _image_input_capability(
            os.environ.get("MODEL_SUPPORTS_IMAGE_INPUT")
        )
    except ValueError:
        print(
            "Error: MODEL_SUPPORTS_IMAGE_INPUT must be true, false, or empty.",
            file=sys.stderr,
        )
        return 2

    if not api_key or not base_url or not model:
        print(
            "Error: MODEL_API_KEY, MODEL_BASE_URL, and MODEL_NAME must be set.",
            file=sys.stderr,
        )
        return 2

    try:
        workspace = Workspace(Path(args.workspace))
        client = OpenAICompatibleClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            supports_image_input=supports_image_input,
        )
        if args.task is None:
            return InteractiveSession(
                client,
                model_name=model,
                workspace=workspace,
                sensitive_values=(api_key,),
                max_steps=args.max_steps,
                max_context_chars=args.max_context_chars,
                max_context_messages=args.max_context_messages,
            ).run()

        registry = ToolRegistry(
            workspace,
            sensitive_values=(api_key,),
            supports_image_input=supports_image_input,
        )
        try:
            renderer: ConsoleRenderer | None = ConsoleRenderer()
        except Exception:
            renderer = None
        if renderer is not None:
            _render_best_effort(
                renderer.show_header,
                model,
                workspace.root,
                args.task,
            )
        presented_client = _NarratingModelClient(
            client,
            getattr(renderer, "show_narration", None)
            if renderer is not None
            else _show_plain_narration,
        )
        result = AgentRunner(
            presented_client,
            tools=registry,
            max_steps=args.max_steps,
            max_context_chars=args.max_context_chars,
            max_context_messages=args.max_context_messages,
            on_tool_event=renderer.show_tool_event if renderer is not None else None,
            on_core_action_event=(
                getattr(renderer, "show_core_action_event", None)
                if renderer is not None
                else None
            ),
            on_recovery_event=(
                getattr(renderer, "show_recovery_event", None)
                if renderer is not None
                else None
            ),
            on_command_approval=(
                ConsoleCommandApprover() if _stdin_is_interactive() else None
            ),
            on_model_request_started=(
                renderer.start_thinking if renderer is not None else None
            ),
            on_model_request_finished=(
                renderer.stop_thinking if renderer is not None else None
            ),
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

    if renderer is None or not _render_best_effort(renderer.show_result, result):
        _show_fallback_result(result)
    return EXIT_CODES[result.status]


def _image_input_capability(value: str | None) -> bool:
    if value is None or value == "" or value.casefold() == "false":
        return False
    if value.casefold() == "true":
        return True
    raise ValueError("invalid image input capability")


def _stdin_is_interactive() -> bool:
    try:
        return sys.stdin.isatty()
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
