"""In-process interactive sessions built around independent agent runs."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from shutil import get_terminal_size
from typing import Any

from ..agent import PublicConversationTurn
from ..agent.context import DEFAULT_MAX_CONTEXT_CHARS, DEFAULT_MAX_CONTEXT_MESSAGES
from ..agent.runner import (
    DEFAULT_MAX_STEPS,
    AgentResult,
    AgentRunner,
    TerminationReason,
)
from ..execution.tools import ToolRegistry
from ..execution.workspace import Workspace, WorkspaceError
from ..model.client import ModelClient
from .approval import ConsoleCommandApprover
from .console import ConsoleRenderer
from .narration import _NarratingModelClient

DEFAULT_MAX_SESSION_PAIRS = 6
DEFAULT_MAX_SESSION_CHARS = 12_000
DEFAULT_MAX_PUBLIC_TEXT_CHARS = 4_000
SESSION_TRUNCATION_MARKER = "... session text truncated ..."

_CONTROLLED_TERMINATIONS: frozenset[TerminationReason] = frozenset(
    {
        "completed",
        "max_steps",
        "repeated_failure",
        "no_progress",
        "user_interrupt",
    }
)
_FATAL_EXIT_CODES: dict[TerminationReason, int] = {
    "fatal_api_error": 2,
    "runtime_error": 1,
}
_NATURAL_WORKSPACE_PREFIXES = (
    "切换到",
    "接下来处理",
    "把工作目录切到",
)


class SessionHistory:
    """Bounded public exchanges; no tool or private runtime state is retained."""

    def __init__(
        self,
        *,
        max_pairs: int = DEFAULT_MAX_SESSION_PAIRS,
        max_chars: int = DEFAULT_MAX_SESSION_CHARS,
        max_text_chars: int = DEFAULT_MAX_PUBLIC_TEXT_CHARS,
    ) -> None:
        if max_pairs < 1 or max_chars < 64 or max_text_chars < 1:
            raise ValueError("Session history budgets must be positive")
        self._max_pairs = max_pairs
        self._max_chars = max_chars
        self._max_text_chars = max_text_chars
        self._turns: list[PublicConversationTurn] = []

    def add(self, user: str, assistant: str) -> None:
        turn = self._fit_single_turn(
            PublicConversationTurn(
                user=_truncate_text(user, self._max_text_chars),
                assistant=_truncate_text(assistant, self._max_text_chars),
            )
        )
        self._turns.append(turn)
        while (
            len(self._turns) > self._max_pairs
            or _serialized_turn_chars(self._turns) > self._max_chars
        ):
            self._turns.pop(0)

    def snapshot(self) -> tuple[PublicConversationTurn, ...]:
        """Return immutable value objects for one new AgentRunner run."""
        return tuple(self._turns)

    def _fit_single_turn(
        self,
        turn: PublicConversationTurn,
    ) -> PublicConversationTurn:
        if _serialized_turn_chars([turn]) <= self._max_chars:
            return turn

        low = 0
        high = max(len(turn.user), len(turn.assistant))
        fitted = PublicConversationTurn(user="", assistant="")
        while low <= high:
            cap = (low + high) // 2
            candidate = PublicConversationTurn(
                user=_truncate_text(turn.user, cap),
                assistant=_truncate_text(turn.assistant, cap),
            )
            if _serialized_turn_chars([candidate]) <= self._max_chars:
                fitted = candidate
                low = cap + 1
            else:
                high = cap - 1
        return fitted


@dataclass(frozen=True)
class WorkspaceSwitch:
    """One unambiguous Interaction-layer workspace switch request."""

    path: Path
    task: str | None = None


class InteractiveSession:
    """REPL that owns only public session data and fixed run configuration."""

    def __init__(
        self,
        client: ModelClient,
        *,
        model_name: str,
        workspace: Workspace,
        sensitive_values: tuple[str, ...] = (),
        max_steps: int = DEFAULT_MAX_STEPS,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        max_context_messages: int = DEFAULT_MAX_CONTEXT_MESSAGES,
        read_line: Callable[[str], str] | None = None,
        write_line: Callable[[str], None] | None = None,
        renderer_factory: Callable[[], ConsoleRenderer | None] | None = None,
    ) -> None:
        self._client = client
        self._model_name = model_name
        self._workspace = workspace
        self._sensitive_values = tuple(sensitive_values)
        self._max_steps = max_steps
        self._max_context_chars = max_context_chars
        self._max_context_messages = max_context_messages
        self._uses_default_read_line = read_line is None
        self._read_line = read_line or input
        self._write_line = write_line or print
        self._renderer_factory = renderer_factory or _new_renderer
        self._command_approver = ConsoleCommandApprover(
            read_line=self._read_line,
            write_line=self._write_line,
        )
        self._history = SessionHistory()

    @property
    def workspace(self) -> Workspace:
        return self._workspace

    def run(self) -> int:
        renderer = self._create_renderer()
        self._show_startup_banner(renderer)
        renderer_has_task_state = False

        while True:
            self._show_input_top_rule(renderer)
            try:
                raw = self._read_interactive_line(renderer)
            except EOFError:
                return 0
            except KeyboardInterrupt:
                self._write_line("Interrupted.")
                return 130

            text = raw.strip()
            if not text:
                continue

            self._show_input_bottom_rule(renderer)
            if text.casefold() in {"exit", "quit"}:
                self._show_goodbye(renderer)
                return 0

            command_result = self._handle_command(text)
            if command_result is not None:
                if command_result >= 0:
                    return command_result
                continue

            if _is_redundant_codeloop_invocation(text):
                self._write_line("Already in interactive mode.")
                self._write_line(
                    "Use /workspace ABSOLUTE_PATH to switch projects."
                )
                continue

            switch = parse_natural_workspace_switch(text)
            task = text
            if switch is not None:
                if not self._switch_workspace(switch.path):
                    continue
                if switch.task is None:
                    continue
                task = switch.task

            if renderer_has_task_state:
                renderer = self._create_renderer()
            result = self._run_task(task, renderer)
            renderer_has_task_state = True
            if result.status in _CONTROLLED_TERMINATIONS:
                continue
            return _FATAL_EXIT_CODES.get(result.status, 1)

    def _create_renderer(self) -> ConsoleRenderer | None:
        try:
            return self._renderer_factory()
        except Exception:
            return None

    def _show_startup_banner(
        self,
        renderer: ConsoleRenderer | None,
    ) -> None:
        callback = (
            getattr(renderer, "show_startup_banner", None)
            if renderer is not None
            else None
        )
        if callback is not None and _best_effort(
            callback,
            self._model_name,
            self._workspace.root,
            "interactive",
        ):
            return
        self._write_line("CodeLoop")
        self._write_line("")
        self._write_line("Welcome to CodeLoop")
        self._write_line("")
        self._write_line(f"model: {self._model_name}")
        self._write_line(f"workspace: {self._workspace.root}")
        self._write_line("mode: interactive")
        self._write_line("")

    def _show_input_top_rule(
        self,
        renderer: ConsoleRenderer | None,
    ) -> None:
        callback = (
            getattr(renderer, "show_input_top_rule", None)
            if renderer is not None
            else None
        )
        if callback is None or not _best_effort(callback):
            self._write_line(_fallback_input_rule())

    def _read_interactive_line(
        self,
        renderer: ConsoleRenderer | None,
    ) -> str:
        if not self._uses_default_read_line:
            return self._read_line("> ")
        callback = (
            getattr(renderer, "read_user_input", None)
            if renderer is not None
            else None
        )
        if callback is not None:
            try:
                return callback()
            except (EOFError, KeyboardInterrupt):
                raise
            except Exception:
                # Only the default terminal path may fall back to builtin
                # input. An injected read_line is handled above and is never
                # replaced when it raises.
                return self._read_line("> ")
        return self._read_line("> ")

    def _show_input_bottom_rule(
        self,
        renderer: ConsoleRenderer | None,
    ) -> None:
        callback = (
            getattr(renderer, "show_input_bottom_rule", None)
            if renderer is not None
            else None
        )
        if callback is not None and _best_effort(callback):
            return
        self._write_line(_fallback_input_rule())
        self._write_line("")

    def _show_goodbye(self, renderer: ConsoleRenderer | None) -> None:
        callback = (
            getattr(renderer, "show_goodbye", None)
            if renderer is not None
            else None
        )
        if callback is None or not _best_effort(callback):
            self._write_line("Bye.")

    def _handle_command(self, text: str) -> int | None:
        if not text.startswith("/"):
            return None

        parts = text.split(maxsplit=1)
        command = parts[0]
        argument = parts[1].strip() if len(parts) == 2 else ""
        if command == "/exit":
            if argument:
                self._write_line("Usage: /exit")
                return -1
            return 0
        if command == "/help":
            if argument:
                self._write_line("Usage: /help")
                return -1
            self._write_line(
                "/help · /workspace ABSOLUTE_PATH · /new · /exit"
            )
            return -1
        if command == "/new":
            if argument:
                self._write_line("Usage: /new")
                return -1
            self._history = SessionHistory()
            self._write_line("Started a new session context.")
            return -1
        if command == "/workspace":
            path = parse_workspace_argument(argument)
            if path is None:
                self._write_line("Usage: /workspace ABSOLUTE_PATH")
                return -1
            self._switch_workspace(path)
            return -1

        self._write_line("Unknown command. Type /help for available commands.")
        return -1

    def _switch_workspace(self, path: Path) -> bool:
        try:
            replacement = Workspace(path)
        except WorkspaceError as exc:
            self._write_line(f"Workspace error: {exc.message}")
            return False
        self._workspace = replacement
        # Replacing, rather than clearing, ensures no previous public-context
        # container remains reachable through the session object.
        self._history = SessionHistory()
        self._write_line(f"Workspace: {replacement.root}")
        return True

    def _run_task(
        self,
        task: str,
        renderer: ConsoleRenderer | None,
    ) -> AgentResult:
        previous_turns = self._history.snapshot()
        registry = ToolRegistry(
            self._workspace,
            sensitive_values=self._sensitive_values,
            supports_image_input=bool(
                getattr(self._client, "supports_image_input", False)
            ),
        )
        narration_callback = (
            getattr(renderer, "show_narration", None)
            if renderer is not None
            else self._show_plain_narration
        )
        runner = AgentRunner(
            _NarratingModelClient(
                self._client,
                narration_callback,
            ),
            tools=registry,
            max_steps=self._max_steps,
            max_context_chars=self._max_context_chars,
            max_context_messages=self._max_context_messages,
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
            on_command_approval=self._command_approver,
            on_model_request_started=(
                renderer.start_thinking if renderer is not None else None
            ),
            on_model_request_finished=(
                renderer.stop_thinking if renderer is not None else None
            ),
        )
        try:
            result = runner.run(task, previous_turns=previous_turns)
            if renderer is None or not _best_effort(renderer.show_result, result):
                self._write_line(_fallback_result_text(result))
            self._history.add(task, _public_assistant_text(result))
            return result
        finally:
            callback = (
                getattr(renderer, "close", None)
                if renderer is not None
                else None
            )
            if callback is not None:
                _best_effort(callback)

    def _show_plain_narration(self, text: str) -> None:
        narration = text.strip()
        if narration:
            self._write_line(narration)
            self._write_line("")


def parse_workspace_argument(argument: str) -> Path | None:
    """Parse a command path only when its absolute boundary is unambiguous."""
    value = argument.strip()
    if not value:
        return None
    if value[0] in {"'", '"'}:
        quote = value[0]
        if len(value) < 2 or value[-1] != quote:
            return None
        value = value[1:-1]
        if not value or quote in value:
            return None
    elif "'" in value or '"' in value:
        return None
    path = Path(value)
    return path if path.is_absolute() else None


def parse_natural_workspace_switch(text: str) -> WorkspaceSwitch | None:
    """Conservatively parse one anchored Chinese workspace-switch sentence."""
    prefix = next(
        (candidate for candidate in _NATURAL_WORKSPACE_PREFIXES if text.startswith(candidate)),
        None,
    )
    if prefix is None:
        return None
    body = text[len(prefix) :].strip()
    if not body:
        return None

    task: str | None = None
    if body[0] in {"'", '"'}:
        quote = body[0]
        closing = body.find(quote, 1)
        if closing < 0:
            return None
        path_text = body[1:closing]
        remainder = body[closing + 1 :].strip()
        if remainder:
            if remainder[0] not in {",", "，"}:
                return None
            task = _normalized_followup_task(remainder[1:])
            if task is None:
                return None
    else:
        if "'" in body or '"' in body:
            return None
        separator_index = _first_comma_index(body)
        if separator_index is None:
            path_text = body.strip()
        else:
            path_text = body[:separator_index].strip()
            task = _normalized_followup_task(body[separator_index + 1 :])
            if task is None:
                return None

    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        return None
    return WorkspaceSwitch(path=path, task=task)


def _normalized_followup_task(value: str) -> str | None:
    task = value.strip()
    if task.startswith("然后"):
        task = task[2:].strip()
    return task or None


def _first_comma_index(value: str) -> int | None:
    indexes = [index for marker in (",", "，") if (index := value.find(marker)) >= 0]
    return min(indexes) if indexes else None


def _truncate_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    if limit <= len(SESSION_TRUNCATION_MARKER):
        return SESSION_TRUNCATION_MARKER[:limit]
    available = limit - len(SESSION_TRUNCATION_MARKER) - 1
    return f"{value[:available]}\n{SESSION_TRUNCATION_MARKER}"


def _serialized_turn_chars(turns: list[PublicConversationTurn]) -> int:
    return len(
        json.dumps(
            [
                {"user": turn.user, "assistant": turn.assistant}
                for turn in turns
            ],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _public_assistant_text(result: AgentResult) -> str:
    if result.status == "completed":
        return result.answer or "Task completed."
    message = result.message or "The task stopped before completion."
    return f"{result.status}: {message}"


def _fallback_result_text(result: AgentResult) -> str:
    if result.status == "completed":
        lines: list[str] = []
        if result.verification_status == "verified":
            lines.extend(("VERIFICATION", "✓ Latest recorded command passed"))
        elif result.verification_status == "unverified":
            lines.extend(
                (
                    "VERIFICATION",
                    "⚠ Managed changes are not verified at the current revision",
                )
            )
        lines.extend(("DONE", f"✓ Task completed · {result.steps} steps"))
        if result.answer:
            lines.extend(("", result.answer))
        return "\n".join(lines)
    lines = ["STOPPED", f"✗ {result.status}"]
    if result.message:
        lines.append(result.message)
    return "\n".join(lines)


def _is_redundant_codeloop_invocation(value: str) -> bool:
    command = value.split(maxsplit=1)[0]
    return command.casefold() == "codeloop"


def _new_renderer() -> ConsoleRenderer | None:
    try:
        return ConsoleRenderer(live=True)
    except Exception:
        return None


def _fallback_input_rule() -> str:
    width = get_terminal_size(fallback=(80, 24)).columns
    return "-" * max(20, min(width, 120))


def _best_effort(callback: Callable[..., None], *args: Any) -> bool:
    try:
        callback(*args)
    except Exception:
        return False
    return True
