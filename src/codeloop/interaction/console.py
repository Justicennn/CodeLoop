from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.status import Status
from rich.style import Style
from rich.syntax import SyntaxTheme
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from codeloop.agent.events import CoreActionEvent, RecoveryEvent, ToolEvent
from codeloop.agent.runner import AgentResult

from .presentation import PresentationBlock, PresentationLine, PresentationState


FAILURE_OUTPUT_LIMIT = 1_000
SUCCESS_OUTPUT_LIMIT = 300
TRUNCATION_MARKER = "... output truncated ..."
FAILURE_EVIDENCE_CHARS = FAILURE_OUTPUT_LIMIT
SUCCESS_EVIDENCE_CHARS = SUCCESS_OUTPUT_LIMIT
OUTPUT_TRUNCATION_MARKER = TRUNCATION_MARKER
_ACCENT_STYLE = "cyan"
_PRIMARY_STYLE = "bold white"
_MUTED_STYLE = "dim"
_SUCCESS_STYLE = "green"
_WARNING_STYLE = "yellow"
_ERROR_STYLE = "red"
_READ_ONLY_TOOLS = frozenset(
    {
        "repository_overview",
        "list_files",
        "read_file",
        "read_document",
        "read_webpage",
        "read_image",
        "search_code",
    }
)


class _TerminalNativeSyntaxTheme(SyntaxTheme):
    """Keep fenced code transparent and readable with terminal defaults."""

    def get_style_for_token(self, token_type: Any) -> Style:
        return Style.null()

    def get_background_style(self) -> Style:
        return Style.null()


_TERMINAL_NATIVE_SYNTAX_THEME = _TerminalNativeSyntaxTheme()
_BACKGROUND_FREE_MARKDOWN_STYLES = {
    "markdown.code": Style(bold=True),
    "markdown.code_block": Style.null(),
}


class ConsoleRenderer:
    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self._thinking: Status | None = None
        self._presentation = PresentationState()

    def show_header(self, model: str, workspace: Path, task: str) -> None:
        title = Text("CodeLoop", style="bold cyan")
        title.append(" · ")
        title.append(model, style="blue")
        self.console.print(title)
        self.console.print(Text(str(workspace), style="dim"))
        self.console.print()
        task_line = Text("codeloop > ", style="green")
        task_line.append(task)
        self.console.print(task_line)
        self.console.print()

    def show_startup_banner(
        self,
        model: str,
        workspace: Path,
        mode: str = "interactive",
    ) -> None:
        """Render session chrome without touching task PresentationState."""
        metadata = Table.grid(padding=(0, 2), expand=False)
        metadata.add_column(style=_MUTED_STYLE, no_wrap=True)
        metadata.add_column(style="white", overflow="fold")
        metadata.add_row("model", model)
        metadata.add_row("workspace", str(workspace))
        metadata.add_row("mode", mode)

        content = Group(
            Text("Welcome to CodeLoop", style=_PRIMARY_STYLE),
            Text(""),
            metadata,
        )
        self.console.print(
            Panel(
                Padding(content, (0, 1)),
                title=Text(" CodeLoop ", style=f"bold {_ACCENT_STYLE}"),
                title_align="left",
                border_style=_ACCENT_STYLE,
                padding=(1, 1),
                expand=True,
            )
        )
        self.console.print()

    def show_input_top_rule(self) -> None:
        """Start one terminal-native user input region."""
        self.console.print(Rule(style=f"{_MUTED_STYLE} {_ACCENT_STYLE}"))

    def read_user_input(self) -> str:
        """Read from the renderer's real console with an accented prompt."""
        return self.console.input(Text("> ", style=f"bold {_ACCENT_STYLE}"))

    def show_input_bottom_rule(self) -> None:
        """Close one submitted user input region."""
        self.console.print(Rule(style=f"{_MUTED_STYLE} {_ACCENT_STYLE}"))
        self.console.print()

    def show_goodbye(self) -> None:
        self.console.print(Text("Bye.", style=_MUTED_STYLE))

    def start_thinking(self) -> None:
        if self._thinking is None:
            self._thinking = self.console.status(
                "[cyan]Working...[/cyan]", spinner="dots"
            )
            self._thinking.start()

    def stop_thinking(self) -> None:
        if self._thinking is not None:
            self._thinking.stop()
            self._thinking = None

    def show_narration(self, narration: str) -> None:
        """Render model-provided public narration without inferring intent."""
        self.stop_thinking()
        text = _compact_answer_spacing(narration)
        if not text:
            return
        with self.console.use_theme(
            Theme(_BACKGROUND_FREE_MARKDOWN_STYLES), inherit=True
        ):
            self.console.print(
                Markdown(text, code_theme=_TERMINAL_NATIVE_SYNTAX_THEME)
            )
        self.console.print()

    def show_tool_event(self, event: ToolEvent) -> None:
        self.stop_thinking()
        self._presentation.record_managed_change(event)
        section = self._presentation.section_for_tool(event)
        if section is None:
            return
        self._render_section(section)
        name = event.tool_call.name

        if name == "run_command":
            self._render_command(event)
        elif name == "edit_file":
            self._render_edit(event)
        elif name == "write_file":
            self._render_write(event)
        elif name == "make_directory":
            self._render_make_directory(event)
        elif name in _READ_ONLY_TOOLS:
            arguments = _object_arguments(event.tool_call.arguments)
            data = _result_data(event)
            path = (
                _string(data.get("path"))
                or _string(arguments.get("path"))
                or _string(data.get("requested_url"))
                or _string(arguments.get("url"))
                or name
            )
            self._render_tool_failure(event, path)
        else:
            self._render_other_tool(event)
        self.console.print()

    def show_core_action_event(self, event: CoreActionEvent) -> None:
        self.stop_thinking()
        block = self._presentation.record_core_action(event)
        if block is not None:
            self._render_block(block)

    def show_recovery_event(self, event: RecoveryEvent) -> None:
        self.stop_thinking()
        self._render_block(self._presentation.record_recovery(event))

    def show_result(self, result: AgentResult) -> None:
        self.stop_thinking()
        changed = self._presentation.changed_block()
        if changed is not None:
            self._render_block(changed)

        if (
            result.verification_status == "verified"
            and not self._presentation.has_verification_attempt
        ):
            self._render_block(
                PresentationBlock(
                    "VERIFICATION",
                    (
                        PresentationLine(
                            "✓",
                            "Latest recorded command passed",
                            "green",
                        ),
                    ),
                )
            )
        elif result.verification_status == "unverified":
            self._render_block(
                PresentationBlock(
                    "VERIFICATION",
                    (
                        PresentationLine(
                            "⚠",
                            "Managed changes are not verified at the current revision",
                            "yellow",
                        ),
                    ),
                )
            )

        section = "DONE" if result.status == "completed" else "STOPPED"
        self._render_section(section)
        if result.status == "completed":
            self._render_heading(
                "✓",
                f"Task completed · {result.steps} steps",
                "green",
            )
        else:
            self._render_heading("✗", result.status, "red")
        if result.message:
            self.console.print(Text(f"  {result.message}"))

        if result.status == "completed" and result.answer:
            answer = result.answer
            if answer.strip():
                self.console.print()
                with self.console.use_theme(
                    Theme(_BACKGROUND_FREE_MARKDOWN_STYLES), inherit=True
                ):
                    self.console.print(
                        Markdown(
                            answer,
                            code_theme=_TERMINAL_NATIVE_SYNTAX_THEME,
                        )
                    )
        self.console.print()

    def _render_block(self, block: PresentationBlock) -> None:
        self._render_section(block.section)
        for line in block.lines:
            self._render_heading(line.marker, line.text, line.style)
            if line.detail:
                self.console.print(Text(f"  {line.detail}", style="dim"))
        self.console.print()

    def _render_section(self, title: str) -> None:
        self.console.print(Rule(title, align="left", style="dim"))

    def _render_command(self, event: ToolEvent) -> None:
        raw_data = event.result.get("data")
        data = raw_data if isinstance(raw_data, dict) else {}
        command = _safe_command(data.get("command"))
        error_code = _string(event.result.get("error_code"))
        message = _string(event.result.get("message"))

        if command is None:
            if event.result.get("ok") is True:
                self._render_heading("✓", "Command completed", _SUCCESS_STYLE)
            else:
                title = "run_command"
                if error_code:
                    title += f" · {error_code}"
                self._render_heading("⚠", title, _WARNING_STYLE)
                if message:
                    self.console.print(Text(f"  {message}"))
            return

        if event.result.get("ok") is True:
            self._render_heading("✓", command, _SUCCESS_STYLE)
            evidence = _select_success_evidence(data)
            limit = SUCCESS_OUTPUT_LIMIT
        else:
            title = command
            exit_code = _integer(data.get("exit_code"))
            if error_code == "command_failed" and exit_code is not None:
                title += f" · exit {exit_code}"
            elif error_code:
                title += f" · {error_code}"
            self._render_heading("✗", title, _ERROR_STYLE)
            evidence = _select_failure_evidence(data)
            limit = FAILURE_OUTPUT_LIMIT

        if evidence:
            self._render_evidence(evidence, limit)
        elif event.result.get("ok") is not True and message:
            self.console.print(Text(f"  {message}", style="red"))

    def _render_edit(self, event: ToolEvent) -> None:
        arguments = _object_arguments(event.tool_call.arguments)
        data = _result_data(event)
        path = _string(data.get("path")) or _string(arguments.get("path")) or "edit_file"
        if event.result.get("ok") is not True:
            self._render_tool_failure(event, path)
            return

        if data.get("workspace_changed") is not True:
            self._render_heading("✓", f"Unchanged {path}", "green")
            return
        title = f"Updated {path}"
        replacements = _integer(data.get("replacements"))
        if replacements is not None:
            noun = "replacement" if replacements == 1 else "replacements"
            title += f" · {replacements} {noun}"
        self._render_heading("M", title, "green")

    def _render_write(self, event: ToolEvent) -> None:
        arguments = _object_arguments(event.tool_call.arguments)
        data = _result_data(event)
        path = _string(data.get("path")) or _string(arguments.get("path")) or "write_file"
        if event.result.get("ok") is True:
            self._render_heading("+", f"Created {path}", "green")
        else:
            self._render_tool_failure(event, path)

    def _render_make_directory(self, event: ToolEvent) -> None:
        arguments = _object_arguments(event.tool_call.arguments)
        data = _result_data(event)
        path = _string(data.get("path")) or _string(arguments.get("path")) or "make_directory"
        if event.result.get("ok") is not True:
            self._render_tool_failure(event, path)
            return
        action = "Created" if data.get("workspace_changed") is True else "Directory ready"
        self._render_heading("✓", f"{action} {path}", "green")

    def _render_other_tool(self, event: ToolEvent) -> None:
        arguments = _object_arguments(event.tool_call.arguments)
        path = _string(arguments.get("path"))
        title = event.tool_call.name
        if path:
            title += f" · {path}"
        marker = "✓" if event.result.get("ok") is True else "✗"
        color = "green" if event.result.get("ok") is True else "red"
        self._render_heading(marker, title, color)
        if event.result.get("ok") is not True:
            self._render_failure_detail(event)

    def _render_tool_failure(self, event: ToolEvent, path: str) -> None:
        error_code = _string(event.result.get("error_code"))
        title = event.tool_call.name
        if path != event.tool_call.name:
            title += f" · {path}"
        if error_code:
            title += f" · {error_code}"
        self._render_heading("✗", title, "red")
        message = _string(event.result.get("message"))
        if message:
            self.console.print(Text(f"  {message}", style="red"))

    def _render_failure_detail(self, event: ToolEvent) -> None:
        detail = " · ".join(
            part
            for part in (
                _string(event.result.get("error_code")),
                _string(event.result.get("message")),
            )
            if part
        )
        if detail:
            self.console.print(Text(f"  {detail}", style="red"))

    def _render_heading(self, marker: str, title: str, color: str) -> None:
        line = Text(f"{marker} ", style=f"bold {color}")
        line.append(title)
        self.console.print(line)

    def _render_evidence(self, evidence: str, limit: int) -> None:
        truncated = len(evidence) > limit
        text = _bounded_text(evidence, limit, TRUNCATION_MARKER)
        for line in text.splitlines():
            style: str | None = None
            if _is_failure_signal(line) or _is_error_detail(line):
                style = "red"
            elif _is_success_signal(line):
                style = "green"
            self.console.print(Text(f"  {line}", style=style))
        if truncated and TRUNCATION_MARKER not in text:
            self.console.print(Text(f"  {TRUNCATION_MARKER}", style="dim"))


def _object_arguments(raw_arguments: str) -> dict[str, Any]:
    try:
        value = json.loads(raw_arguments)
    except (json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _result_data(event: ToolEvent) -> dict[str, Any]:
    value = event.result.get("data")
    return value if isinstance(value, dict) else {}


def _safe_command(value: Any) -> str | None:
    if not isinstance(value, list) or not value:
        return None
    if not all(isinstance(part, str) and part for part in value):
        return None
    return subprocess.list2cmdline(value)


def _stream_lines(data: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for key in ("stdout", "stderr"):
        value = data.get(key)
        if not isinstance(value, str):
            continue
        for raw_line in value.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped or _is_separator_line(stripped) or line in seen:
                continue
            seen.add(line)
            lines.append(line)
    return lines


def _select_failure_evidence(data: dict[str, Any]) -> str:
    lines = _stream_lines(data)
    outcome_lines = [line for line in lines if _is_failed_test_line(line)]
    detail_lines = [line for line in lines if _is_error_detail(line)]
    summary_lines = [line for line in lines if _is_failure_summary(line)]

    selected_set = set(outcome_lines + detail_lines + summary_lines)
    if selected_set:
        selected = [line for line in lines if line in selected_set]
    else:
        keyword_lines = [line for line in lines if _is_failure_signal(line)]
        selected = keyword_lines if keyword_lines else lines[-4:]
    return "\n".join(selected[:4])


def _select_success_evidence(data: dict[str, Any]) -> str:
    lines = _stream_lines(data)
    summaries = [line for line in lines if _is_success_summary(line)]
    return "\n".join(summaries[:2])


def _is_separator_line(line: str) -> bool:
    return bool(line) and all(character in "=-_*~" for character in line)


def _is_failed_test_line(line: str) -> bool:
    return bool(
        re.search(r"\.{2,}\s*(?:FAIL|ERROR)\s*$", line.strip(), re.IGNORECASE)
    )


def _is_error_detail(line: str) -> bool:
    candidate = line.strip()
    if candidate.startswith(("FAIL:", "ERROR:")):
        return False
    return bool(
        re.search(r"\b[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)\b", candidate)
    )


def _is_failure_summary(line: str) -> bool:
    candidate = line.strip()
    return bool(
        re.match(r"^FAILED\b", candidate, re.IGNORECASE)
        or re.match(r"^ERRORS?\s*(?:\(|=|\d)", candidate, re.IGNORECASE)
        or re.search(r"\b\d+\s+failed\b", candidate, re.IGNORECASE)
        or re.search(
            r"\b(?:failures?|errors?)=\d+\b", candidate, re.IGNORECASE
        )
    )


def _is_success_summary(line: str) -> bool:
    candidate = line.strip()
    return bool(
        re.fullmatch(r"(?:OK|PASS|PASSED)", candidate, re.IGNORECASE)
        or re.fullmatch(
            r"\d+\s+passed(?:\s*[·,]\s*\d+\s+(?:failed|skipped))*",
            candidate,
            re.IGNORECASE,
        )
        or re.fullmatch(
            r"\d+\s+failed\s*[·,]\s*\d+\s+passed",
            candidate,
            re.IGNORECASE,
        )
    )


def _is_failure_signal(line: str) -> bool:
    candidate = line.strip()
    return bool(
        re.search(
            r"\b(?:FAIL|FAILED|ERROR|EXCEPTION|TRACEBACK)\b",
            candidate,
            re.IGNORECASE,
        )
    )


def _is_success_signal(line: str) -> bool:
    return _is_success_summary(line)


def _bounded_text(value: str, limit: int, marker: str) -> str:
    if len(value) <= limit:
        return value
    available = max(0, limit - len(marker) - 1)
    return f"{value[:available].rstrip()}\n{marker}"


def _compact_answer_spacing(answer: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", answer.strip())


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
