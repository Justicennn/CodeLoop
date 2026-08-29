from __future__ import annotations

import difflib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.status import Status
from rich.style import Style
from rich.syntax import SyntaxTheme
from rich.text import Text
from rich.theme import Theme

from codeloop.agent.events import ToolEvent
from codeloop.agent.runner import AgentResult


FAILURE_OUTPUT_LIMIT = 2_000
SUCCESS_OUTPUT_LIMIT = 500
DIFF_OUTPUT_LIMIT = 1_500
FILE_PATH_LIMIT = 5
SEARCH_SUMMARY_LIMIT = 3
TRUNCATION_MARKER = "... output truncated ..."
DIFF_TRUNCATION_MARKER = "... diff truncated ..."
FAILURE_EVIDENCE_CHARS = FAILURE_OUTPUT_LIMIT
SUCCESS_EVIDENCE_CHARS = SUCCESS_OUTPUT_LIMIT
DIFF_PREVIEW_CHARS = DIFF_OUTPUT_LIMIT
OUTPUT_TRUNCATION_MARKER = TRUNCATION_MARKER


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
        self._changed_files: list[str] = []
        self._last_successful_command: str | None = None
        self._pending_file_activity = False
        self._pending_paths: dict[str, None] = {}
        self._pending_searches: list[tuple[str | None, int | None]] = []

    def show_header(self, model: str, workspace: Path, task: str) -> None:
        title = Text("CodeLoop", style="bold cyan")
        title.append(" · ")
        title.append(model, style="blue")
        self.console.print(title)
        self.console.print(Text(str(workspace), style="dim"))
        self.console.print()
        task_line = Text("> ", style="cyan")
        task_line.append(task)
        self.console.print(task_line)
        self.console.print()

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

    def show_tool_event(self, event: ToolEvent) -> None:
        self.stop_thinking()
        if event.result.get("ok") is True and event.tool_call.name in {
            "list_files",
            "read_file",
            "search_code",
        }:
            self._buffer_file_activity(event)
            return

        self._flush_file_activity()
        name = event.tool_call.name
        if name == "run_command":
            self._render_command(event)
        elif name == "edit_file":
            self._render_edit(event)
        elif name == "write_file":
            self._render_write(event)
        else:
            self._render_other_tool(event)
        self._record_success(event)
        self.console.print()

    def show_result(self, result: AgentResult) -> None:
        self.stop_thinking()
        self._flush_file_activity()
        separator_width = min(max(self.console.width, 1), 80)
        self.console.print(Text("─" * separator_width, style="dim"))

        if result.status == "completed":
            status = Text("✓ Done", style="bold green")
            status.append(f" · {result.steps} steps")
        else:
            status = Text("✗ Stopped", style="bold red")
            status.append(f" · {result.status} · {result.steps} steps")
        self.console.print(status)

        if result.message:
            self.console.print(Text(result.message))

        changed = self._changed_summary()
        if changed is not None:
            self.console.print(changed)

        if result.status == "completed" and result.answer:
            answer = _compact_answer_spacing(result.answer)
            if answer:
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

    def _buffer_file_activity(self, event: ToolEvent) -> None:
        raw_data = event.result.get("data")
        data = raw_data if isinstance(raw_data, dict) else {}
        name = event.tool_call.name
        self._pending_file_activity = True

        if name == "list_files":
            entries = data.get("entries")
            if isinstance(entries, list):
                for entry in entries:
                    if not isinstance(entry, dict) or entry.get("type") != "file":
                        continue
                    path = _string(entry.get("path"))
                    if path:
                        self._pending_paths.setdefault(path, None)
        elif name == "read_file":
            path = _string(data.get("path"))
            if path:
                self._pending_paths.setdefault(path, None)
        elif name == "search_code":
            matches = data.get("matches")
            if isinstance(matches, list):
                for match in matches:
                    if not isinstance(match, dict):
                        continue
                    path = _string(match.get("path"))
                    if path:
                        self._pending_paths.setdefault(path, None)
            query = _string(data.get("query"))
            count = _integer(data.get("count"))
            if query is not None or count is not None:
                self._pending_searches.append((query, count))

    def _flush_file_activity(self) -> None:
        if not self._pending_file_activity:
            return

        self.console.print(Text("● Files", style="bold cyan"))
        paths = list(self._pending_paths)
        if paths:
            visible = paths[:FILE_PATH_LIMIT]
            line = Text("  " + " · ".join(visible))
            if len(paths) > FILE_PATH_LIMIT:
                line.append(f" · +{len(paths) - FILE_PATH_LIMIT} more", style="dim")
            self.console.print(line)

        for query, count in self._pending_searches[:SEARCH_SUMMARY_LIMIT]:
            summary = Text("  Search", style="dim")
            if query is not None:
                summary.append(f" {query}")
            if count is not None:
                summary.append(f" · {count} matches", style="dim")
            self.console.print(summary)
        if len(self._pending_searches) > SEARCH_SUMMARY_LIMIT:
            self.console.print(
                Text(
                    f"  +{len(self._pending_searches) - SEARCH_SUMMARY_LIMIT} searches",
                    style="dim",
                )
            )

        self.console.print()
        self._pending_file_activity = False
        self._pending_paths.clear()
        self._pending_searches.clear()

    def _render_command(self, event: ToolEvent) -> None:
        raw_data = event.result.get("data")
        data = raw_data if isinstance(raw_data, dict) else {}
        command = _safe_command(data.get("command"))
        error_code = _string(event.result.get("error_code"))
        message = _string(event.result.get("message"))

        if command is None:
            if event.result.get("ok") is True:
                self._render_heading("✓", "Command completed", "green")
            else:
                title = "Command not executed"
                if error_code:
                    title += f" · {error_code}"
                self._render_heading("⚠", title, "yellow")
                if message:
                    self.console.print(Text(f"  {message}"))
            return

        marker = "✓" if event.result.get("ok") is True else "✗"
        color = "green" if event.result.get("ok") is True else "red"
        self._render_heading(marker, command, color)

        meta_rendered = False
        if event.result.get("ok") is not True and error_code not in (None, "command_failed"):
            self._render_compact_failure(error_code, message)
            meta_rendered = True

        if event.result.get("ok") is True:
            evidence = _select_success_evidence(data)
            limit = SUCCESS_OUTPUT_LIMIT
        else:
            evidence = _select_failure_evidence(data)
            limit = FAILURE_OUTPUT_LIMIT

        if evidence:
            self._render_evidence(evidence, limit)
        elif event.result.get("ok") is not True and not meta_rendered:
            self._render_compact_failure(error_code, message)

    def _render_edit(self, event: ToolEvent) -> None:
        arguments = _object_arguments(event.tool_call.arguments)
        path = _string(arguments.get("path")) or "edit_file"
        if event.result.get("ok") is not True:
            self._render_heading("✗", path, "red")
            self._render_compact_failure(
                _string(event.result.get("error_code")),
                _string(event.result.get("message")),
            )
            return

        self._render_heading("◆", path, "green")
        old_text = arguments.get("old_text")
        new_text = arguments.get("new_text")
        if not isinstance(old_text, str) or not isinstance(new_text, str):
            return

        diff_lines = []
        for line in difflib.ndiff(old_text.splitlines(), new_text.splitlines()):
            if line.startswith("? "):
                continue
            if line.startswith("- "):
                diff_lines.append("-" + line[2:])
            elif line.startswith("+ "):
                diff_lines.append("+" + line[2:])
        if not diff_lines:
            return

        raw_diff = "\n".join(diff_lines)
        truncated = len(raw_diff) > DIFF_OUTPUT_LIMIT
        diff_text = _bounded_text(
            raw_diff, DIFF_OUTPUT_LIMIT, DIFF_TRUNCATION_MARKER
        )
        for line in diff_text.splitlines():
            if line.startswith("-"):
                style = "red"
            elif line.startswith("+"):
                style = "green"
            else:
                style = "dim"
            self.console.print(Text(f"  {line}", style=style))
        if truncated and DIFF_TRUNCATION_MARKER not in diff_text:
            self.console.print(Text(f"  {DIFF_TRUNCATION_MARKER}", style="dim"))

    def _render_write(self, event: ToolEvent) -> None:
        arguments = _object_arguments(event.tool_call.arguments)
        path = _string(arguments.get("path")) or "write_file"
        if event.result.get("ok") is True:
            self._render_heading("◆", f"Created {path}", "green")
        else:
            self._render_heading("✗", path, "red")
            self._render_compact_failure(
                _string(event.result.get("error_code")),
                _string(event.result.get("message")),
            )

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
            self._render_compact_failure(
                _string(event.result.get("error_code")),
                _string(event.result.get("message")),
            )

    def _render_compact_failure(
        self, error_code: str | None, message: str | None
    ) -> None:
        detail = " · ".join(part for part in (error_code, message) if part)
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

    def _record_success(self, event: ToolEvent) -> None:
        if event.result.get("ok") is not True:
            return
        if event.tool_call.name in {"edit_file", "write_file"}:
            arguments = _object_arguments(event.tool_call.arguments)
            path = _string(arguments.get("path"))
            if path and path not in self._changed_files:
                self._changed_files.append(path)
        elif event.tool_call.name == "run_command":
            raw_data = event.result.get("data")
            data = raw_data if isinstance(raw_data, dict) else {}
            command = _safe_command(data.get("command"))
            if command:
                self._last_successful_command = command

    def _changed_summary(self) -> Text | None:
        count = len(self._changed_files)
        if count == 0:
            return None
        if count == 1:
            value = self._changed_files[0]
            return Text(f"Changed {value}")
        if count <= 3:
            return Text("Changed " + " · ".join(self._changed_files))
        return Text(f"Changed {count} files")


def _object_arguments(raw_arguments: str) -> dict[str, Any]:
    try:
        value = json.loads(raw_arguments)
    except (json.JSONDecodeError, TypeError):
        return {}
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
        selected = keyword_lines if keyword_lines else lines[-6:]
    return "\n".join(selected[:8])


def _select_success_evidence(data: dict[str, Any]) -> str:
    lines = _stream_lines(data)
    summaries = [line for line in lines if _is_success_summary(line)]
    selected = summaries[:3] if summaries else lines[-3:]
    return "\n".join(selected)


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
        re.match(r"^Ran\s+\d+\s+tests?\b", candidate, re.IGNORECASE)
        or re.match(r"^(?:OK|PASS|PASSED)$", candidate, re.IGNORECASE)
        or re.search(r"\b\d+\s+passed\b", candidate, re.IGNORECASE)
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
    candidate = line.strip()
    return bool(
        re.search(r"\b(?:OK|PASS|PASSED)\b", candidate, re.IGNORECASE)
        or re.match(r"^Ran\s+\d+\s+tests?\b", candidate, re.IGNORECASE)
    )


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
