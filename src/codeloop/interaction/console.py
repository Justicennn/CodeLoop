from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.live import Live
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
from rich.tree import Tree

from codeloop.agent.events import CoreActionEvent, RecoveryEvent, ToolEvent
from codeloop.agent.plan import PlanStep
from codeloop.agent.runner import AgentResult

from .presentation import (
    PresentationAction,
    PresentationBlock,
    PresentationLine,
    PresentationSnapshot,
    PresentationState,
    PresentationUpdate,
)


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
    def __init__(
        self,
        console: Console | None = None,
        *,
        live: bool = False,
    ) -> None:
        self.console = console or Console()
        self._thinking: Status | None = None
        self._presentation = PresentationState()
        self._live_capable = live and self.console.is_terminal
        self._live: Live | None = None
        self._live_disabled = False
        self._live_failed = False
        self._used_live = False

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
        banner_width = min(self.console.width, 104)
        self.console.print(
            Panel(
                Padding(content, (0, 1)),
                title=Text(" CodeLoop ", style=f"bold {_ACCENT_STYLE}"),
                title_align="left",
                border_style=_ACCENT_STYLE,
                padding=(1, 1),
                width=banner_width,
                expand=False,
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
        if self._live_capable and not self._live_disabled:
            if self._ensure_live():
                return
        if self._thinking is None:
            self._thinking = self.console.status(
                "[cyan]Working...[/cyan]", spinner="dots"
            )
            self._thinking.start()

    def stop_thinking(self) -> None:
        if self._live is not None:
            return
        self._stop_status()

    def close(self) -> None:
        """Idempotently stop every transient Presentation primitive."""
        self._stop_status()
        live = self._live
        self._live = None
        if live is not None:
            try:
                live.stop()
            except Exception:
                self._live_failed = True

    def _stop_status(self) -> None:
        thinking = self._thinking
        self._thinking = None
        if thinking is not None:
            try:
                thinking.stop()
            except Exception:
                pass

    def show_narration(self, narration: str) -> None:
        """Render model-provided public narration without inferring intent."""
        self.stop_thinking()
        text = _compact_answer_spacing(narration)
        if not text:
            return
        try:
            update = self._presentation.observe_narration(text)
        except Exception:
            self._disable_live()
            self._render_linear_narration(text)
            return
        if self._present_live(update):
            return
        self._render_linear_narration(update.linear_narration or text)

    def _render_linear_narration(self, text: str) -> None:
        with self.console.use_theme(
            Theme(_BACKGROUND_FREE_MARKDOWN_STYLES), inherit=True
        ):
            self.console.print(
                Markdown(text, code_theme=_TERMINAL_NATIVE_SYNTAX_THEME)
            )
        self.console.print()

    def show_tool_event(self, event: ToolEvent) -> None:
        self.stop_thinking()
        try:
            update = self._presentation.observe_tool(event)
        except Exception:
            self._disable_live()
            self._render_linear_tool(event, _fallback_tool_section(event))
            return
        if self._present_live(update):
            return
        self._render_linear_tool(event, update.linear_tool_section)

    def _render_linear_tool(
        self,
        event: ToolEvent,
        section: str | None,
    ) -> None:
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
        try:
            update = self._presentation.observe_core_action(event)
        except Exception:
            self._disable_live()
            self._render_core_failure_fallback(event)
            return
        if self._present_live(update):
            return
        for block in update.linear_blocks:
            self._render_block(block)

    def show_recovery_event(self, event: RecoveryEvent) -> None:
        self.stop_thinking()
        try:
            update = self._presentation.observe_recovery(event)
        except Exception:
            self._disable_live()
            self._render_heading(
                "⚠",
                "Recovery requested after no material progress",
                _WARNING_STYLE,
            )
            return
        if self._present_live(update):
            return
        for block in update.linear_blocks:
            self._render_block(block)

    def show_result(self, result: AgentResult) -> None:
        live_was_clean = self._used_live and not self._live_failed
        self.close()
        if live_was_clean and not self._live_failed:
            self._render_live_result(result)
            return
        self._render_linear_result(result)

    def _render_live_result(self, result: AgentResult) -> None:
        if result.status != "completed":
            line = Text("✗ ", style=f"bold {_ERROR_STYLE}")
            line.append(f"Stopped · {result.status}")
            self.console.print(line)
            if result.message:
                self.console.print(Text(result.message))
            self.console.print()
            return
        answer = result.answer or ""
        if not answer.strip():
            self._render_heading("✓", "Done", _SUCCESS_STYLE)
            self.console.print()
            return
        with self.console.use_theme(
            Theme(_BACKGROUND_FREE_MARKDOWN_STYLES), inherit=True
        ):
            self.console.print(
                Markdown(answer, code_theme=_TERMINAL_NATIVE_SYNTAX_THEME)
            )
        self.console.print()

    def _render_linear_result(self, result: AgentResult) -> None:
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

    def _ensure_live(self) -> bool:
        if not self._live_capable or self._live_disabled:
            return False
        if self._live is not None:
            return True
        try:
            initial = self._build_live_renderable(
                self._presentation.snapshot(),
                initial=True,
            )
            live = Live(
                initial,
                console=self.console,
                transient=True,
                auto_refresh=False,
            )
            self._live = live
            self._used_live = True
            live.start(refresh=False)
            live.refresh()
            return True
        except Exception:
            self._disable_live()
            return False

    def _present_live(self, update: PresentationUpdate) -> bool:
        if not self._live_capable or self._live_disabled:
            return False
        if not self._ensure_live():
            return False
        live = self._live
        if live is None:
            return False
        try:
            renderable = self._build_live_renderable(update.snapshot)
            live.update(renderable, refresh=False)
            live.refresh()
            return True
        except Exception:
            self._disable_live()
            return False

    def _disable_live(self) -> None:
        self._live_disabled = True
        self._live_failed = self._used_live or self._live_capable
        live = self._live
        self._live = None
        if live is not None:
            try:
                live.stop()
            except Exception:
                pass

    def _build_live_renderable(
        self,
        snapshot: PresentationSnapshot,
        *,
        initial: bool = False,
    ) -> Group:
        phase = snapshot.phase or "Understanding"
        root_label = Text("● ", style=f"bold {_ACCENT_STYLE}")
        root_label.append(phase, style=f"bold {_ACCENT_STYLE}")
        tree = Tree(root_label, guide_style=_MUTED_STYLE)

        if snapshot.plan_steps:
            for step in snapshot.plan_steps:
                tree.add(_plan_step_text(step))
            if snapshot.hidden_plan_steps:
                tree.add(
                    Text(
                        f"… {snapshot.hidden_plan_steps} more plan steps",
                        style=_MUTED_STYLE,
                    )
                )

        action_parent = tree
        if snapshot.plan_steps and snapshot.actions:
            action_parent = tree.add(Text("Evidence", style=_MUTED_STYLE))
        self._append_live_actions(action_parent, snapshot.actions)
        if snapshot.hidden_actions:
            action_parent.add(
                Text(
                    f"… {snapshot.hidden_actions} earlier actions",
                    style=_MUTED_STYLE,
                )
            )

        facts = Table.grid(padding=(0, 2), expand=False)
        facts.add_column(style=_MUTED_STYLE, no_wrap=True)
        facts.add_column(overflow="fold")
        for index, finding in enumerate(snapshot.findings):
            label = "发现" if index == 0 else ""
            finding_text = Text()
            finding_text.append(f"{finding.priority.upper()} · ", style=_MUTED_STYLE)
            finding_text.append(finding.title)
            facts.add_row(label, finding_text)
        if snapshot.hidden_findings:
            facts.add_row("", Text(f"… {snapshot.hidden_findings} more", style=_MUTED_STYLE))

        current = snapshot.current
        if initial and current is None:
            current = "正在理解任务和已有上下文"
        if current:
            facts.add_row("当前", Text(current, style=_MUTED_STYLE))
        if snapshot.next_step:
            facts.add_row("下一步", Text(snapshot.next_step, style=_MUTED_STYLE))

        if facts.row_count:
            return Group(tree, Text(""), facts)
        return Group(tree)

    def _append_live_actions(
        self,
        parent: Tree,
        actions: tuple[PresentationAction, ...],
    ) -> None:
        grouped_reads = [action for action in actions if action.label == "Read file"]
        emitted_read_group = False
        for action in actions:
            if action.label == "Read file" and len(grouped_reads) > 1:
                if emitted_read_group:
                    continue
                emitted_read_group = True
                status = (
                    "failure"
                    if any(item.status == "failure" for item in grouped_reads)
                    else "success"
                )
                branch = parent.add(_action_text("Read files", status, None, 1))
                for item in grouped_reads:
                    child = branch.add(
                        _action_text("", item.status, item.target, item.count)
                    )
                    for detail in item.details:
                        child.add(Text(detail, style=_MUTED_STYLE))
                continue
            node = parent.add(
                _action_text(
                    action.label,
                    action.status,
                    action.target,
                    action.count,
                )
            )
            for detail in action.details:
                node.add(Text(detail, style=_MUTED_STYLE))

    def _render_core_failure_fallback(self, event: CoreActionEvent) -> None:
        marker = "✓" if event.result.get("ok") is True else "✗"
        color = _SUCCESS_STYLE if event.result.get("ok") is True else _ERROR_STYLE
        self._render_heading(marker, event.name, color)
        message = event.result.get("message")
        if isinstance(message, str) and message:
            self.console.print(Text(f"  {message}", style=color))

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


def _plan_step_text(step: PlanStep) -> Text:
    marker, style = {
        "completed": ("✓", _SUCCESS_STYLE),
        "in_progress": ("●", _ACCENT_STYLE),
        "pending": ("○", _MUTED_STYLE),
        "blocked": ("⚠", _WARNING_STYLE),
    }[step.status]
    line = Text(f"{marker} ", style=f"bold {style}")
    line.append(step.description, style=style if step.status != "completed" else None)
    if step.status == "blocked" and step.blocked_reason:
        line.append(f" · {step.blocked_reason}", style=_MUTED_STYLE)
    return line


def _action_text(
    label: str,
    status: str,
    target: str | None,
    count: int,
) -> Text:
    marker, style = {
        "success": ("✓", _SUCCESS_STYLE),
        "failure": ("✗", _ERROR_STYLE),
        "warning": ("⚠", _WARNING_STYLE),
    }.get(status, ("✗", _ERROR_STYLE))
    line = Text(f"{marker} ", style=f"bold {style}")
    if label:
        line.append(label)
    if target:
        if label:
            line.append(" · ", style=_MUTED_STYLE)
        line.append(target, style=_MUTED_STYLE)
    if count > 1:
        line.append(f" · {count}x", style=_MUTED_STYLE)
    return line


def _fallback_tool_section(event: ToolEvent) -> str | None:
    name = event.tool_call.name
    if event.result.get("ok") is True and name in _READ_ONLY_TOOLS:
        return None
    if name == "run_command":
        if event.result.get("error_code") in {
            "user_denied",
            "approval_unavailable",
        }:
            return "WORKING"
        return "VERIFICATION"
    if (
        event.result.get("ok") is not True
        and name in {"read_document", "read_webpage", "read_image"}
    ):
        return "UNDERSTANDING"
    return "WORKING"


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
