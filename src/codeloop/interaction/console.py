from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.control import Control
from rich.live import Live
from rich.markdown import Heading, Markdown, Paragraph
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.segment import ControlType
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
from codeloop.control import InteractionAction, InteractionRequest

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
_ACCENT_STYLE = "orange3"
_PRIMARY_STYLE = "grey93"
_MUTED_STYLE = "grey62 dim"
_SUCCESS_STYLE = "green"
_WARNING_STYLE = "gold3"
_ERROR_STYLE = "red"
_USER_MESSAGE_STYLE = Style(color="grey93", bgcolor="grey11")
_CONTENT_BASE_WIDTH = 88
_CONTENT_GROWTH_RATIO = 0.45
_MAX_INPUT_REDRAW_LINES = 6
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
_CODELOOP_MARKDOWN_STYLES = {
    **{
        f"markdown.h{level}": Style(
            color=_ACCENT_STYLE,
            bold=True,
            underline=False,
            reverse=False,
        )
        for level in range(1, 7)
    },
    "markdown.paragraph": Style(color=_PRIMARY_STYLE),
    "markdown.code": Style(color="grey93"),
    "markdown.code_block": Style.null(),
    "markdown.block_quote": Style(color="grey62", dim=True),
    "markdown.item.bullet": Style(color="grey62"),
    "markdown.item.number": Style(color="grey62"),
    "markdown.table.border": Style(color="grey50"),
    "markdown.table.header": Style(color="grey93", bold=True),
    "markdown.link": Style(color="bright_blue"),
    "markdown.link_url": Style(color="blue", underline=True),
}


class _CodeLoopHeading(Heading):
    """Keep every Markdown heading left-aligned in the terminal."""

    def __rich_console__(self, console: Console, options: Any):
        del console, options
        text = self.text
        text.justify = "left"
        text.no_wrap = False
        text.overflow = "fold"
        yield text


class _CodeLoopParagraph(Paragraph):
    """Prevent inherited Markdown layout from centering prose."""

    def __rich_console__(self, console: Console, options: Any):
        del console, options
        text = self.text
        text.justify = "left"
        text.no_wrap = False
        text.overflow = "fold"
        yield text


class _CodeLoopMarkdown(Markdown):
    elements = {
        **Markdown.elements,
        "heading_open": _CodeLoopHeading,
        "paragraph_open": _CodeLoopParagraph,
    }


def _get_safe_terminal_width(console_width: int) -> int:
    """Return a positive render width with one TTY-edge cell reserved."""
    width = max(1, console_width)
    return max(1, width - 1)


def _get_content_width(safe_width: int) -> int:
    """Pure responsive reading-width calculation for major UI blocks."""
    width = max(1, safe_width)
    if width <= _CONTENT_BASE_WIDTH:
        return width
    grown = _CONTENT_BASE_WIDTH + round(
        (width - _CONTENT_BASE_WIDTH) * _CONTENT_GROWTH_RATIO
    )
    return min(width, max(1, grown))


def _get_horizontal_margin(
    safe_width: int,
    content_width: int,
) -> tuple[int, int]:
    """Left-anchor bounded content and leave unused width on the right."""
    available = max(1, safe_width)
    content = min(available, max(1, content_width))
    return 0, available - content


@dataclass(frozen=True)
class _LayoutWidths:
    """One render-time projection of viewport and reading geometry."""

    viewport: int
    reading: int
    left: int
    right: int


def _get_layout_widths(console_width: int) -> _LayoutWidths:
    """Purely derive all outer widths from one current console width."""
    viewport = _get_safe_terminal_width(console_width)
    reading = _get_content_width(viewport)
    left, right = _get_horizontal_margin(viewport, reading)
    return _LayoutWidths(viewport, reading, left, right)


def _wrapping_text(
    value: str = "",
    *,
    style: str | Style | None = None,
) -> Text:
    """Create left-aligned text that delegates cell wrapping to Rich."""
    text = Text(value, style=style, justify="left")
    text.no_wrap = False
    text.overflow = "fold"
    return text


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
        self._live_suspended = False
        self._used_live = False
        self._input_region_active = False
        self._input_owned = False
        self._input_start_width: int | None = None
        self._input_line_count: int | None = None
        self._input_had_presentation_output = False

    def show_header(self, model: str, workspace: Path, task: str) -> None:
        title = _wrapping_text("CodeLoop", style=f"bold {_ACCENT_STYLE}")
        title.append(" · ")
        title.append(model, style=_MUTED_STYLE)
        self._print_viewport(title)
        self._print_viewport(_wrapping_text(str(workspace), style=_MUTED_STYLE))
        self.console.print()
        task_line = _wrapping_text("codeloop > ", style=_ACCENT_STYLE)
        task_line.append(task, style=_PRIMARY_STYLE)
        self._print_viewport(task_line)
        self.console.print()

    def show_startup_banner(
        self,
        model: str,
        workspace: Path,
        mode: str = "interactive",
    ) -> None:
        """Render session chrome without touching task PresentationState."""
        metadata = Table.grid(padding=(0, 2), expand=True)
        metadata.add_column(
            style=_MUTED_STYLE,
            no_wrap=True,
            justify="left",
        )
        metadata.add_column(
            style=_PRIMARY_STYLE,
            ratio=1,
            no_wrap=False,
            overflow="fold",
            justify="left",
        )
        metadata.add_row("model", model)
        metadata.add_row("workspace", str(workspace))
        metadata.add_row("mode", mode)

        content = Group(
            _wrapping_text(
                "Welcome to CodeLoop", style=f"bold {_PRIMARY_STYLE}"
            ),
            Text(""),
            metadata,
        )
        layout = self._layout_widths()
        panel = Panel(
            Padding(content, (0, 1)),
            title=Text(" CodeLoop ", style=f"bold {_ACCENT_STYLE}"),
            title_align="left",
            border_style=_ACCENT_STYLE,
            padding=(1, 1),
            width=layout.reading,
            expand=False,
        )
        self.console.print(
            Padding(
                panel,
                (0, layout.right, 0, layout.left),
                expand=False,
            ),
            width=layout.viewport,
        )
        self.console.print()

    def show_input_top_rule(self) -> None:
        """Start one terminal-native user input region."""
        self._reset_input_region()
        if not self.console.is_terminal:
            return
        safe_width = self._safe_terminal_width()
        self._input_region_active = True
        self._input_start_width = safe_width
        # The Rule and input measurement share this render-time safe viewport.
        self._print_viewport(Rule(style=_MUTED_STYLE))

    def read_user_input(self) -> str:
        """Read from the renderer's real console with an accented prompt."""
        if not self.console.is_terminal:
            return self.console.input("")
        self._input_region_active = True
        self._input_owned = True
        self._input_start_width = self._safe_terminal_width()
        self._input_line_count = 1
        value = self.console.input(Text("> ", style=f"bold {_ACCENT_STYLE}"))
        try:
            self._input_line_count = self._measure_input_lines(
                f"> {value}",
                self._input_start_width,
            )
        except Exception:
            # Measurement only decides whether a redraw is safe. It must never
            # cause InteractiveSession to ask the user for the same input twice.
            self._input_line_count = None
        return value

    def show_input_bottom_rule(self) -> None:
        """Legacy compatibility wrapper for callers outside InteractiveSession."""
        self.cancel_input_area()

    def show_submitted_user_message(self, text: str) -> None:
        """Best-effort conversion from owned input chrome to a user block."""
        if not self.console.is_terminal:
            self._reset_input_region()
            line = _wrapping_text("> ", style=_ACCENT_STYLE)
            line.append(text, style=_PRIMARY_STYLE)
            self._print_viewport(line)
            self.console.print()
            return

        cleared = self._clear_owned_input_area(submitted_text=text)
        self._reset_input_region()
        if not cleared:
            self.console.print()
            return

        try:
            line = _wrapping_text("❯ ", style=f"bold {_ACCENT_STYLE}")
            line.append(text, style=_PRIMARY_STYLE)
            message = Table.grid(expand=True, padding=0)
            message.add_column(
                ratio=1,
                no_wrap=False,
                overflow="fold",
                justify="left",
            )
            message.add_row(line)
            block = Padding(
                message,
                (0, 1),
                style=_USER_MESSAGE_STYLE,
                expand=True,
            )
            # The submitted-message bar belongs to the full-width input
            # chrome, not to the narrower long-form reading column used by
            # Banner and Final. Let Rich resolve the current viewport on every
            # render so terminal resizes are reflected without cached widths.
            self._print_viewport(block)
            self.console.print()
        except Exception:
            # Clearing is best-effort, but once it succeeded the submitted
            # text must still remain visible if the richer block cannot render.
            line = _wrapping_text("> ", style=_ACCENT_STYLE)
            line.append(text, style=_PRIMARY_STYLE)
            self._print_viewport(line)
            self.console.print()

    def cancel_input_area(self) -> None:
        """Clear owned waiting chrome when the bounded redraw is safe."""
        if self.console.is_terminal:
            self._clear_owned_input_area()
        self._reset_input_region()

    def show_goodbye(self) -> None:
        self._print_viewport(_wrapping_text("Bye.", style=_MUTED_STYLE))

    def _safe_terminal_width(self) -> int:
        return _get_safe_terminal_width(self.console.size.width)

    def _layout_widths(self) -> _LayoutWidths:
        return _get_layout_widths(self.console.size.width)

    def _print_viewport(self, renderable: Any) -> None:
        """Print once against the current safe terminal viewport."""
        self.console.print(renderable, width=self._layout_widths().viewport)

    def _print_reading(self, renderable: Any) -> None:
        """Print long-form content in the current responsive reading column."""
        layout = self._layout_widths()
        self.console.print(
            Padding(
                renderable,
                (0, layout.right, 0, layout.left),
                expand=True,
            ),
            width=layout.viewport,
        )

    def _measure_input_lines(self, text: str, width: int) -> int:
        options = self.console.options.update(width=max(1, width))
        return max(
            1,
            len(
                self.console.render_lines(
                    _wrapping_text(text),
                    options,
                    pad=False,
                )
            ),
        )

    def _clear_owned_input_area(
        self,
        *,
        submitted_text: str | None = None,
    ) -> bool:
        current_width = self._safe_terminal_width()
        line_count = self._input_line_count
        if (
            not self._input_region_active
            or not self._input_owned
            or self._input_had_presentation_output
            or line_count is None
            or line_count > _MAX_INPUT_REDRAW_LINES
        ):
            return False

        if self._input_start_width != current_width:
            if submitted_text is None:
                return False
            try:
                resized_line_count = self._measure_input_lines(
                    f"> {submitted_text}",
                    current_width,
                )
            except Exception:
                return False
            # Resizing is safe to tolerate only when terminal reflow cannot
            # have changed the number of occupied prompt lines.
            if resized_line_count != line_count:
                return False

        codes: list[tuple[ControlType, int]] = []
        for _ in range(line_count + 1):
            codes.extend(
                (
                    (ControlType.CURSOR_UP, 1),
                    (ControlType.CURSOR_MOVE_TO_COLUMN, 0),
                    (ControlType.ERASE_IN_LINE, 2),
                )
            )
        try:
            self.console.control(Control(*codes))
        except Exception:
            return False
        return True

    def _reset_input_region(self) -> None:
        self._input_region_active = False
        self._input_owned = False
        self._input_start_width = None
        self._input_line_count = None
        self._input_had_presentation_output = False

    def _note_presentation_output(self) -> None:
        if self._input_region_active:
            self._input_had_presentation_output = True

    def start_thinking(self) -> None:
        self._note_presentation_output()
        if self._live_capable and not self._live_disabled:
            if self._ensure_live():
                return
        if self._thinking is None:
            self._thinking = self.console.status(
                f"[{_ACCENT_STYLE}]Working...[/{_ACCENT_STYLE}]",
                spinner="dots",
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

    def suspend_live_for_interaction(self) -> None:
        """Idempotently clear transient Live before permanent human I/O."""
        if self._live_suspended:
            return
        self._live_suspended = True
        self._stop_status()
        live = self._live
        self._live = None
        if live is not None:
            try:
                live.stop()
            except Exception:
                self._live_failed = True

    def resume_live_after_interaction(self) -> None:
        """Allow the next real Presentation event to restart Live."""
        self._live_suspended = False

    def show_interaction_request(self, request: InteractionRequest) -> None:
        """Permanently render one left-aligned Human Interaction checkpoint."""
        detailed_reapproval = (
            request.kind == "re_approve"
            and request.action is not None
            and bool(request.action.previous_command)
            and request.action.scope_change is not None
        )
        heading = {
            "inform": "通知",
            "approve": "需要确认",
            "re_approve": (
                "测试范围需要扩大"
                if request.action is not None
                and request.action.category == "test"
                else "执行范围需要扩大"
            ),
            "clarify": "需要澄清",
            "choose": "请选择一个选项",
        }[request.kind]
        title = _wrapping_text("？ ", style=f"bold {_ACCENT_STYLE}")
        title.append(heading, style=f"bold {_PRIMARY_STYLE}")
        self._print_viewport(title)
        if not detailed_reapproval:
            self._print_viewport(
                _wrapping_text(request.prompt, style=_PRIMARY_STYLE)
            )
        if detailed_reapproval and request.action is not None:
            self._render_scope_change(request.action)
            self.console.print()
            return
        if request.action is not None:
            rows: list[tuple[str, Text]] = [
                (
                    "操作",
                    _wrapping_text(
                        request.action.description,
                        style=_MUTED_STYLE,
                    ),
                )
            ]
            if request.action.command:
                rows.append(
                    (
                        "命令",
                        _wrapping_text(
                            subprocess.list2cmdline(request.action.command),
                            style=_PRIMARY_STYLE,
                        ),
                    )
                )
            if request.action.cwd:
                rows.extend(
                    self._interaction_cwd_rows(
                        request.action,
                        request.action.cwd,
                    )
                )
            self._render_interaction_rows(rows)
        if request.options:
            options = Table.grid(expand=True, padding=(0, 1))
            options.add_column(
                style=_ACCENT_STYLE,
                no_wrap=True,
                justify="right",
            )
            options.add_column(
                ratio=1,
                no_wrap=False,
                overflow="fold",
                justify="left",
            )
            for index, option in enumerate(request.options, start=1):
                line = _wrapping_text(option.label, style=_PRIMARY_STYLE)
                if option.description:
                    line.append(f" — {option.description}", style=_MUTED_STYLE)
                options.add_row(f"{index}.", line)
            self._print_viewport(options)
        self.console.print()

    def _render_scope_change(self, action: InteractionAction) -> None:
        self._print_viewport(_wrapping_text("之前已允许：", style=_MUTED_STYLE))
        previous_rows: list[tuple[str, Text]] = [
            (
                "命令",
                _wrapping_text(
                    subprocess.list2cmdline(action.previous_command),
                    style=_PRIMARY_STYLE,
                ),
            )
        ]
        if action.previous_cwd != action.cwd:
            previous_rows.extend(
                self._interaction_cwd_rows(action, action.previous_cwd)
            )
        self._render_interaction_rows(previous_rows)
        self._print_viewport(
            _wrapping_text("现在准备运行：", style=_MUTED_STYLE)
        )
        current_rows = [
            (
                "命令",
                _wrapping_text(
                    subprocess.list2cmdline(action.command),
                    style=_PRIMARY_STYLE,
                ),
            ),
            *self._interaction_cwd_rows(action, action.cwd),
        ]
        self._render_interaction_rows(current_rows)
        self._print_viewport(_wrapping_text("变化：", style=_MUTED_STYLE))
        self._render_interaction_rows(
            [
                (
                    "范围",
                    _wrapping_text(
                        action.scope_change or "",
                        style=_PRIMARY_STYLE,
                    ),
                )
            ]
        )

    def _interaction_cwd_rows(
        self,
        action: InteractionAction,
        cwd: str | None,
    ) -> list[tuple[str, Text]]:
        if cwd in {None, "", "."}:
            rows = [
                (
                    "工作目录",
                    _wrapping_text("当前项目根目录", style=_PRIMARY_STYLE),
                )
            ]
            if action.workspace_root:
                rows.append(
                    (
                        "",
                        _wrapping_text(action.workspace_root, style=_MUTED_STYLE),
                    )
                )
            return rows
        return [
            (
                "工作目录",
                _wrapping_text(
                    f"{cwd}（相对于项目根目录）",
                    style=_MUTED_STYLE,
                ),
            )
        ]

    def _render_interaction_rows(
        self,
        rows: list[tuple[str, Text]],
    ) -> None:
        details = Table.grid(expand=True, padding=(0, 2))
        details.add_column(
            style=_MUTED_STYLE,
            no_wrap=True,
            justify="left",
        )
        details.add_column(
            ratio=1,
            no_wrap=False,
            overflow="fold",
            justify="left",
        )
        for label, value in rows:
            details.add_row(label, value)
        self._print_viewport(details)

    def read_interaction_input(self, prompt: str) -> str:
        return self.console.input(
            Text(prompt, style=f"bold {_ACCENT_STYLE}")
        )

    def show_interaction_response(self, text: str, positive: bool) -> None:
        line = _wrapping_text("❯ ", style=f"bold {_ACCENT_STYLE}")
        line.append(text, style=_SUCCESS_STYLE if positive else _WARNING_STYLE)
        self._print_viewport(line)
        self.console.print()

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
        self._note_presentation_output()
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
        self._render_markdown(text, responsive=False)
        self.console.print()

    def _render_markdown(self, text: str, *, responsive: bool) -> None:
        source = _compact_answer_spacing(text)
        markdown = _CodeLoopMarkdown(
            source,
            code_theme=_TERMINAL_NATIVE_SYNTAX_THEME,
        )
        with self.console.use_theme(
            Theme(_CODELOOP_MARKDOWN_STYLES), inherit=True
        ):
            if responsive and self.console.is_terminal:
                # The outer reading column owns the width. Markdown children
                # receive one shared remaining ConsoleOptions.max_width and
                # Rich handles mixed CJK, Latin, emoji, and inline code cells.
                self._print_reading(markdown)
                return
            if self.console.is_terminal:
                self._print_viewport(markdown)
                return
            self.console.print(markdown)

    def _render_final_markdown(self, text: str) -> None:
        """Normalize model soft breaks only for public Final Markdown."""
        source = _normalize_final_markdown_soft_breaks(text)
        self._render_markdown(source, responsive=True)

    def show_tool_event(self, event: ToolEvent) -> None:
        self._note_presentation_output()
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
        self._note_presentation_output()
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
        self._note_presentation_output()
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
        self._note_presentation_output()
        live_was_clean = self._used_live and not self._live_failed
        self.close()
        if live_was_clean and not self._live_failed:
            self._render_live_result(result)
            return
        self._render_linear_result(result)

    def _render_live_result(self, result: AgentResult) -> None:
        if result.status != "completed":
            line = _wrapping_text("✗ ", style=f"bold {_ERROR_STYLE}")
            line.append(f"Stopped · {result.status}")
            self._print_viewport(line)
            if result.message:
                self._print_viewport(_wrapping_text(result.message))
            self.console.print()
            return
        answer = result.answer or ""
        if not answer.strip():
            self._render_heading("✓", "Done", _SUCCESS_STYLE)
            self.console.print()
            return
        self._render_final_markdown(answer)
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
                            _WARNING_STYLE,
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
            self._print_viewport(_wrapping_text(result.message))

        if result.status == "completed" and result.answer:
            answer = result.answer
            if answer.strip():
                self.console.print()
                self._render_final_markdown(answer)
        self.console.print()

    def _ensure_live(self) -> bool:
        if not self._live_capable or self._live_disabled or self._live_suspended:
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
    ) -> Padding:
        phase = snapshot.phase or "Understanding"
        root_label = _wrapping_text("● ", style=f"bold {_ACCENT_STYLE}")
        root_label.append(phase, style=f"bold {_ACCENT_STYLE}")
        tree = Tree(root_label, guide_style=_MUTED_STYLE)

        if snapshot.plan_steps:
            for step in snapshot.plan_steps:
                tree.add(_plan_step_text(step))
            if snapshot.hidden_plan_steps:
                tree.add(
                    _wrapping_text(
                        f"… {snapshot.hidden_plan_steps} more plan steps",
                        style=_MUTED_STYLE,
                    )
                )

        action_parent = tree
        if snapshot.plan_steps and snapshot.actions:
            action_parent = tree.add(
                _wrapping_text("Evidence", style=_MUTED_STYLE)
            )
        self._append_live_actions(action_parent, snapshot.actions)
        if snapshot.hidden_actions:
            action_parent.add(
                _wrapping_text(
                    f"… {snapshot.hidden_actions} earlier actions",
                    style=_MUTED_STYLE,
                )
            )

        facts = Table.grid(padding=(0, 2), expand=True)
        facts.add_column(
            style=_MUTED_STYLE,
            no_wrap=True,
            justify="left",
        )
        facts.add_column(
            ratio=1,
            no_wrap=False,
            overflow="fold",
            justify="left",
        )
        for index, finding in enumerate(snapshot.findings):
            label = "发现" if index == 0 else ""
            finding_text = _wrapping_text()
            finding_text.append(f"{finding.priority.upper()} · ", style=_MUTED_STYLE)
            finding_text.append(finding.title)
            facts.add_row(label, finding_text)
        if snapshot.hidden_findings:
            facts.add_row(
                "",
                _wrapping_text(
                    f"… {snapshot.hidden_findings} more",
                    style=_MUTED_STYLE,
                ),
            )

        current = snapshot.current
        if initial and current is None:
            current = "正在理解任务和已有上下文"
        if current:
            facts.add_row(
                "当前", _wrapping_text(current, style=_MUTED_STYLE)
            )
        if snapshot.next_step:
            facts.add_row(
                "下一步",
                _wrapping_text(snapshot.next_step, style=_MUTED_STYLE),
            )

        content: Group
        if facts.row_count:
            content = Group(tree, Text(""), facts)
        else:
            content = Group(tree)
        # Live itself owns the full Console. A one-cell right inset produces
        # the same safe viewport as permanent output while every nested Tree
        # child continues to use Rich's remaining width after guides/indent.
        return Padding(content, (0, 1, 0, 0), expand=True)

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
                        child.add(
                            _wrapping_text(detail, style=_MUTED_STYLE)
                        )
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
                node.add(_wrapping_text(detail, style=_MUTED_STYLE))

    def _render_core_failure_fallback(self, event: CoreActionEvent) -> None:
        marker = "✓" if event.result.get("ok") is True else "✗"
        color = _SUCCESS_STYLE if event.result.get("ok") is True else _ERROR_STYLE
        self._render_heading(marker, event.name, color)
        message = event.result.get("message")
        if isinstance(message, str) and message:
            self._print_indented(message, style=color)

    def _render_block(self, block: PresentationBlock) -> None:
        self._render_section(block.section)
        for line in block.lines:
            self._render_heading(line.marker, line.text, line.style)
            if line.detail:
                self._print_indented(line.detail, style=_MUTED_STYLE)
        self.console.print()

    def _render_section(self, title: str) -> None:
        self._print_viewport(
            Rule(title, align="left", style=_MUTED_STYLE)
        )

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
                    self._print_indented(message)
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
            self._print_indented(message, style=_ERROR_STYLE)

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
            self._print_indented(message, style=_ERROR_STYLE)

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
            self._print_indented(detail, style=_ERROR_STYLE)

    def _render_heading(self, marker: str, title: str, color: str) -> None:
        line = _wrapping_text(f"{marker} ", style=f"bold {color}")
        line.append(title)
        self._print_viewport(line)

    def _print_indented(
        self,
        value: str,
        *,
        style: str | Style | None = None,
        indent: int = 2,
    ) -> None:
        """Indent without changing the shared right viewport boundary."""
        self._print_viewport(
            Padding(
                _wrapping_text(value, style=style),
                (0, 0, 0, max(0, indent)),
                expand=True,
            )
        )

    def _render_evidence(self, evidence: str, limit: int) -> None:
        truncated = len(evidence) > limit
        text = _bounded_text(evidence, limit, TRUNCATION_MARKER)
        for line in text.splitlines():
            style: str | None = None
            if _is_failure_signal(line) or _is_error_detail(line):
                style = "red"
            elif _is_success_signal(line):
                style = "green"
            self._print_indented(line, style=style)
        if truncated and TRUNCATION_MARKER not in text:
            self._print_indented(TRUNCATION_MARKER, style=_MUTED_STYLE)


def _plan_step_text(step: PlanStep) -> Text:
    marker, style = {
        "completed": ("✓", _SUCCESS_STYLE),
        "in_progress": ("●", _ACCENT_STYLE),
        "pending": ("○", _MUTED_STYLE),
        "blocked": ("⚠", _WARNING_STYLE),
    }[step.status]
    line = _wrapping_text(f"{marker} ", style=f"bold {style}")
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
    line = _wrapping_text(f"{marker} ", style=f"bold {style}")
    if label:
        line.append(label)
    if target:
        if label:
            line.append(" · ", style=_MUTED_STYLE)
        line.append(target, style=_MUTED_STYLE)
    if count > 1:
        line.append(f" · ×{count}", style=_MUTED_STYLE)
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


_MARKDOWN_FENCE_OPEN_RE = re.compile(
    r"^ {0,3}(?P<marker>`{3,}|~{3,})"
)
_MARKDOWN_ATX_HEADING_RE = re.compile(r"^ {0,3}#{1,6}(?:\s|$)")
_MARKDOWN_BLOCKQUOTE_RE = re.compile(r"^ {0,3}>")
_MARKDOWN_LIST_ITEM_RE = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s+")
_MARKDOWN_HORIZONTAL_RULE_RE = re.compile(
    r"^ {0,3}(?:(?:\*\s*){3,}|(?:-\s*){3,}|(?:_\s*){3,})$"
)
_MARKDOWN_INDENTED_CODE_RE = re.compile(r"^(?: {4}|\t)")
_MARKDOWN_TABLE_DELIMITER_RE = re.compile(
    r"^ {0,3}\|?\s*:?-{3,}:?\s*"
    r"(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)
_MARKDOWN_HARD_BREAK_RE = re.compile(r"(?: {2,}|\\)$")
_CJK_CHARACTER_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"
)


def _normalize_final_markdown_soft_breaks(text: str) -> str:
    """Join model-created prose soft breaks without changing Markdown blocks."""
    normalized: list[str] = []
    block_kind: str | None = None
    fence_marker: str | None = None
    in_table = False

    lines = iter(text.splitlines())
    line = next(lines, None)
    while line is not None:
        next_line = next(lines, None)

        if fence_marker is not None:
            normalized.append(line)
            if _is_markdown_fence_close(line, fence_marker):
                fence_marker = None
            line = next_line
            continue

        opening_fence = _MARKDOWN_FENCE_OPEN_RE.match(line)
        if opening_fence is not None:
            normalized.append(line)
            fence_marker = opening_fence.group("marker")
            block_kind = None
            in_table = False
            line = next_line
            continue

        if in_table:
            if _is_markdown_table_row(line):
                normalized.append(line)
                block_kind = None
                line = next_line
                continue
            in_table = False

        if not line.strip():
            normalized.append(line)
            block_kind = None
            line = next_line
            continue

        if (
            next_line is not None
            and _is_markdown_table_header(line)
            and _MARKDOWN_TABLE_DELIMITER_RE.match(next_line) is not None
        ):
            normalized.append(line)
            block_kind = None
            in_table = True
            line = next_line
            continue

        if _is_markdown_structural_line(line):
            normalized.append(line)
            block_kind = None
            line = next_line
            continue

        if _MARKDOWN_LIST_ITEM_RE.match(line) is not None:
            normalized.append(line)
            block_kind = "list"
            line = next_line
            continue

        if block_kind in {"paragraph", "list"} and normalized:
            if _MARKDOWN_HARD_BREAK_RE.search(normalized[-1]) is not None:
                normalized.append(line)
            else:
                previous = normalized[-1].rstrip()
                continuation = line.strip()
                separator = _markdown_soft_break_separator(
                    previous,
                    continuation,
                )
                normalized[-1] = f"{previous}{separator}{continuation}"
        else:
            normalized.append(line)
            block_kind = "paragraph"
        line = next_line

    return "\n".join(normalized)


def _is_markdown_fence_close(line: str, marker: str) -> bool:
    marker_character = re.escape(marker[0])
    return bool(
        re.match(
            rf"^ {{0,3}}{re.escape(marker)}{marker_character}*\s*$",
            line,
        )
    )


def _is_markdown_table_header(line: str) -> bool:
    return "|" in line and bool(line.strip())


def _is_markdown_table_row(line: str) -> bool:
    return "|" in line and bool(line.strip())


def _is_markdown_structural_line(line: str) -> bool:
    return bool(
        _MARKDOWN_ATX_HEADING_RE.match(line)
        or _MARKDOWN_BLOCKQUOTE_RE.match(line)
        or _MARKDOWN_HORIZONTAL_RULE_RE.match(line)
        or _MARKDOWN_INDENTED_CODE_RE.match(line)
    )


def _markdown_soft_break_separator(
    previous: str,
    continuation: str,
) -> str:
    """Avoid inserting artificial whitespace inside a CJK word."""
    if not previous or not continuation:
        return ""
    if (
        _CJK_CHARACTER_RE.fullmatch(previous[-1]) is not None
        and _CJK_CHARACTER_RE.fullmatch(continuation[0]) is not None
    ):
        return ""
    return " "


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
