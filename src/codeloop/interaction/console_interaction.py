"""CLI implementation of the pure Human Interaction protocol."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from typing import TYPE_CHECKING

from ..control import (
    InteractionAction,
    InteractionProvider,
    InteractionRequest,
    InteractionResponse,
)

if TYPE_CHECKING:
    from .console import ConsoleRenderer


class ConsoleInteractionProvider(InteractionProvider):
    """Suspend transient output, show one public request, and read one answer."""

    def __init__(
        self,
        *,
        read_line: Callable[[str], str] | None = None,
        write_line: Callable[[str], None] | None = None,
        renderer: ConsoleRenderer | None = None,
    ) -> None:
        self._read_line = read_line
        self._write_line = write_line or print
        self._renderer = renderer

    def interact(self, request: InteractionRequest) -> InteractionResponse:
        self._suspend_live()
        try:
            self._render_request(request)
            if request.kind == "inform":
                return InteractionResponse(status="answered", answer="acknowledged")
            if request.kind in {"approve", "re_approve"}:
                return self._read_approval()
            if request.kind == "choose":
                return self._read_choice(request)
            return self._read_clarification()
        finally:
            self._resume_live()

    def _render_request(self, request: InteractionRequest) -> None:
        callback = getattr(self._renderer, "show_interaction_request", None)
        if callback is not None:
            try:
                callback(request)
                return
            except Exception:
                pass
        detailed_reapproval = (
            request.kind == "re_approve"
            and request.action is not None
            and bool(request.action.previous_command)
            and request.action.scope_change is not None
        )
        labels = {
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
        }
        self._write_line(labels[request.kind])
        if not detailed_reapproval:
            self._write_line(request.prompt)
        if detailed_reapproval and request.action is not None:
            self._write_line("之前已允许：")
            self._write_line(
                "  " + subprocess.list2cmdline(request.action.previous_command)
            )
            if request.action.previous_cwd != request.action.cwd:
                _write_plain_cwd(
                    self._write_line,
                    request.action,
                    request.action.previous_cwd,
                    prefix="  工作目录：",
                )
            self._write_line("现在准备运行：")
            self._write_line(
                "  " + subprocess.list2cmdline(request.action.command)
            )
            _write_plain_cwd(
                self._write_line,
                request.action,
                request.action.cwd,
                prefix="  工作目录：",
            )
            self._write_line("变化：")
            self._write_line(f"  {request.action.scope_change}")
            return
        if request.action is not None:
            self._write_line(f"操作：{request.action.description}")
            if request.action.command:
                self._write_line(
                    "命令：" + subprocess.list2cmdline(request.action.command)
                )
            if request.action.cwd:
                _write_plain_cwd(
                    self._write_line,
                    request.action,
                    request.action.cwd,
                    prefix="工作目录：",
                )
        for index, option in enumerate(request.options, start=1):
            suffix = f" — {option.description}" if option.description else ""
            self._write_line(f"  {index}. {option.label}{suffix}")

    def _read_approval(self) -> InteractionResponse:
        try:
            value = self._read("是否继续？[y/N]：")
        except KeyboardInterrupt:
            return InteractionResponse(status="interrupted")
        except EOFError:
            return InteractionResponse(status="unavailable")
        approved = value.strip().casefold() in {"y", "yes"}
        self._render_response("已批准。" if approved else "已拒绝。", approved)
        return InteractionResponse(
            status="answered",
            answer="approved" if approved else "denied",
            approved=approved,
        )

    def _read_choice(self, request: InteractionRequest) -> InteractionResponse:
        try:
            value = self._read(
                f"请选择 [1-{len(request.options)}]："
            ).strip()
        except KeyboardInterrupt:
            return InteractionResponse(status="interrupted")
        except EOFError:
            return InteractionResponse(status="unavailable")
        try:
            index = int(value) - 1
        except ValueError:
            return InteractionResponse(status="answered", answer=value)
        if not 0 <= index < len(request.options):
            return InteractionResponse(status="answered", answer=value)
        option = request.options[index]
        self._render_response(f"已选择：{option.label}", True)
        return InteractionResponse(
            status="answered",
            answer=option.label,
            selected_option_id=option.id,
        )

    def _read_clarification(self) -> InteractionResponse:
        try:
            answer = self._read("请输入说明：").strip()
        except KeyboardInterrupt:
            return InteractionResponse(status="interrupted")
        except EOFError:
            return InteractionResponse(status="unavailable")
        answer = answer[:2_000]
        self._render_response("已记录回复。", True)
        return InteractionResponse(status="answered", answer=answer)

    def _read(self, prompt: str) -> str:
        if self._read_line is not None:
            return self._read_line(prompt)
        callback = getattr(self._renderer, "read_interaction_input", None)
        if callback is not None:
            return callback(prompt)
        return input(prompt)

    def _render_response(self, text: str, positive: bool) -> None:
        callback = getattr(self._renderer, "show_interaction_response", None)
        if callback is not None:
            try:
                callback(text, positive)
                return
            except Exception:
                pass
        self._write_line(text)

    def _suspend_live(self) -> None:
        callback = getattr(self._renderer, "suspend_live_for_interaction", None)
        if callback is not None:
            try:
                callback()
            except Exception:
                pass

    def _resume_live(self) -> None:
        callback = getattr(self._renderer, "resume_live_after_interaction", None)
        if callback is not None:
            try:
                callback()
            except Exception:
                pass


class NonInteractiveInteractionProvider(InteractionProvider):
    """One-shot provider: notifications work, responses fail closed."""

    def __init__(self, write_line: Callable[[str], None] | None = None) -> None:
        self._write_line = write_line or print

    def interact(self, request: InteractionRequest) -> InteractionResponse:
        if request.kind == "inform":
            self._write_line(request.prompt)
            if request.action is not None:
                self._write_line(f"操作：{request.action.description}")
                if request.action.command:
                    self._write_line(
                        "命令："
                        + subprocess.list2cmdline(request.action.command)
                    )
                if request.action.cwd is not None:
                    _write_plain_cwd(
                        self._write_line,
                        request.action,
                        request.action.cwd,
                        prefix="工作目录：",
                    )
            return InteractionResponse(status="answered", answer="acknowledged")
        return InteractionResponse(status="unavailable")


def _human_cwd(
    action: InteractionAction,
    cwd: str | None,
) -> tuple[str, str | None]:
    workspace_root = action.workspace_root
    if cwd in {None, "", "."}:
        return (
            "当前项目根目录",
            workspace_root if isinstance(workspace_root, str) else None,
        )
    return (f"{cwd}（相对于项目根目录）", None)


def _write_plain_cwd(
    write_line: Callable[[str], None],
    action: InteractionAction,
    cwd: str | None,
    *,
    prefix: str,
) -> None:
    label, absolute_path = _human_cwd(action, cwd)
    write_line(f"{prefix}{label}")
    if absolute_path:
        write_line(absolute_path)
