"""Narrow events emitted by Agent Core without a presentation dependency."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..execution.command_policy import CommandApprovalRequest
from ..execution.tools import ToolResult
from ..model.client import ToolCall


@dataclass(frozen=True)
class ToolEvent:
    tool_call: ToolCall
    result: ToolResult
    dispatch_duration_ms: int
    truncated: bool


ToolEventHandler = Callable[[ToolEvent], None]
ModelRequestHandler = Callable[[], None]
CommandApprovalHandler = Callable[[CommandApprovalRequest], bool]
