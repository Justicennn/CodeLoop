"""Narrow events emitted by Agent Core without a presentation dependency."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from ..execution.command_policy import CommandApprovalRequest
from ..execution.tools import ToolResult
from ..model.client import ToolCall
from .plan import PlanStep


@dataclass(frozen=True)
class ToolEvent:
    tool_call: ToolCall
    result: ToolResult
    dispatch_duration_ms: int
    truncated: bool


CoreActionName = Literal[
    "update_plan",
    "update_requirements",
    "update_working_set",
    "update_review_findings",
]


@dataclass(frozen=True)
class ReviewFindingProjection:
    title: str
    finding_type: Literal["issue", "enhancement"]
    priority: Literal["high", "medium", "low"]


@dataclass(frozen=True)
class CoreActionEvent:
    """One Core Action result plus only its bounded presentation projection."""

    name: CoreActionName
    call_id: str
    result: ToolResult
    requirement_count: int | None = None
    requirement_sources: tuple[str, ...] | None = None
    plan_steps: tuple[PlanStep, ...] | None = None
    review_findings: tuple[ReviewFindingProjection, ...] | None = None


@dataclass(frozen=True)
class RecoveryEvent:
    """An explicit recovery request emitted by the existing ProgressTracker."""

    reason: Literal["no_progress"] = "no_progress"


ToolEventHandler = Callable[[ToolEvent], None]
CoreActionEventHandler = Callable[[CoreActionEvent], None]
RecoveryEventHandler = Callable[[RecoveryEvent], None]
ModelRequestHandler = Callable[[], None]
CommandApprovalHandler = Callable[[CommandApprovalRequest], bool]
