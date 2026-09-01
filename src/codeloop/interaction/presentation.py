"""Task-local aggregation of narrow Agent Core presentation events."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..agent.events import CoreActionEvent, RecoveryEvent, ToolEvent
from ..agent.plan import PlanStep


@dataclass(frozen=True)
class PresentationLine:
    marker: str
    text: str
    style: str
    detail: str | None = None


@dataclass(frozen=True)
class PresentationBlock:
    section: str
    lines: tuple[PresentationLine, ...]


class PresentationState:
    """Aggregate only already-explicit Runtime facts for one task."""

    def __init__(self) -> None:
        self._requirements_signature: tuple[object, ...] | None = None
        self._plan_by_id: dict[str, PlanStep] | None = None
        self._review_signature: tuple[object, ...] | None = None
        self._changed_files: dict[str, str] = {}
        self._displayed_command_calls: set[str] = set()
        self._structured_task = False

    def record_core_action(
        self,
        event: CoreActionEvent,
    ) -> PresentationBlock | None:
        name = event.name
        if event.result.get("ok") is not True:
            return PresentationBlock(
                _core_section(name),
                (_failure_line(name, event.result),),
            )

        data = _result_data(event.result)
        if data.get("changed") is False:
            return None

        if (
            name == "update_requirements"
            and event.requirement_count is not None
            and event.requirement_sources is not None
        ):
            signature = (
                event.requirement_count,
                event.requirement_sources,
            )
            if signature == self._requirements_signature:
                return None
            self._requirements_signature = signature
            self._structured_task = True
            lines = [
                PresentationLine("✓", f"Source: {source}", "green")
                for source in event.requirement_sources
            ]
            count = event.requirement_count
            noun = "requirement" if count == 1 else "requirements"
            lines.append(
                PresentationLine(
                    "✓",
                    f"{count} {noun} registered",
                    "green",
                )
            )
            return PresentationBlock("UNDERSTANDING", tuple(lines))

        if name == "update_plan" and event.plan_steps is not None:
            current = {step.id: step for step in event.plan_steps}
            if self._plan_by_id is None:
                changed_steps = event.plan_steps
                removed_lines: tuple[PresentationLine, ...] = ()
            else:
                changed_steps = tuple(
                    step
                    for step in event.plan_steps
                    if self._plan_by_id.get(step.id) != step
                )
                current_ids = {step.id for step in event.plan_steps}
                if tuple(self._plan_by_id) != tuple(
                    step.id for step in event.plan_steps
                ):
                    changed_steps = event.plan_steps
                removed_lines = tuple(
                    PresentationLine(
                        "○",
                        f"Removed from plan · {step.description}",
                        "dim",
                    )
                    for step_id, step in self._plan_by_id.items()
                    if step_id not in current_ids
                )
            self._plan_by_id = current
            self._structured_task = True
            if not changed_steps and not removed_lines:
                return None
            return PresentationBlock(
                "PLAN",
                tuple(_plan_line(step) for step in changed_steps) + removed_lines,
            )

        if name == "update_review_findings" and event.review_findings is not None:
            signature = tuple(event.review_findings)
            if signature == self._review_signature:
                return None
            self._review_signature = signature
            self._structured_task = True
            ordered = sorted(
                event.review_findings,
                key=lambda finding: {"high": 0, "medium": 1, "low": 2}[
                    finding.priority
                ],
            )
            if not ordered:
                return PresentationBlock(
                    "REVIEW",
                    (PresentationLine("✓", "0 findings registered", "green"),),
                )
            return PresentationBlock(
                "REVIEW",
                tuple(
                    PresentationLine(
                        "⚠" if finding.priority == "high" else "●",
                        (
                            f"{finding.priority.upper()} · "
                            f"{finding.finding_type} · {finding.title}"
                        ),
                        "yellow" if finding.priority == "high" else "cyan",
                    )
                    for finding in ordered
                ),
            )

        return None

    def section_for_tool(self, event: ToolEvent) -> str | None:
        name = event.tool_call.name
        if event.result.get("ok") is True and name in _QUIET_READ_TOOLS:
            return None
        if name == "run_command":
            # Runner emits dispatched command events only after recording the
            # VerificationAttempt. Approval failures are explicit non-attempts.
            if event.result.get("error_code") in {
                "user_denied",
                "approval_unavailable",
            }:
                return "WORKING"
            if event.tool_call.id in self._displayed_command_calls:
                return None
            self._displayed_command_calls.add(event.tool_call.id)
            return "VERIFICATION"
        if (
            event.result.get("ok") is not True
            and name in _SOURCE_READ_TOOLS
        ):
            return "UNDERSTANDING"
        return "WORKING"

    def record_managed_change(self, event: ToolEvent) -> None:
        if (
            event.result.get("ok") is not True
            or event.tool_call.name not in {"write_file", "edit_file"}
        ):
            return
        data = _result_data(event.result)
        if data.get("workspace_changed") is not True:
            return
        path = data.get("path")
        if not isinstance(path, str) or not path:
            return
        marker = "+" if event.tool_call.name == "write_file" else "M"
        if path not in self._changed_files:
            self._changed_files[path] = marker
        elif self._changed_files[path] != "+":
            self._changed_files[path] = marker

    def record_recovery(self, event: RecoveryEvent) -> PresentationBlock:
        del event
        return PresentationBlock(
            "REPAIR",
            (
                PresentationLine(
                    "⚠",
                    "Recovery requested after no material progress",
                    "yellow",
                ),
            ),
        )

    def changed_block(self) -> PresentationBlock | None:
        if not self._changed_files:
            return None
        if len(self._changed_files) == 1 and not self._structured_task:
            return None
        count = len(self._changed_files)
        return PresentationBlock(
            f"CHANGED · {count} {'file' if count == 1 else 'files'}",
            tuple(
                PresentationLine(
                    marker,
                    f"{'created' if marker == '+' else 'modified'} · {path}",
                    "green" if marker == "+" else "magenta",
                )
                for path, marker in self._changed_files.items()
            ),
        )

    @property
    def has_verification_attempt(self) -> bool:
        return bool(self._displayed_command_calls)


_QUIET_READ_TOOLS = frozenset(
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
_SOURCE_READ_TOOLS = frozenset({"read_document", "read_webpage", "read_image"})


def _core_section(name: str) -> str:
    return {
        "update_requirements": "UNDERSTANDING",
        "update_plan": "PLAN",
        "update_review_findings": "REVIEW",
    }.get(name, "WORKING")


def _failure_line(name: str, result: Mapping[str, Any]) -> PresentationLine:
    error_code = result.get("error_code")
    message = result.get("message")
    suffix = f" · {error_code}" if isinstance(error_code, str) else ""
    return PresentationLine(
        "✗",
        f"{name}{suffix}",
        "red",
        message if isinstance(message, str) else None,
    )


def _plan_line(step: PlanStep) -> PresentationLine:
    marker, style = {
        "completed": ("✓", "green"),
        "in_progress": ("●", "cyan"),
        "pending": ("○", "dim"),
        "blocked": ("⚠", "yellow"),
    }[step.status]
    detail = step.blocked_reason if step.status == "blocked" else None
    return PresentationLine(marker, step.description, style, detail)


def _result_data(result: Mapping[str, Any]) -> Mapping[str, Any]:
    data = result.get("data")
    return data if isinstance(data, dict) else {}
