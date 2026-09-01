"""Task-local aggregation of narrow Agent Core presentation events."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from ..agent.events import (
    CoreActionEvent,
    RecoveryEvent,
    ReviewFindingProjection,
    ToolEvent,
)
from ..agent.plan import PlanStep


_MAX_VISIBLE_PLAN_STEPS = 6
_MAX_STORED_ACTIONS = 12
_MAX_VISIBLE_ACTIONS = 8
_MAX_ACTION_DETAILS = 4
_MAX_VISIBLE_FINDINGS = 4
_MAX_STATUS_CHARS = 240
_MAX_TARGET_CHARS = 160


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


ActionStatus = Literal["success", "failure", "warning"]


@dataclass(frozen=True)
class PresentationAction:
    """One bounded, factual completed Action for the transient view."""

    key: str
    label: str
    status: ActionStatus
    target: str | None = None
    details: tuple[str, ...] = ()
    count: int = 1


@dataclass(frozen=True)
class PresentationSnapshot:
    """Immutable render input built from already-public Runtime facts."""

    phase: str | None
    plan_steps: tuple[PlanStep, ...]
    hidden_plan_steps: int
    actions: tuple[PresentationAction, ...]
    hidden_actions: int
    findings: tuple[ReviewFindingProjection, ...]
    hidden_findings: int
    current: str | None
    next_step: str | None


@dataclass(frozen=True)
class PresentationUpdate:
    """One event consumption plus both live and linear projections."""

    snapshot: PresentationSnapshot
    linear_blocks: tuple[PresentationBlock, ...] = ()
    linear_tool_section: str | None = None
    linear_narration: str | None = None


class PresentationState:
    """Aggregate only already-explicit Runtime facts for one task."""

    def __init__(self) -> None:
        self._requirements_signature: tuple[object, ...] | None = None
        self._plan_by_id: dict[str, PlanStep] | None = None
        self._review_signature: tuple[object, ...] | None = None
        self._review_findings: tuple[ReviewFindingProjection, ...] = ()
        self._changed_files: dict[str, str] = {}
        self._displayed_command_calls: set[str] = set()
        self._structured_task = False
        self._phase: str | None = None
        self._narration: str | None = None
        self._actions: dict[str, PresentationAction] = {}
        self._hidden_action_count = 0

    def snapshot(self) -> PresentationSnapshot:
        plan_steps = tuple(self._plan_by_id.values()) if self._plan_by_id else ()
        visible_plan = _visible_plan_steps(plan_steps)
        actions = tuple(self._actions.values())
        visible_actions = actions[-_MAX_VISIBLE_ACTIONS:]
        ordered_findings = tuple(
            sorted(
                self._review_findings,
                key=lambda finding: {"high": 0, "medium": 1, "low": 2}[
                    finding.priority
                ],
            )
        )
        visible_findings = ordered_findings[:_MAX_VISIBLE_FINDINGS]
        active = next(
            (step.description for step in plan_steps if step.status == "in_progress"),
            None,
        )
        pending = next(
            (step.description for step in plan_steps if step.status == "pending"),
            None,
        )
        return PresentationSnapshot(
            phase=self._phase,
            plan_steps=visible_plan,
            hidden_plan_steps=max(0, len(plan_steps) - len(visible_plan)),
            actions=visible_actions,
            hidden_actions=(
                self._hidden_action_count
                + max(0, len(actions) - len(visible_actions))
            ),
            findings=visible_findings,
            hidden_findings=max(0, len(ordered_findings) - len(visible_findings)),
            current=self._narration or active,
            next_step=pending,
        )

    def observe_narration(self, narration: str) -> PresentationUpdate:
        self._narration = _bounded_text(narration, _MAX_STATUS_CHARS)
        return PresentationUpdate(
            snapshot=self.snapshot(),
            linear_narration=self._narration,
        )

    def observe_core_action(self, event: CoreActionEvent) -> PresentationUpdate:
        self._narration = None
        self._phase = _core_phase(event.name)
        block = self._record_core_action(event)
        if event.name != "update_plan" or event.result.get("ok") is not True:
            self._record_action(_core_action_projection(event))
        return PresentationUpdate(
            snapshot=self.snapshot(),
            linear_blocks=(block,) if block is not None else (),
        )

    def observe_tool(self, event: ToolEvent) -> PresentationUpdate:
        self._narration = None
        self._phase = _tool_phase(event)
        self.record_managed_change(event)
        self._record_action(_tool_action_projection(event))
        return PresentationUpdate(
            snapshot=self.snapshot(),
            linear_tool_section=self._linear_section_for_tool(event),
        )

    def observe_recovery(self, event: RecoveryEvent) -> PresentationUpdate:
        del event
        self._narration = None
        self._phase = "Repairing"
        block = PresentationBlock(
            "REPAIR",
            (
                PresentationLine(
                    "⚠",
                    "Recovery requested after no material progress",
                    "gold3",
                ),
            ),
        )
        self._record_action(
            PresentationAction(
                key="recovery:no_progress",
                label="Request bounded recovery",
                status="warning",
                details=("no material progress",),
            )
        )
        return PresentationUpdate(snapshot=self.snapshot(), linear_blocks=(block,))

    def _record_core_action(
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
            signature = (event.requirement_count, event.requirement_sources)
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
                PresentationLine("✓", f"{count} {noun} registered", "green")
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
            self._review_findings = tuple(event.review_findings)
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
                        "gold3" if finding.priority == "high" else "orange3",
                    )
                    for finding in ordered
                ),
            )

        return None

    def _linear_section_for_tool(self, event: ToolEvent) -> str | None:
        name = event.tool_call.name
        if event.result.get("ok") is True and name in _QUIET_READ_TOOLS:
            return None
        if name == "run_command":
            if event.result.get("error_code") in {
                "user_denied",
                "approval_unavailable",
            }:
                return "WORKING"
            if event.tool_call.id in self._displayed_command_calls:
                return None
            self._displayed_command_calls.add(event.tool_call.id)
            return "VERIFICATION"
        if event.result.get("ok") is not True and name in _SOURCE_READ_TOOLS:
            return "UNDERSTANDING"
        return "WORKING"

    def _record_action(self, action: PresentationAction) -> None:
        previous = self._actions.get(action.key)
        if previous is not None:
            self._actions[action.key] = PresentationAction(
                key=action.key,
                label=action.label,
                status=action.status,
                target=action.target,
                details=action.details,
                count=previous.count + 1,
            )
            return
        if len(self._actions) >= _MAX_STORED_ACTIONS:
            oldest = next(iter(self._actions))
            del self._actions[oldest]
            self._hidden_action_count += 1
        self._actions[action.key] = action

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
                    "green" if marker == "+" else "orange3",
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


def _core_phase(name: str) -> str:
    return {
        "update_requirements": "Understanding requirements",
        "update_plan": "Planning work",
        "update_working_set": "Inspecting workspace",
        "update_review_findings": "Reviewing project",
    }.get(name, "Working")


def _tool_phase(event: ToolEvent) -> str:
    name = event.tool_call.name
    if name in _SOURCE_READ_TOOLS:
        return "Understanding sources"
    if name in _QUIET_READ_TOOLS:
        return "Inspecting workspace"
    if name == "run_command":
        if event.result.get("error_code") in {
            "user_denied",
            "approval_unavailable",
        }:
            return "Working"
        return "Verifying"
    return "Working"


def _core_action_projection(event: CoreActionEvent) -> PresentationAction:
    ok = event.result.get("ok") is True
    label = {
        "update_requirements": "Register requirements",
        "update_plan": "Update plan",
        "update_working_set": "Update working set",
        "update_review_findings": "Register review findings",
    }.get(event.name, event.name)
    details: list[str] = []
    if event.name == "update_requirements":
        if event.requirement_count is not None:
            details.append(f"{event.requirement_count} requirements")
        if event.requirement_sources:
            details.extend(event.requirement_sources[:_MAX_ACTION_DETAILS])
    elif event.name == "update_review_findings" and event.review_findings is not None:
        details.append(f"{len(event.review_findings)} findings")
    if not ok:
        details.extend(_error_details(event.result))
    return PresentationAction(
        key=f"core:{event.name}",
        label=label,
        status="success" if ok else "failure",
        details=tuple(
            _bounded_text(detail, _MAX_TARGET_CHARS)
            for detail in details[:_MAX_ACTION_DETAILS]
        ),
    )


def _tool_action_projection(event: ToolEvent) -> PresentationAction:
    name = event.tool_call.name
    data = _result_data(event.result)
    arguments = _object_arguments(event.tool_call.arguments)
    label = {
        "repository_overview": "Inspect repository structure",
        "list_files": "List files",
        "read_file": "Read file",
        "read_document": "Read document",
        "read_webpage": "Read webpage",
        "read_image": "Read image",
        "search_code": "Search code",
        "make_directory": "Prepare directory",
        "write_file": "Create file",
        "edit_file": "Update file",
        "run_command": "Run command",
    }.get(name, name)
    target = _primary_target(name, data, arguments)
    details = _tool_details(name, data, event.result)
    key = f"tool:{name}:{target or ''}"
    return PresentationAction(
        key=key,
        label=label,
        status="success" if event.result.get("ok") is True else "failure",
        target=target,
        details=details,
    )


def _primary_target(
    name: str,
    data: Mapping[str, Any],
    arguments: Mapping[str, Any],
) -> str | None:
    if name == "run_command":
        command = data.get("command")
        if (
            isinstance(command, list)
            and command
            and all(isinstance(part, str) and part for part in command)
        ):
            return _bounded_text(subprocess.list2cmdline(command), _MAX_TARGET_CHARS)
    for field in ("path", "requested_url", "url", "query"):
        value = data.get(field) or arguments.get(field)
        if isinstance(value, str) and value:
            return _bounded_text(value, _MAX_TARGET_CHARS)
    return None


def _tool_details(
    name: str,
    data: Mapping[str, Any],
    result: Mapping[str, Any],
) -> tuple[str, ...]:
    details: list[str] = []
    if name == "list_files":
        details.extend(_paths_from_items(data.get("entries")))
    elif name == "repository_overview":
        anchors = data.get("anchors")
        if isinstance(anchors, dict):
            details.extend(_string_items(anchors.get("items")))
    elif name == "search_code":
        details.extend(_paths_from_items(data.get("matches")))
    elif name == "run_command":
        exit_code = data.get("exit_code")
        if isinstance(exit_code, int) and not isinstance(exit_code, bool):
            details.append(f"exit {exit_code}")
    if result.get("ok") is not True:
        details.extend(_error_details(result))
    if data.get("truncated") is True or result.get("truncated") is True:
        details.append("output truncated")
    return tuple(
        _bounded_text(detail, _MAX_TARGET_CHARS)
        for detail in details[:_MAX_ACTION_DETAILS]
    )


def _error_details(result: Mapping[str, Any]) -> list[str]:
    details: list[str] = []
    for field in ("error_code", "message"):
        value = result.get(field)
        if isinstance(value, str) and value:
            details.append(value)
    return details


def _paths_from_items(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    paths: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        if isinstance(path, str) and path and path not in paths:
            paths.append(path)
        if len(paths) >= _MAX_ACTION_DETAILS:
            break
    return paths


def _string_items(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item][
        :_MAX_ACTION_DETAILS
    ]


def _visible_plan_steps(steps: tuple[PlanStep, ...]) -> tuple[PlanStep, ...]:
    if len(steps) <= _MAX_VISIBLE_PLAN_STEPS:
        return steps
    focus = next(
        (index for index, step in enumerate(steps) if step.status == "in_progress"),
        next(
            (index for index, step in enumerate(steps) if step.status == "pending"),
            len(steps) - 1,
        ),
    )
    start = max(0, min(focus - 2, len(steps) - _MAX_VISIBLE_PLAN_STEPS))
    return steps[start : start + _MAX_VISIBLE_PLAN_STEPS]


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
        "in_progress": ("●", "orange3"),
        "pending": ("○", "dim"),
        "blocked": ("⚠", "gold3"),
    }[step.status]
    detail = step.blocked_reason if step.status == "blocked" else None
    return PresentationLine(marker, step.description, style, detail)


def _object_arguments(raw_arguments: str) -> dict[str, Any]:
    try:
        value = json.loads(raw_arguments)
    except (json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _result_data(result: Mapping[str, Any]) -> Mapping[str, Any]:
    data = result.get("data")
    return data if isinstance(data, dict) else {}


def _bounded_text(value: str, limit: int) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."
