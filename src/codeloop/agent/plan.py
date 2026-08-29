"""Structured task plans and the reserved ``update_plan`` core action."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from .task_state import TaskState

PlanStatus = Literal["pending", "in_progress", "completed", "blocked"]
PlanMode = Literal["create", "update", "replan"]
PlanState = Literal["active", "completed", "terminal_with_blocks"]

UPDATE_PLAN_ACTION_NAME = "update_plan"
MAX_PLAN_STEPS = 12
MAX_STEP_ID_CHARS = 64
MAX_STEP_DESCRIPTION_CHARS = 500
MAX_BLOCKED_REASON_CHARS = 500
MAX_REPLAN_EXPLANATION_CHARS = 1_000

_VALID_STATUSES = {"pending", "in_progress", "completed", "blocked"}
_VALID_MODES = {"create", "update", "replan"}
_NORMAL_TRANSITIONS: dict[str, set[str]] = {
    "pending": {"pending", "in_progress", "completed", "blocked"},
    "in_progress": {"pending", "in_progress", "completed", "blocked"},
    "completed": {"completed"},
    "blocked": {"blocked"},
}


UPDATE_PLAN_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": UPDATE_PLAN_ACTION_NAME,
        "description": (
            "Create, update, or explicitly re-plan the current high-level task plan. "
            "Use explanation only for replan; do not include reasoning or chain-of-thought."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["create", "update", "replan"],
                },
                "steps": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": MAX_PLAN_STEPS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": MAX_STEP_ID_CHARS,
                            },
                            "description": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": MAX_STEP_DESCRIPTION_CHARS,
                            },
                            "status": {
                                "type": "string",
                                "enum": [
                                    "pending",
                                    "in_progress",
                                    "completed",
                                    "blocked",
                                ],
                            },
                            "blocked_reason": {
                                "type": "string",
                                "maxLength": MAX_BLOCKED_REASON_CHARS,
                                "description": (
                                    "Required for blocked status; omit for all other statuses."
                                ),
                            },
                        },
                        "required": ["id", "description", "status"],
                        "additionalProperties": False,
                    },
                },
                "explanation": {
                    "type": "string",
                    "maxLength": MAX_REPLAN_EXPLANATION_CHARS,
                    "description": "Required only for a real replan.",
                },
            },
            "required": ["mode", "steps"],
            "additionalProperties": False,
        },
    },
}


class PlanValidationError(Exception):
    """A safe, recoverable plan-action validation failure."""

    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message


@dataclass(frozen=True)
class PlanStep:
    id: str
    description: str
    status: PlanStatus
    blocked_reason: str | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_string("step id", self.id, MAX_STEP_ID_CHARS)
        _validate_non_empty_string(
            "step description",
            self.description,
            MAX_STEP_DESCRIPTION_CHARS,
        )
        if not isinstance(self.status, str) or self.status not in _VALID_STATUSES:
            raise PlanValidationError("invalid_plan", "Plan step status is invalid.")
        if self.status == "blocked":
            _validate_non_empty_string(
                "blocked_reason",
                self.blocked_reason,
                MAX_BLOCKED_REASON_CHARS,
            )
        elif self.blocked_reason is not None:
            raise PlanValidationError(
                "invalid_plan",
                "Only a blocked step may contain blocked_reason.",
            )

    def to_snapshot(self) -> dict[str, Any]:
        snapshot = {
            "id": self.id,
            "description": self.description,
            "status": self.status,
        }
        if self.blocked_reason is not None:
            snapshot["blocked_reason"] = self.blocked_reason
        return snapshot


@dataclass(frozen=True)
class TaskPlan:
    revision: int
    steps: tuple[PlanStep, ...]
    last_explanation: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 1
        ):
            raise PlanValidationError("invalid_plan", "Plan revision must be positive.")
        _validate_steps(self.steps)
        if self.last_explanation is not None:
            _validate_non_empty_string(
                "last_explanation",
                self.last_explanation,
                MAX_REPLAN_EXPLANATION_CHARS,
            )

    @classmethod
    def create(cls, steps: tuple[PlanStep, ...]) -> TaskPlan:
        return cls(revision=1, steps=steps)

    @property
    def status(self) -> PlanState:
        statuses = {step.status for step in self.steps}
        if statuses == {"completed"}:
            return "completed"
        if not statuses.intersection({"pending", "in_progress"}):
            return "terminal_with_blocks"
        return "active"

    def apply(
        self,
        mode: Literal["update", "replan"],
        steps: tuple[PlanStep, ...],
        explanation: str | None,
    ) -> tuple[TaskPlan, bool, tuple[str, ...]]:
        _validate_steps(steps)
        if mode == "update":
            if explanation not in (None, ""):
                raise PlanValidationError(
                    "invalid_arguments",
                    "explanation is only allowed for replan.",
                )
            self._validate_normal_update(steps)
        else:
            _validate_non_empty_string(
                "replan explanation",
                explanation,
                MAX_REPLAN_EXPLANATION_CHARS,
            )
            self._validate_replan(steps)

        if steps == self.steps:
            return self, False, ()

        changed_ids = _changed_step_ids(self.steps, steps)
        next_explanation = explanation if mode == "replan" else self.last_explanation
        return (
            TaskPlan(
                revision=self.revision + 1,
                steps=steps,
                last_explanation=next_explanation,
            ),
            True,
            changed_ids,
        )

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "status": self.status,
            "steps": [step.to_snapshot() for step in self.steps],
            "last_explanation": self.last_explanation,
        }

    def _validate_normal_update(self, steps: tuple[PlanStep, ...]) -> None:
        old_by_id = {step.id: step for step in self.steps}
        new_by_id = {step.id: step for step in steps}
        if not old_by_id.keys() <= new_by_id.keys():
            raise PlanValidationError(
                "invalid_plan_transition",
                "Normal update cannot remove existing plan steps.",
            )
        retained_order = tuple(step.id for step in steps if step.id in old_by_id)
        if retained_order != tuple(step.id for step in self.steps):
            raise PlanValidationError(
                "invalid_plan_transition",
                "Normal update cannot reorder existing plan steps.",
            )
        for step_id, old_step in old_by_id.items():
            new_step = new_by_id[step_id]
            if new_step.description != old_step.description:
                raise PlanValidationError(
                    "invalid_plan_transition",
                    "Normal update cannot change an existing step description.",
                )
            if new_step.status not in _NORMAL_TRANSITIONS[old_step.status]:
                raise PlanValidationError(
                    "invalid_plan_transition",
                    f"Normal update cannot reopen terminal step: {step_id}.",
                )

    def _validate_replan(self, steps: tuple[PlanStep, ...]) -> None:
        new_ids = {step.id for step in steps}
        missing_terminal_ids = [
            step.id
            for step in self.steps
            if step.status in {"completed", "blocked"} and step.id not in new_ids
        ]
        if missing_terminal_ids:
            raise PlanValidationError(
                "invalid_plan_transition",
                "Replan cannot remove existing terminal step IDs: "
                + ", ".join(missing_terminal_ids),
            )


def apply_plan_action(task_state: TaskState, arguments_json: str) -> dict[str, Any]:
    """Validate and atomically apply one model-native update_plan action."""
    try:
        arguments = _parse_action_arguments(arguments_json)
        mode = arguments["mode"]
        steps = _parse_steps(arguments["steps"])
        explanation = arguments.get("explanation")

        if mode == "create":
            if task_state.plan is not None:
                raise PlanValidationError(
                    "plan_already_exists",
                    "A task plan already exists; use update or replan.",
                )
            if explanation not in (None, ""):
                raise PlanValidationError(
                    "invalid_arguments",
                    "explanation is only allowed for replan.",
                )
            plan = TaskPlan.create(steps)
            changed = True
            changed_ids = tuple(step.id for step in steps)
        else:
            if task_state.plan is None:
                raise PlanValidationError(
                    "plan_not_found",
                    "No task plan exists; use create first.",
                )
            plan, changed, changed_ids = task_state.plan.apply(
                mode,
                steps,
                explanation,
            )

        task_state.replace_plan(plan)
        return {
            "ok": True,
            "data": {
                "changed": changed,
                "mode": mode,
                "revision": plan.revision,
                "plan_status": plan.status,
                "changed_step_ids": list(changed_ids),
            },
        }
    except PlanValidationError as exc:
        return {
            "ok": False,
            "error_code": exc.error_code,
            "message": exc.message,
        }


def _parse_action_arguments(arguments_json: str) -> dict[str, Any]:
    try:
        arguments = json.loads(arguments_json)
    except (json.JSONDecodeError, TypeError) as exc:
        raise PlanValidationError(
            "invalid_arguments",
            "Action arguments must be a JSON object.",
        ) from exc
    if not isinstance(arguments, dict):
        raise PlanValidationError(
            "invalid_arguments",
            "Action arguments must be a JSON object.",
        )
    unknown = set(arguments) - {"mode", "steps", "explanation"}
    if unknown or "mode" not in arguments or "steps" not in arguments:
        raise PlanValidationError(
            "invalid_arguments",
            "update_plan requires only mode, steps, and optional explanation.",
        )
    mode = arguments["mode"]
    if not isinstance(mode, str) or mode not in _VALID_MODES:
        raise PlanValidationError("invalid_arguments", "Plan mode is invalid.")
    explanation = arguments.get("explanation")
    if explanation is not None and not isinstance(explanation, str):
        raise PlanValidationError(
            "invalid_arguments",
            "explanation must be a string or null.",
        )
    if isinstance(explanation, str) and len(explanation) > MAX_REPLAN_EXPLANATION_CHARS:
        raise PlanValidationError(
            "invalid_arguments",
            "explanation is too long.",
        )
    return arguments


def _parse_steps(value: Any) -> tuple[PlanStep, ...]:
    if not isinstance(value, list):
        raise PlanValidationError("invalid_arguments", "steps must be an array.")
    parsed: list[PlanStep] = []
    for item in value:
        if not isinstance(item, dict):
            raise PlanValidationError(
                "invalid_arguments",
                "Each plan step must be an object.",
            )
        unknown = set(item) - {"id", "description", "status", "blocked_reason"}
        if unknown or not {"id", "description", "status"} <= item.keys():
            raise PlanValidationError(
                "invalid_arguments",
                "Each plan step requires id, description, and status only.",
            )
        parsed.append(
            PlanStep(
                id=item["id"],
                description=item["description"],
                status=item["status"],
                blocked_reason=item.get("blocked_reason"),
            )
        )
    return tuple(parsed)


def _validate_steps(steps: tuple[PlanStep, ...]) -> None:
    if not steps or len(steps) > MAX_PLAN_STEPS:
        raise PlanValidationError(
            "invalid_plan",
            f"A task plan must contain between 1 and {MAX_PLAN_STEPS} steps.",
        )
    ids = [step.id for step in steps]
    if len(set(ids)) != len(ids):
        raise PlanValidationError(
            "duplicate_step_id",
            "Plan step IDs must be unique.",
        )
    if sum(step.status == "in_progress" for step in steps) > 1:
        raise PlanValidationError(
            "invalid_plan",
            "A task plan may contain at most one in_progress step.",
        )


def _validate_non_empty_string(name: str, value: Any, maximum: int) -> None:
    if not isinstance(value, str) or not value.strip():
        raise PlanValidationError("invalid_plan", f"{name} must be a non-empty string.")
    if len(value) > maximum:
        raise PlanValidationError("invalid_plan", f"{name} is too long.")


def _changed_step_ids(
    old_steps: tuple[PlanStep, ...],
    new_steps: tuple[PlanStep, ...],
) -> tuple[str, ...]:
    old_by_id = {step.id: step for step in old_steps}
    new_by_id = {step.id: step for step in new_steps}
    changed = [
        step.id for step in new_steps if old_by_id.get(step.id) != step
    ]
    if tuple(old_by_id) != tuple(new_by_id):
        changed.extend(step.id for step in new_steps if step.id not in changed)
    changed.extend(step.id for step in old_steps if step.id not in new_by_id)
    return tuple(dict.fromkeys(changed))
