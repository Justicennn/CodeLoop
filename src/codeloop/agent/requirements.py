"""Bounded source requirements and the reserved core action."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, TYPE_CHECKING

from .repository import (
    RepositoryStateValidationError,
    normalize_workspace_relative_path,
)

if TYPE_CHECKING:
    from .task_state import TaskState

RequirementKind = Literal["functional", "constraint", "acceptance", "reference"]

UPDATE_REQUIREMENTS_ACTION_NAME = "update_requirements"
MAX_REQUIREMENTS = 32
MAX_REQUIREMENT_ID_CHARS = 64
MAX_REQUIREMENT_DESCRIPTION_CHARS = 400
MAX_REQUIREMENT_LOCATOR_CHARS = 160

_REQUIREMENT_KINDS = {"functional", "constraint", "acceptance", "reference"}

UPDATE_REQUIREMENTS_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": UPDATE_REQUIREMENTS_ACTION_NAME,
        "description": (
            "Replace the current task's bounded requirements extracted from "
            "successfully read local source documents."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "requirements": {
                    "type": "array",
                    "maxItems": MAX_REQUIREMENTS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": MAX_REQUIREMENT_ID_CHARS,
                            },
                            "kind": {
                                "type": "string",
                                "enum": sorted(_REQUIREMENT_KINDS),
                            },
                            "description": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": MAX_REQUIREMENT_DESCRIPTION_CHARS,
                            },
                            "source": {
                                "type": "object",
                                "properties": {
                                    "path": {
                                        "type": "string",
                                        "minLength": 1,
                                        "maxLength": 1_000,
                                    },
                                    "locator": {
                                        "type": "string",
                                        "minLength": 1,
                                        "maxLength": MAX_REQUIREMENT_LOCATOR_CHARS,
                                    },
                                },
                                "required": ["path"],
                                "additionalProperties": False,
                            },
                        },
                        "required": ["id", "kind", "description", "source"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["requirements"],
            "additionalProperties": False,
        },
    },
}


class RequirementValidationError(Exception):
    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message


@dataclass(frozen=True)
class RequirementSource:
    path: str
    locator: str | None = None

    def __post_init__(self) -> None:
        normalized = normalize_workspace_relative_path(self.path)
        if normalized != self.path:
            raise RequirementValidationError(
                "invalid_requirements",
                "Requirement source paths must be normalized.",
            )
        if self.locator is not None:
            _text("source locator", self.locator, MAX_REQUIREMENT_LOCATOR_CHARS)

    def to_snapshot(self) -> dict[str, str]:
        snapshot = {"path": self.path}
        if self.locator is not None:
            snapshot["locator"] = self.locator
        return snapshot


@dataclass(frozen=True)
class Requirement:
    id: str
    kind: RequirementKind
    description: str
    source: RequirementSource

    def __post_init__(self) -> None:
        _text("requirement id", self.id, MAX_REQUIREMENT_ID_CHARS)
        _text(
            "requirement description",
            self.description,
            MAX_REQUIREMENT_DESCRIPTION_CHARS,
        )
        if self.kind not in _REQUIREMENT_KINDS:
            raise RequirementValidationError(
                "invalid_requirements",
                "Requirement kind is invalid.",
            )

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "description": self.description,
            "source": self.source.to_snapshot(),
        }


@dataclass(frozen=True)
class RequirementState:
    revision: int = 0
    requirements: tuple[Requirement, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 0
        ):
            raise RequirementValidationError(
                "invalid_requirements",
                "Requirement revision must be a non-negative integer.",
            )
        if len(self.requirements) > MAX_REQUIREMENTS:
            raise RequirementValidationError(
                "invalid_requirements",
                f"Requirement state cannot exceed {MAX_REQUIREMENTS} items.",
            )
        ids = [requirement.id for requirement in self.requirements]
        if len(set(ids)) != len(ids):
            raise RequirementValidationError(
                "duplicate_requirement_id",
                "Requirement IDs must be unique.",
            )

    def replace(
        self,
        requirements: tuple[Requirement, ...],
    ) -> tuple[RequirementState, bool]:
        if requirements == self.requirements:
            return self, False
        return RequirementState(self.revision + 1, requirements), True

    def to_snapshot(self) -> list[dict[str, Any]] | None:
        if not self.requirements:
            return None
        return [requirement.to_snapshot() for requirement in self.requirements]


def apply_requirements_action(
    task_state: TaskState,
    arguments_json: str,
) -> dict[str, Any]:
    """Validate source eligibility and atomically replace requirements."""
    try:
        arguments = _parse_arguments(arguments_json)
        requirements = _parse_requirements(arguments["requirements"])
        eligible = set(task_state.read_source_paths)
        unobserved = sorted(
            {
                requirement.source.path
                for requirement in requirements
                if requirement.source.path not in eligible
            }
        )
        if unobserved:
            raise RequirementValidationError(
                "unobserved_requirement_source",
                "Requirement sources must first be read successfully: "
                + ", ".join(unobserved),
            )

        replacement, changed = task_state.requirements.replace(requirements)
        task_state.requirements = replacement
        return {
            "ok": True,
            "data": {
                "changed": changed,
                "revision": replacement.revision,
                "requirement_count": len(replacement.requirements),
            },
        }
    except (RequirementValidationError, RepositoryStateValidationError) as exc:
        return {
            "ok": False,
            "error_code": exc.error_code,
            "message": exc.message,
        }


def _parse_arguments(arguments_json: str) -> dict[str, Any]:
    try:
        arguments = json.loads(arguments_json)
    except (json.JSONDecodeError, TypeError) as exc:
        raise RequirementValidationError(
            "invalid_arguments",
            "Action arguments must be a JSON object.",
        ) from exc
    if not isinstance(arguments, dict) or set(arguments) != {"requirements"}:
        raise RequirementValidationError(
            "invalid_arguments",
            "update_requirements requires only requirements.",
        )
    if not isinstance(arguments["requirements"], list):
        raise RequirementValidationError(
            "invalid_arguments",
            "requirements must be an array.",
        )
    return arguments


def _parse_requirements(raw_requirements: list[Any]) -> tuple[Requirement, ...]:
    if len(raw_requirements) > MAX_REQUIREMENTS:
        raise RequirementValidationError(
            "invalid_requirements",
            f"Requirement state cannot exceed {MAX_REQUIREMENTS} items.",
        )
    parsed: list[Requirement] = []
    ids: set[str] = set()
    for raw in raw_requirements:
        if (
            not isinstance(raw, dict)
            or set(raw) != {"id", "kind", "description", "source"}
        ):
            raise RequirementValidationError(
                "invalid_arguments",
                "Each requirement requires only id, kind, description, and source.",
            )
        requirement_id = _text(
            "requirement id",
            raw["id"],
            MAX_REQUIREMENT_ID_CHARS,
        )
        if requirement_id in ids:
            raise RequirementValidationError(
                "duplicate_requirement_id",
                f"Requirement IDs must be unique: {requirement_id}.",
            )
        ids.add(requirement_id)
        source = _parse_source(raw["source"])
        kind = raw["kind"]
        if not isinstance(kind, str) or kind not in _REQUIREMENT_KINDS:
            raise RequirementValidationError(
                "invalid_requirements",
                "Requirement kind is invalid.",
            )
        parsed.append(
            Requirement(
                id=requirement_id,
                kind=kind,
                description=_text(
                    "requirement description",
                    raw["description"],
                    MAX_REQUIREMENT_DESCRIPTION_CHARS,
                ),
                source=source,
            )
        )
    return tuple(parsed)


def _parse_source(raw: Any) -> RequirementSource:
    if (
        not isinstance(raw, dict)
        or "path" not in raw
        or set(raw) - {"path", "locator"}
    ):
        raise RequirementValidationError(
            "invalid_arguments",
            "Requirement source requires path and optional locator.",
        )
    locator = raw.get("locator")
    return RequirementSource(
        path=normalize_workspace_relative_path(raw["path"]),
        locator=(
            _text("source locator", locator, MAX_REQUIREMENT_LOCATOR_CHARS)
            if locator is not None
            else None
        ),
    )


def _text(name: str, value: Any, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RequirementValidationError(
            "invalid_requirements",
            f"{name} must be a non-empty string.",
        )
    if len(value) > maximum:
        raise RequirementValidationError(
            "invalid_requirements",
            f"{name} is too long.",
        )
    return value
