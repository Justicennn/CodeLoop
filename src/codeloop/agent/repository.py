"""Bounded repository focus state and its reserved Core Action."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .task_state import TaskState

UPDATE_WORKING_SET_ACTION_NAME = "update_working_set"
MAX_WORKING_SET_ENTRIES = 16
MAX_REPOSITORY_PATH_CHARS = 1_000
MAX_WORKING_SET_REASON_CHARS = 160

UPDATE_WORKING_SET_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": UPDATE_WORKING_SET_ACTION_NAME,
        "description": (
            "Replace the current task's bounded repository focus. This is a focus "
            "aid, not a file-access allowlist."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entries": {
                    "type": "array",
                    "maxItems": MAX_WORKING_SET_ENTRIES,
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": MAX_REPOSITORY_PATH_CHARS,
                            },
                            "reason": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": MAX_WORKING_SET_REASON_CHARS,
                            },
                        },
                        "required": ["path", "reason"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["entries"],
            "additionalProperties": False,
        },
    },
}


class RepositoryStateValidationError(Exception):
    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message


@dataclass(frozen=True)
class WorkingSetEntry:
    path: str
    reason: str

    def __post_init__(self) -> None:
        normalized = normalize_workspace_relative_path(self.path)
        if normalized != self.path:
            raise RepositoryStateValidationError(
                "invalid_working_set",
                "Working-set paths must be normalized.",
            )
        _bounded_text("reason", self.reason, MAX_WORKING_SET_REASON_CHARS)

    def to_snapshot(self) -> dict[str, str]:
        return {"path": self.path, "reason": self.reason}


@dataclass(frozen=True)
class RepositoryWorkingSet:
    revision: int = 0
    entries: tuple[WorkingSetEntry, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 0
        ):
            raise RepositoryStateValidationError(
                "invalid_working_set",
                "Working-set revision must be a non-negative integer.",
            )
        if len(self.entries) > MAX_WORKING_SET_ENTRIES:
            raise RepositoryStateValidationError(
                "invalid_working_set",
                f"Working set cannot exceed {MAX_WORKING_SET_ENTRIES} entries.",
            )
        paths = [entry.path for entry in self.entries]
        if len(set(paths)) != len(paths):
            raise RepositoryStateValidationError(
                "duplicate_working_set_path",
                "Working-set paths must be unique.",
            )

    def replace(
        self,
        entries: tuple[WorkingSetEntry, ...],
    ) -> tuple[RepositoryWorkingSet, bool]:
        if entries == self.entries:
            return self, False
        return RepositoryWorkingSet(self.revision + 1, entries), True

    def to_snapshot(self) -> list[dict[str, str]] | None:
        if not self.entries:
            return None
        return [entry.to_snapshot() for entry in self.entries]


def apply_working_set_action(
    task_state: TaskState,
    arguments_json: str,
) -> dict[str, Any]:
    """Validate and atomically replace the current repository focus."""
    try:
        arguments = _parse_object(arguments_json, {"entries"})
        raw_entries = arguments.get("entries")
        if not isinstance(raw_entries, list):
            raise RepositoryStateValidationError(
                "invalid_arguments",
                "entries must be an array.",
            )
        if len(raw_entries) > MAX_WORKING_SET_ENTRIES:
            raise RepositoryStateValidationError(
                "invalid_working_set",
                f"Working set cannot exceed {MAX_WORKING_SET_ENTRIES} entries.",
            )

        parsed: list[WorkingSetEntry] = []
        paths: set[str] = set()
        for raw in raw_entries:
            if not isinstance(raw, dict) or set(raw) != {"path", "reason"}:
                raise RepositoryStateValidationError(
                    "invalid_arguments",
                    "Each working-set entry requires only path and reason.",
                )
            path = normalize_workspace_relative_path(raw["path"])
            reason = _bounded_text(
                "reason",
                raw["reason"],
                MAX_WORKING_SET_REASON_CHARS,
            )
            if path in paths:
                raise RepositoryStateValidationError(
                    "duplicate_working_set_path",
                    f"Working-set paths must be unique: {path}.",
                )
            paths.add(path)
            parsed.append(WorkingSetEntry(path=path, reason=reason))

        replacement, changed = task_state.working_set.replace(tuple(parsed))
        task_state.working_set = replacement
        return {
            "ok": True,
            "data": {
                "changed": changed,
                "revision": replacement.revision,
                "entry_count": len(replacement.entries),
            },
        }
    except RepositoryStateValidationError as exc:
        return {
            "ok": False,
            "error_code": exc.error_code,
            "message": exc.message,
        }


def normalize_workspace_relative_path(value: Any) -> str:
    """Normalize a lexical Workspace-relative path without filesystem access."""
    if not isinstance(value, str) or not value.strip():
        raise RepositoryStateValidationError(
            "invalid_path",
            "Path must be a non-empty string.",
        )
    if len(value) > MAX_REPOSITORY_PATH_CHARS:
        raise RepositoryStateValidationError("invalid_path", "Path is too long.")
    if "\x00" in value:
        raise RepositoryStateValidationError(
            "invalid_path",
            "Path contains a NUL byte.",
        )

    normalized = value.replace("\\", "/")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        raise RepositoryStateValidationError(
            "invalid_path",
            "Absolute paths are not allowed.",
        )
    path = PurePosixPath(normalized)
    if any(part == ".." for part in path.parts):
        raise RepositoryStateValidationError(
            "invalid_path",
            "Parent traversal is not allowed.",
        )
    collapsed = path.as_posix()
    return "." if collapsed in {"", "."} else collapsed


def _parse_object(arguments_json: str, required: set[str]) -> dict[str, Any]:
    try:
        arguments = json.loads(arguments_json)
    except (json.JSONDecodeError, TypeError) as exc:
        raise RepositoryStateValidationError(
            "invalid_arguments",
            "Action arguments must be a JSON object.",
        ) from exc
    if not isinstance(arguments, dict) or set(arguments) != required:
        raise RepositoryStateValidationError(
            "invalid_arguments",
            f"Action requires only: {', '.join(sorted(required))}.",
        )
    return arguments


def _bounded_text(name: str, value: Any, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RepositoryStateValidationError(
            "invalid_arguments",
            f"{name} must be a non-empty string.",
        )
    if len(value) > maximum:
        raise RepositoryStateValidationError(
            "invalid_arguments",
            f"{name} is too long.",
        )
    return value
