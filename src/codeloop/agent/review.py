"""Bounded evidence-backed project review state and Core Action."""

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

FindingType = Literal["issue", "enhancement"]
FindingPriority = Literal["high", "medium", "low"]
FindingCategory = Literal[
    "correctness",
    "reliability",
    "maintainability",
    "performance",
    "security",
    "usability",
    "testing",
    "architecture",
]

UPDATE_REVIEW_FINDINGS_ACTION_NAME = "update_review_findings"
MAX_REVIEW_FINDINGS = 16
MAX_FINDING_ID_CHARS = 64
MAX_FINDING_TITLE_CHARS = 160
MAX_FINDING_EVIDENCE = 4
MAX_EVIDENCE_SYMBOL_CHARS = 160
MAX_EVIDENCE_DESCRIPTION_CHARS = 500
MAX_FINDING_IMPACT_CHARS = 600
MAX_FINDING_RECOMMENDATION_CHARS = 800

_FINDING_TYPES = {"issue", "enhancement"}
_PRIORITIES = {"high", "medium", "low"}
_CATEGORIES = {
    "correctness",
    "reliability",
    "maintainability",
    "performance",
    "security",
    "usability",
    "testing",
    "architecture",
}

UPDATE_REVIEW_FINDINGS_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": UPDATE_REVIEW_FINDINGS_ACTION_NAME,
        "description": (
            "Replace the current task's bounded evidence-backed review findings. "
            "Every evidence path must already have successful read/search evidence."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "findings": {
                    "type": "array",
                    "maxItems": MAX_REVIEW_FINDINGS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "minLength": 1, "maxLength": MAX_FINDING_ID_CHARS},
                            "finding_type": {"type": "string", "enum": ["issue", "enhancement"]},
                            "title": {"type": "string", "minLength": 1, "maxLength": MAX_FINDING_TITLE_CHARS},
                            "category": {"type": "string", "enum": sorted(_CATEGORIES)},
                            "evidence": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": MAX_FINDING_EVIDENCE,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "path": {"type": "string", "minLength": 1, "maxLength": 1_000},
                                        "symbol": {"type": "string", "minLength": 1, "maxLength": MAX_EVIDENCE_SYMBOL_CHARS},
                                        "description": {"type": "string", "minLength": 1, "maxLength": MAX_EVIDENCE_DESCRIPTION_CHARS},
                                    },
                                    "required": ["path", "description"],
                                    "additionalProperties": False,
                                },
                            },
                            "impact": {"type": "string", "minLength": 1, "maxLength": MAX_FINDING_IMPACT_CHARS},
                            "recommendation": {"type": "string", "minLength": 1, "maxLength": MAX_FINDING_RECOMMENDATION_CHARS},
                            "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                        },
                        "required": ["id", "finding_type", "title", "category", "evidence", "impact", "recommendation", "priority"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["findings"],
            "additionalProperties": False,
        },
    },
}


class ReviewValidationError(Exception):
    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message


@dataclass(frozen=True)
class FindingEvidence:
    path: str
    description: str
    symbol: str | None = None

    def __post_init__(self) -> None:
        normalized = normalize_workspace_relative_path(self.path)
        if normalized != self.path:
            raise ReviewValidationError(
                "invalid_review_findings",
                "Evidence paths must be normalized.",
            )
        _text(
            "evidence description",
            self.description,
            MAX_EVIDENCE_DESCRIPTION_CHARS,
        )
        if self.symbol is not None:
            _text("symbol", self.symbol, MAX_EVIDENCE_SYMBOL_CHARS)


@dataclass(frozen=True)
class ReviewFinding:
    id: str
    finding_type: FindingType
    title: str
    category: FindingCategory
    evidence: tuple[FindingEvidence, ...]
    impact: str
    recommendation: str
    priority: FindingPriority

    def __post_init__(self) -> None:
        _text("id", self.id, MAX_FINDING_ID_CHARS)
        _text("title", self.title, MAX_FINDING_TITLE_CHARS)
        _enum("finding_type", self.finding_type, _FINDING_TYPES)
        _enum("category", self.category, _CATEGORIES)
        _enum("priority", self.priority, _PRIORITIES)
        if not self.evidence or len(self.evidence) > MAX_FINDING_EVIDENCE:
            raise ReviewValidationError(
                "invalid_review_findings",
                f"Each finding requires 1 to {MAX_FINDING_EVIDENCE} evidence items.",
            )
        _text("impact", self.impact, MAX_FINDING_IMPACT_CHARS)
        _text(
            "recommendation",
            self.recommendation,
            MAX_FINDING_RECOMMENDATION_CHARS,
        )

    def to_snapshot(self) -> dict[str, str]:
        return {
            "id": self.id,
            "finding_type": self.finding_type,
            "priority": self.priority,
            "title": self.title,
        }


@dataclass(frozen=True)
class ReviewState:
    revision: int = 0
    findings: tuple[ReviewFinding, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 0
        ):
            raise ReviewValidationError(
                "invalid_review_findings",
                "Review revision must be a non-negative integer.",
            )
        if len(self.findings) > MAX_REVIEW_FINDINGS:
            raise ReviewValidationError(
                "invalid_review_findings",
                f"Review state cannot exceed {MAX_REVIEW_FINDINGS} findings.",
            )
        ids = [finding.id for finding in self.findings]
        if len(set(ids)) != len(ids):
            raise ReviewValidationError(
                "duplicate_finding_id",
                "Finding IDs must be unique.",
            )

    def replace(self, findings: tuple[ReviewFinding, ...]) -> tuple[ReviewState, bool]:
        if findings == self.findings:
            return self, False
        return ReviewState(self.revision + 1, findings), True

    def invalidate_path(self, path: str) -> tuple[ReviewState, bool]:
        retained = tuple(
            finding
            for finding in self.findings
            if all(evidence.path != path for evidence in finding.evidence)
        )
        return self.replace(retained)

    def to_snapshot(self) -> list[dict[str, str]] | None:
        if not self.findings:
            return None
        return [finding.to_snapshot() for finding in self.findings]


def apply_review_findings_action(
    task_state: TaskState,
    arguments_json: str,
) -> dict[str, Any]:
    """Validate evidence eligibility and atomically replace review findings."""
    try:
        arguments = _parse_arguments(arguments_json)
        findings = _parse_findings(arguments["findings"])
        eligible = set(task_state.inspected_evidence_paths)
        unobserved = sorted(
            {
                evidence.path
                for finding in findings
                for evidence in finding.evidence
                if evidence.path not in eligible
            }
        )
        if unobserved:
            raise ReviewValidationError(
                "unobserved_review_evidence",
                "Review evidence must first be obtained through read_file or "
                "search_code: " + ", ".join(unobserved),
            )

        replacement, changed = task_state.review_state.replace(findings)
        task_state.review_state = replacement
        return {
            "ok": True,
            "data": {
                "changed": changed,
                "revision": replacement.revision,
                "finding_count": len(replacement.findings),
            },
        }
    except (ReviewValidationError, RepositoryStateValidationError) as exc:
        return {
            "ok": False,
            "error_code": exc.error_code,
            "message": exc.message,
        }


def _parse_arguments(arguments_json: str) -> dict[str, Any]:
    try:
        arguments = json.loads(arguments_json)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ReviewValidationError(
            "invalid_arguments",
            "Action arguments must be a JSON object.",
        ) from exc
    if not isinstance(arguments, dict) or set(arguments) != {"findings"}:
        raise ReviewValidationError(
            "invalid_arguments",
            "update_review_findings requires only findings.",
        )
    if not isinstance(arguments["findings"], list):
        raise ReviewValidationError("invalid_arguments", "findings must be an array.")
    return arguments


def _parse_findings(raw_findings: list[Any]) -> tuple[ReviewFinding, ...]:
    if len(raw_findings) > MAX_REVIEW_FINDINGS:
        raise ReviewValidationError(
            "invalid_review_findings",
            f"Review state cannot exceed {MAX_REVIEW_FINDINGS} findings.",
        )
    parsed: list[ReviewFinding] = []
    ids: set[str] = set()
    required = {
        "id", "finding_type", "title", "category", "evidence",
        "impact", "recommendation", "priority",
    }
    for raw in raw_findings:
        if not isinstance(raw, dict) or set(raw) != required:
            raise ReviewValidationError(
                "invalid_arguments",
                "Each finding must contain exactly the documented finding fields.",
            )
        finding_id = _text("id", raw["id"], MAX_FINDING_ID_CHARS)
        if finding_id in ids:
            raise ReviewValidationError(
                "duplicate_finding_id",
                f"Finding IDs must be unique: {finding_id}.",
            )
        ids.add(finding_id)
        finding_type = _enum("finding_type", raw["finding_type"], _FINDING_TYPES)
        category = _enum("category", raw["category"], _CATEGORIES)
        priority = _enum("priority", raw["priority"], _PRIORITIES)
        parsed.append(
            ReviewFinding(
                id=finding_id,
                finding_type=finding_type,
                title=_text("title", raw["title"], MAX_FINDING_TITLE_CHARS),
                category=category,
                evidence=_parse_evidence(raw["evidence"]),
                impact=_text("impact", raw["impact"], MAX_FINDING_IMPACT_CHARS),
                recommendation=_text(
                    "recommendation",
                    raw["recommendation"],
                    MAX_FINDING_RECOMMENDATION_CHARS,
                ),
                priority=priority,
            )
        )
    return tuple(parsed)


def _parse_evidence(raw_evidence: Any) -> tuple[FindingEvidence, ...]:
    if (
        not isinstance(raw_evidence, list)
        or not raw_evidence
        or len(raw_evidence) > MAX_FINDING_EVIDENCE
    ):
        raise ReviewValidationError(
            "invalid_review_findings",
            f"Each finding requires 1 to {MAX_FINDING_EVIDENCE} evidence items.",
        )
    parsed: list[FindingEvidence] = []
    for raw in raw_evidence:
        if (
            not isinstance(raw, dict)
            or not {"path", "description"} <= set(raw)
            or set(raw) - {"path", "description", "symbol"}
        ):
            raise ReviewValidationError(
                "invalid_arguments",
                "Evidence requires path, description, and optional symbol.",
            )
        symbol = raw.get("symbol")
        parsed.append(
            FindingEvidence(
                path=normalize_workspace_relative_path(raw["path"]),
                description=_text(
                    "evidence description",
                    raw["description"],
                    MAX_EVIDENCE_DESCRIPTION_CHARS,
                ),
                symbol=(
                    _text("symbol", symbol, MAX_EVIDENCE_SYMBOL_CHARS)
                    if symbol is not None
                    else None
                ),
            )
        )
    return tuple(parsed)


def _text(name: str, value: Any, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReviewValidationError(
            "invalid_review_findings",
            f"{name} must be a non-empty string.",
        )
    if len(value) > maximum:
        raise ReviewValidationError(
            "invalid_review_findings",
            f"{name} is too long.",
        )
    return value


def _enum(name: str, value: Any, allowed: set[str]) -> Any:
    if not isinstance(value, str) or value not in allowed:
        raise ReviewValidationError(
            "invalid_review_findings",
            f"{name} is invalid.",
        )
    return value
