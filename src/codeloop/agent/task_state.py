"""Explicit state owned by one AgentRunner run."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Literal

from .plan import PlanStep, TaskPlan
from .progress import ProgressState
from .repository import (
    RepositoryStateValidationError,
    RepositoryWorkingSet,
    normalize_workspace_relative_path,
)
from .review import ReviewState
from .requirements import RequirementState
from .verification import VerificationState, VerificationStatus

PlanOutcome = Literal[
    "not_planned",
    "active",
    "completed",
    "terminal_with_blocks",
]
CompletionReviewFingerprint = tuple[int, int, VerificationStatus, tuple[str, ...]]
_COMPLETION_REVIEW_INSTRUCTION = (
    "Continue the task: finish or block active plan steps, run a relevant "
    "verification command, or explain why completion remains limited."
)
MAX_INSPECTED_EVIDENCE_PATHS = 512
MAX_READ_SOURCE_PATHS = 128
MAX_READ_SOURCE_URLS = 128
MAX_READ_SOURCE_URL_CHARS = 2_000
_TEXT_SOURCE_EXTENSIONS = {".txt", ".md", ".json", ".yaml", ".yml"}


@dataclass
class TaskState:
    """Aggregate only the persistent state implemented by the current stage."""

    plan: TaskPlan | None = None
    workspace_revision: int = 0
    verification: VerificationState = field(default_factory=VerificationState)
    progress: ProgressState = field(default_factory=ProgressState)
    working_set: RepositoryWorkingSet = field(default_factory=RepositoryWorkingSet)
    review_state: ReviewState = field(default_factory=ReviewState)
    requirements: RequirementState = field(default_factory=RequirementState)
    inspected_evidence_paths: tuple[str, ...] = ()
    read_source_paths: tuple[str, ...] = ()
    read_source_urls: tuple[str, ...] = ()
    last_completion_review_fingerprint: CompletionReviewFingerprint | None = None
    pending_completion_review: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if len(self.inspected_evidence_paths) > MAX_INSPECTED_EVIDENCE_PATHS:
            raise RepositoryStateValidationError(
                "invalid_review_evidence",
                "Inspected evidence paths exceed the task-local limit.",
            )
        normalized = tuple(
            normalize_workspace_relative_path(path)
            for path in self.inspected_evidence_paths
        )
        if len(set(normalized)) != len(normalized):
            raise RepositoryStateValidationError(
                "invalid_review_evidence",
                "Inspected evidence paths must be unique.",
            )
        self.inspected_evidence_paths = normalized
        if len(self.read_source_paths) > MAX_READ_SOURCE_PATHS:
            raise RepositoryStateValidationError(
                "invalid_requirement_sources",
                "Read source paths exceed the task-local limit.",
            )
        normalized_sources = tuple(
            normalize_workspace_relative_path(path) for path in self.read_source_paths
        )
        if len(set(normalized_sources)) != len(normalized_sources):
            raise RepositoryStateValidationError(
                "invalid_requirement_sources",
                "Read source paths must be unique.",
            )
        self.read_source_paths = normalized_sources
        if len(self.read_source_urls) > MAX_READ_SOURCE_URLS:
            raise RepositoryStateValidationError(
                "invalid_requirement_sources",
                "Read source URLs exceed the task-local limit.",
            )
        if any(
            not isinstance(url, str)
            or not url
            or len(url) > MAX_READ_SOURCE_URL_CHARS
            for url in self.read_source_urls
        ):
            raise RepositoryStateValidationError(
                "invalid_requirement_sources",
                "Read source URLs must be non-empty bounded strings.",
            )
        if len(set(self.read_source_urls)) != len(self.read_source_urls):
            raise RepositoryStateValidationError(
                "invalid_requirement_sources",
                "Read source URLs must be unique.",
            )

    def replace_plan(self, plan: TaskPlan) -> None:
        self.plan = plan

    def record_workspace_change(self, path: str | None = None) -> None:
        self.workspace_revision += 1
        self.verification.require_verification()
        if path is not None:
            self.invalidate_review_evidence(path)

    def record_execution_evidence(
        self,
        *,
        tool_name: str,
        result: dict[str, Any],
    ) -> None:
        """Remember bounded evidence and source locators from successful reads."""
        if result.get("ok") is not True:
            return
        data = result.get("data")
        if not isinstance(data, dict):
            return
        paths: list[str] = []
        if tool_name == "read_file":
            if isinstance(data.get("path"), str):
                paths.append(data["path"])
        elif tool_name == "search_code":
            matches = data.get("matches")
            if isinstance(matches, list):
                paths.extend(
                    match["path"]
                    for match in matches
                    if isinstance(match, dict) and isinstance(match.get("path"), str)
                )
        for path in paths:
            try:
                normalized = normalize_workspace_relative_path(path)
            except RepositoryStateValidationError:
                continue
            if normalized in self.inspected_evidence_paths:
                continue
            self.inspected_evidence_paths = (
                *self.inspected_evidence_paths,
                normalized,
            )[-MAX_INSPECTED_EVIDENCE_PATHS:]

        source_path = data.get("path")
        if not isinstance(source_path, str):
            if tool_name == "read_webpage":
                self._record_web_source_urls(data)
            return
        if tool_name == "read_document":
            eligible_source = True
        elif tool_name == "read_file":
            eligible_source = (
                PurePosixPath(source_path).suffix.casefold()
                in _TEXT_SOURCE_EXTENSIONS
            )
        else:
            eligible_source = False
        if not eligible_source:
            return
        try:
            normalized_source = normalize_workspace_relative_path(source_path)
        except RepositoryStateValidationError:
            return
        if normalized_source in self.read_source_paths:
            return
        self.read_source_paths = (
            *self.read_source_paths,
            normalized_source,
        )[-MAX_READ_SOURCE_PATHS:]

    def _record_web_source_urls(self, data: dict[str, Any]) -> None:
        for key in ("requested_url", "final_url"):
            url = data.get(key)
            if (
                not isinstance(url, str)
                or not url
                or len(url) > MAX_READ_SOURCE_URL_CHARS
                or url in self.read_source_urls
            ):
                continue
            self.read_source_urls = (
                *self.read_source_urls,
                url,
            )[-MAX_READ_SOURCE_URLS:]

    def invalidate_review_evidence(self, path: str) -> None:
        """Invalidate exact-path eligibility and findings after a managed edit."""
        try:
            normalized = normalize_workspace_relative_path(path)
        except RepositoryStateValidationError:
            return
        self.inspected_evidence_paths = tuple(
            existing
            for existing in self.inspected_evidence_paths
            if existing != normalized
        )
        replacement, _ = self.review_state.invalidate_path(normalized)
        self.review_state = replacement

    def record_run_command(
        self,
        *,
        model_step: int,
        result: dict[str, Any],
    ) -> None:
        """Record every dispatched run_command without inferring filesystem effects."""
        self.verification.record_attempt(
            workspace_revision=self.workspace_revision,
            model_step=model_step,
            result=result,
        )

    @property
    def verification_status(self) -> VerificationStatus:
        return self.verification.status(self.workspace_revision)

    @property
    def verified_revision(self) -> int | None:
        return self.verification.verified_revision(self.workspace_revision)

    @property
    def plan_status(self) -> PlanOutcome:
        if self.plan is None:
            return "not_planned"
        return self.plan.status

    @property
    def unfinished_steps(self) -> tuple[PlanStep, ...]:
        if self.plan is None:
            return ()
        return tuple(
            step
            for step in self.plan.steps
            if step.status in {"pending", "in_progress"}
        )

    @property
    def blocked_steps(self) -> tuple[PlanStep, ...]:
        if self.plan is None:
            return ()
        return tuple(step for step in self.plan.steps if step.status == "blocked")

    def request_completion_review(self) -> bool:
        reasons = self._completion_review_reasons()
        if not reasons:
            return False
        fingerprint = self._completion_review_fingerprint()
        if fingerprint == self.last_completion_review_fingerprint:
            return False
        self.last_completion_review_fingerprint = fingerprint
        self.pending_completion_review = {
            "reasons": list(reasons),
            "instruction": _COMPLETION_REVIEW_INSTRUCTION,
        }
        return True

    def clear_pending_completion_review(self) -> None:
        self.pending_completion_review = None

    def snapshot_for_model(self) -> dict[str, Any] | None:
        snapshot: dict[str, Any] = {}
        if self.plan is not None:
            snapshot["plan"] = self.plan.to_snapshot()
        if self.verification.required:
            snapshot["verification"] = self.verification.to_snapshot(
                self.workspace_revision
            )
        if self.pending_completion_review is not None:
            snapshot["completion_review"] = dict(self.pending_completion_review)
        progress_snapshot = self.progress.snapshot_for_model()
        if progress_snapshot is not None:
            snapshot["progress"] = progress_snapshot
        repository_focus = self.working_set.to_snapshot()
        if repository_focus is not None:
            snapshot["repository_focus"] = repository_focus
        review_findings = self.review_state.to_snapshot()
        if review_findings is not None:
            snapshot["review_findings"] = review_findings
        requirements = self.requirements.to_snapshot()
        if requirements is not None:
            snapshot["requirements"] = requirements
        return snapshot or None

    def _completion_review_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.plan_status == "active":
            reasons.append("active_plan")
        if self.verification_status == "unverified":
            reasons.append("unverified_workspace")
        return tuple(reasons)

    def _completion_review_fingerprint(self) -> CompletionReviewFingerprint:
        plan_revision = self.plan.revision if self.plan is not None else 0
        active_step_ids = tuple(step.id for step in self.unfinished_steps)
        return (
            plan_revision,
            self.workspace_revision,
            self.verification_status,
            active_step_ids,
        )
