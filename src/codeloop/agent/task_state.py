"""Explicit state owned by one AgentRunner run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .plan import PlanStep, TaskPlan
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


@dataclass
class TaskState:
    """Aggregate only the persistent state implemented by the current stage."""

    plan: TaskPlan | None = None
    workspace_revision: int = 0
    verification: VerificationState = field(default_factory=VerificationState)
    last_completion_review_fingerprint: CompletionReviewFingerprint | None = None
    pending_completion_review: dict[str, Any] | None = None

    def replace_plan(self, plan: TaskPlan) -> None:
        self.plan = plan

    def record_workspace_change(self) -> None:
        self.workspace_revision += 1
        self.verification.require_verification()

    def record_run_command(
        self,
        *,
        model_step: int,
        result: dict[str, Any],
    ) -> None:
        """Record every run_command Action without inferring filesystem effects."""
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
