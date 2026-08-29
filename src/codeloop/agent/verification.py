"""Run-local verification facts for CodeLoop-managed workspace revisions.

Verification deliberately inherits the Stage 7B blind spot: ``run_command``
may change files, but CodeLoop does not scan or diff the filesystem around a
command.  An attempt is therefore evidence about the managed revision recorded
at dispatch time, not proof that the command left the filesystem unchanged or
that the user's semantic requirements are correct.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

VerificationStatus = Literal["not_required", "verified", "unverified"]


@dataclass(frozen=True)
class VerificationAttempt:
    """Minimal metadata for one run_command Action and its result."""

    workspace_revision: int
    model_step: int
    succeeded: bool
    exit_code: int | None
    timed_out: bool
    error_code: str | None

    @classmethod
    def from_result(
        cls,
        *,
        workspace_revision: int,
        model_step: int,
        result: Mapping[str, Any],
    ) -> VerificationAttempt:
        data = result.get("data")
        result_data = data if isinstance(data, Mapping) else {}
        raw_exit_code = result_data.get("exit_code")
        exit_code = (
            raw_exit_code
            if isinstance(raw_exit_code, int) and not isinstance(raw_exit_code, bool)
            else None
        )
        timed_out = result_data.get("timed_out") is True
        succeeded = (
            result.get("ok") is True
            and exit_code == 0
            and not timed_out
        )
        raw_error_code = result.get("error_code")
        error_code = raw_error_code if isinstance(raw_error_code, str) else None
        return cls(
            workspace_revision=workspace_revision,
            model_step=model_step,
            succeeded=succeeded,
            exit_code=exit_code,
            timed_out=timed_out,
            error_code=error_code,
        )

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "workspace_revision": self.workspace_revision,
            "model_step": self.model_step,
            "succeeded": self.succeeded,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "error_code": self.error_code,
        }


@dataclass
class VerificationState:
    """Track only the latest command attempt and historical latest success.

    ``verified`` means that the latest run_command attempt for the current
    CodeLoop-managed workspace revision succeeded.  It says nothing about
    command relevance, semantic correctness, or unobserved command side effects.
    """

    required: bool = False
    last_attempt: VerificationAttempt | None = None
    last_success: VerificationAttempt | None = None

    def require_verification(self) -> None:
        self.required = True

    def record_attempt(
        self,
        *,
        workspace_revision: int,
        model_step: int,
        result: Mapping[str, Any],
    ) -> VerificationAttempt:
        attempt = VerificationAttempt.from_result(
            workspace_revision=workspace_revision,
            model_step=model_step,
            result=result,
        )
        self.last_attempt = attempt
        if attempt.succeeded:
            self.last_success = attempt
        return attempt

    def status(self, current_revision: int) -> VerificationStatus:
        if not self.required:
            return "not_required"
        attempt = self.last_attempt
        if (
            attempt is not None
            and attempt.workspace_revision == current_revision
            and attempt.succeeded
        ):
            return "verified"
        return "unverified"

    def verified_revision(self, current_revision: int) -> int | None:
        if self.status(current_revision) == "verified":
            return current_revision
        return None

    def to_snapshot(self, current_revision: int) -> dict[str, Any]:
        return {
            "required": self.required,
            "status": self.status(current_revision),
            "workspace_revision": current_revision,
            "verified_revision": self.verified_revision(current_revision),
            "last_attempt": (
                self.last_attempt.to_snapshot()
                if self.last_attempt is not None
                else None
            ),
        }
