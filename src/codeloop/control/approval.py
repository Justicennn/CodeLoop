"""Pure approval decisions over objective action facts and task authorization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .authorization import ActionFingerprint, AuthorizationScope, TestScope

ApprovalDecision = Literal["allow", "inform", "ask", "deny"]


@dataclass(frozen=True)
class ApprovalSubject:
    description: str
    category: str
    command: tuple[str, ...]
    cwd: str
    fingerprint: ActionFingerprint
    reason: str
    test_scope: TestScope | None = None
    authorization_basis: str | None = None
    forbidden: bool = False


class ApprovalPolicy:
    """Decide authorization without performing I/O or executing the action."""

    def decide(
        self,
        subject: ApprovalSubject,
        scope: AuthorizationScope,
    ) -> ApprovalDecision:
        if subject.forbidden:
            return "deny"
        if subject.fingerprint in scope.denied_fingerprints:
            return "deny"
        if subject.category == "read_only_git":
            return "allow"
        if scope.authorizes(
            subject.category,
            subject.fingerprint,
            test_scope=subject.test_scope,
        ):
            return "inform"
        if subject.fingerprint in scope.one_shot_approvals:
            return "inform"
        if subject.authorization_basis is not None:
            return "inform"
        return "ask"
