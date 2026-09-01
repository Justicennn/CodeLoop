"""Task-local authorization facts; never session or presentation state."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ActionFingerprint:
    value: str

    @classmethod
    def command(cls, command: tuple[str, ...], cwd: str) -> ActionFingerprint:
        serialized = json.dumps(
            {"command": list(command), "cwd": cwd},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return cls(hashlib.sha256(serialized.encode("utf-8")).hexdigest())


@dataclass(frozen=True)
class TestScope:
    fingerprint: ActionFingerprint
    family: str | None = None
    cwd: str = "."
    all_tests: bool = False
    targets: frozenset[str] = frozenset()
    command: tuple[str, ...] = ()

    def contains(self, requested: TestScope) -> bool:
        if self.fingerprint == requested.fingerprint:
            return True
        if (
            self.family is None
            or requested.family is None
            or self.family != requested.family
            or self.cwd != requested.cwd
        ):
            return False
        if self.all_tests:
            return True
        if requested.all_tests:
            return False
        return bool(requested.targets) and requested.targets.issubset(self.targets)


@dataclass
class AuthorizationScope:
    """Authorization accumulated during exactly one AgentRunner.run()."""

    test_scopes: set[TestScope] = field(default_factory=set)
    program_actions: set[ActionFingerprint] = field(default_factory=set)
    program_action_details: dict[
        ActionFingerprint, tuple[tuple[str, ...], str]
    ] = field(default_factory=dict)
    one_shot_approvals: set[ActionFingerprint] = field(default_factory=set)
    used_one_shot_bases: set[ActionFingerprint] = field(default_factory=set)
    denied_fingerprints: set[ActionFingerprint] = field(default_factory=set)
    answered_interaction_fingerprints: set[str] = field(default_factory=set)
    human_response_bases: list[str] = field(default_factory=list)

    def authorizes(
        self,
        category: str,
        fingerprint: ActionFingerprint,
        *,
        test_scope: TestScope | None = None,
    ) -> bool:
        if category == "test":
            requested = test_scope or TestScope(fingerprint)
            return any(scope.contains(requested) for scope in self.test_scopes)
        if category == "program_execution":
            return fingerprint in self.program_actions
        return False

    def approve_reusable(
        self,
        category: str,
        fingerprint: ActionFingerprint,
        *,
        test_scope: TestScope | None = None,
        command: tuple[str, ...] = (),
        cwd: str = ".",
    ) -> None:
        if category == "test":
            self.test_scopes.add(test_scope or TestScope(
                fingerprint=fingerprint,
                cwd=cwd,
                command=command,
            ))
        elif category == "program_execution":
            self.program_actions.add(fingerprint)
            self.program_action_details[fingerprint] = (command, cwd)

    def related_test_scope(self, requested: TestScope) -> TestScope | None:
        if not self.test_scopes:
            return None
        ordered = sorted(
            self.test_scopes,
            key=lambda scope: (
                scope.family != requested.family,
                scope.cwd != requested.cwd,
                scope.all_tests,
                tuple(sorted(scope.targets)),
                scope.fingerprint.value,
            ),
        )
        return ordered[0]

    def related_program_action(
        self,
    ) -> tuple[tuple[str, ...], str] | None:
        if not self.program_action_details:
            return None
        fingerprint = sorted(
            self.program_action_details,
            key=lambda item: item.value,
        )[0]
        return self.program_action_details[fingerprint]

    def deny(self, fingerprint: ActionFingerprint) -> None:
        self.denied_fingerprints.add(fingerprint)

    def approve_one_shot(self, fingerprint: ActionFingerprint) -> None:
        self.one_shot_approvals.add(fingerprint)

    def consume_one_shot(self, fingerprint: ActionFingerprint) -> bool:
        if fingerprint not in self.one_shot_approvals:
            return False
        self.one_shot_approvals.remove(fingerprint)
        return True

    def mark_one_shot_basis_used(self, fingerprint: ActionFingerprint) -> None:
        self.used_one_shot_bases.add(fingerprint)

    def record_interaction(self, fingerprint: str) -> None:
        self.answered_interaction_fingerprints.add(fingerprint)

    def record_human_response_basis(self, response: str) -> None:
        bounded = response.strip()[:2_000]
        if bounded and bounded not in self.human_response_bases:
            self.human_response_bases.append(bounded)
            del self.human_response_bases[:-32]

    def basis_is_traceable(self, basis: str, task: str) -> bool:
        candidate = basis.strip()
        if not candidate:
            return False
        return candidate in task or any(
            candidate in response for response in self.human_response_bases
        )

    def clear(self) -> None:
        self.test_scopes.clear()
        self.program_actions.clear()
        self.program_action_details.clear()
        self.one_shot_approvals.clear()
        self.used_one_shot_bases.clear()
        self.denied_fingerprints.clear()
        self.answered_interaction_fingerprints.clear()
        self.human_response_bases.clear()
