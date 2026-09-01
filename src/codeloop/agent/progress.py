"""Conservative, run-local detection of obvious repetition and stalls.

Material Observation novelty is only an information-gain heuristic.  It does
not establish semantic task progress: a model can still wander through new but
irrelevant files until the independent ``max_steps`` limit stops the run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal, Mapping

from .plan import TaskPlan, UPDATE_PLAN_ACTION_NAME
from .repository import UPDATE_WORKING_SET_ACTION_NAME
from .review import UPDATE_REVIEW_FINDINGS_ACTION_NAME
from .requirements import UPDATE_REQUIREMENTS_ACTION_NAME
from .verification import VerificationStatus

ProgressStatus = Literal["active", "possible_stall"]
ProgressReason = Literal["repeating_pattern", "extended_no_progress"]
ProgressDecision = Literal[
    "continue",
    "request_recovery",
    "terminate_no_progress",
]

REPEATING_NO_PROGRESS_TURNS = 3
EXTENDED_NO_PROGRESS_TURNS = 5
RECENT_DIGEST_LIMIT = 32

_PROGRESS_INSTRUCTION = (
    "Reconsider the current assumptions and take a materially different action. "
    "If a plan exists, use replan with a short explanation; otherwise create a "
    "high-level strategy plan if useful. Do not use repeated calls, no-ops, or "
    "cosmetic plan updates to imitate progress."
)
_REVISION_SCOPED_TOOLS = {
    "repository_overview",
    "list_files",
    "read_file",
    "read_document",
    "read_webpage",
    "read_image",
    "search_code",
    "run_command",
}
_MUTATION_TOOLS = {"edit_file", "write_file", "make_directory"}
_CORE_STATE_ACTIONS = {
    UPDATE_PLAN_ACTION_NAME,
    UPDATE_REQUIREMENTS_ACTION_NAME,
    UPDATE_WORKING_SET_ACTION_NAME,
    UPDATE_REVIEW_FINDINGS_ACTION_NAME,
}
_NON_DISPATCH_CONTROL_ERRORS = {
    "approval_unavailable",
    "interaction_required",
    "permission_denied",
    "user_denied",
}


@dataclass
class ProgressState:
    """Serializable progress bookkeeping aggregated by TaskState.

    This object stores facts only.  ProgressTracker owns all fingerprinting,
    signal classification, streak, recovery, and termination decisions.
    """

    status: ProgressStatus = "active"
    no_progress_turns: int = 0
    repeating_pattern_turns: int = 0
    last_no_progress_pattern: str | None = None
    recovery_active: bool = False
    recovery_reason: ProgressReason | None = None
    recent_observation_digests: tuple[str, ...] = ()
    recent_active_transition_digests: tuple[str, ...] = ()

    def snapshot_for_model(self) -> dict[str, str] | None:
        """Expose only the bounded recovery instruction, never bookkeeping."""
        if self.status != "possible_stall" or self.recovery_reason is None:
            return None
        return {
            "status": self.status,
            "reason": self.recovery_reason,
            "instruction": _PROGRESS_INSTRUCTION,
        }


@dataclass(frozen=True)
class ProgressFacts:
    workspace_revision: int
    verification_status: VerificationStatus


@dataclass(frozen=True)
class ProgressAction:
    """One completed Action and the facts needed for turn evaluation."""

    name: str
    arguments: str
    result: Mapping[str, Any]
    workspace_revision: int
    plan_before: TaskPlan | None
    plan_after: TaskPlan | None


class ProgressTracker:
    """Evaluate completed turns without orchestrating Runner, Context, or Tools."""

    def evaluate_turn(
        self,
        state: ProgressState,
        *,
        before: ProgressFacts,
        after: ProgressFacts,
        actions: tuple[ProgressAction, ...],
    ) -> ProgressDecision:
        if not actions:
            raise ValueError("Progress evaluation requires at least one Action")

        # VerificationAttempt fields are intentionally absent from ProgressFacts:
        # only a real derived status transition is independent progress. New
        # command evidence can still qualify through Observation novelty below.
        strong_progress = (
            after.workspace_revision != before.workspace_revision
            or after.verification_status != before.verification_status
        )
        weak_progress = False
        pattern_parts: list[tuple[str, str]] = []

        for action in actions:
            observation = observation_digest(action)
            pattern_parts.append(
                (action_digest(action.name, action.arguments), observation)
            )

            if self._is_material_observation(action):
                if observation not in state.recent_observation_digests:
                    strong_progress = True
                state.recent_observation_digests = _remember_digest(
                    state.recent_observation_digests,
                    observation,
                )

            plan_strong, transition = _plan_progress(action)
            strong_progress = strong_progress or plan_strong
            if transition is not None:
                transition_digest = _digest_json(transition)
                if transition_digest not in state.recent_active_transition_digests:
                    weak_progress = True
                state.recent_active_transition_digests = _remember_digest(
                    state.recent_active_transition_digests,
                    transition_digest,
                )

        if strong_progress or weak_progress:
            _clear_streak_and_recovery(state)
            return "continue"

        turn_pattern = _digest_json(pattern_parts)
        state.no_progress_turns += 1
        if turn_pattern == state.last_no_progress_pattern:
            state.repeating_pattern_turns += 1
        else:
            state.last_no_progress_pattern = turn_pattern
            state.repeating_pattern_turns = 1

        reason: ProgressReason | None = None
        if state.repeating_pattern_turns >= REPEATING_NO_PROGRESS_TURNS:
            reason = "repeating_pattern"
        elif state.no_progress_turns >= EXTENDED_NO_PROGRESS_TURNS:
            reason = "extended_no_progress"
        if reason is None:
            return "continue"

        if state.recovery_active:
            state.status = "possible_stall"
            state.recovery_reason = reason
            return "terminate_no_progress"

        state.status = "possible_stall"
        state.recovery_active = True
        state.recovery_reason = reason
        state.no_progress_turns = 0
        state.repeating_pattern_turns = 0
        state.last_no_progress_pattern = None
        return "request_recovery"

    @staticmethod
    def _is_material_observation(action: ProgressAction) -> bool:
        if action.result.get("error_code") in _NON_DISPATCH_CONTROL_ERRORS:
            # Human-control outcomes are important recoverable Observations,
            # but no Execution action ran and no new workspace/runtime evidence
            # was obtained. Repeating them must remain eligible for bounded
            # no-progress recovery rather than refreshing progress once.
            return False
        if action.name in _CORE_STATE_ACTIONS:
            return action.result.get("ok") is False
        if action.name in _MUTATION_TOOLS:
            data = _result_data(action.result)
            if action.result.get("ok") is True:
                return data.get("workspace_changed") is True
        return True


def action_digest(name: str, arguments: str) -> str:
    """Hash an Action name and canonical arguments without retaining arguments."""
    try:
        parsed = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        canonical = f"raw:{arguments}"
    else:
        canonical = json.dumps(
            parsed,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return _digest_text(f"{name}\x00{canonical}")


def observation_digest(action: ProgressAction) -> str:
    """Hash a narrow material projection; the projection is never persisted."""
    result = action.result
    data = _result_data(result)
    projection: dict[str, Any] = {
        "ok": result.get("ok"),
        "error_code": result.get("error_code"),
    }

    if action.name == "repository_overview":
        projection["data"] = _select(
            data,
            "path",
            "scan",
            "tree",
            "anchors",
            "directory_candidates",
            "extension_stats",
            "truncated",
            "truncation_reasons",
        )
    elif action.name == "list_files":
        projection["data"] = _select(
            data,
            "path",
            "entries",
            "truncated",
        )
    elif action.name == "read_file":
        projection["data"] = _select(
            data,
            "path",
            "start_line",
            "end_line",
            "total_lines",
            "content",
            "truncated",
        )
    elif action.name == "read_document":
        projection["data"] = _select(
            data,
            "path",
            "document_type",
            "text",
            "position",
            "truncated",
            "next_cursor",
        )
    elif action.name == "read_webpage":
        projection["data"] = _select(
            data,
            "requested_url",
            "final_url",
            "title",
            "content_type",
            "text",
            "position",
            "truncated",
            "next_cursor",
        )
    elif action.name == "read_image":
        projection["data"] = _select(
            data,
            "path",
            "image_type",
            "mime_type",
            "size_bytes",
        )
    elif action.name == "search_code":
        projection["data"] = _select(
            data,
            "matches",
            "count",
            "files",
            "truncated",
        )
    elif action.name == "run_command":
        projection["data"] = _select(
            data,
            "exit_code",
            "stdout",
            "stderr",
            "stdout_truncated",
            "stderr_truncated",
            "timed_out",
            "direct_child_reaped",
        )
    elif action.name in _MUTATION_TOOLS:
        projection["data"] = _select(
            data,
            "path",
            "workspace_changed",
            "matches",
            "replacements",
            "before_chars",
            "after_chars",
            "characters",
            "created_directories",
            "created_count",
        )
    elif action.name == UPDATE_PLAN_ACTION_NAME:
        if result.get("ok") is True:
            projection["data"] = {"result": "successful_plan_action"}
        else:
            projection["data"] = _select(data, "changed_step_ids")
    elif action.name in {
        UPDATE_REQUIREMENTS_ACTION_NAME,
        UPDATE_WORKING_SET_ACTION_NAME,
        UPDATE_REVIEW_FINDINGS_ACTION_NAME,
    }:
        if result.get("ok") is True:
            projection["data"] = {"result": "successful_core_state_action"}
        else:
            projection["data"] = _stable_error_data(data)
    else:
        projection["data"] = _stable_error_data(data)

    if action.name in _REVISION_SCOPED_TOOLS:
        projection["workspace_revision"] = action.workspace_revision
    return _digest_json(projection)


def _plan_progress(action: ProgressAction) -> tuple[bool, tuple[str, str] | None]:
    if action.name != UPDATE_PLAN_ACTION_NAME or action.result.get("ok") is not True:
        return False, None
    data = _result_data(action.result)
    if data.get("mode") != "update":
        return False, None
    before = action.plan_before
    after = action.plan_after
    if before is None or after is None:
        return False, None

    before_by_id = {step.id: step for step in before.steps}
    after_by_id = {step.id: step for step in after.steps}
    terminal_advance = any(
        step_id in after_by_id
        and old.status in {"pending", "in_progress"}
        and after_by_id[step_id].status in {"completed", "blocked"}
        for step_id, old in before_by_id.items()
    )

    old_active = next(
        (step.id for step in before.steps if step.status == "in_progress"),
        "",
    )
    new_active = next(
        (step.id for step in after.steps if step.status == "in_progress"),
        "",
    )
    transition = None
    if new_active and new_active != old_active:
        transition = (old_active, new_active)
    return terminal_advance, transition


def _clear_streak_and_recovery(state: ProgressState) -> None:
    state.status = "active"
    state.no_progress_turns = 0
    state.repeating_pattern_turns = 0
    state.last_no_progress_pattern = None
    state.recovery_active = False
    state.recovery_reason = None


def _remember_digest(existing: tuple[str, ...], digest: str) -> tuple[str, ...]:
    if digest in existing:
        return existing
    return (*existing, digest)[-RECENT_DIGEST_LIMIT:]


def _result_data(result: Mapping[str, Any]) -> Mapping[str, Any]:
    data = result.get("data")
    return data if isinstance(data, Mapping) else {}


def _select(data: Mapping[str, Any], *fields: str) -> dict[str, Any]:
    return {field: data[field] for field in fields if field in data}


def _stable_error_data(data: Mapping[str, Any]) -> dict[str, Any]:
    volatile = {
        "command",
        "cwd",
        "duration_ms",
        "engine",
        "query",
        "timeout_seconds",
    }
    return {
        key: value
        for key, value in data.items()
        if key not in volatile
    }


def _digest_json(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _digest_text(serialized)


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
