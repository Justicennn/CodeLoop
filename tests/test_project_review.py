"""Stage 9 working-set, review-state, and Runtime integration tests."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import codeloop.agent.runner as runner_module
from codeloop.agent.prompt import SYSTEM_PROMPT
from codeloop.agent.progress import ProgressAction, ProgressFacts, ProgressTracker
from codeloop.agent.repository import (
    MAX_WORKING_SET_ENTRIES,
    RepositoryStateValidationError,
    RepositoryWorkingSet,
    UPDATE_WORKING_SET_ACTION_NAME,
    WorkingSetEntry,
    apply_working_set_action,
)
from codeloop.agent.review import (
    FindingEvidence,
    ReviewFinding,
    ReviewState,
    ReviewValidationError,
    UPDATE_REVIEW_FINDINGS_ACTION_NAME,
    apply_review_findings_action,
)
from codeloop.agent.runner import AgentRunner
from codeloop.agent.task_state import MAX_INSPECTED_EVIDENCE_PATHS, TaskState
from codeloop.execution.tools import ToolRegistry
from codeloop.execution.workspace import Workspace
from codeloop.model.client import ModelAPIError, ModelResponse, ToolCall


class FakeClient:
    def __init__(self, responses: list[ModelResponse | BaseException]) -> None:
        self._responses = iter(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        self.calls.append({"messages": deepcopy(messages), "tools": deepcopy(tools)})
        response = next(self._responses)
        if isinstance(response, BaseException):
            raise response
        return response


def _finding(path: str, *, finding_id: str = "F1") -> dict[str, Any]:
    return {
        "id": finding_id,
        "finding_type": "issue",
        "title": "Repeated rebuild",
        "category": "performance",
        "evidence": [
            {
                "path": path,
                "symbol": "render",
                "description": "render recreates every child after a local change",
            }
        ],
        "impact": "Normal updates perform avoidable full-tree work.",
        "recommendation": "Update only the affected child.",
        "priority": "high",
    }


def _review_action(state: TaskState, findings: list[dict[str, Any]]) -> dict[str, Any]:
    return apply_review_findings_action(
        state,
        json.dumps({"findings": findings}),
    )


def test_working_set_replace_clear_noop_and_bounds() -> None:
    state = TaskState()
    entries = [{"path": "src\\app.py", "reason": "request handling"}]
    created = apply_working_set_action(state, json.dumps({"entries": entries}))
    noop = apply_working_set_action(state, json.dumps({"entries": entries}))
    cleared = apply_working_set_action(state, json.dumps({"entries": []}))

    assert created["data"] == {"changed": True, "revision": 1, "entry_count": 1}
    assert state.working_set.entries == ()
    assert noop["data"]["changed"] is False
    assert cleared["data"] == {"changed": True, "revision": 2, "entry_count": 0}

    duplicate = apply_working_set_action(
        TaskState(),
        json.dumps(
            {"entries": [
                {"path": "src/app.py", "reason": "one"},
                {"path": "src\\app.py", "reason": "two"},
            ]}
        ),
    )
    too_many = apply_working_set_action(
        TaskState(),
        json.dumps(
            {"entries": [
                {"path": f"f{index}.py", "reason": "focus"}
                for index in range(MAX_WORKING_SET_ENTRIES + 1)
            ]}
        ),
    )
    assert duplicate["error_code"] == "duplicate_working_set_path"
    assert too_many["ok"] is False

    with pytest.raises(RepositoryStateValidationError):
        RepositoryWorkingSet(
            entries=(WorkingSetEntry("src/app.py", "x" * 161),)
        )


def test_working_set_is_not_an_allowlist_and_state_is_task_local(tmp_path: Path) -> None:
    (tmp_path / "outside_focus.py").write_text("value = 1", encoding="utf-8")
    focused = TaskState()
    apply_working_set_action(
        focused,
        json.dumps({"entries": [{"path": "inside.py", "reason": "initial focus"}]}),
    )

    result = ToolRegistry(Workspace(tmp_path)).dispatch(
        "read_file",
        json.dumps({"path": "outside_focus.py"}),
    )
    fresh = TaskState()

    assert result["ok"] is True
    assert focused.working_set.entries[0].path == "inside.py"
    assert fresh.working_set.entries == ()
    assert fresh.review_state.findings == ()
    assert fresh.inspected_evidence_paths == ()


def test_overview_and_list_paths_are_not_review_evidence(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("def render(): pass", encoding="utf-8")
    registry = ToolRegistry(Workspace(tmp_path))
    state = TaskState()
    for name in ("repository_overview", "list_files"):
        result = registry.dispatch(name, "{}")
        state.record_execution_evidence(tool_name=name, result=result)

    rejected = _review_action(state, [_finding("app.py")])

    assert state.inspected_evidence_paths == ()
    assert rejected["error_code"] == "unobserved_review_evidence"
    assert state.review_state.findings == ()


def test_read_and_search_register_path_eligibility_not_complete_understanding(
    tmp_path: Path,
) -> None:
    source = tmp_path / "app.py"
    source.write_text("def render():\n    return 'node'\n", encoding="utf-8")
    registry = ToolRegistry(Workspace(tmp_path))
    state = TaskState()

    partial_read = registry.dispatch(
        "read_file",
        json.dumps({"path": "app.py", "start_line": 1, "end_line": 1}),
    )
    state.record_execution_evidence(tool_name="read_file", result=partial_read)
    accepted = _review_action(state, [_finding("app.py")])

    assert state.inspected_evidence_paths == ("app.py",)
    assert accepted["ok"] is True

    search_state = TaskState()
    matched = registry.dispatch("search_code", json.dumps({"query": "render"}))
    search_state.record_execution_evidence(tool_name="search_code", result=matched)
    assert search_state.inspected_evidence_paths == ("app.py",)

    empty_state = TaskState()
    empty = registry.dispatch("search_code", json.dumps({"query": "missing"}))
    empty_state.record_execution_evidence(tool_name="search_code", result=empty)
    assert empty_state.inspected_evidence_paths == ()


def test_review_replace_is_atomic_for_unobserved_evidence() -> None:
    state = TaskState(inspected_evidence_paths=("a.py",))
    initial = _review_action(state, [_finding("a.py")])
    rejected = _review_action(
        state,
        [_finding("a.py"), _finding("unread.py", finding_id="F2")],
    )

    assert initial["ok"] is True
    assert rejected["error_code"] == "unobserved_review_evidence"
    assert [finding.id for finding in state.review_state.findings] == ["F1"]
    assert state.review_state.revision == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("finding_type", "defect"),
        ("category", "style"),
        ("priority", "critical"),
    ],
)
def test_review_enums_and_ids_are_strict(field: str, value: str) -> None:
    state = TaskState(inspected_evidence_paths=("a.py",))
    invalid = _finding("a.py")
    invalid[field] = value
    rejected = _review_action(state, [invalid])
    assert rejected["error_code"] == "invalid_review_findings"
    assert state.review_state.findings == ()

    duplicate = _review_action(
        state,
        [_finding("a.py"), _finding("a.py")],
    )
    assert duplicate["error_code"] == "duplicate_finding_id"


def test_review_finding_and_evidence_counts_are_bounded() -> None:
    eligible = tuple(f"f{index}.py" for index in range(17))
    state = TaskState(inspected_evidence_paths=eligible)
    too_many = _review_action(
        state,
        [_finding(path, finding_id=f"F{index}") for index, path in enumerate(eligible)],
    )
    assert too_many["ok"] is False
    assert state.review_state.findings == ()

    excessive_evidence = _finding("f0.py")
    excessive_evidence["evidence"] = [
        {"path": f"f{index}.py", "description": "evidence"}
        for index in range(5)
    ]
    rejected = _review_action(state, [excessive_evidence])
    assert rejected["error_code"] == "invalid_review_findings"

    enhancement = _finding("f0.py", finding_id="E1")
    enhancement.update(
        {
            "finding_type": "enhancement",
            "category": "usability",
            "priority": "low",
            "title": "Optional interaction polish",
        }
    )
    accepted = _review_action(state, [enhancement])
    assert accepted["ok"] is True
    assert state.review_state.findings[0].finding_type == "enhancement"
    assert state.review_state.findings[0].priority == "low"

    with pytest.raises(ReviewValidationError):
        ReviewState(
            findings=(
                ReviewFinding(
                    id="F",
                    finding_type="issue",
                    title="title",
                    category="performance",
                    evidence=(FindingEvidence("f0.py", "x" * 501),),
                    impact="impact",
                    recommendation="recommendation",
                    priority="high",
                ),
            )
        )


def test_evidence_fifo_is_bounded_unique_and_not_snapshotted() -> None:
    state = TaskState()
    for index in range(MAX_INSPECTED_EVIDENCE_PATHS + 2):
        state.record_execution_evidence(
            tool_name="read_file",
            result={"ok": True, "data": {"path": f"src/f{index}.py"}},
        )
    state.record_execution_evidence(
        tool_name="read_file",
        result={"ok": True, "data": {"path": "src/f2.py"}},
    )

    assert len(state.inspected_evidence_paths) == MAX_INSPECTED_EVIDENCE_PATHS
    assert state.inspected_evidence_paths[0] == "src/f2.py"
    assert state.inspected_evidence_paths.count("src/f2.py") == 1
    assert state.snapshot_for_model() is None


def test_managed_change_invalidates_exact_path_and_dependent_findings() -> None:
    state = TaskState(inspected_evidence_paths=("a.py", "b.py"))
    finding = _finding("a.py")
    finding["evidence"].append(
        {"path": "b.py", "description": "caller repeats the operation"}
    )
    assert _review_action(state, [finding])["ok"] is True

    state.record_workspace_change("a.py")

    assert state.inspected_evidence_paths == ("b.py",)
    assert state.review_state.findings == ()
    assert state.review_state.revision == 2
    assert _review_action(state, [finding])["error_code"] == "unobserved_review_evidence"

    state.record_execution_evidence(
        tool_name="read_file",
        result={"ok": True, "data": {"path": "a.py"}},
    )
    assert _review_action(state, [finding])["ok"] is True


def test_noop_workspace_result_does_not_invalidate_evidence(tmp_path: Path) -> None:
    path = tmp_path / "a.py"
    path.write_text("same", encoding="utf-8")
    registry = ToolRegistry(Workspace(tmp_path))
    state = TaskState(inspected_evidence_paths=("a.py",))
    assert _review_action(state, [_finding("a.py")])["ok"] is True

    result = registry.dispatch(
        "edit_file",
        json.dumps({"path": "a.py", "old_text": "same", "new_text": "same"}),
    )
    assert registry.confirmed_workspace_change("edit_file", result) is False
    assert state.inspected_evidence_paths == ("a.py",)
    assert len(state.review_state.findings) == 1


def test_snapshots_are_compact_and_replace_without_full_finding_fields() -> None:
    state = TaskState(inspected_evidence_paths=("app.py",))
    apply_working_set_action(
        state,
        json.dumps({"entries": [{"path": "app.py", "reason": "render path"}]}),
    )
    assert _review_action(state, [_finding("app.py")])["ok"] is True

    snapshot = state.snapshot_for_model()
    assert snapshot is not None
    assert snapshot["repository_focus"] == [
        {"path": "app.py", "reason": "render path"}
    ]
    assert snapshot["review_findings"] == [
        {
            "id": "F1",
            "finding_type": "issue",
            "priority": "high",
            "title": "Repeated rebuild",
        }
    ]
    serialized = json.dumps(snapshot)
    for forbidden in ("evidence", "impact", "recommendation", "inspected_evidence"):
        assert forbidden not in serialized


def test_core_actions_are_not_execution_tools_and_runner_routes_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "app.py").write_text("def render(): pass", encoding="utf-8")
    state = TaskState()
    monkeypatch.setattr(runner_module, "TaskState", lambda: state)
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[ToolCall("read", "read_file", '{"path":"app.py"}')]
            ),
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        "focus",
                        UPDATE_WORKING_SET_ACTION_NAME,
                        '{"entries":[{"path":"app.py","reason":"render"}]}',
                    ),
                    ToolCall(
                        "finding",
                        UPDATE_REVIEW_FINDINGS_ACTION_NAME,
                        json.dumps({"findings": [_finding("app.py")]}),
                    ),
                ]
            ),
            ModelResponse(text="The current issue is the repeated rebuild."),
        ]
    )
    registry = ToolRegistry(Workspace(tmp_path))

    result = AgentRunner(client, tools=registry).run("Review this project")

    assert result.status == "completed"
    assert UPDATE_WORKING_SET_ACTION_NAME not in registry.names
    assert UPDATE_REVIEW_FINDINGS_ACTION_NAME not in registry.names
    assert state.working_set.entries[0].path == "app.py"
    assert state.review_state.findings[0].id == "F1"
    result_messages = client.calls[2]["messages"][-2:]
    assert [message["tool_call_id"] for message in result_messages] == [
        "focus",
        "finding",
    ]
    names = [schema["function"]["name"] for schema in client.calls[0]["tools"]]
    assert names[:4] == [
        "update_plan",
        "update_requirements",
        UPDATE_WORKING_SET_ACTION_NAME,
        UPDATE_REVIEW_FINDINGS_ACTION_NAME,
    ]


def test_review_fix_invalidates_current_finding_then_reports_fixed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "app.py").write_text("return 'slow'\n", encoding="utf-8")
    state = TaskState()
    monkeypatch.setattr(runner_module, "TaskState", lambda: state)
    verify_arguments = json.dumps(
        {"command": [sys.executable, "-c", "print('verified')"]}
    )
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[ToolCall("read-old", "read_file", '{"path":"app.py"}')]
            ),
            ModelResponse(
                tool_calls=[ToolCall(
                    "finding-old",
                    UPDATE_REVIEW_FINDINGS_ACTION_NAME,
                    json.dumps({"findings": [_finding("app.py")]}),
                )]
            ),
            ModelResponse(
                tool_calls=[ToolCall(
                    "edit",
                    "edit_file",
                    json.dumps(
                        {
                            "path": "app.py",
                            "old_text": "return 'slow'",
                            "new_text": "return 'focused'",
                        }
                    ),
                )]
            ),
            ModelResponse(
                tool_calls=[ToolCall("verify", "run_command", verify_arguments)]
            ),
            ModelResponse(
                tool_calls=[ToolCall("read-new", "read_file", '{"path":"app.py"}')]
            ),
            ModelResponse(
                tool_calls=[ToolCall(
                    "clear-current",
                    UPDATE_REVIEW_FINDINGS_ACTION_NAME,
                    '{"findings":[]}',
                )]
            ),
            ModelResponse(text="The repeated rebuild issue was fixed and verified."),
        ]
    )

    result = AgentRunner(
        client,
        tools=ToolRegistry(Workspace(tmp_path)),
    ).run("Review and fix the most serious issue")

    assert result.status == "completed"
    assert result.verification_status == "verified"
    assert state.review_state.findings == ()
    assert state.inspected_evidence_paths == ("app.py",)
    assert result.answer is not None
    assert "was fixed" in result.answer
    assert "current defect" not in result.answer


def test_prompt_requires_evidence_guided_navigation_and_post_edit_recheck() -> None:
    prompt = SYSTEM_PROMPT
    assert "consider repository_overview first" in prompt
    assert "Navigate from confirmed repository evidence" in prompt
    assert "structure returned by repository_overview" in prompt
    assert "entries returned by a successful list_files call" in prompt
    assert "match paths returned by search_code" in prompt
    assert "paths returned by a successful read_file call" in prompt
    assert "Conventions may form investigation hypotheses" in prompt
    assert "they are not filesystem facts" in prompt
    assert "this is not a mandatory mechanical tool order" in prompt
    assert "only the current investigation focus" in prompt
    assert "does not prove that a path exists" in prompt
    assert "is not an allowlist" in prompt
    assert "Confirm a not-yet-confirmed working-set path" in prompt
    assert (
        "treat file_not_found from list_files or read_file as navigation evidence"
        in prompt
    )
    assert "Do not fabricate the path" in prompt
    assert "immediately guess a sequence of similar directories" in prompt
    assert "Return to the most recent confirmed repository structure" in prompt
    assert "Workspace Root remains the only filesystem access boundary" in prompt
    assert "only an investigation lead" in prompt
    assert "A search_code match can support a finding directly" in prompt
    assert "depends on broader control flow, state changes, call relationships" in prompt
    assert "continue with read_file" in prompt
    assert "you remain responsible for evidence sufficiency" in prompt
    assert "Re-read or search the modified implementation" in prompt
    assert "report it as fixed rather than listing it as a current defect" in prompt
    assert "run_command filesystem side effects are not tracked" in prompt


def test_successful_review_bookkeeping_does_not_imitate_material_progress() -> None:
    state = TaskState()
    tracker = ProgressTracker()
    facts = ProgressFacts(workspace_revision=0, verification_status="not_required")
    action = ProgressAction(
        name=UPDATE_WORKING_SET_ACTION_NAME,
        arguments='{"entries":[]}',
        result={
            "ok": True,
            "data": {"changed": False, "revision": 0, "entry_count": 0},
        },
        workspace_revision=0,
        plan_before=None,
        plan_after=None,
    )

    assert tracker.evaluate_turn(
        state.progress,
        before=facts,
        after=facts,
        actions=(action,),
    ) == "continue"
    assert tracker.evaluate_turn(
        state.progress,
        before=facts,
        after=facts,
        actions=(action,),
    ) == "continue"
    assert tracker.evaluate_turn(
        state.progress,
        before=facts,
        after=facts,
        actions=(action,),
    ) == "request_recovery"


def test_review_snapshot_and_all_action_schemas_are_frozen_across_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = TaskState(inspected_evidence_paths=("app.py",))
    apply_working_set_action(
        state,
        json.dumps({"entries": [{"path": "app.py", "reason": "review focus"}]}),
    )
    assert _review_action(state, [_finding("app.py")])["ok"] is True
    monkeypatch.setattr(runner_module, "TaskState", lambda: state)
    monkeypatch.setattr(runner_module, "sleep", lambda _seconds: None)
    client = FakeClient(
        [
            ModelAPIError(
                "temporary_api_error",
                "temporary",
                classification="retryable",
            ),
            ModelResponse(text="done"),
        ]
    )

    result = AgentRunner(
        client,
        tools=ToolRegistry(Workspace(tmp_path)),
    ).run("Review")

    assert result.status == "completed"
    assert client.calls[0] == client.calls[1]
    system = client.calls[0]["messages"][0]["content"]
    assert '"repository_focus"' in system
    assert '"review_findings"' in system
    assert "render recreates every child after a local change" not in system
    assert "Normal updates perform avoidable full-tree work." not in system
    assert "Update only the affected child." not in system
