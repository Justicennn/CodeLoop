"""Stage 10A source-guided coding and requirement-state tests."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from docx import Document

import codeloop.agent.runner as runner_module
from codeloop.agent.plan import UPDATE_PLAN_ACTION_NAME
from codeloop.agent.progress import (
    ProgressAction,
    ProgressFacts,
    ProgressState,
    ProgressTracker,
)
from codeloop.prompts import SYSTEM_PROMPT
from codeloop.agent.requirements import (
    MAX_REQUIREMENTS,
    UPDATE_REQUIREMENTS_ACTION_NAME,
    apply_requirements_action,
)
from codeloop.agent.runner import AgentRunner
from codeloop.agent.task_state import TaskState
from codeloop.execution.tools import ToolRegistry
from codeloop.execution.web_sources import ExtractedWebPage
from codeloop.execution.workspace import Workspace
from codeloop.interaction.session import SessionHistory
from codeloop.model.client import ModelAPIError, ModelResponse, ToolCall


class FakeClient:
    def __init__(self, actions: list[ModelResponse | BaseException]) -> None:
        self._actions = iter(actions)
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        self.calls.append(
            {"messages": deepcopy(messages), "tools": deepcopy(tools)}
        )
        action = next(self._actions)
        if isinstance(action, BaseException):
            raise action
        return action


def _source_result(path: str) -> dict[str, Any]:
    return {
        "ok": True,
        "data": {
            "path": path,
            "document_type": "pdf",
            "text": "authoritative document body",
            "position": {
                "start_cursor": 0,
                "end_cursor": 27,
                "total_chars": 27,
                "first_unit": {"kind": "page", "index": 1},
                "last_unit": {"kind": "page", "index": 1},
            },
            "truncated": False,
            "next_cursor": None,
        },
    }


def _requirement(
    requirement_id: str = "R1",
    *,
    path: str = "requirements.pdf",
    kind: str = "functional",
    description: str = "Add products",
    locator: str | None = "page 1",
) -> dict[str, Any]:
    source: dict[str, str] = {"path": path}
    if locator is not None:
        source["locator"] = locator
    return {
        "id": requirement_id,
        "kind": kind,
        "description": description,
        "source": source,
    }


def _web_requirement(
    requirement_id: str = "W1",
    *,
    url: str = "https://competition.example/spec",
    locator: str | None = "Acceptance Criteria",
) -> dict[str, Any]:
    source: dict[str, str] = {"url": url}
    if locator is not None:
        source["locator"] = locator
    return {
        "id": requirement_id,
        "kind": "acceptance",
        "description": "The implementation must pass the published checks",
        "source": source,
    }


def _arguments(requirements: list[dict[str, Any]]) -> str:
    return json.dumps({"requirements": requirements}, ensure_ascii=False)


def _plan_arguments(steps: list[dict[str, str]], mode: str = "create") -> str:
    return json.dumps({"mode": mode, "steps": steps})


def test_requirement_replace_clear_noop_and_snapshot() -> None:
    state = TaskState()
    state.record_execution_evidence(
        tool_name="read_document",
        result=_source_result("requirements.pdf"),
    )
    item = _requirement()

    created = apply_requirements_action(state, _arguments([item]))
    assert created == {
        "ok": True,
        "data": {"changed": True, "revision": 1, "requirement_count": 1},
    }
    assert state.plan is None
    assert state.workspace_revision == 0
    assert state.verification.required is False
    assert state.snapshot_for_model() == {
        "requirements": [
            {
                "id": "R1",
                "kind": "functional",
                "description": "Add products",
                "source": {"path": "requirements.pdf", "locator": "page 1"},
            }
        ]
    }

    unchanged = apply_requirements_action(state, _arguments([item]))
    assert unchanged["ok"] is True
    assert unchanged["data"]["changed"] is False
    assert unchanged["data"]["revision"] == 1

    cleared = apply_requirements_action(state, _arguments([]))
    assert cleared == {
        "ok": True,
        "data": {"changed": True, "revision": 2, "requirement_count": 0},
    }
    assert state.snapshot_for_model() is None


def test_requirement_validation_is_bounded_and_atomic() -> None:
    state = TaskState(read_source_paths=("requirements.pdf",))
    valid = apply_requirements_action(state, _arguments([_requirement()]))
    assert valid["ok"] is True
    original = state.requirements

    invalid_cases = [
        [_requirement(kind="unknown")],
        [_requirement(description="x" * 401)],
        [_requirement("x" * 65)],
        [_requirement(locator="x" * 161)],
        [_requirement("R1"), _requirement("R1")],
        [_requirement(str(index)) for index in range(MAX_REQUIREMENTS + 1)],
    ]
    expected_codes = {
        "invalid_requirements",
        "duplicate_requirement_id",
    }
    for requirements in invalid_cases:
        result = apply_requirements_action(state, _arguments(requirements))
        assert result["ok"] is False
        assert result["error_code"] in expected_codes
        assert state.requirements is original

    maximum = [_requirement(f"R{index}") for index in range(MAX_REQUIREMENTS)]
    accepted = apply_requirements_action(state, _arguments(maximum))
    assert accepted["ok"] is True
    snapshot = state.requirements.to_snapshot()
    assert snapshot is not None
    assert len(snapshot) == MAX_REQUIREMENTS
    assert "authoritative document body" not in repr(snapshot)


def test_requirement_sources_require_successful_eligible_reads() -> None:
    state = TaskState()
    text_sources = (
        "requirements.txt",
        "design.md",
        "rules.json",
        "config.yaml",
        "more.yml",
    )
    for path in text_sources:
        state.record_execution_evidence(
            tool_name="read_file",
            result={"ok": True, "data": {"path": path}},
        )
    state.record_execution_evidence(
        tool_name="read_file",
        result={"ok": True, "data": {"path": "app.py"}},
    )
    state.record_execution_evidence(
        tool_name="read_document",
        result=_source_result("requirements.docx"),
    )
    state.record_execution_evidence(
        tool_name="read_document",
        result={"ok": False, "error_code": "malformed_document"},
    )

    assert state.read_source_paths == (*text_sources, "requirements.docx")
    accepted = apply_requirements_action(
        state,
        _arguments(
            [
                _requirement("R1", path="design.md"),
                _requirement("R2", path="requirements.docx"),
            ]
        ),
    )
    assert accepted["ok"] is True
    original = state.requirements

    rejected = apply_requirements_action(
        state,
        _arguments([_requirement(path="app.py")]),
    )
    assert rejected["ok"] is False
    assert rejected["error_code"] == "unobserved_requirement_source"
    assert state.requirements is original

    other_task = TaskState()
    isolated = apply_requirements_action(
        other_task,
        _arguments([_requirement(path="design.md")]),
    )
    assert isolated["ok"] is False
    assert isolated["error_code"] == "unobserved_requirement_source"


def test_visual_requirement_sources_use_separate_successful_read_eligibility() -> None:
    state = TaskState()
    unread = apply_requirements_action(
        state,
        _arguments([_requirement(path="1.png", locator="visible requirements text")]),
    )
    assert unread["error_code"] == "unobserved_requirement_source"

    state.record_execution_evidence(
        tool_name="read_image",
        result={
            "ok": True,
            "data": {
                "path": "1.png",
                "image_type": "png",
                "mime_type": "image/png",
                "size_bytes": 12,
            },
        },
    )
    accepted = apply_requirements_action(
        state,
        _arguments(
            [
                _requirement(
                    "V1",
                    path="1.png",
                    kind="functional",
                    locator="visible requirements text",
                ),
                _requirement(
                    "V2",
                    path="1.png",
                    kind="reference",
                    description="Use the visible sidebar and card layout",
                    locator="overall visible layout",
                ),
            ]
        ),
    )
    assert accepted["ok"] is True
    assert {item.kind for item in state.requirements.requirements} == {
        "functional",
        "reference",
    }
    snapshot = state.snapshot_for_model()
    assert snapshot is not None
    assert "read_visual_source_paths" not in snapshot
    assert "image/png" not in repr(snapshot)
    assert "base64" not in repr(snapshot)

    other_task = TaskState()
    rejected = apply_requirements_action(
        other_task,
        _arguments([_requirement(path="1.png")]),
    )
    assert rejected["error_code"] == "unobserved_requirement_source"


def test_web_requirement_sources_are_eligible_without_redirect_rewriting() -> None:
    requested = "https://competition.example/spec"
    final = "https://static.example/spec-v2"
    state = TaskState()
    unread = apply_requirements_action(
        state,
        _arguments([_web_requirement(url=requested)]),
    )
    assert unread["ok"] is False
    assert unread["error_code"] == "unobserved_requirement_source"

    state.record_execution_evidence(
        tool_name="read_webpage",
        result={
            "ok": True,
            "data": {
                "requested_url": requested,
                "final_url": final,
                "title": "Competition",
                "content_type": "text/html",
                "text": "private webpage body",
                "position": {
                    "start_cursor": 0,
                    "end_cursor": 20,
                    "total_chars": 20,
                },
                "truncated": False,
                "next_cursor": None,
            },
        },
    )
    assert state.read_source_urls == (requested, final)

    requested_result = apply_requirements_action(
        state,
        _arguments([_web_requirement(url=requested)]),
    )
    assert requested_result["ok"] is True
    requested_source = state.requirements.requirements[0].source
    assert requested_source.url == requested
    assert requested_source.path is None
    snapshot = state.snapshot_for_model()
    assert snapshot is not None
    assert snapshot["requirements"][0]["source"] == {
        "url": requested,
        "locator": "Acceptance Criteria",
    }
    assert "private webpage body" not in repr(snapshot)

    state.record_execution_evidence(
        tool_name="read_webpage",
        result={
            "ok": True,
            "data": {"requested_url": requested, "final_url": final},
        },
    )
    assert state.read_source_urls == (requested, final)
    assert state.requirements.requirements[0].source.url == requested

    final_result = apply_requirements_action(
        state,
        _arguments([_web_requirement(url=final)]),
    )
    assert final_result["ok"] is True
    assert state.requirements.requirements[0].source.url == final


def test_web_source_fifo_dedup_snapshot_and_task_isolation() -> None:
    state = TaskState()
    urls = [f"https://example.com/spec/{index}" for index in range(130)]
    for url in urls:
        state.record_execution_evidence(
            tool_name="read_webpage",
            result={
                "ok": True,
                "data": {"requested_url": url, "final_url": url},
            },
        )
    assert len(state.read_source_urls) == 128
    assert state.read_source_urls == tuple(urls[-128:])
    state.record_execution_evidence(
        tool_name="read_webpage",
        result={
            "ok": True,
            "data": {
                "requested_url": urls[-1],
                "final_url": urls[-1],
            },
        },
    )
    assert state.read_source_urls == tuple(urls[-128:])
    assert state.snapshot_for_model() is None

    other_task = TaskState()
    rejected = apply_requirements_action(
        other_task,
        _arguments([_web_requirement(url=urls[-1])]),
    )
    assert rejected["error_code"] == "unobserved_requirement_source"


def test_requirement_source_requires_exactly_one_path_or_url() -> None:
    state = TaskState(
        read_source_paths=("requirements.pdf",),
        read_source_urls=("https://example.com/spec",),
    )
    neither = _requirement()
    neither["source"] = {"locator": "section"}
    both = _requirement()
    both["source"] = {
        "path": "requirements.pdf",
        "url": "https://example.com/spec",
    }
    for item in (neither, both):
        result = apply_requirements_action(state, _arguments([item]))
        assert result["ok"] is False
        assert result["error_code"] == "invalid_arguments"


def test_requirement_bookkeeping_does_not_count_as_material_progress() -> None:
    state = TaskState(read_source_paths=("requirements.pdf",))
    result = apply_requirements_action(state, _arguments([_requirement()]))
    progress = ProgressState()
    decision = ProgressTracker().evaluate_turn(
        progress,
        before=ProgressFacts(0, "not_required"),
        after=ProgressFacts(0, "not_required"),
        actions=(
            ProgressAction(
                name=UPDATE_REQUIREMENTS_ACTION_NAME,
                arguments=_arguments([_requirement()]),
                result=result,
                workspace_revision=0,
                plan_before=None,
                plan_after=None,
            ),
        ),
    )
    assert decision == "continue"
    assert progress.no_progress_turns == 1


def test_repeated_webpage_chunk_does_not_refresh_material_progress() -> None:
    result = {
        "ok": True,
        "data": {
            "requested_url": "https://example.com/spec",
            "final_url": "https://example.com/spec",
            "title": "Spec",
            "content_type": "text/plain",
            "text": "same chunk",
            "position": {
                "start_cursor": 0,
                "end_cursor": 10,
                "total_chars": 10,
            },
            "truncated": False,
            "next_cursor": None,
        },
    }
    action = ProgressAction(
        name="read_webpage",
        arguments='{"url":"https://example.com/spec"}',
        result=result,
        workspace_revision=0,
        plan_before=None,
        plan_after=None,
    )
    tracker = ProgressTracker()
    progress = ProgressState()
    facts = ProgressFacts(0, "not_required")

    assert tracker.evaluate_turn(
        progress,
        before=facts,
        after=facts,
        actions=(action,),
    ) == "continue"
    assert progress.no_progress_turns == 0
    assert tracker.evaluate_turn(
        progress,
        before=facts,
        after=facts,
        actions=(action,),
    ) == "continue"
    assert progress.no_progress_turns == 1


def test_core_action_schema_retry_and_session_isolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner_module, "sleep", lambda _delay: None)
    temporary = ModelAPIError(
        "temporary_api_error",
        "temporary",
        classification="retryable",
    )
    client = FakeClient([temporary, ModelResponse(text="done")])
    registry = ToolRegistry(Workspace(tmp_path))

    result = AgentRunner(client, tools=registry).run("Read requirements")

    assert result.status == "completed"
    assert client.calls[0] == client.calls[1]
    names = [schema["function"]["name"] for schema in client.calls[0]["tools"]]
    assert names[:4] == [
        UPDATE_PLAN_ACTION_NAME,
        UPDATE_REQUIREMENTS_ACTION_NAME,
        "update_working_set",
        "update_review_findings",
    ]
    assert UPDATE_REQUIREMENTS_ACTION_NAME not in registry.names
    assert "read_document" in registry.names
    assert "read_webpage" in registry.names
    assert "read_image" in registry.names

    history = SessionHistory()
    history.add("Build from requirements.pdf", "Implemented R1 and verified it.")
    serialized = repr(history.snapshot())
    assert "authoritative document body" not in serialized
    assert "private webpage body" not in serialized
    assert "Implemented R1" in serialized


def test_source_guided_fake_client_uses_existing_coding_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_path = tmp_path / "requirements.docx"
    document = Document()
    document.add_paragraph("Replace old with new and run an automated test.")
    document.save(document_path)
    (tmp_path / "app.txt").write_text("old\n", encoding="utf-8")

    state = TaskState()
    monkeypatch.setattr(runner_module, "TaskState", lambda: state)
    plan_created = [
        {"id": "implement", "description": "Implement R1", "status": "in_progress"},
        {"id": "verify", "description": "Verify R1", "status": "pending"},
    ]
    plan_verifying = [
        {"id": "implement", "description": "Implement R1", "status": "completed"},
        {"id": "verify", "description": "Verify R1", "status": "in_progress"},
    ]
    plan_completed = [
        {"id": "implement", "description": "Implement R1", "status": "completed"},
        {"id": "verify", "description": "Verify R1", "status": "completed"},
    ]
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall("document", "read_document", '{"path":"requirements.docx"}')
                ]
            ),
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        "requirements",
                        UPDATE_REQUIREMENTS_ACTION_NAME,
                        _arguments(
                            [
                                _requirement(
                                    path="requirements.docx",
                                    description="Replace old with new and test it",
                                    locator="paragraph 1",
                                )
                            ]
                        ),
                    )
                ]
            ),
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        "plan",
                        UPDATE_PLAN_ACTION_NAME,
                        _plan_arguments(plan_created),
                    )
                ]
            ),
            ModelResponse(
                tool_calls=[
                    ToolCall("inspect", "read_file", '{"path":"app.txt"}')
                ]
            ),
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        "edit",
                        "edit_file",
                        '{"path":"app.txt","old_text":"old\\n","new_text":"new\\n"}',
                    )
                ]
            ),
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        "plan-progress",
                        UPDATE_PLAN_ACTION_NAME,
                        _plan_arguments(plan_verifying, mode="update"),
                    ),
                    ToolCall(
                        "verify",
                        "run_command",
                        json.dumps(
                            {"command": [sys.executable, "-c", "print('1 passed')"]}
                        ),
                    ),
                ]
            ),
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        "plan-complete",
                        UPDATE_PLAN_ACTION_NAME,
                        _plan_arguments(plan_completed, mode="update"),
                    )
                ]
            ),
            ModelResponse(text="Implemented R1 and verified it."),
        ]
    )

    result = AgentRunner(client, tools=ToolRegistry(Workspace(tmp_path))).run(
        "Implement requirements.docx and verify it"
    )

    assert result.status == "completed"
    assert result.verification_status == "verified"
    assert result.plan_status == "completed"
    assert (tmp_path / "app.txt").read_text(encoding="utf-8") == "new\n"
    assert state.requirements.requirements[0].source.path == "requirements.docx"
    matching_results = client.calls[6]["messages"][-2:]
    assert [message["tool_call_id"] for message in matching_results] == [
        "plan-progress",
        "verify",
    ]


def test_web_source_fake_client_uses_same_coding_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested = "https://competition.example/spec"
    final = "https://static.example/spec-v2"
    (tmp_path / "app.txt").write_text("old\n", encoding="utf-8")
    state = TaskState()
    monkeypatch.setattr(runner_module, "TaskState", lambda: state)

    class Adapter:
        def extract(self, _url: str) -> ExtractedWebPage:
            return ExtractedWebPage(
                requested_url=requested,
                final_url=final,
                title="Competition Requirements",
                content_type="text/plain",
                text="Replace old with web and run an automated check.",
            )

    plan_created = [
        {"id": "implement", "description": "Implement W1", "status": "in_progress"},
        {"id": "verify", "description": "Verify W1", "status": "pending"},
    ]
    plan_verifying = [
        {"id": "implement", "description": "Implement W1", "status": "completed"},
        {"id": "verify", "description": "Verify W1", "status": "in_progress"},
    ]
    plan_completed = [
        {"id": "implement", "description": "Implement W1", "status": "completed"},
        {"id": "verify", "description": "Verify W1", "status": "completed"},
    ]
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[ToolCall("web", "read_webpage", json.dumps({"url": requested}))]
            ),
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        "requirements",
                        UPDATE_REQUIREMENTS_ACTION_NAME,
                        _arguments([_web_requirement(url=requested)]),
                    )
                ]
            ),
            ModelResponse(
                tool_calls=[
                    ToolCall("plan", UPDATE_PLAN_ACTION_NAME, _plan_arguments(plan_created))
                ]
            ),
            ModelResponse(
                tool_calls=[ToolCall("inspect", "read_file", '{"path":"app.txt"}')]
            ),
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        "edit",
                        "edit_file",
                        '{"path":"app.txt","old_text":"old\\n","new_text":"web\\n"}',
                    )
                ]
            ),
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        "plan-progress",
                        UPDATE_PLAN_ACTION_NAME,
                        _plan_arguments(plan_verifying, mode="update"),
                    ),
                    ToolCall(
                        "verify",
                        "run_command",
                        json.dumps({"command": [sys.executable, "-c", "print('1 passed')"]}),
                    ),
                ]
            ),
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        "plan-complete",
                        UPDATE_PLAN_ACTION_NAME,
                        _plan_arguments(plan_completed, mode="update"),
                    )
                ]
            ),
            ModelResponse(text="Implemented W1 from the requested URL and verified it."),
        ]
    )
    registry = ToolRegistry(Workspace(tmp_path), webpage_adapter=Adapter())  # type: ignore[arg-type]

    result = AgentRunner(client, tools=registry).run(
        f"Implement the requirements at {requested} and verify them"
    )

    assert result.status == "completed"
    assert result.verification_status == "verified"
    assert result.plan_status == "completed"
    assert (tmp_path / "app.txt").read_text(encoding="utf-8") == "web\n"
    assert state.requirements.requirements[0].source.url == requested
    assert state.read_source_urls == (requested, final)


def test_web_read_and_requirement_action_share_sequential_turn_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested = "https://example.com/spec"
    state = TaskState()
    monkeypatch.setattr(runner_module, "TaskState", lambda: state)

    class Adapter:
        def extract(self, _url: str) -> ExtractedWebPage:
            return ExtractedWebPage(
                requested,
                requested,
                "Spec",
                "text/plain",
                "Requirement body",
            )

    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall("web", "read_webpage", json.dumps({"url": requested})),
                    ToolCall(
                        "requirements",
                        UPDATE_REQUIREMENTS_ACTION_NAME,
                        _arguments([_web_requirement(url=requested)]),
                    ),
                ]
            ),
            ModelResponse(text="Requirements recorded."),
        ]
    )
    result = AgentRunner(
        client,
        tools=ToolRegistry(
            Workspace(tmp_path),
            webpage_adapter=Adapter(),  # type: ignore[arg-type]
        ),
    ).run(f"Read {requested}")

    assert result.status == "completed"
    assert state.requirements.requirements[0].source.url == requested
    observations = client.calls[1]["messages"][-2:]
    assert [message["tool_call_id"] for message in observations] == [
        "web",
        "requirements",
    ]


def test_prompt_fixes_source_and_verification_policy() -> None:
    prompt = SYSTEM_PROMPT
    assert "next_cursor" in prompt
    assert "不得声称已经理解整份文档" in prompt
    for kind in ("functional", "constraint", "acceptance", "reference"):
        assert kind in prompt
    assert "来源需求与 Agent 的设计决策" in prompt
    assert "TaskPlan" in prompt and "编码循环" in prompt
    assert "已实现但未自动验证" in prompt
    assert "read_webpage" in prompt
    assert "不得发现链接、爬取站点" in prompt
    assert "requested_url" in prompt and "final_url" in prompt
    assert "read_image" in prompt
    assert "文件名只是来源标签" in prompt
    assert "同一张图片" in prompt
    assert "只包含 `read_image` 调用" in prompt
    assert "其他 Tool 或 Core Action" in prompt
