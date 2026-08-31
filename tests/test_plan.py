from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import codeloop.agent.runner as agent_module
from codeloop.agent.context import ConversationContext
from codeloop.agent.plan import UPDATE_PLAN_ACTION_NAME, apply_plan_action
from codeloop.agent.runner import AgentRunner, _FailureTracker
from codeloop.agent.task_state import TaskState
from codeloop.execution.tools import ToolRegistry
from codeloop.execution.workspace import Workspace
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


def _step(
    step_id: str,
    description: str,
    status: str,
    blocked_reason: str | None = None,
) -> dict[str, Any]:
    step = {
        "id": step_id,
        "description": description,
        "status": status,
    }
    if blocked_reason is not None:
        step["blocked_reason"] = blocked_reason
    return step


def _action(mode: str, steps: list[dict[str, Any]], **extra: Any) -> str:
    return json.dumps(
        {"mode": mode, "steps": steps, **extra},
        ensure_ascii=False,
    )


def _tool_call(call_id: str, arguments: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        name=UPDATE_PLAN_ACTION_NAME,
        arguments=arguments,
    )


def test_plan_create_update_add_block_and_noop() -> None:
    state = TaskState()
    created = apply_plan_action(
        state,
        _action("create", [_step("inspect", "Inspect project", "in_progress")]),
    )
    assert created["ok"] is True
    assert created["data"]["revision"] == 1
    assert state.plan is not None
    assert state.plan.last_explanation is None

    duplicate_create = apply_plan_action(
        state,
        _action("create", [_step("inspect", "Inspect project", "in_progress")]),
    )
    assert duplicate_create["ok"] is False
    assert duplicate_create["error_code"] == "plan_already_exists"

    rejected_reasoning = apply_plan_action(
        state,
        _action(
            "update",
            [_step("inspect", "Inspect project", "in_progress")],
            explanation="Ordinary update reasoning must not be stored.",
        ),
    )
    assert rejected_reasoning["ok"] is False
    assert rejected_reasoning["error_code"] == "invalid_arguments"
    assert state.plan.revision == 1

    updated_steps = [
        _step("inspect", "Inspect project", "completed"),
        _step("implement", "Implement change", "in_progress"),
    ]
    updated = apply_plan_action(state, _action("update", updated_steps))
    assert updated == {
        "ok": True,
        "data": {
            "changed": True,
            "mode": "update",
            "revision": 2,
            "plan_status": "active",
            "changed_step_ids": ["inspect", "implement"],
        },
    }

    no_op = apply_plan_action(state, _action("update", updated_steps))
    assert no_op["ok"] is True
    assert no_op["data"]["changed"] is False
    assert state.plan.revision == 2

    pending_again = [
        _step("inspect", "Inspect project", "completed"),
        _step("implement", "Implement change", "pending"),
    ]
    assert apply_plan_action(state, _action("update", pending_again))["ok"] is True
    blocked = [
        _step("inspect", "Inspect project", "completed"),
        _step("implement", "Implement change", "blocked", "Missing dependency"),
    ]
    result = apply_plan_action(state, _action("update", blocked))
    assert result["ok"] is True
    assert result["data"]["plan_status"] == "terminal_with_blocks"
    assert state.plan.last_explanation is None


def test_terminal_steps_require_explicit_replan_and_remain_present() -> None:
    state = TaskState()
    terminal = [
        _step("done", "Finished stage", "completed"),
        _step("blocked", "Blocked stage", "blocked", "Unavailable input"),
    ]
    assert apply_plan_action(state, _action("create", terminal))["ok"] is True

    reopened = [
        _step("done", "Finished stage", "in_progress"),
        _step("blocked", "Blocked stage", "pending"),
    ]
    ordinary = apply_plan_action(state, _action("update", reopened))
    assert ordinary["ok"] is False
    assert ordinary["error_code"] == "invalid_plan_transition"
    assert state.plan is not None and state.plan.revision == 1

    missing_reason = apply_plan_action(state, _action("replan", reopened))
    assert missing_reason["ok"] is False
    assert state.plan.last_explanation is None

    missing_terminal = apply_plan_action(
        state,
        _action(
            "replan",
            [_step("done", "Finished stage", "in_progress")],
            explanation="New evidence makes the work actionable.",
        ),
    )
    assert missing_terminal["ok"] is False
    assert missing_terminal["error_code"] == "invalid_plan_transition"

    replanned = apply_plan_action(
        state,
        _action(
            "replan",
            reopened,
            explanation="New evidence makes both stages actionable.",
        ),
    )
    assert replanned["ok"] is True
    assert replanned["data"]["revision"] == 2
    assert state.plan.last_explanation == (
        "New evidence makes both stages actionable."
    )

    no_op = apply_plan_action(
        state,
        _action(
            "replan",
            reopened,
            explanation="A different explanation without a plan change.",
        ),
    )
    assert no_op["ok"] is True
    assert no_op["data"]["changed"] is False
    assert state.plan.revision == 2
    assert state.plan.last_explanation == (
        "New evidence makes both stages actionable."
    )


@pytest.mark.parametrize(
    ("arguments", "error_code"),
    [
        ("not-json", "invalid_arguments"),
        (_action("create", []), "invalid_plan"),
        (
            _action(
                "create",
                [
                    _step("same", "One", "pending"),
                    _step("same", "Two", "pending"),
                ],
            ),
            "duplicate_step_id",
        ),
        (_action("create", [_step("one", " ", "pending")]), "invalid_plan"),
        (_action("create", [_step("one", "One", "unknown")]), "invalid_plan"),
        (
            _action(
                "create",
                [
                    _step("one", "One", "in_progress"),
                    _step("two", "Two", "in_progress"),
                ],
            ),
            "invalid_plan",
        ),
        (
            _action("create", [_step("one", "One", "blocked")]),
            "invalid_plan",
        ),
        (
            _action("create", [_step("one", "One", "pending", "stale")]),
            "invalid_plan",
        ),
    ],
)
def test_invalid_plan_actions_are_recoverable(
    arguments: str,
    error_code: str,
) -> None:
    state = TaskState()
    result = apply_plan_action(state, arguments)
    assert result["ok"] is False
    assert result["error_code"] == error_code
    assert state.plan is None


def test_update_requires_an_existing_plan() -> None:
    result = apply_plan_action(
        TaskState(),
        _action("update", [_step("one", "One", "pending")]),
    )
    assert result["ok"] is False
    assert result["error_code"] == "plan_not_found"


def test_context_replaces_current_plan_and_trims_only_complete_cycles() -> None:
    context = ConversationContext(
        "base system",
        "original task",
        max_chars=1_600,
        max_messages=10,
    )
    assert context.messages_for_model() == [
        {"role": "system", "content": "base system"},
        {"role": "user", "content": "original task"},
    ]

    for index in range(4):
        call_id = f"call-{index}"
        context.add_tool_cycle(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
            [
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": "read_file",
                    "content": "x" * 250,
                }
            ],
        )

    context.set_runtime_state(
        {"plan": {"revision": 1, "status": "active", "steps": []}}
    )
    first_system = context.messages_for_model()[0]
    assert first_system["role"] == "system"
    assert first_system["content"].startswith("base system")
    assert '"revision":1' in first_system["content"]

    context.set_runtime_state(
        {"plan": {"revision": 2, "status": "completed", "steps": []}}
    )
    messages = context.messages_for_model()
    assert messages[0]["content"].count("Runtime task state:") == 1
    assert '"revision":2' in messages[0]["content"]
    assert '"revision":1' not in messages[0]["content"]
    assert {"role": "user", "content": "original task"} in messages

    declared_ids = [
        call["id"]
        for message in messages
        if message["role"] == "assistant"
        for call in message["tool_calls"]
    ]
    result_ids = [
        message["tool_call_id"]
        for message in messages
        if message["role"] == "tool"
    ]
    assert declared_ids
    assert result_ids == declared_ids
    assert len(
        json.dumps(
            messages,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    ) <= 1_600


def test_oversized_runtime_state_is_pinned_and_marks_overflow() -> None:
    context = ConversationContext(
        "base system",
        "original task",
        max_chars=1_000,
        max_messages=5,
    )
    context.set_runtime_state(
        {"plan": {"revision": 1, "description": "x" * 1_200}}
    )

    messages = context.messages_for_model()
    assert "Runtime task state:" in messages[0]["content"]
    notice = json.loads(messages[1]["content"].split(": ", 1)[1])
    assert notice["overflow"] is True
    assert notice["conversation_history_trimmed"] is False
    assert messages[2] == {"role": "user", "content": "original task"}


def test_plan_guided_react_preserves_multi_action_cycles(
    tmp_path: Path,
) -> None:
    (tmp_path / "sample.txt").write_text("hello\n", encoding="utf-8")
    create = _action(
        "create",
        [
            _step("inspect", "Inspect sample", "in_progress"),
            _step("finish", "Finish task", "pending"),
        ],
    )
    complete = _action(
        "update",
        [
            _step("inspect", "Inspect sample", "completed"),
            _step("finish", "Finish task", "completed"),
        ],
    )
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[
                    _tool_call("plan-create", create),
                    ToolCall(
                        id="read-sample",
                        name="read_file",
                        arguments='{"path":"sample.txt"}',
                    ),
                ]
            ),
            ModelResponse(tool_calls=[_tool_call("plan-complete", complete)]),
            ModelResponse(text="Finished."),
        ]
    )
    tool_events: list[str] = []
    result = AgentRunner(
        client,
        tools=ToolRegistry(Workspace(tmp_path)),
        on_tool_event=lambda event: tool_events.append(event.tool_call.name),
    ).run("Complete a small multi-step task.")

    assert result.status == "completed"
    assert result.answer == "Finished."
    assert tool_events == ["read_file"]
    assert len(client.calls) == 3
    assert "Runtime task state:" not in client.calls[0]["messages"][0]["content"]
    assert '"revision":1' in client.calls[1]["messages"][0]["content"]
    assert '"revision":2' in client.calls[2]["messages"][0]["content"]
    assert '"status":"completed"' in client.calls[2]["messages"][0]["content"]
    assert client.calls[1]["messages"][1] == {
        "role": "user",
        "content": "Complete a small multi-step task.",
    }

    first_cycle = client.calls[1]["messages"][-3:]
    assert [call["id"] for call in first_cycle[0]["tool_calls"]] == [
        "plan-create",
        "read-sample",
    ]
    assert [message["tool_call_id"] for message in first_cycle[1:]] == [
        "plan-create",
        "read-sample",
    ]
    schema_names = [
        schema["function"]["name"] for schema in client.calls[0]["tools"]
    ]
    assert schema_names[0] == UPDATE_PLAN_ACTION_NAME
    assert schema_names.count(UPDATE_PLAN_ACTION_NAME) == 1
    assert len(schema_names) == 14


def test_agent_result_exposes_blocked_plan_facts(tmp_path: Path) -> None:
    blocked = _action(
        "create",
        [
            _step(
                "blocked-step",
                "Use an unavailable external service",
                "blocked",
                blocked_reason="The required service is unavailable.",
            )
        ],
    )
    client = FakeClient(
        [
            ModelResponse(tool_calls=[_tool_call("plan-blocked", blocked)]),
            ModelResponse(text="Cannot continue without the service."),
        ]
    )

    result = AgentRunner(
        client,
        tools=ToolRegistry(Workspace(tmp_path)),
    ).run("Use the required service.")

    assert result.status == "completed"
    assert result.plan_status == "terminal_with_blocks"
    assert result.unfinished_steps == ()
    assert [step.id for step in result.blocked_steps] == ["blocked-step"]
    assert result.verification_status == "not_required"


def test_plan_snapshot_and_schemas_are_stable_across_api_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temporary = ModelAPIError(
        "temporary_api_error",
        "temporary",
        classification="retryable",
    )
    create = _action(
        "create",
        [_step("inspect", "Inspect project", "in_progress")],
    )
    client = FakeClient(
        [
            ModelResponse(tool_calls=[_tool_call("plan-create", create)]),
            temporary,
            temporary,
            ModelResponse(text="Recovered."),
            ModelResponse(text="Accepted with unfinished plan work."),
        ]
    )
    delays: list[float] = []
    monkeypatch.setattr(agent_module, "sleep", delays.append)

    result = AgentRunner(
        client,
        tools=ToolRegistry(Workspace(tmp_path)),
    ).run("Use a plan, then retry.")

    assert result.status == "completed"
    assert delays == [0.5, 1.0]
    retry_calls = client.calls[1:4]
    assert retry_calls[0] == retry_calls[1] == retry_calls[2]
    assert '"revision":1' in retry_calls[0]["messages"][0]["content"]
    assert "completion_review" in client.calls[4]["messages"][0]["content"]


def test_identical_update_plan_failures_use_existing_termination(
    tmp_path: Path,
) -> None:
    invalid = _tool_call("bad-1", "not-json")
    client = FakeClient(
        [
            ModelResponse(tool_calls=[invalid]),
            ModelResponse(tool_calls=[_tool_call("bad-2", "not-json")]),
            ModelResponse(tool_calls=[_tool_call("bad-3", "not-json")]),
        ]
    )
    result = AgentRunner(
        client,
        tools=ToolRegistry(Workspace(tmp_path)),
    ).run("Keep submitting the same invalid plan action.")

    assert result.status == "repeated_failure"
    assert result.steps == 3
    observations = [
        json.loads(call["messages"][-1]["content"])
        for call in client.calls[1:]
    ]
    assert all(item["error_code"] == "invalid_arguments" for item in observations)


def test_plan_failure_fingerprint_ignores_observation_details() -> None:
    tracker = _FailureTracker()
    arguments = '{"mode":"update","steps":[]}'
    calls = [
        _tool_call("one", arguments),
        _tool_call("two", '{"steps":[],"mode":"update"}'),
        _tool_call("three", arguments),
    ]
    results = [
        {
            "ok": False,
            "error_code": "invalid_plan",
            "message": "first wording",
            "data": {"revision": 1},
        },
        {
            "ok": False,
            "error_code": "invalid_plan",
            "message": "second wording",
            "data": {"revision": 2},
        },
        {
            "ok": False,
            "error_code": "invalid_plan",
            "message": "third wording",
            "data": {"format": "different"},
        },
    ]

    assert tracker.record(calls[0], results[0]) is False
    assert tracker.record(calls[1], results[1]) is False
    assert tracker.record(calls[2], results[2]) is True


def test_simple_task_stays_on_the_existing_react_path(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("CodeLoop\n", encoding="utf-8")
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="read-readme",
                        name="read_file",
                        arguments='{"path":"README.md"}',
                    )
                ]
            ),
            ModelResponse(text="The project is CodeLoop."),
        ]
    )
    result = AgentRunner(
        client,
        tools=ToolRegistry(Workspace(tmp_path)),
    ).run("Read the project name.")

    assert result.status == "completed"
    assert all(
        "Runtime task state:" not in call["messages"][0]["content"]
        for call in client.calls
    )


def test_execution_registry_cannot_claim_reserved_action(tmp_path: Path) -> None:
    registry = ToolRegistry(Workspace(tmp_path))
    registry._tools[UPDATE_PLAN_ACTION_NAME] = next(iter(registry._tools.values()))
    with pytest.raises(ValueError, match="reserved action"):
        AgentRunner(FakeClient([]), tools=registry)
