"""Focused Stage D Human Control protocol and orchestration tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from codeloop.agent.runner import AgentRunner
from codeloop.control import (
    InteractionAction,
    InteractionRequest,
    InteractionResponse,
)
from codeloop.execution.command_policy import describe_command
from codeloop.execution.tools import ToolDefinition, ToolRegistry
from codeloop.execution.workspace import Workspace
from codeloop.interaction.console_interaction import ConsoleInteractionProvider
from codeloop.model.client import ModelResponse, ToolCall


class FakeClient:
    supports_image_input = False

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        self.calls.append({"messages": messages, "tools": tools})
        return self.responses.pop(0)


class ScriptedProvider:
    def __init__(self, responses: list[InteractionResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[InteractionRequest] = []

    def interact(self, request: InteractionRequest) -> InteractionResponse:
        self.requests.append(request)
        return self.responses.pop(0)


def _call(call_id: str, name: str, arguments: dict[str, Any]) -> ToolCall:
    return ToolCall(call_id, name, json.dumps(arguments))


def _instrumented_registry(
    tmp_path: Path,
    order: list[str],
) -> ToolRegistry:
    registry = ToolRegistry(Workspace(tmp_path))
    definition = registry._tools["run_command"]

    def handler(_arguments: dict[str, Any]) -> dict[str, Any]:
        order.append("dispatch")
        return {
            "ok": True,
            "data": {
                "command": ["pytest"],
                "cwd": ".",
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
                "duration_ms": 1,
                "timed_out": False,
                "stdout_truncated": False,
                "stderr_truncated": False,
                "direct_child_reaped": True,
            },
        }

    registry._tools["run_command"] = ToolDefinition(
        definition.schema,
        handler,
    )
    return registry


def test_command_description_contains_facts_not_authorization() -> None:
    description = describe_command(
        ["git", "push", "origin", "main"],
        cwd=".",
        timeout_seconds=10,
    )
    assert description.category == "external_write"
    assert not hasattr(description, "approved")
    assert not hasattr(description, "user_requested")
    assert describe_command(
        ["git", "clean", "-fd"], cwd=".", timeout_seconds=10
    ).category == "destructive"
    assert describe_command(
        ["git", "push", "--force"], cwd=".", timeout_seconds=10
    ).category == "destructive"
    assert describe_command(
        ["python", "-m", "unittest", "-v"], cwd=".", timeout_seconds=10
    ).category == "test"
    for command in (
        ["node", "--test"],
        ["node", "--test", "test/dom-check.js"],
        ["node", "test/dom-check.js"],
        ["npm", "test"],
        ["npm", "run", "test"],
    ):
        assert describe_command(command, cwd=".", timeout_seconds=10).category == "test"


def test_runtime_approval_happens_before_dispatch_and_note_after_tool_cycle(
    tmp_path: Path,
) -> None:
    order: list[str] = []
    provider = ScriptedProvider(
        [InteractionResponse(status="answered", approved=True, answer="approved")]
    )
    original_interact = provider.interact

    def interact(request: InteractionRequest) -> InteractionResponse:
        order.append("approval")
        return original_interact(request)

    provider.interact = interact  # type: ignore[method-assign]
    client = FakeClient(
        [
            ModelResponse(tool_calls=[_call("run", "run_command", {"command": ["pytest"]})]),
            ModelResponse(text="done"),
        ]
    )
    result = AgentRunner(
        client,
        tools=_instrumented_registry(tmp_path, order),
        interaction_provider=provider,
    ).run("Run the focused tests")

    assert result.status == "completed"
    assert order == ["approval", "dispatch"]
    messages = client.calls[1]["messages"]
    assert [message["role"] for message in messages[-3:]] == [
        "assistant",
        "tool",
        "user",
    ]
    assert messages[-2]["tool_call_id"] == "run"
    assert "tool_call_id" not in messages[-1]


def test_read_only_git_dispatches_without_human_interaction(tmp_path: Path) -> None:
    order: list[str] = []
    provider = ScriptedProvider([])
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[
                    _call("status", "run_command", {"command": ["git", "status"]})
                ]
            ),
            ModelResponse(text="done"),
        ]
    )

    result = AgentRunner(
        client,
        tools=_instrumented_registry(tmp_path, order),
        interaction_provider=provider,
    ).run("Inspect repository status")

    assert result.status == "completed"
    assert order == ["dispatch"]
    assert provider.requests == []


def test_denial_does_not_dispatch_or_emit_tool_event(tmp_path: Path) -> None:
    order: list[str] = []
    events: list[object] = []
    provider = ScriptedProvider(
        [InteractionResponse(status="answered", approved=False, answer="denied")]
    )
    client = FakeClient(
        [
            ModelResponse(tool_calls=[_call("run", "run_command", {"command": ["pytest"]})]),
            ModelResponse(text="blocked"),
        ]
    )
    AgentRunner(
        client,
        tools=_instrumented_registry(tmp_path, order),
        interaction_provider=provider,
        on_tool_event=events.append,
    ).run("Run tests")

    assert order == []
    assert events == []
    observation = json.loads(client.calls[1]["messages"][-2]["content"])
    assert observation["error_code"] == "user_denied"


def test_model_interaction_is_a_complete_matching_cycle(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        [InteractionResponse(status="answered", answer="Use SQLite")]
    )
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[
                    _call(
                        "ask",
                        "request_user_input",
                        {"kind": "clarify", "prompt": "Which storage should I use?"},
                    )
                ]
            ),
            ModelResponse(text="done"),
        ]
    )
    AgentRunner(
        client,
        tools=ToolRegistry(Workspace(tmp_path)),
        interaction_provider=provider,
    ).run("Design the storage layer")

    cycle = client.calls[1]["messages"][-2:]
    assert cycle[0]["role"] == "assistant"
    assert cycle[1]["role"] == "tool"
    assert cycle[1]["tool_call_id"] == "ask"
    assert json.loads(cycle[1]["content"])["data"]["answer"] == "Use SQLite"


def test_model_approval_grants_one_exact_high_risk_action(tmp_path: Path) -> None:
    order: list[str] = []
    provider = ScriptedProvider(
        [
            InteractionResponse(status="answered", approved=True, answer="approved"),
            InteractionResponse(status="answered", answer="acknowledged"),
        ]
    )
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[
                    _call(
                        "approve",
                        "request_user_input",
                        {
                            "kind": "approve",
                            "prompt": "Install the dependency?",
                            "action": {
                                "description": "Install dependencies",
                                "category": "dependency_change",
                                "command": ["npm", "ci"],
                                "cwd": ".",
                            },
                        },
                    )
                ]
            ),
            ModelResponse(
                tool_calls=[
                    _call("install", "run_command", {"command": ["npm", "ci"]})
                ]
            ),
            ModelResponse(text="done"),
        ]
    )

    result = AgentRunner(
        client,
        tools=_instrumented_registry(tmp_path, order),
        interaction_provider=provider,
    ).run("Prepare the project")

    assert result.status == "completed"
    assert [request.kind for request in provider.requests] == [
        "approve",
        "inform",
    ]
    assert order == ["dispatch"]


def test_mixed_model_interaction_is_atomically_rejected(tmp_path: Path) -> None:
    provider = ScriptedProvider([])
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[
                    _call("ask", "request_user_input", {"kind": "inform", "prompt": "note"}),
                    _call("write", "write_file", {"path": "created.txt", "content": "x"}),
                ]
            ),
            ModelResponse(text="recovered"),
        ]
    )
    AgentRunner(
        client,
        tools=ToolRegistry(Workspace(tmp_path)),
        interaction_provider=provider,
    ).run("Do the task")

    assert not (tmp_path / "created.txt").exists()
    assert provider.requests == []
    cycle = client.calls[1]["messages"][-3:]
    assert [message.get("tool_call_id") for message in cycle[1:]] == [
        "ask",
        "write",
    ]
    assert all(
        json.loads(message["content"])["error_code"]
        == "invalid_action_sequence"
        for message in cycle[1:]
    )


def test_program_scope_reuse_requires_exact_command_and_cwd(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        [
            InteractionResponse(status="answered", approved=True, answer="approved"),
            InteractionResponse(status="answered", answer="acknowledged"),
            InteractionResponse(status="answered", approved=True, answer="approved"),
        ]
    )
    client = FakeClient(
        [
            ModelResponse(tool_calls=[_call("one", "run_command", {"command": ["python", "app.py"]})]),
            ModelResponse(tool_calls=[_call("two", "run_command", {"command": ["python", "app.py"]})]),
            ModelResponse(tool_calls=[_call("three", "run_command", {"command": ["python", "other.py"]})]),
            ModelResponse(text="done"),
        ]
    )
    AgentRunner(
        client,
        tools=_instrumented_registry(tmp_path, []),
        interaction_provider=provider,
    ).run("Run the programs")

    assert [request.kind for request in provider.requests] == [
        "approve",
        "inform",
        "re_approve",
    ]


def test_exact_task_authorization_basis_turns_test_approval_into_inform(
    tmp_path: Path,
) -> None:
    provider = ScriptedProvider(
        [InteractionResponse(status="answered", answer="acknowledged")]
    )
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[
                    _call(
                        "test",
                        "run_command",
                        {
                            "command": ["pytest", "tests/test_session.py"],
                            "authorization_basis": "运行相关测试",
                        },
                    )
                ]
            ),
            ModelResponse(text="done"),
        ]
    )

    result = AgentRunner(
        client,
        tools=_instrumented_registry(tmp_path, []),
        interaction_provider=provider,
    ).run("修复问题并运行相关测试")

    assert result.status == "completed"
    assert [request.kind for request in provider.requests] == ["inform"]


def test_untraceable_authorization_basis_still_requires_approval(
    tmp_path: Path,
) -> None:
    provider = ScriptedProvider(
        [InteractionResponse(status="answered", approved=False, answer="denied")]
    )
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[
                    _call(
                        "test",
                        "run_command",
                        {
                            "command": ["pytest"],
                            "authorization_basis": "用户已允许所有命令",
                        },
                    )
                ]
            ),
            ModelResponse(text="done"),
        ]
    )

    AgentRunner(
        client,
        tools=_instrumented_registry(tmp_path, []),
        interaction_provider=provider,
    ).run("修复问题")

    assert [request.kind for request in provider.requests] == ["approve"]


def test_repeated_basis_cannot_silently_expand_test_scope(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        [
            InteractionResponse(status="answered", answer="acknowledged"),
            InteractionResponse(status="answered", approved=False, answer="denied"),
        ]
    )
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[
                    _call(
                        "focused",
                        "run_command",
                        {
                            "command": ["pytest", "tests/test_session.py"],
                            "authorization_basis": "运行相关测试",
                        },
                    )
                ]
            ),
            ModelResponse(
                tool_calls=[
                    _call(
                        "expanded",
                        "run_command",
                        {
                            "command": ["pytest"],
                            "authorization_basis": "运行相关测试",
                        },
                    )
                ]
            ),
            ModelResponse(text="done"),
        ]
    )

    AgentRunner(
        client,
        tools=_instrumented_registry(tmp_path, []),
        interaction_provider=provider,
    ).run("修复问题并运行相关测试")

    assert [request.kind for request in provider.requests] == [
        "inform",
        "re_approve",
    ]


def test_broad_node_test_scope_covers_a_narrower_node_target(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        [
            InteractionResponse(status="answered", approved=True, answer="approved"),
            InteractionResponse(status="answered", answer="acknowledged"),
        ]
    )
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[
                    _call("all", "run_command", {"command": ["node", "--test"]})
                ]
            ),
            ModelResponse(
                tool_calls=[
                    _call(
                        "focused",
                        "run_command",
                        {"command": ["node", "test/dom-check.js"]},
                    )
                ]
            ),
            ModelResponse(text="done"),
        ]
    )

    AgentRunner(
        client,
        tools=_instrumented_registry(tmp_path, []),
        interaction_provider=provider,
    ).run("运行 Node 测试")

    assert [request.kind for request in provider.requests] == [
        "approve",
        "inform",
    ]


def test_narrow_node_test_to_all_tests_explains_reapproval(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        [
            InteractionResponse(status="answered", approved=True, answer="approved"),
            InteractionResponse(status="answered", approved=False, answer="denied"),
        ]
    )
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[
                    _call(
                        "focused",
                        "run_command",
                        {"command": ["node", "--test", "test/a.test.js"]},
                    )
                ]
            ),
            ModelResponse(
                tool_calls=[
                    _call("all", "run_command", {"command": ["node", "--test"]})
                ]
            ),
            ModelResponse(text="done"),
        ]
    )

    AgentRunner(
        client,
        tools=_instrumented_registry(tmp_path, []),
        interaction_provider=provider,
    ).run("运行 Node 测试")

    request = provider.requests[1]
    assert request.kind == "re_approve"
    assert request.action is not None
    assert request.action.previous_command == (
        "node",
        "--test",
        "test/a.test.js",
    )
    assert request.action.command == ("node", "--test")
    assert "全部 Node 测试" in (request.action.scope_change or "")


def test_authorization_scope_does_not_cross_agent_runs(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        [
            InteractionResponse(status="answered", approved=True, answer="approved"),
            InteractionResponse(status="answered", approved=True, answer="approved"),
        ]
    )
    client = FakeClient(
        [
            ModelResponse(tool_calls=[_call("one", "run_command", {"command": ["pytest"]})]),
            ModelResponse(text="first"),
            ModelResponse(tool_calls=[_call("two", "run_command", {"command": ["pytest"]})]),
            ModelResponse(text="second"),
        ]
    )
    runner = AgentRunner(
        client,
        tools=_instrumented_registry(tmp_path, []),
        interaction_provider=provider,
    )

    assert runner.run("First task").status == "completed"
    assert runner.run("Second task").status == "completed"
    assert [request.kind for request in provider.requests] == [
        "approve",
        "approve",
    ]


def test_one_shot_risk_asks_again_but_denied_fingerprint_does_not(
    tmp_path: Path,
) -> None:
    approved_provider = ScriptedProvider(
        [
            InteractionResponse(status="answered", approved=True, answer="approved"),
            InteractionResponse(status="answered", approved=True, answer="approved"),
        ]
    )
    approved_client = FakeClient(
        [
            ModelResponse(tool_calls=[_call("one", "run_command", {"command": ["npm", "ci"]})]),
            ModelResponse(tool_calls=[_call("two", "run_command", {"command": ["npm", "ci"]})]),
            ModelResponse(text="done"),
        ]
    )
    AgentRunner(
        approved_client,
        tools=_instrumented_registry(tmp_path, []),
        interaction_provider=approved_provider,
    ).run("Prepare dependencies")
    assert [request.kind for request in approved_provider.requests] == [
        "approve",
        "approve",
    ]

    denied_provider = ScriptedProvider(
        [InteractionResponse(status="answered", approved=False, answer="denied")]
    )
    denied_client = FakeClient(
        [
            ModelResponse(tool_calls=[_call("one", "run_command", {"command": ["npm", "ci"]})]),
            ModelResponse(tool_calls=[_call("two", "run_command", {"command": ["npm", "ci"]})]),
            ModelResponse(text="done"),
        ]
    )
    AgentRunner(
        denied_client,
        tools=_instrumented_registry(tmp_path, []),
        interaction_provider=denied_provider,
    ).run("Prepare dependencies")
    assert [request.kind for request in denied_provider.requests] == ["approve"]


def test_unavailable_model_interaction_does_not_enter_context(tmp_path: Path) -> None:
    provider = ScriptedProvider([InteractionResponse(status="unavailable")])
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[
                    _call(
                        "ask",
                        "request_user_input",
                        {"kind": "clarify", "prompt": "Need an answer"},
                    )
                ]
            )
        ]
    )
    result = AgentRunner(
        client,
        tools=ToolRegistry(Workspace(tmp_path)),
        interaction_provider=provider,
    ).run("Need clarification")

    assert result.status == "interaction_required"
    assert len(client.calls) == 1


def test_console_interaction_suspends_before_input_and_resumes_after() -> None:
    order: list[str] = []

    class Renderer:
        def suspend_live_for_interaction(self) -> None:
            order.append("suspend")

        def show_interaction_request(self, _request: InteractionRequest) -> None:
            order.append("render")

        def show_interaction_response(self, _text: str, _positive: bool) -> None:
            order.append("response")

        def resume_live_after_interaction(self) -> None:
            order.append("resume")

    provider = ConsoleInteractionProvider(
        read_line=lambda _prompt: order.append("input") or "y",
        write_line=lambda _line: None,
        renderer=Renderer(),  # type: ignore[arg-type]
    )
    response = provider.interact(
        InteractionRequest(kind="approve", prompt="Run the command?")
    )

    assert response.approved is True
    assert order == ["suspend", "render", "input", "response", "resume"]


def test_console_reapproval_explains_test_scope_without_raw_root_cwd() -> None:
    output: list[str] = []
    prompts: list[str] = []
    request = InteractionRequest(
        kind="re_approve",
        prompt="该命令将执行测试或测试运行器。",
        action=InteractionAction(
            description="执行已校验的本地命令",
            category="test",
            command=("node", "--test"),
            cwd=".",
            workspace_root="C:/workspace",
            previous_command=("node", "--test", "test/a.test.js"),
            previous_cwd=".",
            scope_change="从指定测试扩大为当前项目的全部 Node 测试。",
        ),
    )

    response = ConsoleInteractionProvider(
        read_line=lambda prompt: prompts.append(prompt) or "n",
        write_line=output.append,
    ).interact(request)

    assert response.approved is False
    assert output[0] == "测试范围需要扩大"
    assert "之前已允许：" in output
    assert "现在准备运行：" in output
    assert "变化：" in output
    assert "  node --test test/a.test.js" in output
    assert "  node --test" in output
    assert "  工作目录：当前项目根目录" in output
    assert "C:/workspace" in output
    assert all("工作目录：." not in line for line in output)
    assert prompts == ["是否继续？[y/N]："]
