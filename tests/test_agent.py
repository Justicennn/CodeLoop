from __future__ import annotations

from copy import deepcopy
from typing import Any

from codeloop.agent import AgentRunner
from codeloop.llm import ModelResponse, ToolCall


class FakeClient:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = iter(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        self.calls.append({"messages": deepcopy(messages), "tools": deepcopy(tools)})
        return next(self._responses)


def test_normal_agent_loop_returns_tool_observation_to_model() -> None:
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(id="call-echo-1", name="echo", arguments='{"text":"CodeLoop"}')
                ]
            ),
            ModelResponse(text="Echo returned CodeLoop."),
        ]
    )

    result = AgentRunner(client).run("Use echo, then report its result.")

    assert result.status == "completed"
    assert result.answer == "Echo returned CodeLoop."
    assert len(client.calls) == 2
    second_messages = client.calls[1]["messages"]
    assistant_message = second_messages[-2]
    tool_message = second_messages[-1]
    assert assistant_message["tool_calls"][0]["id"] == "call-echo-1"
    assert tool_message["tool_call_id"] == "call-echo-1"
    assert tool_message["content"] == '{"ok": true, "data": {"text": "CodeLoop"}}'


def test_unknown_tool_becomes_structured_observation() -> None:
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[ToolCall(id="call-unknown-1", name="missing", arguments="{}")]
            ),
            ModelResponse(text="The requested tool was unavailable."),
        ]
    )

    result = AgentRunner(client).run("Try a missing tool and recover.")

    assert result.status == "completed"
    tool_message = client.calls[1]["messages"][-1]
    assert tool_message["tool_call_id"] == "call-unknown-1"
    assert tool_message["content"] == (
        '{"ok": false, "error_code": "unknown_tool", '
        '"message": "Unknown tool: missing"}'
    )


def test_max_steps_stops_repeated_tool_requests() -> None:
    repeated_call = ModelResponse(
        tool_calls=[ToolCall(id="call-repeat", name="echo", arguments='{"text":"again"}')]
    )
    client = FakeClient([repeated_call, repeated_call, repeated_call])

    result = AgentRunner(client, max_steps=3).run("Keep calling echo.")

    assert result.status == "max_steps"
    assert result.answer is None
    assert result.steps == 3
    assert len(client.calls) == 3
