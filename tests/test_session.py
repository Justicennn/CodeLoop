from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import codeloop.agent.runner as runner_module
import codeloop.interaction.cli as cli_module
from codeloop.agent.context import ConversationContext
from codeloop.agent import PublicConversationTurn
from codeloop.agent.runner import AgentRunner
from codeloop.execution.tools import ToolRegistry
from codeloop.execution.workspace import Workspace
from codeloop.interaction.session import (
    SESSION_TRUNCATION_MARKER,
    InteractiveSession,
    SessionHistory,
    parse_natural_workspace_switch,
    parse_workspace_argument,
)
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


def _input(values: list[str | BaseException]):
    remaining = iter(values)

    def read_line(_prompt: str) -> str:
        value = next(remaining)
        if isinstance(value, BaseException):
            raise value
        return value

    return read_line


def _cycle(call_id: str, content: str = "ok"):
    return (
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "list_files", "arguments": "{}"},
                }
            ],
        },
        [
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": "list_files",
                "content": content,
            }
        ],
    )


def test_public_turns_use_agent_roles_and_one_shot_shape_is_unchanged(
    tmp_path: Path,
) -> None:
    one_shot = FakeClient([ModelResponse(text="done")])
    AgentRunner(one_shot, tools=ToolRegistry(Workspace(tmp_path))).run("current")
    assert [message["role"] for message in one_shot.calls[0]["messages"]] == [
        "system",
        "user",
    ]

    contextual = FakeClient([ModelResponse(text="done")])
    previous = (PublicConversationTurn("first", "public answer"),)
    AgentRunner(contextual, tools=ToolRegistry(Workspace(tmp_path))).run(
        "current",
        previous_turns=previous,
    )
    messages = contextual.calls[0]["messages"]
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages[1:] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "public answer"},
        {"role": "user", "content": "current"},
    ]
    assert all("tool_call_id" not in message for message in messages)


def test_previous_pairs_trim_before_current_tool_cycles() -> None:
    context = ConversationContext(
        "system",
        "current",
        previous_turns=(
            PublicConversationTurn("u1", "a1"),
            PublicConversationTurn("u2", "a2"),
        ),
        max_chars=10_000,
        max_messages=9,
    )
    for call_id in ("one", "two", "three"):
        assistant, results = _cycle(call_id)
        context.add_tool_cycle(assistant, results)

    messages = context.messages_for_model()
    notice = json.loads(messages[1]["content"].split(": ", 1)[1])
    assert notice["removed_previous_pairs"] == 2
    assert notice["removed_previous_messages"] == 4
    assert notice["removed_cycles"] == 0
    assert [
        message["tool_call_id"]
        for message in messages
        if message["role"] == "tool"
    ] == ["one", "two", "three"]

    assistant, results = _cycle("four")
    context.add_tool_cycle(assistant, results)
    messages = context.messages_for_model()
    notice = json.loads(messages[1]["content"].split(": ", 1)[1])
    assert notice["removed_cycles"] == 1
    assert [
        message["tool_call_id"]
        for message in messages
        if message["role"] == "tool"
    ] == ["two", "three", "four"]


def test_previous_pairs_share_character_budget_and_latest_cycle_is_pinned() -> None:
    context = ConversationContext(
        "system",
        "current task",
        previous_turns=(PublicConversationTurn("old" * 500, "answer" * 500),),
        max_chars=1_000,
        max_messages=10,
    )
    assistant, results = _cycle("latest", "x" * 2_000)
    context.add_tool_cycle(assistant, results)
    messages = context.messages_for_model()
    notice = json.loads(messages[1]["content"].split(": ", 1)[1])

    assert notice["removed_previous_pairs"] == 1
    assert notice["overflow"] is True
    assert messages[0]["role"] == "system"
    assert messages[2] == {"role": "user", "content": "current task"}
    assert messages[-2]["tool_calls"][0]["id"] == "latest"
    assert messages[-1]["tool_call_id"] == "latest"


def test_session_history_is_pair_bounded_and_deterministically_truncated() -> None:
    history = SessionHistory(max_pairs=2, max_chars=180, max_text_chars=4_000)
    history.add("old", "old answer")
    history.add("middle", "middle answer")
    history.add("new" * 200, "answer" * 200)

    turns = history.snapshot()
    assert len(turns) == 1
    assert SESSION_TRUNCATION_MARKER in turns[0].user
    assert SESSION_TRUNCATION_MARKER in turns[0].assistant
    assert isinstance(turns, tuple)

    default_history = SessionHistory()
    for index in range(7):
        default_history.add(f"u{index}", f"a{index}")
    assert len(default_history.snapshot()) == 6
    assert default_history.snapshot()[0].user == "u1"


def test_workspace_argument_and_natural_switch_are_conservative(
    tmp_path: Path,
) -> None:
    absolute = str(tmp_path)
    assert parse_workspace_argument(absolute) == tmp_path
    assert parse_workspace_argument(f'"{absolute}"') == tmp_path
    assert parse_workspace_argument("relative/path") is None
    assert parse_workspace_argument(f'"{absolute}') is None
    assert parse_workspace_argument(f'"{absolute}" extra') is None

    pure = parse_natural_workspace_switch(f"切换到 {absolute}")
    assert pure is not None and pure.path == tmp_path and pure.task is None
    combined = parse_natural_workspace_switch(
        f'接下来处理 "{absolute}"，然后修复失败测试'
    )
    assert combined is not None
    assert combined.path == tmp_path
    assert combined.task == "修复失败测试"
    assert parse_natural_workspace_switch("切换到 relative/path") is None
    assert parse_natural_workspace_switch(f'切换到 "{absolute}') is None
    assert parse_natural_workspace_switch(f'切换到 "{absolute}" extra') is None
    assert parse_natural_workspace_switch(f"切换到 {absolute}，") is None
    assert parse_natural_workspace_switch(f"请查看路径 {absolute}") is None


def test_interactive_tasks_receive_only_bounded_public_results(
    tmp_path: Path,
) -> None:
    client = FakeClient(
        [ModelResponse(text="first answer"), ModelResponse(text="second answer")]
    )
    output: list[str] = []
    session = InteractiveSession(
        client,
        model_name="fake",
        workspace=Workspace(tmp_path),
        read_line=_input(["first task", "second task", "/exit"]),
        write_line=output.append,
        renderer_factory=lambda: None,
    )

    assert session.run() == 0
    assert len(client.calls) == 2
    second = client.calls[1]["messages"]
    assert second[1:] == [
        {"role": "user", "content": "first task"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second task"},
    ]
    assert all(message.get("role") != "tool" for message in second)
    assert any("CodeLoop interactive" in line for line in output)


def test_each_task_gets_a_fresh_renderer_and_empty_input_is_ignored(
    tmp_path: Path,
) -> None:
    client = FakeClient([ModelResponse(text="one"), ModelResponse(text="two")])
    renderers: list[object] = []

    class Renderer:
        def start_thinking(self) -> None:
            pass

        def stop_thinking(self) -> None:
            pass

        def show_tool_event(self, _event: object) -> None:
            pass

        def show_result(self, _result: object) -> None:
            pass

    def factory():
        renderer = Renderer()
        renderers.append(renderer)
        return renderer

    session = InteractiveSession(
        client,
        model_name="fake",
        workspace=Workspace(tmp_path),
        read_line=_input(["", "one", "   ", "two", "/exit"]),
        write_line=lambda _line: None,
        renderer_factory=factory,
    )
    assert session.run() == 0
    assert len(renderers) == 2
    assert renderers[0] is not renderers[1]


@pytest.mark.parametrize("reset_input", ["/new", "workspace"])
def test_new_and_workspace_switch_remove_all_previous_public_context(
    tmp_path: Path,
    reset_input: str,
) -> None:
    other = tmp_path / "other"
    other.mkdir()
    reset = "/new" if reset_input == "/new" else f'/workspace "{other}"'
    client = FakeClient([ModelResponse(text="old answer"), ModelResponse(text="new")])
    session = InteractiveSession(
        client,
        model_name="fake",
        workspace=Workspace(tmp_path),
        read_line=_input(["old task", reset, "new task", "/exit"]),
        write_line=lambda _line: None,
        renderer_factory=lambda: None,
    )

    assert session.run() == 0
    second_messages = client.calls[1]["messages"]
    assert [message["role"] for message in second_messages] == ["system", "user"]
    assert second_messages[-1]["content"] == "new task"
    assert "old task" not in json.dumps(second_messages, ensure_ascii=False)
    expected_root = other.resolve() if reset_input == "workspace" else tmp_path.resolve()
    assert session.workspace.root == expected_root


def test_unknown_and_malformed_commands_never_start_agent(tmp_path: Path) -> None:
    client = FakeClient([])
    output: list[str] = []
    session = InteractiveSession(
        client,
        model_name="fake",
        workspace=Workspace(tmp_path),
        read_line=_input(["/wat", "/help extra", "/workspace relative", "/exit"]),
        write_line=output.append,
        renderer_factory=lambda: None,
    )

    assert session.run() == 0
    assert client.calls == []
    assert "Unknown command. Type /help for available commands." in output


def test_invalid_clear_natural_switch_preserves_workspace_and_skips_agent(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    client = FakeClient([])
    session = InteractiveSession(
        client,
        model_name="fake",
        workspace=Workspace(tmp_path),
        read_line=_input([f"切换到 {missing}", "/exit"]),
        write_line=lambda _line: None,
        renderer_factory=lambda: None,
    )

    assert session.run() == 0
    assert client.calls == []
    assert session.workspace.root == tmp_path.resolve()


def test_failed_workspace_switch_preserves_existing_public_history(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    client = FakeClient([ModelResponse(text="remembered"), ModelResponse(text="used")])
    session = InteractiveSession(
        client,
        model_name="fake",
        workspace=Workspace(tmp_path),
        read_line=_input(["first", f'/workspace "{missing}"', "second", "/exit"]),
        write_line=lambda _line: None,
        renderer_factory=lambda: None,
    )
    assert session.run() == 0
    assert client.calls[1]["messages"][1:3] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "remembered"},
    ]
    assert session.workspace.root == tmp_path.resolve()


def test_ambiguous_natural_switch_is_submitted_as_the_original_task(
    tmp_path: Path,
) -> None:
    text = '切换到 "未闭合路径'
    client = FakeClient([ModelResponse(text="handled")])
    session = InteractiveSession(
        client,
        model_name="fake",
        workspace=Workspace(tmp_path),
        read_line=_input([text, "/exit"]),
        write_line=lambda _line: None,
        renderer_factory=lambda: None,
    )

    assert session.run() == 0
    assert client.calls[0]["messages"][-1] == {"role": "user", "content": text}
    assert session.workspace.root == tmp_path.resolve()


def test_natural_switch_with_task_uses_new_workspace_and_new_history(
    tmp_path: Path,
) -> None:
    other = tmp_path / "new root"
    other.mkdir()
    client = FakeClient([ModelResponse(text="old"), ModelResponse(text="new")])
    session = InteractiveSession(
        client,
        model_name="fake",
        workspace=Workspace(tmp_path),
        read_line=_input(
            ["old task", f'把工作目录切到 "{other}"，然后检查项目', "/exit"]
        ),
        write_line=lambda _line: None,
        renderer_factory=lambda: None,
    )

    assert session.run() == 0
    assert client.calls[1]["messages"][-1] == {
        "role": "user",
        "content": "检查项目",
    }
    assert len(client.calls[1]["messages"]) == 2
    assert session.workspace.root == other.resolve()


def test_task_runtime_state_and_tool_cycles_do_not_leak_to_next_task(
    tmp_path: Path,
) -> None:
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="old-write-id",
                        name="write_file",
                        arguments='{"path":"created.txt","content":"hello"}',
                    )
                ]
            ),
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="old-run-id",
                        name="run_command",
                        arguments=json.dumps(
                            {"command": [sys.executable, "-c", "pass"]}
                        ),
                    )
                ]
            ),
            ModelResponse(text="first complete"),
            ModelResponse(text="second complete"),
        ]
    )
    session = InteractiveSession(
        client,
        model_name="fake",
        workspace=Workspace(tmp_path),
        read_line=_input(["build", "review", "/exit"]),
        write_line=lambda _line: None,
        renderer_factory=lambda: None,
    )

    assert session.run() == 0
    second_task_request = client.calls[3]["messages"]
    serialized = json.dumps(second_task_request, ensure_ascii=False)
    assert "old-write-id" not in serialized
    assert "old-run-id" not in serialized
    assert "Runtime task state:" not in second_task_request[0]["content"]
    assert all(message["role"] != "tool" for message in second_task_request)


def test_controlled_termination_continues_but_fatal_termination_ends(
    tmp_path: Path,
) -> None:
    controlled = FakeClient(
        [
            ModelResponse(
                tool_calls=[ToolCall("read", "list_files", "{}")]
            ),
            ModelResponse(text="next completed"),
        ]
    )
    continuing = InteractiveSession(
        controlled,
        model_name="fake",
        workspace=Workspace(tmp_path),
        max_steps=1,
        read_line=_input(["limited", "next", "/exit"]),
        write_line=lambda _line: None,
        renderer_factory=lambda: None,
    )
    assert continuing.run() == 0
    assert len(controlled.calls) == 2

    fatal = FakeClient(
        [
            ModelAPIError(
                "fatal",
                "safe fatal",
                classification="fatal",
            )
        ]
    )
    ending = InteractiveSession(
        fatal,
        model_name="fake",
        workspace=Workspace(tmp_path),
        read_line=_input(["task"]),
        write_line=lambda _line: None,
        renderer_factory=lambda: None,
    )
    assert ending.run() == 2


def test_eof_and_prompt_interrupt_have_fixed_exit_codes(tmp_path: Path) -> None:
    for exception, expected in ((EOFError(), 0), (KeyboardInterrupt(), 130)):
        session = InteractiveSession(
            FakeClient([]),
            model_name="fake",
            workspace=Workspace(tmp_path),
            read_line=_input([exception]),
            write_line=lambda _line: None,
            renderer_factory=lambda: None,
        )
        assert session.run() == expected


def test_task_interrupt_returns_to_the_session_prompt(tmp_path: Path) -> None:
    client = FakeClient([KeyboardInterrupt(), ModelResponse(text="recovered")])
    session = InteractiveSession(
        client,
        model_name="fake",
        workspace=Workspace(tmp_path),
        read_line=_input(["interrupt this task", "next task", "/exit"]),
        write_line=lambda _line: None,
        renderer_factory=lambda: None,
    )
    assert session.run() == 0
    assert len(client.calls) == 2
    assert client.calls[1]["messages"][1]["content"].startswith("interrupt this task")
    assert client.calls[1]["messages"][2]["content"].startswith("user_interrupt:")


def test_previous_turns_are_frozen_across_api_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(runner_module, "sleep", lambda _seconds: None)
    retry = ModelAPIError(
        "temporary",
        "retry",
        classification="retryable",
    )
    client = FakeClient([retry, retry, ModelResponse(text="done")])
    previous = (PublicConversationTurn("earlier", "public result"),)

    result = AgentRunner(client, tools=ToolRegistry(Workspace(tmp_path))).run(
        "current",
        previous_turns=previous,
    )

    assert result.status == "completed"
    assert len(client.calls) == 3
    assert client.calls[0]["messages"] == client.calls[1]["messages"]
    assert client.calls[1]["messages"] == client.calls[2]["messages"]
    assert client.calls[0]["tools"] == client.calls[1]["tools"]
    assert client.calls[1]["tools"] == client.calls[2]["tools"]


def test_cli_task_is_optional_and_dispatches_interactive_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assert cli_module._parser().parse_args([]).task is None
    assert cli_module._parser().parse_args(["one shot"]).task == "one shot"

    monkeypatch.setenv("MODEL_API_KEY", "key")
    monkeypatch.setenv("MODEL_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("MODEL_NAME", "fake")
    monkeypatch.setattr(cli_module, "OpenAICompatibleClient", lambda **_kwargs: object())
    observed: dict[str, Any] = {}

    class StubSession:
        def __init__(self, _client: object, **kwargs: Any) -> None:
            observed.update(kwargs)

        def run(self) -> int:
            return 17

    monkeypatch.setattr(cli_module, "InteractiveSession", StubSession)
    assert cli_module.main(["--workspace", str(tmp_path)]) == 17
    assert observed["workspace"].root == tmp_path.resolve()


def test_cli_one_shot_branch_keeps_original_message_shape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = FakeClient([ModelResponse(text="done")])
    monkeypatch.setenv("MODEL_API_KEY", "key")
    monkeypatch.setenv("MODEL_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("MODEL_NAME", "fake")
    monkeypatch.setattr(cli_module, "OpenAICompatibleClient", lambda **_kwargs: client)
    monkeypatch.setattr(cli_module, "ConsoleRenderer", lambda: None)

    assert cli_module.main(["one shot", "--workspace", str(tmp_path)]) == 0
    assert [message["role"] for message in client.calls[0]["messages"]] == [
        "system",
        "user",
    ]
    assert client.calls[0]["messages"][-1]["content"] == "one shot"
