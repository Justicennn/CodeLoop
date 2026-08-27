from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from codeloop.agent import AgentRunner
from codeloop.llm import ModelResponse, ToolCall
from codeloop.tools import ToolRegistry
from codeloop.workspace import Workspace, WorkspaceError


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


def test_workspace_rejects_escape(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("outside", encoding="utf-8")

    with pytest.raises(WorkspaceError):
        workspace.resolve("../outside.txt")
    with pytest.raises(WorkspaceError):
        workspace.resolve(str(outside))

    link = tmp_path / "escape-link.txt"
    try:
        link.symlink_to(outside)
    except (NotImplementedError, OSError):
        return
    with pytest.raises(WorkspaceError):
        workspace.resolve("escape-link.txt")


def test_edit_file_unique_mismatch_and_ambiguous(tmp_path: Path) -> None:
    registry = ToolRegistry(Workspace(tmp_path))
    target = tmp_path / "target.txt"
    target.write_text("alpha beta", encoding="utf-8")

    success = registry.dispatch(
        "edit_file",
        json.dumps(
            {"path": "target.txt", "old_text": "alpha", "new_text": "gamma"}
        ),
    )
    assert success["ok"] is True
    assert target.read_text(encoding="utf-8") == "gamma beta"

    mismatch = registry.dispatch(
        "edit_file",
        json.dumps(
            {"path": "target.txt", "old_text": "missing", "new_text": "changed"}
        ),
    )
    assert mismatch["error_code"] == "edit_mismatch"
    assert target.read_text(encoding="utf-8") == "gamma beta"

    repeated = tmp_path / "repeated.txt"
    repeated.write_text("same same", encoding="utf-8")
    ambiguous = registry.dispatch(
        "edit_file",
        json.dumps(
            {"path": "repeated.txt", "old_text": "same", "new_text": "different"}
        ),
    )
    assert ambiguous["error_code"] == "edit_ambiguous"
    assert repeated.read_text(encoding="utf-8") == "same same"


def test_write_file_never_overwrites(tmp_path: Path) -> None:
    registry = ToolRegistry(Workspace(tmp_path))
    arguments = json.dumps({"path": "new.txt", "content": "first"})

    created = registry.dispatch("write_file", arguments)
    rejected = registry.dispatch(
        "write_file",
        json.dumps({"path": "new.txt", "content": "second"}),
    )

    assert created["ok"] is True
    assert rejected["error_code"] == "file_exists"
    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "first"


def test_run_command_success_and_failure(tmp_path: Path) -> None:
    registry = ToolRegistry(Workspace(tmp_path))
    work = tmp_path / "work"
    work.mkdir()

    success = registry.dispatch(
        "run_command",
        json.dumps(
            {
                "command": [
                    sys.executable,
                    "-c",
                    "import os; print(os.path.basename(os.getcwd()))",
                ],
                "cwd": "work",
            }
        ),
    )
    failure = registry.dispatch(
        "run_command",
        json.dumps(
            {
                "command": [
                    sys.executable,
                    "-c",
                    "import sys; print('failed', file=sys.stderr); sys.exit(3)",
                ]
            }
        ),
    )

    assert success["ok"] is True
    assert success["data"]["exit_code"] == 0
    assert success["data"]["stdout"].strip() == "work"
    assert failure["error_code"] == "command_failed"
    assert failure["data"]["exit_code"] == 3
    assert "failed" in failure["data"]["stderr"]


def test_real_tool_agent_loop_returns_observation_to_model(tmp_path: Path) -> None:
    (tmp_path / "sample.txt").write_text("hello\n", encoding="utf-8")
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call-read-1",
                        name="read_file",
                        arguments='{"path":"sample.txt"}',
                    )
                ]
            ),
            ModelResponse(text="The file says hello."),
        ]
    )
    registry = ToolRegistry(Workspace(tmp_path))

    result = AgentRunner(client, tools=registry).run("Read sample.txt.")

    assert result.status == "completed"
    assert result.answer == "The file says hello."
    assert len(client.calls) == 2
    second_messages = client.calls[1]["messages"]
    assistant_message = second_messages[-2]
    tool_message = second_messages[-1]
    observation = json.loads(tool_message["content"])
    assert assistant_message["tool_calls"][0]["id"] == "call-read-1"
    assert tool_message["tool_call_id"] == "call-read-1"
    assert observation["ok"] is True
    assert "1: hello" in observation["data"]["content"]


def test_max_steps_stops_repeated_tool_requests(tmp_path: Path) -> None:
    repeated_call = ModelResponse(
        tool_calls=[ToolCall(id="call-repeat", name="list_files", arguments="{}")]
    )
    client = FakeClient([repeated_call, repeated_call, repeated_call])
    registry = ToolRegistry(Workspace(tmp_path))

    result = AgentRunner(client, tools=registry, max_steps=3).run(
        "Keep listing files."
    )

    assert result.status == "max_steps"
    assert result.answer is None
    assert result.steps == 3
    assert len(client.calls) == 3
