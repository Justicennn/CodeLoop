"""Stage 1 temporary tool registry and dispatcher."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

ToolResult = dict[str, Any]
ToolHandler = Callable[[dict[str, Any]], ToolResult]


@dataclass(frozen=True)
class ToolDefinition:
    schema: dict[str, Any]
    handler: ToolHandler


def _error(error_code: str, message: str) -> ToolResult:
    return {"ok": False, "error_code": error_code, "message": message}


def _echo(arguments: dict[str, Any]) -> ToolResult:
    text = arguments.get("text")
    if not isinstance(text, str):
        return _error("invalid_arguments", "echo requires a string field named 'text'")
    return {"ok": True, "data": {"text": text}}


ECHO_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "echo",
        "description": "Return the supplied text unchanged.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
    },
}


class ToolRegistry:
    """A direct name-to-schema-and-callable registry."""

    def __init__(self) -> None:
        self._tools = {
            "echo": ToolDefinition(schema=ECHO_SCHEMA, handler=_echo),
        }

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [definition.schema for definition in self._tools.values()]

    def dispatch(self, name: str, arguments_json: str) -> ToolResult:
        definition = self._tools.get(name)
        if definition is None:
            return _error("unknown_tool", f"Unknown tool: {name}")

        try:
            arguments = json.loads(arguments_json)
        except (json.JSONDecodeError, TypeError):
            return _error("invalid_arguments", "Tool arguments must be a JSON object")

        if not isinstance(arguments, dict):
            return _error("invalid_arguments", "Tool arguments must be a JSON object")

        return definition.handler(arguments)
