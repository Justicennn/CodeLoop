"""Reserved model-native Human Interaction core action."""

from __future__ import annotations

import json
from typing import Any

from ..control import (
    InteractionAction,
    InteractionOption,
    InteractionRequest,
)

REQUEST_USER_INPUT_ACTION_NAME = "request_user_input"
MAX_INTERACTION_PROMPT_CHARS = 1_000
MAX_INTERACTION_ANSWER_CHARS = 2_000
MAX_INTERACTION_OPTIONS = 5

REQUEST_USER_INPUT_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": REQUEST_USER_INPUT_ACTION_NAME,
        "description": (
            "Return control to the user for one notification, clarification, "
            "choice, approval, or scope re-approval. This action must be the "
            "only call in the model decision."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": [
                        "inform",
                        "approve",
                        "re_approve",
                        "clarify",
                        "choose",
                    ],
                },
                "prompt": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_INTERACTION_PROMPT_CHARS,
                },
                "options": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": MAX_INTERACTION_OPTIONS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "minLength": 1, "maxLength": 64},
                            "label": {"type": "string", "minLength": 1, "maxLength": 160},
                            "description": {"type": "string", "maxLength": 300},
                        },
                        "required": ["id", "label"],
                        "additionalProperties": False,
                    },
                },
                "action": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string", "minLength": 1, "maxLength": 500},
                        "category": {"type": "string", "maxLength": 64},
                        "command": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 128,
                            "items": {"type": "string", "maxLength": 1_000},
                        },
                        "cwd": {"type": "string", "maxLength": 1_000},
                        "authorization_basis": {"type": "string", "maxLength": 1_000},
                    },
                    "required": ["description"],
                    "additionalProperties": False,
                },
            },
            "required": ["kind", "prompt"],
            "additionalProperties": False,
        },
    },
}


class InteractionValidationError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def parse_interaction_request(arguments_json: str) -> InteractionRequest:
    try:
        value = json.loads(arguments_json)
    except (json.JSONDecodeError, TypeError) as exc:
        raise InteractionValidationError(
            "Action arguments must be a JSON object."
        ) from exc
    if not isinstance(value, dict):
        raise InteractionValidationError("Action arguments must be a JSON object.")
    if set(value) - {"kind", "prompt", "options", "action"}:
        raise InteractionValidationError("request_user_input contains unknown fields.")
    kind = value.get("kind")
    if kind not in {"inform", "approve", "re_approve", "clarify", "choose"}:
        raise InteractionValidationError("Interaction kind is invalid.")
    prompt = _bounded_text(value.get("prompt"), "prompt", MAX_INTERACTION_PROMPT_CHARS)
    options = _parse_options(value.get("options"))
    action = _parse_action(value.get("action"))
    if kind == "choose" and len(options) < 2:
        raise InteractionValidationError("choose requires at least two options.")
    if kind != "choose" and options:
        raise InteractionValidationError("options are only allowed for choose.")
    if kind in {"approve", "re_approve"} and action is None:
        raise InteractionValidationError("approve and re_approve require action.")
    if kind not in {"inform", "approve", "re_approve"} and action is not None:
        raise InteractionValidationError(
            "action is only allowed for inform or approval kinds."
        )
    return InteractionRequest(kind=kind, prompt=prompt, options=options, action=action)


def _parse_options(value: Any) -> tuple[InteractionOption, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not 2 <= len(value) <= MAX_INTERACTION_OPTIONS:
        raise InteractionValidationError("options must contain between 2 and 5 items.")
    result: list[InteractionOption] = []
    for item in value:
        if not isinstance(item, dict) or set(item) - {"id", "label", "description"}:
            raise InteractionValidationError("Each option must be a bounded object.")
        result.append(
            InteractionOption(
                id=_bounded_text(item.get("id"), "option id", 64),
                label=_bounded_text(item.get("label"), "option label", 160),
                description=_optional_text(item.get("description"), "description", 300) or "",
            )
        )
    if len({item.id for item in result}) != len(result):
        raise InteractionValidationError("Option IDs must be unique.")
    return tuple(result)


def _parse_action(value: Any) -> InteractionAction | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - {
        "description", "category", "command", "cwd", "authorization_basis"
    }:
        raise InteractionValidationError("action must be a bounded object.")
    command_value = value.get("command")
    command: tuple[str, ...] = ()
    if command_value is not None:
        if (
            not isinstance(command_value, list)
            or not command_value
            or len(command_value) > 128
            or any(
                not isinstance(part, str)
                or not part
                or "\x00" in part
                or len(part) > 1_000
                for part in command_value
            )
        ):
            raise InteractionValidationError("action.command is invalid.")
        command = tuple(command_value)
    return InteractionAction(
        description=_bounded_text(value.get("description"), "action description", 500),
        category=_optional_text(value.get("category"), "category", 64),
        command=command,
        cwd=_optional_text(value.get("cwd"), "cwd", 1_000),
        authorization_basis=_optional_text(
            value.get("authorization_basis"), "authorization_basis", 1_000
        ),
    )


def _bounded_text(value: Any, name: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise InteractionValidationError(f"{name} must be a bounded non-empty string.")
    return value.strip()


def _optional_text(value: Any, name: str, limit: int) -> str | None:
    if value is None:
        return None
    return _bounded_text(value, name, limit)
