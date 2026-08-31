"""Model-boundary coverage for Stage 10C native image input."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import codeloop.model.client as client_module
from codeloop.interaction.cli import _image_input_capability
from codeloop.interaction.narration import _NarratingModelClient
from codeloop.model.client import ModelAPIError, ModelResponse, OpenAICompatibleClient


class _Completions:
    def __init__(self, response: Any = None, error: BaseException | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class _SDK:
    def __init__(self, completions: _Completions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def _response(content: str = "seen") -> Any:
    message = SimpleNamespace(content=content, tool_calls=[])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_capability_defaults_false_and_can_be_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(client_module, "OpenAI", lambda **_kwargs: _SDK(_Completions()))
    disabled = OpenAICompatibleClient(
        api_key="key",
        base_url="https://example.invalid",
        model="model",
    )
    enabled = OpenAICompatibleClient(
        api_key="key",
        base_url="https://example.invalid",
        model="model",
        supports_image_input=True,
    )
    assert disabled.supports_image_input is False
    assert enabled.supports_image_input is True


@pytest.mark.parametrize("value", [None, "", "false", "FALSE"])
def test_environment_capability_disabled_values(value: str | None) -> None:
    assert _image_input_capability(value) is False


@pytest.mark.parametrize("value", ["true", "TRUE", "TrUe"])
def test_environment_capability_enabled_values(value: str) -> None:
    assert _image_input_capability(value) is True


@pytest.mark.parametrize("value", ["1", "yes", " true ", "off"])
def test_environment_capability_rejects_other_values(value: str) -> None:
    with pytest.raises(ValueError):
        _image_input_capability(value)


def test_narration_wrapper_forwards_capability() -> None:
    class Delegate:
        supports_image_input = True

        def complete(self, _messages: list[dict[str, Any]], _tools: list[dict[str, Any]]) -> ModelResponse:
            return ModelResponse(text="done")

    assert _NarratingModelClient(Delegate(), None).supports_image_input is True  # type: ignore[arg-type]


def test_multimodal_message_is_passed_to_openai_compatible_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completions = _Completions(response=_response())
    monkeypatch.setattr(client_module, "OpenAI", lambda **_kwargs: _SDK(completions))
    client = OpenAICompatibleClient(
        api_key="key",
        base_url="https://example.invalid",
        model="model",
        supports_image_input=True,
    )
    messages = [
        {"role": "user", "content": "task"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Visual source: 1.png"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,iVBORw=="},
                },
            ],
        },
    ]
    result = client.complete(messages, [])

    assert result == ModelResponse(text="seen")
    assert completions.calls[0]["messages"] == messages
    assert completions.calls[0]["messages"][1]["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )


def test_provider_image_rejection_remains_model_api_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RejectedStatusError(Exception):
        status_code = 400

    completions = _Completions(error=RejectedStatusError("provider detail"))
    monkeypatch.setattr(client_module, "APIStatusError", RejectedStatusError)
    monkeypatch.setattr(client_module, "OpenAI", lambda **_kwargs: _SDK(completions))
    client = OpenAICompatibleClient(
        api_key="key",
        base_url="https://example.invalid",
        model="model",
        supports_image_input=True,
    )

    with pytest.raises(ModelAPIError) as captured:
        client.complete(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,iVBORw=="},
                        }
                    ],
                }
            ],
            [],
        )
    assert captured.value.error_code == "model_api_error"
    assert captured.value.classification == "fatal"
    assert "provider detail" not in captured.value.safe_message
