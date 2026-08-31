"""Optional public narration passthrough owned by the Interaction Layer."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..model.client import ModelClient, ModelResponse


class _NarratingModelClient:
    """Observe successful public text without changing the model decision.

    The delegate's response is returned by identity.  Failed API attempts never
    reach the observer, and observer failures are isolated from Agent Runtime.
    """

    def __init__(
        self,
        delegate: ModelClient,
        on_narration: Callable[[str], None] | None,
    ) -> None:
        self._delegate = delegate
        self._on_narration = on_narration

    @property
    def supports_image_input(self) -> bool:
        return bool(getattr(self._delegate, "supports_image_input", False))

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        response = self._delegate.complete(messages, tools)
        text = response.text
        if response.tool_calls and isinstance(text, str) and text.strip():
            self._notify(text)
        return response

    def _notify(self, text: str) -> None:
        if self._on_narration is None:
            return
        try:
            self._on_narration(text)
        except Exception:
            pass
