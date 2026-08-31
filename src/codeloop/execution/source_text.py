"""Pure deterministic slicing for normalized source text."""

from __future__ import annotations

from dataclasses import dataclass


class SourceTextSliceError(Exception):
    """A cursor cannot address the supplied normalized source text."""

    def __init__(self, message: str, *, total_chars: int) -> None:
        super().__init__(message)
        self.message = message
        self.total_chars = total_chars


@dataclass(frozen=True)
class SourceTextSlice:
    text: str
    start_cursor: int
    end_cursor: int
    total_chars: int
    truncated: bool
    next_cursor: int | None


def bounded_text_slice(
    text: str,
    *,
    cursor: int,
    max_chars: int,
) -> SourceTextSlice:
    """Return one bounded cursor slice from an already normalized source."""
    total_chars = len(text)
    if cursor > total_chars:
        raise SourceTextSliceError(
            "cursor cannot exceed the source's total_chars.",
            total_chars=total_chars,
        )
    end_cursor = min(total_chars, cursor + max_chars)
    truncated = end_cursor < total_chars
    return SourceTextSlice(
        text=text[cursor:end_cursor],
        start_cursor=cursor,
        end_cursor=end_cursor,
        total_chars=total_chars,
        truncated=truncated,
        next_cursor=end_cursor if truncated else None,
    )
