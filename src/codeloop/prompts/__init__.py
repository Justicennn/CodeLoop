"""Packaged prompt assets used by CodeLoop."""

from importlib.resources import files


SYSTEM_PROMPT = files(__package__).joinpath("system.md").read_text(
    encoding="utf-8"
)

__all__ = ["SYSTEM_PROMPT"]
