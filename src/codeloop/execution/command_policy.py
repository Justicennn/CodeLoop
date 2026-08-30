"""Deterministic confirmation policy for explicit dependency commands."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

ApprovalCategory = Literal["dependency_mutation"]

DEPENDENCY_MUTATION_REASON = (
    "This command will modify the current dependency environment."
)


@dataclass(frozen=True)
class CommandApprovalRequest:
    """One per-command request produced without executing the command."""

    command: tuple[str, ...]
    category: ApprovalCategory = field(
        default="dependency_mutation",
        init=False,
    )
    reason: str = field(
        default=DEPENDENCY_MUTATION_REASON,
        init=False,
    )


_DIRECT_ACTIONS: dict[str, frozenset[str]] = {
    "conda": frozenset({"install", "remove", "update"}),
    "npm": frozenset({"ci", "install", "remove", "uninstall", "update"}),
    "pnpm": frozenset({"add", "install", "remove", "update"}),
    "poetry": frozenset({"add", "install", "remove"}),
    "yarn": frozenset({"add", "install", "remove", "update", "upgrade"}),
}
_UV_ACTIONS = frozenset({"add", "remove", "sync"})
_UV_PIP_ACTIONS = frozenset({"install", "sync", "uninstall"})
_PIP_ACTIONS = frozenset({"install", "uninstall"})
_SHELL_WRAPPERS = frozenset({"bash", "cmd", "powershell", "pwsh", "sh", "zsh"})
_WINDOWS_EXECUTABLE_SUFFIXES = (".exe", ".cmd", ".bat")
_PIP_NAME = re.compile(r"^pip(?:\d+(?:\.\d+)*)?$")
_PYTHON_NAME = re.compile(r"^python(?:\d+(?:\.\d+)*)?$")
_WRAPPER_TOKEN = re.compile(r"[A-Za-z0-9_.+\-]+")


def dependency_mutation_request(
    command: Sequence[str],
) -> CommandApprovalRequest | None:
    """Classify only the explicit dependency-mutating forms documented here."""
    frozen = tuple(command)
    if not frozen:
        return None
    executable = _executable_name(frozen[0])
    if executable in _SHELL_WRAPPERS:
        tokens = tuple(_WRAPPER_TOKEN.findall(" ".join(frozen[1:])))
        if any(_is_direct_mutation(tokens[index:]) for index in range(len(tokens))):
            return CommandApprovalRequest(command=frozen)
        return None
    if _is_direct_mutation(frozen):
        return CommandApprovalRequest(command=frozen)
    return None


def _is_direct_mutation(command: Sequence[str]) -> bool:
    if not command:
        return False
    executable = _executable_name(command[0])
    arguments = tuple(part.casefold() for part in command[1:])

    if _PIP_NAME.fullmatch(executable):
        return bool(arguments) and arguments[0] in _PIP_ACTIONS
    if executable == "py" or _PYTHON_NAME.fullmatch(executable):
        return _is_python_pip_mutation(arguments)
    if executable == "uv":
        if not arguments:
            return False
        if arguments[0] in _UV_ACTIONS:
            return True
        return (
            len(arguments) >= 2
            and arguments[0] == "pip"
            and arguments[1] in _UV_PIP_ACTIONS
        )
    actions = _DIRECT_ACTIONS.get(executable)
    return actions is not None and bool(arguments) and arguments[0] in actions


def _is_python_pip_mutation(arguments: Sequence[str]) -> bool:
    try:
        module_index = arguments.index("-m")
    except ValueError:
        return False
    return (
        len(arguments) > module_index + 2
        and _PIP_NAME.fullmatch(arguments[module_index + 1]) is not None
        and arguments[module_index + 2] in _PIP_ACTIONS
    )


def _executable_name(value: str) -> str:
    name = re.split(r"[\\/]", value)[-1].casefold()
    for suffix in _WINDOWS_EXECUTABLE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name
