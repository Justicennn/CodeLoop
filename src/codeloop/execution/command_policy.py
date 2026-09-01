"""Deterministic description of command facts, independent of authorization."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

CommandCategory = Literal[
    "read_only_git",
    "test",
    "dependency_change",
    "program_execution",
    "external_write",
    "destructive",
    "unknown",
]

@dataclass(frozen=True)
class CommandTestScope:
    """Deterministic test-family facts, without authorization semantics."""

    family: str
    all_tests: bool
    targets: tuple[str, ...] = ()


@dataclass(frozen=True)
class CommandDescription:
    """Validated command facts; contains no approval or user-intent decision."""

    command: tuple[str, ...]
    display_command: tuple[str, ...]
    cwd: str
    timeout_seconds: int
    category: CommandCategory
    reason: str
    test_scope: CommandTestScope | None = None
    workspace_root: str | None = None


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
_READ_ONLY_GIT_ACTIONS = frozenset(
    {"diff", "log", "show", "status", "rev-parse", "ls-files"}
)
_EXTERNAL_GIT_ACTIONS = frozenset({"push"})
_DESTRUCTIVE_EXECUTABLES = frozenset({"del", "erase", "rmdir", "rm"})
_TEST_EXECUTABLES = frozenset({"pytest", "tox", "nox"})


def describe_command(
    command: Sequence[str],
    *,
    cwd: str,
    timeout_seconds: int,
    display_command: Sequence[str] | None = None,
    workspace_root: str | None = None,
) -> CommandDescription:
    """Classify validated argv by observable behavior, never by user intent."""
    frozen = tuple(command)
    category = _command_category(frozen)
    test_scope = _test_command_scope(frozen) if category == "test" else None
    reasons = {
        "read_only_git": "该命令只读取 Git 仓库信息。",
        "test": "该命令将执行测试或测试运行器。",
        "dependency_change": "该命令将更改依赖环境。",
        "program_execution": "该命令将执行本地程序。",
        "external_write": "该命令将写入外部系统。",
        "destructive": "该命令可能对本地状态造成破坏性更改。",
        "unknown": "该命令的影响不属于当前已知的低风险类别。",
    }
    return CommandDescription(
        command=frozen,
        display_command=tuple(display_command) if display_command is not None else frozen,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        category=category,
        reason=reasons[category],
        test_scope=test_scope,
        workspace_root=workspace_root,
    )


def _command_category(command: tuple[str, ...]) -> CommandCategory:
    if is_dependency_mutation(command):
        return "dependency_change"
    if not command:
        return "unknown"
    executable = _executable_name(command[0])
    arguments = tuple(part.casefold() for part in command[1:])
    if _test_command_scope(command) is not None:
        return "test"
    if executable == "node" and arguments and arguments[0] == "--test":
        # Flags can make target containment ambiguous, but the command is
        # still objectively a Node test invocation and receives an exact scope.
        return "test"
    if executable == "git" and arguments:
        if _is_destructive_git(arguments):
            return "destructive"
        if arguments[0] in _READ_ONLY_GIT_ACTIONS:
            return "read_only_git"
        if arguments[0] in _EXTERNAL_GIT_ACTIONS:
            return "external_write"
    if executable in _DESTRUCTIVE_EXECUTABLES:
        return "destructive"
    if executable in _TEST_EXECUTABLES:
        return "test"
    if executable in {"python", "py"} or _PYTHON_NAME.fullmatch(executable):
        if (
            len(arguments) >= 2
            and arguments[0] == "-m"
            and arguments[1] in {"pytest", "unittest"}
        ):
            return "test"
        return "program_execution"
    if executable in {"node", "java", "dotnet", "cargo", "go"}:
        if arguments and arguments[0] == "test":
            return "test"
        return "program_execution"
    if executable in {"npm", "pnpm", "yarn"} and arguments:
        if arguments[0] == "test" or arguments[:2] == ("run", "test"):
            return "test"
        if arguments[0] == "run":
            return "program_execution"
    return "unknown"


def _test_command_scope(command: tuple[str, ...]) -> CommandTestScope | None:
    """Recognize only test forms with a reliable bounded scope meaning."""
    if not command:
        return None
    executable = _executable_name(command[0])
    arguments = command[1:]
    folded = tuple(part.casefold() for part in arguments)

    if executable == "node":
        if folded and folded[0] == "--test":
            targets = arguments[1:]
            if not targets:
                return CommandTestScope(family="node", all_tests=True)
            if all(not target.startswith("-") for target in targets):
                return CommandTestScope(
                    family="node",
                    all_tests=False,
                    targets=tuple(_normalize_test_target(value) for value in targets),
                )
            return None
        if len(arguments) == 1 and _looks_like_node_test_target(arguments[0]):
            return CommandTestScope(
                family="node",
                all_tests=False,
                targets=(_normalize_test_target(arguments[0]),),
            )

    if executable in {"npm", "pnpm", "yarn"}:
        is_test_script = (
            folded == ("test",)
            or folded == ("run", "test")
        )
        if is_test_script:
            return CommandTestScope(family=f"{executable}:test", all_tests=True)

    if executable == "pytest":
        if not arguments:
            return CommandTestScope(family="pytest", all_tests=True)
        if all(not value.startswith("-") for value in arguments):
            return CommandTestScope(
                family="pytest",
                all_tests=False,
                targets=tuple(_normalize_test_target(value) for value in arguments),
            )

    if (
        (executable == "py" or _PYTHON_NAME.fullmatch(executable))
        and len(folded) >= 2
        and folded[:2] == ("-m", "pytest")
    ):
        targets = arguments[2:]
        if not targets:
            return CommandTestScope(family="pytest", all_tests=True)
        if all(not value.startswith("-") for value in targets):
            return CommandTestScope(
                family="pytest",
                all_tests=False,
                targets=tuple(_normalize_test_target(value) for value in targets),
            )
    return None


def _looks_like_node_test_target(value: str) -> bool:
    normalized = _normalize_test_target(value)
    folded = normalized.casefold()
    parts = folded.split("/")
    filename = parts[-1]
    if not filename.endswith((".js", ".mjs", ".cjs")):
        return False
    return (
        parts[0] in {"test", "tests"}
        or ".test." in filename
        or ".spec." in filename
        or filename.startswith("test-")
        or filename.endswith("_test.js")
    )


def _normalize_test_target(value: str) -> str:
    normalized = value.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized or "."


def _is_destructive_git(arguments: tuple[str, ...]) -> bool:
    action = arguments[0]
    remaining = arguments[1:]
    if action == "reset" and "--hard" in remaining:
        return True
    if action == "clean" and any(
        value == "--force"
        or (value.startswith("-") and not value.startswith("--") and "f" in value[1:])
        for value in remaining
    ):
        return True
    return action == "push" and any(
        value in {"--force", "-f", "--force-with-lease"}
        for value in remaining
    )


def is_dependency_mutation(
    command: Sequence[str],
) -> bool:
    """Classify only the explicit dependency-mutating forms documented here."""
    frozen = tuple(command)
    if not frozen:
        return False
    executable = _executable_name(frozen[0])
    if executable in _SHELL_WRAPPERS:
        tokens = tuple(_WRAPPER_TOKEN.findall(" ".join(frozen[1:])))
        if any(_is_direct_mutation(tokens[index:]) for index in range(len(tokens))):
            return True
        return False
    return _is_direct_mutation(frozen)


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
