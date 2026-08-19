"""MCP stdio command validation — allowlist and path canonicalisation.

Issue #764, #778.
"""

from __future__ import annotations

import os
import re
import shlex

__all__ = [
    "MCPCommandError",
    "ALLOWED_MCP_EXECUTABLES",
    "validate_stdio_command",
]

ALLOWED_MCP_EXECUTABLES: frozenset[str] = frozenset(
    {
        "python",
        "python3",
        "python3.11",
        "python3.12",
        "uv",
        "node",
        "npx",
        "npm",
    }
)

_SHELL_META_RE = re.compile(r"[;&|`$<>()\\\n\r]")


class MCPCommandError(ValueError):
    """Raised when an MCP stdio command fails allowlist or injection checks."""


def validate_stdio_command(command: str) -> None:
    """Validate a user-supplied MCP stdio command string.

    Checks (in order):
    1. Non-empty after strip.
    2. Parseable by ``shlex.split``.
    3. Executable basename is in ``ALLOWED_MCP_EXECUTABLES``.
    4. No shell meta-characters in any argument token.

    Raises :class:`MCPCommandError` on any failure.
    """
    if not command or not command.strip():
        raise MCPCommandError("MCP stdio command must not be empty.")

    try:
        parts = shlex.split(command)
    except ValueError as exc:
        raise MCPCommandError(f"MCP command is not parseable: {exc}") from exc

    if not parts:
        raise MCPCommandError("MCP stdio command resolved to an empty argument list.")

    executable = os.path.basename(parts[0])
    if executable not in ALLOWED_MCP_EXECUTABLES:
        raise MCPCommandError(
            f"MCP stdio executable '{executable}' is not in the allowed list. "
            f"Permitted executables: {', '.join(sorted(ALLOWED_MCP_EXECUTABLES))}"
        )

    for token in parts:
        if _SHELL_META_RE.search(token):
            raise MCPCommandError(
                f"MCP command token contains a shell meta-character: {token!r}. "
                "Shell operators are not permitted in MCP stdio commands."
            )
