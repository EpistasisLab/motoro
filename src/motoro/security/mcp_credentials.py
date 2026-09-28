"""Validation for credentials attached to MCP transports.

This module deliberately validates only names and shape.  It never includes a
credential value in an exception or log message.
"""

from __future__ import annotations

import re


class MCPCredentialValidationError(ValueError):
    """A supplied MCP credential cannot safely be passed to its transport."""


_HEADER_NAME = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

MAX_CREDENTIAL_ITEMS = 32
MAX_HEADER_NAME_BYTES = 128
MAX_HEADER_VALUE_BYTES = 8192
MAX_HEADER_TOTAL_BYTES = 32768
MAX_ENV_NAME_BYTES = 128
MAX_ENV_VALUE_BYTES = 16384
MAX_ENV_TOTAL_BYTES = 65536

# These are generated or interpreted by HTTPX/the MCP transport.  Permitting a
# caller to override them creates request smuggling, protocol confusion, or
# session fixation opportunities.
_TRANSPORT_HEADERS = frozenset(
    {
        "host",
        "content-length",
        "transfer-encoding",
        "connection",
        "expect",
        "keep-alive",
        "proxy-connection",
        "accept",
        "content-type",
        "mcp-protocol-version",
        "mcp-session-id",
        "te",
        "trailer",
        "upgrade",
    }
)

# Variables which change how the executable/runtime is located or inject code
# before the MCP server starts.  Matching is case-insensitive so behavior is
# consistent on Windows and POSIX hosts.
_DANGEROUS_ENV_EXACT = frozenset(
    {
        "BASH_ENV",
        "BUNDLE_GEMFILE",
        "CDPATH",
        "CLASSPATH",
        "DOTNET_ADDITIONAL_DEPS",
        "DOTNET_SHARED_STORE",
        "DOTNET_STARTUP_HOOKS",
        "ELECTRON_RUN_AS_NODE",
        "ENV",
        "GCONV_PATH",
        "IFS",
        "JAVA_TOOL_OPTIONS",
        "JDK_JAVA_OPTIONS",
        "LUA_CPATH",
        "LUA_INIT",
        "LUA_PATH",
        "NODE_OPTIONS",
        "NODE_PATH",
        "PERL5LIB",
        "PERL5DB",
        "PERL5OPT",
        "PATH",
        "PHPRC",
        "PHP_INI_SCAN_DIR",
        "PYTHONHOME",
        "PYTHONINSPECT",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "RUBYLIB",
        "RUBYOPT",
        "SHELLOPTS",
        "SSLKEYLOGFILE",
        "_JAVA_OPTIONS",
    }
)
_DANGEROUS_ENV_PREFIXES = ("LD_", "DYLD_", "COMPLUS_", "COR_")


def validate_http_headers(headers: dict[str, str] | None) -> dict[str, str]:
    """Return a defensive copy of safe static HTTP headers."""
    if not headers:
        return {}
    if len(headers) > MAX_CREDENTIAL_ITEMS:
        raise MCPCredentialValidationError(f"At most {MAX_CREDENTIAL_ITEMS} static headers are allowed")

    result: dict[str, str] = {}
    total = 0
    for name, value in headers.items():
        if not isinstance(name, str) or not _HEADER_NAME.fullmatch(name):
            raise MCPCredentialValidationError("Static header name is not a valid HTTP field name")
        name_bytes = len(name.encode("ascii"))
        if name_bytes > MAX_HEADER_NAME_BYTES:
            raise MCPCredentialValidationError("Static header name is too long")
        if name.lower() in _TRANSPORT_HEADERS:
            raise MCPCredentialValidationError(f"Static header '{name}' is controlled by the MCP transport")
        if not isinstance(value, str):
            raise MCPCredentialValidationError(f"Static header '{name}' must have a string value")
        try:
            # HTTPX's string-header API accepts ASCII. obs-text can be valid
            # on the wire but would require a bytes API that this public
            # credential contract deliberately does not expose.
            encoded = value.encode("ascii")
        except UnicodeEncodeError:
            raise MCPCredentialValidationError(f"Static header '{name}' contains unsupported characters") from None
        if len(encoded) > MAX_HEADER_VALUE_BYTES:
            raise MCPCredentialValidationError(f"Static header '{name}' value is too large")
        # Among the ASCII values supported by this API, allow HTAB and
        # SP/VCHAR. Reject every other C0 control and DEL, including CR/LF.
        if any(byte < 0x20 and byte != 0x09 or byte == 0x7F for byte in encoded):
            raise MCPCredentialValidationError(f"Static header '{name}' contains an invalid HTTP field value")
        total += name_bytes + len(encoded)
        if total > MAX_HEADER_TOTAL_BYTES:
            raise MCPCredentialValidationError("Static headers exceed the total size limit")
        result[name] = value
    return result


def validate_stdio_env(server_env: dict[str, str] | None) -> dict[str, str]:
    """Return a defensive copy of safe explicit stdio environment values."""
    if not server_env:
        return {}
    if len(server_env) > MAX_CREDENTIAL_ITEMS:
        raise MCPCredentialValidationError(f"At most {MAX_CREDENTIAL_ITEMS} stdio environment values are allowed")

    result: dict[str, str] = {}
    total = 0
    for name, value in server_env.items():
        if not isinstance(name, str) or not _ENV_NAME.fullmatch(name):
            raise MCPCredentialValidationError("Stdio environment name is invalid")
        normalized = name.upper()
        if normalized in _DANGEROUS_ENV_EXACT or normalized.startswith(_DANGEROUS_ENV_PREFIXES):
            raise MCPCredentialValidationError(f"Stdio environment variable '{name}' may alter process execution")
        if len(name.encode("ascii")) > MAX_ENV_NAME_BYTES:
            raise MCPCredentialValidationError("Stdio environment name is too long")
        if not isinstance(value, str):
            raise MCPCredentialValidationError(f"Stdio environment variable '{name}' must have a string value")
        encoded = value.encode("utf-8")
        if len(encoded) > MAX_ENV_VALUE_BYTES:
            raise MCPCredentialValidationError(f"Stdio environment variable '{name}' value is too large")
        if "\x00" in value or "\r" in value or "\n" in value:
            raise MCPCredentialValidationError(f"Stdio environment variable '{name}' contains an invalid value")
        total += len(name) + len(encoded)
        if total > MAX_ENV_TOTAL_BYTES:
            raise MCPCredentialValidationError("Stdio environment values exceed the total size limit")
        result[name] = value
    return result
