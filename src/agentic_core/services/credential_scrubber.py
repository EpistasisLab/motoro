"""Defence-in-depth credential scrubber for error messages and log output.

Patterns target provider-prefixed key formats only — tight enough to avoid
destroying UUIDs, SHAs, JWTs, and other high-entropy-but-benign strings.
"""

from __future__ import annotations

import re
from collections.abc import MutableMapping
from typing import Any

_REDACTED = "[REDACTED_API_KEY]"

# Provider-prefixed API key patterns.
# Ordered longest-prefix-first to avoid partial matches (sk-ant- before sk-).
_KEY_PATTERNS: list[re.Pattern[str]] = [
    # Anthropic: sk-ant-api03-... or sk-ant-...
    re.compile(r"sk-ant-[A-Za-z0-9_-]{10,}"),
    # OpenAI project keys: sk-proj-...
    re.compile(r"sk-proj-[A-Za-z0-9_-]{10,}"),
    # Generic OpenAI / other OpenAI-style keys: sk-... (≥20 chars after prefix)
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    # AWS access key IDs: AKIA...
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    # Google API keys: AIza...
    re.compile(r"\bAIza[A-Za-z0-9_-]{35}\b"),
    # GitHub personal access tokens: ghp_... / gho_... / ghs_... / ghr_...
    re.compile(r"\bgh[pos]_[A-Za-z0-9]{36,}\b"),
    # Slack bot/user tokens: xoxb-... / xoxp-...
    re.compile(r"\bxox[bpu]-[A-Za-z0-9-]{20,}\b"),
    # HuggingFace tokens: hf_...
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    # Cohere API keys: co-... (lower-case prefix)
    re.compile(r"\bco-[A-Za-z0-9_-]{20,}\b"),
    # OpenRouter keys: sk-or-...
    re.compile(r"\bsk-or-[A-Za-z0-9_-]{20,}"),
]

# Field names whose values should always be redacted regardless of format.
_SENSITIVE_KEYS: frozenset[str] = frozenset(
    {"api_key", "apikey", "api-key", "secret", "password", "passwd", "token", "auth_token"}
)


def scrub(text: str) -> str:
    """Replace provider-prefixed API key patterns in *text* with a redaction marker.

    Does NOT redact arbitrary 40-char strings — that would destroy SHAs, JWTs,
    UUIDs, and other identifiers.  Only patterns with provider-specific prefixes
    are matched.
    """
    for pattern in _KEY_PATTERNS:
        text = pattern.sub(_REDACTED, text)
    return text


def _scrub_value(value: Any) -> Any:
    """Recursively scrub a value: walks nested dicts and lists."""
    if isinstance(value, str):
        return scrub(value)
    if isinstance(value, dict):
        return {k: (_REDACTED if k.lower() in _SENSITIVE_KEYS else _scrub_value(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub_value(item) for item in value]
    return value


def redact_tool_args(args: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *args* safe for persistence and logging.

    Applies the same deep-walk logic as ``_scrub_value``: sensitive key names
    are always replaced by the redaction marker; string values are scrubbed for
    provider-prefixed key patterns; nested dicts/lists are walked recursively.

    Usage::

        from agentic_core.services.credential_scrubber import redact_tool_args
        safe_args = redact_tool_args(step.tool_args or {})
        record = ToolCallRecord(arguments=safe_args, ...)
    """
    return {k: (_REDACTED if k.lower() in _SENSITIVE_KEYS else _scrub_value(v)) for k, v in args.items()}


def scrub_structlog_processor(
    logger: Any,
    method_name: str,
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """structlog processor that scrubs credential-like strings from log values.

    Walks nested dicts and lists so that structured log fields with nested
    objects are also protected.  Sensitive field names (api_key, password,
    secret, token) are always redacted regardless of value format.
    """
    for key, value in list(event_dict.items()):
        if key.lower() in _SENSITIVE_KEYS:
            event_dict[key] = _REDACTED
        elif isinstance(value, str) and len(value) > 20:
            event_dict[key] = scrub(value)
        elif isinstance(value, (dict, list)):
            event_dict[key] = _scrub_value(value)
    return event_dict
