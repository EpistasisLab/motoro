"""Where LLM credentials come from — a product decision, not a core one.

Core knows it needs an API key, a base URL, and sometimes a region or an API
version. It deliberately does not know *where* those live. ARES stores them per
user in a ``user_llm_settings`` table, encrypted at rest; another product might
read one environment variable, or call a secret manager, or hand every call the
same key. None of that belongs in a runtime.

So resolution is a hook. Core's default reads the credential off the
:class:`ModelConfig` and nothing else — no environment fallback, so a
misconfigured call fails loudly instead of silently borrowing a shared key.

A product installs its own resolver at startup::

    from agentic_core.services.credentials import set_credential_resolver

    async def resolve(config, principal_id):
        row = await load_setting(principal_id, config.provider.value)
        if row is None:
            return None                      # fall back to the ModelConfig
        return {"api_key": decrypt(row.key), "api_base": row.base_url,
                "model": None, "api_version": None, "aws_region_name": None}

    set_credential_resolver(resolve)

Returning ``None`` means "no opinion" and defers to the ``ModelConfig``, which
lets a resolver handle only the providers it knows about.

The identifier passed to a resolver is an **opaque principal id**. Core never
joins it to anything; it exists only to be handed back to the product that
issued it.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable

#: Resolved connection details. Keys match the kwargs litellm expects, so a
#: resolver returns the same shape core's default does.
type Connection = dict[str, str | None]

#: ``(model_config, principal_id) -> Connection | None``. ``None`` defers to the
#: credential carried on the ``ModelConfig``.
type CredentialResolver = Callable[..., Awaitable[Connection | None]]

_resolver: CredentialResolver | None = None


def set_credential_resolver(fn: CredentialResolver | None) -> None:
    """Install the resolver consulted when a ``ModelConfig`` carries no key.

    Passing ``None`` restores the default of using the ``ModelConfig`` alone.
    """
    global _resolver
    _resolver = fn


def get_credential_resolver() -> CredentialResolver | None:
    """Return the installed resolver, or ``None`` if credentials come from config."""
    return _resolver


async def resolve(
    default: Connection,
    config: object,
    principal_id: uuid.UUID | None,
) -> Connection:
    """Apply the installed resolver, falling back to *default*.

    *default* is what core's own config-only resolution produced, so a missing or
    abstaining resolver is indistinguishable from having none installed.
    """
    if principal_id is None:
        return default
    resolver = _resolver
    if resolver is None:
        return default
    resolved = await resolver(config, principal_id)
    return resolved if resolved is not None else default
