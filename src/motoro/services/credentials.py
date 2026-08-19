"""Where LLM credentials come from.

Core knows it needs an API key, a base URL, and sometimes a region or an API
version. It does **not** hardcode where those live — resolution is a hook, and a
product can install its own. ARES resolves per user from an encrypted
``user_llm_settings`` table; another product might call a secret manager.

Core ships one resolver by default: :func:`env_credential_resolver`, which reads
:class:`CoreSettings`. Those fields carry bare-name validation aliases
(``ANTHROPIC_API_KEY``, ``ANTHROPIC_FOUNDRY_API_KEY``, …), so putting the
conventional variables in a ``.env`` is enough to make a run work with no code.

**Why a resolver is needed at all, and not just ``ModelConfig.api_key``.**
``ModelConfig.api_key`` is declared ``exclude=True`` — deliberately, so a
credential is never serialized into an API response. But an agent's model config
is *persisted*: ``create_agent`` writes ``model_config_data`` to the database and
``execute_run`` rebuilds the config from it. The key does not survive that round
trip. A key set at agent-creation time is therefore gone by the time the run
executes, and the provider call fails with a confusing 401 rather than a missing
-credential error. Resolution has to happen at call time, from something durable.

Order of precedence, per call:

1. A credential on the :class:`ModelConfig` (an explicit per-call override).
2. The installed resolver — :func:`env_credential_resolver` unless replaced.
3. Nothing, and the provider call fails loudly rather than silently borrowing a
   shared key.

A product installs its own resolver at startup::

    from motoro.services.credentials import set_credential_resolver

    async def resolve(config, principal_id):
        row = await load_setting(principal_id, config.provider.value)
        if row is None:
            return None                      # defer to the ModelConfig
        return {"api_key": decrypt(row.key), "api_base": row.base_url,
                "model": None, "api_version": None, "aws_region_name": None}

    set_credential_resolver(resolve)

Returning ``None`` means "no opinion", which lets a resolver handle only the
providers it knows about.

The identifier passed to a resolver is an **opaque principal id**. Core never
joins it to anything; it exists only to be handed back to whoever issued it.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import uuid

#: Resolved connection details. Keys match the kwargs litellm expects, so a
#: resolver returns the same shape core's default does.
type Connection = dict[str, str | None]

#: ``(model_config, principal_id) -> Connection | None``. ``None`` defers to the
#: credential carried on the ``ModelConfig``.
type CredentialResolver = Callable[..., Awaitable[Connection | None]]

_resolver: CredentialResolver | None = None
_use_default = True


def foundry_api_base(resource: str) -> str:
    """Base URL for an Anthropic-on-Foundry resource.

    The **bare host** — no ``/anthropic`` suffix. litellm's ``azure_ai`` route
    appends ``/anthropic/v1/messages`` itself, so a base that already ends in
    ``/anthropic`` double-counts the segment for Claude and 404s for every
    OpenAI-style model. ARES normalizes the same way
    (``user_llm_settings_service.normalize_azure_foundry_api_base``), because
    Foundry's own console displays the ``/anthropic`` form and users paste it.

    A *resource* that is already a URL is returned as-is, so an unusual endpoint
    can be configured directly.
    """
    resource = resource.strip().rstrip("/")
    if resource.lower().endswith("/anthropic"):
        resource = resource[: -len("/anthropic")]
    if resource.startswith("http"):
        return resource
    return f"https://{resource}.services.ai.azure.com"


async def env_credential_resolver(config: Any, principal_id: uuid.UUID | None = None) -> Connection | None:
    """Core's default resolver: read credentials from :class:`CoreSettings`.

    Ignores *principal_id* — settings are process-wide, so every run in this
    process uses the same credential. A product needing per-principal
    credentials installs its own resolver.

    Returns ``None`` for a provider it has no configured credential for, so the
    caller falls back to whatever the ``ModelConfig`` carries.
    """
    from motoro.config import settings

    provider = getattr(getattr(config, "provider", None), "value", None)
    model = getattr(config, "model", "") or ""
    api_base = getattr(config, "api_base", None)

    if provider == "azure_foundry":
        key = settings.anthropic_foundry_api_key
        resource = settings.anthropic_foundry_resource
        if not key:
            return None
        if not api_base:
            if not resource:
                # A key without a resource cannot be used: the key does not
                # encode an endpoint, and Foundry's host is derived from the
                # resource name. Say so rather than sending a keyed request
                # into the void.
                raise ValueError(
                    "azure_foundry needs ANTHROPIC_FOUNDRY_RESOURCE as well as "
                    "ANTHROPIC_FOUNDRY_API_KEY — the base URL is derived from the "
                    "resource name, so the key alone is not enough."
                )
            api_base = foundry_api_base(resource)
        else:
            api_base = foundry_api_base(api_base)
        return {
            # litellm's azure_ai route needs the prefix even when the credential
            # is supplied explicitly.
            "model": f"azure_ai/{model}",
            "api_key": key,
            "api_base": api_base,
            # Claude on Foundry is served through the Anthropic Messages
            # passthrough, which 404s if an api-version query param is attached.
            "api_version": None,
            "aws_region_name": None,
        }

    if provider == "anthropic":
        key = settings.anthropic_api_key
        if not key:
            return None
        return {"model": None, "api_key": key, "api_base": api_base, "api_version": None, "aws_region_name": None}

    if provider == "openai":
        key = settings.openai_api_key
        if not key:
            return None
        return {"model": None, "api_key": key, "api_base": api_base, "api_version": None, "aws_region_name": None}

    if provider == "bedrock":
        # litellm builds the bedrock/<model> string itself and reads the bearer
        # token from api_key; the region rides on aws_region_name.
        region = settings.bedrock_region
        if not region:
            return None
        return {
            "model": None,
            "api_key": getattr(config, "api_key", None),
            "api_base": api_base,
            "api_version": None,
            "aws_region_name": region,
        }

    return None


def set_credential_resolver(fn: CredentialResolver | None) -> None:
    """Install the resolver consulted when a ``ModelConfig`` carries no key.

    Passing ``None`` restores :func:`env_credential_resolver`. To disable
    resolution entirely — so only an explicit ``ModelConfig.api_key`` is ever
    used — call :func:`disable_credential_resolution`.
    """
    global _resolver, _use_default
    _resolver = fn
    _use_default = fn is None


def disable_credential_resolution() -> None:
    """Use only the credential on the ``ModelConfig``; consult no resolver."""
    global _resolver, _use_default
    _resolver = None
    _use_default = False


def get_credential_resolver() -> CredentialResolver | None:
    """Return the resolver that will be consulted, or ``None`` if disabled."""
    if _resolver is not None:
        return _resolver
    return env_credential_resolver if _use_default else None


async def resolve(default: Connection, config: Any, principal_id: uuid.UUID | None) -> Connection:
    """Apply the active resolver, falling back to *default*.

    *default* is what core's own config-only resolution produced, so a missing or
    abstaining resolver is indistinguishable from having none installed.

    Note this runs regardless of *principal_id* — the default resolver reads
    process settings and needs no principal. A resolver that does need one should
    return ``None`` when it is absent.
    """
    resolver = get_credential_resolver()
    if resolver is None:
        return default
    resolved = await resolver(config, principal_id)
    return resolved if resolved is not None else default
