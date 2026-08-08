"""Core configuration.

Unlike the ARES ``Settings`` this was derived from, ``CoreSettings`` sets **no**
``env_prefix``. The prefix is a product concern: ``ares`` binds ``ARES_``,
``ecoxai`` binds its own, and core must not presume either. A product subclasses
``CoreSettings``, adds its own fields, sets its prefix, and installs the instance
with :func:`configure` before anything reads it.

Every field here is one that a slice-1 module actually reads. The ARES original
carries ~70 fields; the SRPA loop, LLM bridge, and MCP client between them
reference 10. Fields arrive when a slice needs them, not in anticipation — so
this file stays a description of what core does rather than a description of what
ARES happens to configure.

The module-level ``settings`` object is a proxy, not an instance. That keeps the
``from agentic_core.config import settings`` / ``settings.field`` call pattern
working verbatim (there are 62 such sites in ARES) while still letting the
product decide, at startup, what the values are.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CoreSettings(BaseSettings):
    """Settings the core platform reads. Products subclass and extend this."""

    # Environment
    env: str = "development"  # development | production
    debug: bool = False
    log_level: str = "INFO"

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres"
    db_pool_recycle_seconds: int = 3600  # recycle DB connections after 1 hour

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Provider credentials.
    #
    # These carry `validation_alias`, which makes pydantic-settings read the
    # bare environment name and ignore the product's `env_prefix`. That is
    # deliberate: `ANTHROPIC_API_KEY` and friends are conventional names that
    # tooling, CI, and developers already set, and forcing them under a
    # per-product prefix (`ARES_ANTHROPIC_API_KEY`) buys nothing.
    #
    # Read by the built-in resolver in ``services.credentials``. A product that
    # keeps credentials elsewhere — a per-user table, a secret manager — installs
    # its own resolver and leaves these empty.
    anthropic_api_key: str = Field(default="", validation_alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    # Anthropic on Microsoft Foundry. The base URL is derived from the resource
    # name, so the key alone is not enough — it does not encode an endpoint.
    anthropic_foundry_api_key: str = Field(default="", validation_alias="ANTHROPIC_FOUNDRY_API_KEY")
    anthropic_foundry_resource: str = Field(default="", validation_alias="ANTHROPIC_FOUNDRY_RESOURCE")
    # AWS region for Bedrock; litellm reads the bearer token from api_key.
    bedrock_region: str = Field(default="", validation_alias="AWS_REGION")

    # LLM bridge
    #
    # Per-attempt cap on a single completion (asyncio.wait_for around the
    # litellm call; retried up to 3x via tenacity on a timeout, so a stuck
    # attempt costs up to ~3x this before the run fails). Deployment-
    # dependent, not a core concern: extended-thinking / high reasoning-
    # effort completions and large tool-call contexts can legitimately run
    # well past the 120s default. Raise it in your product's own settings
    # (env-prefixed, e.g. ASAREE_LLM_CALL_TIMEOUT_SECONDS) if you see runs
    # failing with a "Hook '...' timed out after {hook_timeout_seconds}s"
    # abort -- that message names hook_timeout_seconds regardless of which
    # inner timeout actually fired, so a fast per-attempt timeout here is a
    # common misdiagnosed cause.
    llm_call_timeout_seconds: int = 120
    # Instructor retries for structured-output validation failures.
    llm_structured_max_retries: int = 3

    # Tool execution
    tool_timeout_seconds: int = 300
    # Per-step cap (chars) on prior tool results replayed into the Act phase's
    # tool-selection context. Tool dumps (full data dictionaries, per-feature
    # tables) can be large; replaying them verbatim for every subsequent step
    # grows a run's own LLM context superlinearly. Head-truncated past this cap
    # so the ids/keys near the start survive. 0 disables truncation.
    act_prior_result_max_chars: int = 4000

    # Pattern engine. Read by the orchestrator's hook pipeline, so these arrived
    # with the pattern engine rather than with the bare SRPA loop.
    #
    # Wraps one hook invocation (e.g. one reason_act PRE_ACT turn), not a
    # whole run and not a whole multi-run workflow. Also deployment-
    # dependent like llm_call_timeout_seconds above -- the 30s default is
    # tuned for a generic core deployment, not for a slow provider or a hook
    # that itself makes a full LLM call. Keep this comfortably above
    # llm_call_timeout_seconds (which it wraps indirectly): if the two are
    # close, a slow completion trips this outer timeout instead of the
    # inner one, and the resulting error is harder to diagnose because it
    # reports this value, not the inner timeout that actually applies.
    hook_timeout_seconds: int = 30
    # When True, a pattern hook returning a wrong-typed value (a BaseModel that
    # is not the expected phase output, or any non-None / non-HookAction object)
    # aborts the run instead of being logged and dropped. False preserves the
    # best-effort behaviour; flip it on in strict environments.
    fail_on_hook_type_mismatch: bool = False

    # MCP subprocess environment isolation. Comma-separated extra env var names
    # allowed through to MCP subprocesses; the built-in allowlist (PATH, HOME,
    # SHELL, TERM, USER, LOGNAME, LANG, LC_ALL) is always included.
    mcp_allowed_env_vars: str = ""
    # Skip the SSRF private-IP/DNS-rebinding check on registered HTTP/SSE server
    # URLs. Only for trusted dev/research deployments that register servers on
    # the host machine or local network — see security.ssrf_guard.
    mcp_allow_private_urls: bool = False

    # Fernet key (or comma-separated keys, for rotation — see
    # services.encryption) encrypting MCP server auth headers at rest. Not a
    # per-user secret: one server-side key for every row core encrypts. Generate
    # with: python -c 'from cryptography.fernet import Fernet;
    # print(Fernet.generate_key().decode())'
    encryption_key: str = ""

    # Memory. A ``sentence-transformers/`` prefix selects the local backend (no
    # API key, runs in-process); anything else is a litellm-supported remote
    # model. ``embedding_dimensions`` must match the model's actual output width —
    # it sizes the ``memory_entries.embedding`` column at migration time, not the
    # other way around.
    embedding_model: str = "sentence-transformers/BAAI/bge-base-en-v1.5"
    embedding_dimensions: int = 768
    # Chunk size for remote batch-embedding calls, to stay under provider limits.
    embedding_batch_max: int = 2048
    # Stamped onto every row alongside embedding_model, so a deliberate re-embed
    # (prompt template change, normalization fix) can be told apart from
    # "same model, same version" rows during search filtering.
    embedding_version: str = "v1"

    # OpenTelemetry. The service name defaults to the library, not a product —
    # a product that does not override this will at least not claim to be ARES.
    otel_service_name: str = "agentic-core"
    # Prometheus instrument-name prefix. Two products sharing one Prometheus
    # registry must not collide, and a product with existing dashboards should
    # set this to whatever those dashboards already query.
    metrics_prefix: str = "agentic_core"
    otel_exporter_otlp_endpoint: str = ""  # e.g. "http://otel-collector:4317"
    otel_sample_rate: float = 1.0  # 1.0 = 100% sampling
    # True (insecure) is safe for an in-cluster collector; set False for an
    # internet-facing endpoint to enable TLS certificate verification.
    otel_exporter_otlp_insecure: bool = True

    # populate_by_name: the credential fields above carry a validation_alias,
    # which by default makes pydantic accept *only* the alias. Without this, a
    # product (or a test) constructing settings explicitly —
    # `Settings(anthropic_api_key=...)` — would have the value silently ignored.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)


_instance: CoreSettings | None = None
_accessed = False


def configure(settings_instance: CoreSettings) -> None:
    """Install the product's settings instance.

    Must be called before any core module reads a setting. Calling it after a
    read has already happened is an error rather than a silent no-op: half the
    process would be running on defaults and the other half on the product's
    values, which is far harder to diagnose than a startup failure.
    """
    global _instance
    if _accessed and _instance is not None and settings_instance is not _instance:
        raise RuntimeError(
            "agentic_core.config.configure() called after settings were already read. "
            "Call configure() first, before importing modules that read settings."
        )
    _instance = settings_instance


def get_settings() -> CoreSettings:
    """Return the active settings, constructing bare defaults if unconfigured.

    Auto-construction keeps tests and scratch scripts usable without a
    composition root. A product is still expected to call :func:`configure`.
    """
    global _instance, _accessed
    _accessed = True
    if _instance is None:
        _instance = CoreSettings()
    return _instance


def reset_for_testing() -> None:
    """Drop the installed instance. Test-support only."""
    global _instance, _accessed
    _instance = None
    _accessed = False


class _SettingsProxy:
    """Attribute-forwarding proxy so ``settings.field`` resolves lazily.

    Exists so that ``from agentic_core.config import settings`` can be imported
    at module scope — as it is throughout the codebase — without freezing the
    values at import time, which would make :func:`configure` useless.
    """

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        return getattr(get_settings(), name)

    def __repr__(self) -> str:
        return f"<settings proxy -> {_instance!r}>"


settings = _SettingsProxy()
