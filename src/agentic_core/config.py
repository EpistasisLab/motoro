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

    # LLM bridge
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

    # MCP subprocess environment isolation. Comma-separated extra env var names
    # allowed through to MCP subprocesses; the built-in allowlist (PATH, HOME,
    # SHELL, TERM, USER, LOGNAME, LANG, LC_ALL) is always included.
    mcp_allowed_env_vars: str = ""

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

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


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
