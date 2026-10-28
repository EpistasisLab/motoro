"""Portable provider configuration and metadata, with no application state."""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, JsonValue, SecretStr, model_validator

from motoro.services.credentials import foundry_api_base
from motoro.services.model_capabilities import ModelCapabilities

type Provenance = Literal["provider", "catalog", "unknown"]
type ErrorKind = Literal[
    "credentials_missing",
    "credential_rejected",
    "rate_limited",
    "timeout",
    "network",
    "http",
    "invalid_response",
    "unavailable",
    "empty",
    "unsupported",
]


class ProviderConfig(BaseModel):
    """Explicit connection only; never reads environment or a credential resolver.

    Bases must be HTTP URLs (Azure inference resource names are also accepted).
    Credentials belong in api_key, never in URLs. No redirects are followed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    provider: Literal["anthropic", "openai", "azure_foundry", "openrouter", "local"]
    api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)
    api_base: str | None = None
    azure_project_endpoint: str | None = None

    @model_validator(mode="after")
    def validate_urls(self) -> ProviderConfig:
        key = self.api_key.get_secret_value() if self.api_key else None
        for name in ("api_base", "azure_project_endpoint"):
            value = getattr(self, name)
            if value is None:
                continue
            if key and key in value:
                raise ValueError("Credentials must be supplied separately from endpoints")
            if name == "api_base" and self.provider == "azure_foundry":
                # Validate the same normalized endpoint inference uses, while
                # retaining the supplied value for inconclusive check results.
                normalized = foundry_api_base(value)
                if not value.strip().startswith(("http://", "https://")) and urlsplit(normalized).path:
                    raise ValueError("Azure resource names must resolve to a host without a path")
                value = normalized
            parts = urlsplit(value)
            if (
                parts.scheme not in {"http", "https"}
                or not parts.hostname
                or parts.username
                or parts.password
                or parts.query
                or parts.fragment
            ):
                raise ValueError("Endpoints must be HTTP URLs without credentials, queries, or fragments")
        return self


class ProviderError(BaseModel):
    kind: ErrorKind
    http_status: int | None = None


class DefaultValue(BaseModel):
    """None alone is unknown. 0, False, '', [] and {} are known values."""

    value: JsonValue = None
    source: Provenance = "unknown"

    @property
    def known(self) -> bool:
        return self.value is not None

    @model_validator(mode="after")
    def consistent_source(self) -> DefaultValue:
        if self.known == (self.source == "unknown"):
            raise ValueError("Known values require provenance; unknown values require source='unknown'")
        return self


class ModelDefaults(BaseModel):
    """Sampling defaults project existing capability fields, never app policy.

    Other advertised defaults are in additional. Token limits are not defaults.
    """

    temperature: DefaultValue = Field(default_factory=DefaultValue)
    effort: DefaultValue = Field(default_factory=DefaultValue)
    additional: dict[str, DefaultValue] = Field(default_factory=dict)


class ModelInfo(BaseModel):
    id: str
    label: str | None = None
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    capability_sources: dict[str, Provenance] = Field(default_factory=dict)
    defaults: ModelDefaults = Field(default_factory=ModelDefaults)


class ModelListing(BaseModel):
    models: list[ModelInfo] = Field(default_factory=list)
    source: Literal["api", "static", "error"]
    note: str | None = None
    error: ProviderError | None = None


class ConnectionCheck(BaseModel):
    status: Literal["ok", "failed", "unknown"]
    detail: str
    endpoint: str | None
    models: list[ModelInfo] | None = None
    error: ProviderError | None = None
