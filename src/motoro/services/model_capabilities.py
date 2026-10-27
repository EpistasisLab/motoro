"""Shared factual sampling metadata and a small curated model registry.

Products can validate live metadata here and populate resolved_capabilities.
Independent selection, fallback policy, and validation live in services.sampling.
LiteLLM's parameter list alone cannot describe reasoning-mode restrictions.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Full Anthropic effort ladder, low -> max. Individual models gate the top rungs
# (``xhigh`` / ``max``); each registry entry lists only the levels it actually accepts.
EFFORT_LEVELS_FULL: list[str] = ["low", "medium", "high", "xhigh", "max"]
DEFAULT_EFFORT: str = "medium"


class ModelCapabilities(BaseModel):
    """What a model accepts, independent of configuration. None means unknown.

    Defaults are advertised facts, never application policy. An absent combination
    allowlist imposes no additional restriction; it does not prove compatibility.
    """

    model_config = ConfigDict(extra="forbid", revalidate_instances="always")

    supports_temperature: bool | None = None
    supports_effort: bool | None = None
    default_temperature: float | None = Field(default=None, ge=0.0, le=2.0, allow_inf_nan=False)
    effort_levels: list[str] | None = Field(default=None, description="Accepted levels; None means unknown.")
    default_effort: str | None = Field(
        default=None,
        description="Advertised provider default; None means unadvertised, 'none' disables reasoning.",
    )
    temperature_effort_levels: list[str] | None = Field(
        default=None,
        description="When supplied, temperature requires resolved effort in this allowlist. "
        "None imposes no additional restriction; [] forbids temperature in every effort mode.",
    )

    @model_validator(mode="after")
    def validate_effort_metadata(self) -> ModelCapabilities:
        for name, levels in (
            ("effort_levels", self.effort_levels),
            ("temperature_effort_levels", self.temperature_effort_levels),
        ):
            if levels is not None:
                if any(not level.strip() or level != level.strip() for level in levels):
                    raise ValueError(f"{name} must contain nonblank, unpadded values")
                if len(set(levels)) != len(levels):
                    raise ValueError(f"{name} must contain unique values")
        if self.default_effort is not None:
            if not self.default_effort.strip() or self.default_effort != self.default_effort.strip():
                raise ValueError("default_effort must be nonblank and unpadded")
            if self.effort_levels is not None and self.default_effort not in self.effort_levels:
                raise ValueError("default_effort must be one of effort_levels")
        if self.supports_effort is not True and (self.effort_levels or self.default_effort is not None):
            raise ValueError("Effort metadata requires known effort support")
        if self.supports_effort is True and self.effort_levels == []:
            raise ValueError("Use effort_levels=None for unknown accepted levels")
        if self.default_temperature is not None and self.supports_temperature is not True:
            raise ValueError("default_temperature requires known temperature support")
        if self.temperature_effort_levels is not None:
            if self.supports_temperature is not True or self.supports_effort is not True:
                raise ValueError("Combination restrictions require both controls to be supported")
            if self.effort_levels is not None and not set(self.temperature_effort_levels) <= set(self.effort_levels):
                raise ValueError("temperature_effort_levels must be a subset of effort_levels")
        return self


class ModelCatalogEntry(BaseModel):
    """A curated, selectable model plus its capabilities (for UI dropdowns / SDK)."""

    provider: str
    model: str
    label: str
    capabilities: ModelCapabilities


# Effort-only models (temperature 400s). Opus 4.7/4.8 and Fable 5 accept the full ladder
# including xhigh and max (verified against litellm's model map).
_EFFORT_ONLY = ModelCapabilities(
    supports_temperature=False,
    supports_effort=True,
    effort_levels=list(EFFORT_LEVELS_FULL),
    default_effort="high",
)

# Explicit override for custom temperature-only models and known registry entries.
TEMPERATURE_ONLY_CAPABILITIES = ModelCapabilities(
    supports_temperature=True,
    supports_effort=False,
    effort_levels=[],
    default_effort=None,
)

# Unrecognized models have unknown support, accepted levels, and defaults.
DEFAULT_CAPABILITIES = ModelCapabilities()

# Original GPT-5 supports minimal, but not the disabled reasoning mode "none".
_OPENAI_REASONING = ModelCapabilities(
    supports_temperature=False,
    supports_effort=True,
    effort_levels=["minimal", "low", "medium", "high"],
    default_effort=DEFAULT_EFFORT,
)

# GPT-5.1/5.2 accept temperature only with effort="none". Original GPT-5
# rejects temperature in every mode. Official documentation: README.md.
_OPENAI_51 = ModelCapabilities(
    supports_temperature=True,
    supports_effort=True,
    effort_levels=["none", "low", "medium", "high"],
    default_effort="none",
    temperature_effort_levels=["none"],
)
_OPENAI_52 = ModelCapabilities(
    supports_temperature=True,
    supports_effort=True,
    effort_levels=["none", "low", "medium", "high", "xhigh"],
    default_effort="none",
    temperature_effort_levels=["none"],
)

# Registry keyed by normalized (lowercase, provider-prefix-stripped) model id.
# Dated deployment variants are also recognized.
_REGISTRY: dict[str, ModelCapabilities] = {
    "claude-opus-5": _EFFORT_ONLY,
    "claude-opus-4-8": _EFFORT_ONLY,
    "claude-opus-4-7": _EFFORT_ONLY,
    "claude-fable-5": _EFFORT_ONLY,
    "claude-sonnet-5": _EFFORT_ONLY,
    # Chat is a distinct non-reasoning model.
    "gpt-5-chat": TEMPERATURE_ONLY_CAPABILITIES,
    "gpt-5-chat-latest": TEMPERATURE_ONLY_CAPABILITIES,
    "gpt-5.2": _OPENAI_52,
    "gpt-5.1": _OPENAI_51,
    "gpt-5-mini": _OPENAI_REASONING,
    "gpt-5-nano": _OPENAI_REASONING,
    "gpt-5": _OPENAI_REASONING,
    "claude-haiku-4-5": TEMPERATURE_ONLY_CAPABILITIES,
    "gpt-4o-mini": TEMPERATURE_ONLY_CAPABILITIES,
    "gpt-4o": TEMPERATURE_ONLY_CAPABILITIES,
}


def _normalize(model: str) -> str:
    """Lowercase and strip any ``provider/`` prefix from a model string."""
    m = (model or "").lower().strip()
    if "/" in m:
        m = m.rsplit("/", 1)[-1]
    return m


def get_capabilities(model: str, *, default: ModelCapabilities = DEFAULT_CAPABILITIES) -> ModelCapabilities:
    """Resolve capabilities, using ``default`` only when no registry entry matches.

    Without a caller override, unknown models have unknown sampling metadata.
    Exact and dated registry matches always take precedence over the default.
    """
    m = _normalize(model)
    if m in _REGISTRY:
        return _REGISTRY[m]
    for key in sorted(_REGISTRY, key=len, reverse=True):
        # Do not apply a base model's rules to unrelated variants (e.g. -pro).
        suffix = m.removeprefix(key)
        if suffix != m and suffix.startswith("-") and suffix[1:5].isdigit():
            return _REGISTRY[key]
    return default


def supports_temperature(model: str) -> bool | None:
    """True if an explicit ``temperature`` should be sent for this model."""
    return get_capabilities(model).supports_temperature


def supports_effort(model: str) -> bool | None:
    """True if this model accepts an effort dial; None if unknown."""
    return get_capabilities(model).supports_effort


def _entry(provider: str, model: str, label: str) -> ModelCatalogEntry:
    return ModelCatalogEntry(
        provider=provider,
        model=model,
        label=label,
        capabilities=get_capabilities(model),
    )


# Curated, selectable models with their capabilities — feeds the GUI's
# sampling controls and the SDK. Arbitrary model strings still resolve via
# get_capabilities (defaulting to unknown support).
CATALOG: list[ModelCatalogEntry] = [
    _entry("anthropic", "claude-opus-5", "Claude Opus 5"),
    _entry("anthropic", "claude-opus-4-8", "Claude Opus 4.8"),
    _entry("anthropic", "claude-opus-4-7", "Claude Opus 4.7"),
    _entry("anthropic", "claude-fable-5", "Claude Fable 5"),
    _entry("anthropic", "claude-sonnet-5", "Claude Sonnet 5"),
    _entry("anthropic", "claude-haiku-4-5", "Claude Haiku 4.5"),
    _entry("openai", "gpt-5", "GPT-5"),
    _entry("openai", "gpt-5-mini", "GPT-5 Mini"),
    _entry("openai", "gpt-5-nano", "GPT-5 Nano"),
    _entry("openai", "gpt-4o", "GPT-4o"),
    _entry("openai", "gpt-4o-mini", "GPT-4o Mini"),
]
