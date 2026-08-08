"""Per-model capability metadata — the single source of truth for which sampling
controls a model accepts.

Background: the newest adaptive-thinking Claude models (Opus 4.7, Opus 4.8, Fable 5)
removed the sampling parameters ``temperature`` / ``top_p`` / ``top_k``. Sending an
explicit ``temperature`` returns HTTP 400 ("temperature is deprecated for this model").
These models are steered instead with Anthropic's ``output_config.effort`` dial
(``low | medium | high | xhigh | max``). litellm maps the unified ``reasoning_effort``
param onto ``output_config.effort`` for these models, so the LLM service sends
``reasoning_effort`` at the call sites for effort-based models.

This module decides, per model, whether to expose **temperature** or **effort** — and
which effort levels are valid (``xhigh`` / ``max`` are gated per model by the provider).
The backend, the SDK, and the GUI all resolve capabilities through here (the GUI/SDK via
the ``/api/model-capabilities`` endpoint) so the temperature-vs-effort decision is made in
exactly one place.

NOTE: litellm's own ``get_supported_openai_params`` is NOT a reliable oracle here — it
reports ``temperature`` as supported for ``claude-opus-4-8`` even though the wire call
400s. Hence this hand-maintained registry.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# Full Anthropic effort ladder, low -> max. Individual models gate the top rungs
# (``xhigh`` / ``max``); each registry entry lists only the levels it actually accepts.
EFFORT_LEVELS_FULL: list[str] = ["low", "medium", "high", "xhigh", "max"]
DEFAULT_EFFORT: str = "medium"


class ModelCapabilities(BaseModel):
    """What sampling controls a given model accepts."""

    supports_temperature: bool = Field(
        default=True,
        description="Whether the model accepts an explicit `temperature`. False for "
        "adaptive-thinking models that 400 on it (Opus 4.7/4.8, Fable 5).",
    )
    supports_effort: bool = Field(
        default=False,
        description="Whether the model accepts an `effort` dial (output_config.effort).",
    )
    effort_levels: list[str] = Field(
        default_factory=list,
        description="Valid effort levels for this model (subset of EFFORT_LEVELS_FULL).",
    )
    default_effort: str | None = Field(
        default=None,
        description="Suggested default effort level when effort is the active control.",
    )


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
    default_effort=DEFAULT_EFFORT,
)

# Temperature-based models (the default for anything not listed below).
DEFAULT_CAPABILITIES = ModelCapabilities(
    supports_temperature=True,
    supports_effort=False,
    effort_levels=[],
    default_effort=None,
)

# Registry keyed by normalized (lowercase, provider-prefix-stripped) model id.
# Substring matching covers date-suffixed / Foundry-deployed variants.
_REGISTRY: dict[str, ModelCapabilities] = {
    "claude-opus-5": _EFFORT_ONLY,
    "claude-opus-4-8": _EFFORT_ONLY,
    "claude-opus-4-7": _EFFORT_ONLY,
    "claude-fable-5": _EFFORT_ONLY,
    "claude-sonnet-5": _EFFORT_ONLY,
}


def _normalize(model: str) -> str:
    """Lowercase and strip any ``provider/`` prefix from a model string."""
    m = (model or "").lower().strip()
    if "/" in m:
        m = m.rsplit("/", 1)[-1]
    return m


def get_capabilities(model: str) -> ModelCapabilities:
    """Resolve capabilities for a model string. Unknown models default to
    temperature-based (the safe, backward-compatible choice)."""
    m = _normalize(model)
    if m in _REGISTRY:
        return _REGISTRY[m]
    for key, caps in _REGISTRY.items():
        if key in m:
            return caps
    return DEFAULT_CAPABILITIES


def supports_temperature(model: str) -> bool:
    """True if an explicit ``temperature`` should be sent for this model."""
    return get_capabilities(model).supports_temperature


def supports_effort(model: str) -> bool:
    """True if this model accepts an ``effort`` dial instead of temperature."""
    return get_capabilities(model).supports_effort


def _entry(provider: str, model: str, label: str) -> ModelCatalogEntry:
    return ModelCatalogEntry(
        provider=provider,
        model=model,
        label=label,
        capabilities=get_capabilities(model),
    )


# Curated, selectable models with their capabilities — feeds the GUI's
# temperature-vs-effort toggle and the SDK. Not exhaustive; arbitrary model strings
# still resolve via get_capabilities (defaulting to temperature-based).
CATALOG: list[ModelCatalogEntry] = [
    _entry("anthropic", "claude-opus-5", "Claude Opus 5"),
    _entry("anthropic", "claude-opus-4-8", "Claude Opus 4.8"),
    _entry("anthropic", "claude-opus-4-7", "Claude Opus 4.7"),
    _entry("anthropic", "claude-fable-5", "Claude Fable 5"),
    _entry("anthropic", "claude-sonnet-5", "Claude Sonnet 5"),
    _entry("anthropic", "claude-haiku-4-5", "Claude Haiku 4.5"),
    _entry("openai", "gpt-4o", "GPT-4o"),
    _entry("openai", "gpt-4o-mini", "GPT-4o Mini"),
]
