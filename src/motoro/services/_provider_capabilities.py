"""Provider capability interpretation; shared with the public provider operations."""

from typing import Any

from motoro.services.model_capabilities import EFFORT_LEVELS_FULL, ModelCapabilities, get_capabilities


def _capabilities_from_anthropic(caps: dict[str, Any], model: str = "") -> ModelCapabilities:
    """Map Anthropic's own capability tree onto ModelCapabilities.

    Shape confirmed live against GET /v1/models::

        "effort": {"supported": true,
                   "low": {"supported": true}, ..., "max": {"supported": true}}

    This is strictly better than Motoro's hand-maintained _REGISTRY for
    this provider: the registry gives every entry one hardcoded effort ladder,
    whereas the real ladders differ per model (opus-4-5 stops at high,
    sonnet-4-6/opus-4-6 have no xhigh, the 5-series has the full set) and
    older models like haiku-4-5/sonnet-4-5 report effort unsupported outright.

    The current API does not state temperature support or sampling defaults.
    Preserve those independent facts from the registry unless explicitly
    advertised, and leave missing defaults unknown for Motoro to resolve.
    """
    registry = get_capabilities(model)
    effort = caps.get("effort") or {}
    supports_effort = effort.get("supported", registry.supports_effort)
    levels = (
        [level for level in EFFORT_LEVELS_FULL if (effort.get(level) or {}).get("supported")]
        if "supported" in effort
        else registry.effort_levels
    )
    default = effort.get("default_effort", registry.default_effort)
    temperature = caps.get("temperature") or {}
    supports_temperature = temperature.get("supported", registry.supports_temperature)
    return ModelCapabilities(
        supports_temperature=supports_temperature,
        default_temperature=temperature.get("default", registry.default_temperature)
        if supports_temperature is True
        else None,
        supports_effort=supports_effort,
        effort_levels=levels if levels or supports_effort is False else None,
        default_effort=default if supports_effort and (levels is None or default in levels) else None,
    )


def _capabilities_from_openrouter(entry: dict[str, Any]) -> ModelCapabilities:
    """Read factual support and defaults independently; Motoro owns fallback policy."""
    fallback = get_capabilities(entry["id"])
    if not {"supported_parameters", "reasoning", "default_parameters"} & entry.keys():
        return fallback

    parameters = entry.get("supported_parameters") or []
    reasoning = entry.get("reasoning") or {}
    defaults_only = "supported_parameters" not in entry and "reasoning" not in entry
    supports_effort = fallback.supports_effort if defaults_only else False
    levels = fallback.effort_levels if defaults_only else None
    if "supported_efforts" in reasoning:
        # null means every gateway effort is accepted; omission means no selector.
        supports_effort = True
        advertised = reasoning["supported_efforts"]
        levels = list(["none", "minimal", *EFFORT_LEVELS_FULL] if advertised is None else advertised)
    elif "reasoning" not in entry and "reasoning_effort" in parameters:
        # Older listings expose the parameter without a per-model effort ladder.
        supports_effort = True
        levels = fallback.effort_levels
    if reasoning.get("mandatory") and levels is not None:
        levels = [level for level in levels if level != "none"]
    if levels == []:
        supports_effort = False
    default = reasoning.get("default_effort", fallback.default_effort if defaults_only else None)
    if not supports_effort or (levels is not None and default not in levels):
        default = None
    supports_temperature = (
        "temperature" in parameters if "supported_parameters" in entry else fallback.supports_temperature
    )
    restrictions = fallback.temperature_effort_levels
    if restrictions is not None and levels is not None:
        restrictions = [level for level in restrictions if level in levels]
    return ModelCapabilities(
        supports_temperature=supports_temperature,
        default_temperature=(entry.get("default_parameters") or {}).get("temperature")
        if supports_temperature is True
        else None,
        supports_effort=supports_effort,
        effort_levels=levels,
        default_effort=default,
        temperature_effort_levels=restrictions if supports_temperature is True and supports_effort else None,
    )
