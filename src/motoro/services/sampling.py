"""Shared sampling resolution for execution, previews, and product integrations.

This module owns policy. It does not discover providers or import execution
schemas. None is unset; the effort string "none" is an explicit disabled mode.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict

from motoro.services.model_capabilities import ModelCapabilities

SamplingSource = Literal["explicit", "advertised_default", "application_fallback", "omitted"]

# Semantic order, never provider-list order. Unknown provider-specific names
# remain usable explicitly or as advertised defaults, but cannot be ranked.
_ACTIVE_EFFORT_ORDER = ("minimal", "low", "medium", "high", "xhigh", "max")


class SamplingConfigurationError(ValueError):
    """Invalid sampling configuration; terminal across retries and fallbacks."""


class UnsupportedEffortError(SamplingConfigurationError):
    """An explicit effort cannot be validated for this model."""


class SamplingCombinationError(SamplingConfigurationError):
    """An explicit temperature conflicts with the resolved reasoning mode."""


class ResolvedSampling(BaseModel):
    """Effective values and provenance; omitted values are never sent on the wire."""

    model_config = ConfigDict(frozen=True)

    temperature: float | None = None
    effort: str | None = None
    temperature_source: SamplingSource = "omitted"
    effort_source: SamplingSource = "omitted"
    temperature_omission_reason: str | None = None
    effort_omission_reason: str | None = None

    def as_kwargs(self) -> dict[str, float | str]:
        """LiteLLM/OpenAI parameter names, containing only effective controls."""
        result: dict[str, float | str] = {}
        if self.temperature is not None:
            result["temperature"] = self.temperature
        if self.effort is not None:
            result["reasoning_effort"] = self.effort
        return result


def resolve_sampling(
    capabilities: ModelCapabilities,
    *,
    temperature: float | None = None,
    effort: str | None = None,
    model: str = "<unspecified>",
) -> ResolvedSampling:
    """Resolve and validate independent controls using factual capabilities.

    Effort: explicit > advertised default > lowest known active level > omit.
    An advertised default is itself evidence that that level is accepted, even
    when the complete level list is unknown. Other explicit unknown levels fail.
    Temperature: explicit > advertised default > 0.5, when supported in the
    resolved mode. A mode restriction suppresses only an *unset* temperature;
    an explicit incompatible temperature raises, never silently disappears.
    Unsupported/unknown temperature support omits legacy saved temperatures.
    """
    caps = ModelCapabilities.model_validate(capabilities)
    if temperature is not None and (not math.isfinite(temperature) or not 0 <= temperature <= 2):
        raise SamplingConfigurationError(f"Invalid temperature {temperature!r} for model {model!r}; expected 0..2")

    resolved_effort = None
    effort_source: SamplingSource = "omitted"
    effort_reason = None
    if effort is not None:
        accepted = (
            caps.effort_levels
            if caps.effort_levels is not None
            else ([caps.default_effort] if caps.default_effort is not None else [])
        )
        if caps.supports_effort is not True or effort not in accepted:
            raise UnsupportedEffortError(
                f"Unsupported effort {effort!r} for model {model!r}; supported levels: {caps.effort_levels}"
            )
        resolved_effort, effort_source = effort, "explicit"
    elif caps.supports_effort is True:
        if caps.default_effort is not None:
            resolved_effort, effort_source = caps.default_effort, "advertised_default"
        elif caps.effort_levels is not None:
            resolved_effort = next((level for level in _ACTIVE_EFFORT_ORDER if level in caps.effort_levels), None)
            if resolved_effort is not None:
                effort_source = "application_fallback"
            else:
                effort_reason = "no_ranked_active_level"
        else:
            effort_reason = "unknown_levels"
    else:
        effort_reason = "unsupported" if caps.supports_effort is False else "unknown_support"

    resolved_temperature = None
    temperature_source: SamplingSource = "omitted"
    temperature_reason = None
    if caps.supports_temperature is True:
        allowed = caps.temperature_effort_levels
        if allowed is not None and resolved_effort not in allowed:
            if temperature is not None:
                raise SamplingCombinationError(
                    f"Temperature {temperature!r} for model {model!r} is incompatible with "
                    f"effort {resolved_effort!r}; temperature requires effort in {allowed}"
                )
            temperature_reason = "incompatible_effort"
        elif temperature is not None:
            resolved_temperature, temperature_source = temperature, "explicit"
        elif caps.default_temperature is not None:
            resolved_temperature, temperature_source = caps.default_temperature, "advertised_default"
        else:
            resolved_temperature, temperature_source = 0.5, "application_fallback"
    else:
        temperature_reason = "unsupported" if caps.supports_temperature is False else "unknown_support"

    return ResolvedSampling(
        temperature=resolved_temperature,
        effort=resolved_effort,
        temperature_source=temperature_source,
        effort_source=effort_source,
        temperature_omission_reason=temperature_reason,
        effort_omission_reason=effort_reason,
    )
