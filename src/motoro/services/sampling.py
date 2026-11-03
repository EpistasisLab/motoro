"""Shared sampling resolution for execution, previews, and product integrations.

This module owns policy. It does not discover providers or import execution
schemas. None is unset; the effort string "none" is an explicit disabled mode.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict

from motoro.services.model_capabilities import ModelCapabilities
from motoro.services.provider_types import ModelInfo

SamplingSource = Literal["explicit", "advertised_default", "capability_override", "application_fallback", "omitted"]


class SamplingConfigurationError(ValueError):
    """Invalid sampling configuration; terminal across retries and fallbacks."""


class UnsupportedEffortError(SamplingConfigurationError):
    """An explicit effort contradicts known capabilities for this model."""


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
    capabilities: ModelCapabilities | ModelInfo,
    *,
    temperature: float | None = None,
    effort: str | None = None,
    model: str = "<unspecified>",
) -> ResolvedSampling:
    """Explicit values > provider defaults > omission, without guessed defaults.

    ModelInfo retains discovery provenance: only provider-sourced facts constrain
    resolution. Bare ModelCapabilities is an explicit caller override; its
    defaults are labelled capability_override, never advertised_default.
    Unknown support/levels defer validation of explicit controls to the provider.
    """
    default_source: SamplingSource = "capability_override"
    if isinstance(capabilities, ModelInfo):
        info = capabilities
        # Old cached/catalog guesses must not reject user input or become defaults.
        caps = ModelCapabilities.model_validate(
            {
                name: value
                for name, value in info.capabilities.model_dump().items()
                if info.capability_sources.get(name) == "provider"
            }
        )
        effort_default = info.defaults.effort
        temperature_default = info.defaults.temperature
        default_effort = effort_default.value if effort_default.source == "provider" else caps.default_effort
        default_temperature = (
            temperature_default.value if temperature_default.source == "provider" else caps.default_temperature
        )
        enabled = info.defaults.additional.get("reasoning.default_enabled")
        if enabled is not None and enabled.source == "provider" and enabled.value is False:
            default_effort = None  # This effort default applies only when reasoning is enabled.
        default_source = "advertised_default"
    else:
        caps = ModelCapabilities.model_validate(capabilities)
        default_effort, default_temperature = caps.default_effort, caps.default_temperature

    if temperature is not None and (not math.isfinite(temperature) or not 0 <= temperature <= 2):
        raise SamplingConfigurationError(f"Invalid temperature {temperature!r} for model {model!r}; expected 0..2")
    if effort is not None and (not effort.strip() or effort != effort.strip()):
        raise SamplingConfigurationError(f"Invalid effort {effort!r} for model {model!r}")

    resolved_effort = None
    effort_source: SamplingSource = "omitted"
    effort_reason: str | None = "unsupported" if caps.supports_effort is False else "unknown_support"
    if effort is not None:
        if (
            caps.supports_effort is False
            or effort in (caps.unsupported_effort_levels or [])
            or (caps.effort_levels_complete and caps.effort_levels is not None and effort not in caps.effort_levels)
        ):
            raise UnsupportedEffortError(
                f"Unsupported effort {effort!r} for model {model!r}; supported levels: {caps.effort_levels}"
            )
        resolved_effort, effort_source, effort_reason = effort, "explicit", None
    elif caps.supports_effort is not False:
        if isinstance(default_effort, str) and (
            (caps.effort_levels is None or not caps.effort_levels_complete or default_effort in caps.effort_levels)
            and default_effort not in (caps.unsupported_effort_levels or [])
        ):
            resolved_effort, effort_source, effort_reason = default_effort, default_source, None
        elif caps.supports_effort is True:
            effort_reason = "unadvertised_default"

    resolved_temperature = None
    temperature_source: SamplingSource = "omitted"
    temperature_reason: str | None = "unsupported" if caps.supports_temperature is False else "unknown_support"
    allowed = caps.temperature_effort_levels
    incompatible = allowed is not None and (
        not allowed or resolved_effort is not None and resolved_effort not in allowed
    )
    uncertain_mode = allowed is not None and resolved_effort is None
    if temperature is not None:
        if caps.supports_temperature is False:
            raise SamplingConfigurationError(f"Unsupported temperature {temperature!r} for model {model!r}")
        if incompatible:
            raise SamplingCombinationError(
                f"Temperature {temperature!r} for model {model!r} is incompatible with "
                f"effort {resolved_effort!r}; temperature requires effort in {allowed}"
            )
        resolved_temperature, temperature_source, temperature_reason = temperature, "explicit", None
    elif caps.supports_temperature is not False:
        if incompatible or uncertain_mode:
            temperature_reason = "incompatible_effort" if incompatible else "unknown_effort"
        elif isinstance(default_temperature, (float, int)) and not isinstance(default_temperature, bool):
            if not math.isfinite(default_temperature) or not 0 <= default_temperature <= 2:
                raise SamplingConfigurationError(f"Invalid advertised temperature for model {model!r}")
            resolved_temperature, temperature_source, temperature_reason = default_temperature, default_source, None
        elif caps.supports_temperature is True:
            temperature_reason = "unadvertised_default"

    return ResolvedSampling(
        temperature=resolved_temperature,
        effort=resolved_effort,
        temperature_source=temperature_source,
        effort_source=effort_source,
        temperature_omission_reason=temperature_reason,
        effort_omission_reason=effort_reason,
    )
