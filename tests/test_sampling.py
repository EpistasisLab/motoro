"""Sampling respects explicit values, unknown facts, and default provenance."""

import pytest
from pydantic import ValidationError

from motoro.schemas.agent import ModelConfig, PartialModelConfig
from motoro.services.model_capabilities import ModelCapabilities, get_capabilities
from motoro.services.provider_types import DefaultValue, ModelDefaults, ModelInfo
from motoro.services.sampling import (
    SamplingCombinationError,
    SamplingConfigurationError,
    UnsupportedEffortError,
    resolve_sampling,
)


@pytest.mark.parametrize("support", [None, True, False])
def test_unset_controls_never_invent_defaults(support: bool | None) -> None:
    caps = ModelCapabilities(supports_temperature=support, supports_effort=support)
    assert resolve_sampling(caps).as_kwargs() == {}
    if support is not False:
        assert resolve_sampling(caps, temperature=0, effort="future-effort").as_kwargs() == {
            "temperature": 0,
            "reasoning_effort": "future-effort",
        }
    else:
        with pytest.raises(UnsupportedEffortError):
            resolve_sampling(caps, effort="medium")
        with pytest.raises(SamplingConfigurationError, match="Unsupported temperature"):
            resolve_sampling(caps, temperature=0)


@pytest.mark.parametrize("levels", [["high", "low"], ["none"], ["future-effort"], None])
def test_no_lowest_effort_policy(levels: list[str] | None) -> None:
    assert resolve_sampling(ModelCapabilities(supports_effort=True, effort_levels=levels)).effort is None


def test_unknown_and_unsupported_remain_distinct() -> None:
    assert resolve_sampling(ModelCapabilities()).effort_omission_reason == "unknown_support"
    assert resolve_sampling(ModelCapabilities(supports_effort=False)).effort_omission_reason == "unsupported"
    assert resolve_sampling(ModelCapabilities(supports_effort=True)).effort_omission_reason == "unadvertised_default"


@pytest.mark.parametrize("source", ["provider", "catalog", "unknown"])
def test_provenance_controls_defaults_and_restrictions(source: str) -> None:
    caps = ModelCapabilities(
        supports_temperature=True,
        default_temperature=0,
        supports_effort=True,
        effort_levels=["special"],
        default_effort="special",
    )
    info = ModelInfo(
        id="future",
        capabilities=caps,
        capability_sources={name: source for name in caps.model_dump()},
        defaults=ModelDefaults(
            temperature=DefaultValue(value=0, source=source) if source != "unknown" else DefaultValue(),
            effort=DefaultValue(value="special", source=source) if source != "unknown" else DefaultValue(),
        ),
    )
    result = resolve_sampling(info)
    assert result.as_kwargs() == ({"temperature": 0, "reasoning_effort": "special"} if source == "provider" else {})
    assert result.effort_source == ("advertised_default" if source == "provider" else "omitted")
    if source == "provider":
        with pytest.raises(UnsupportedEffortError):
            resolve_sampling(info, effort="other")
    else:
        assert resolve_sampling(info, effort="other").effort == "other"
    override = resolve_sampling(caps)
    assert override.effort_source == override.temperature_source == "capability_override"


def test_partial_provider_default_does_not_imply_exhaustive_ladder() -> None:
    info = ModelInfo(id="future", defaults=ModelDefaults(effort=DefaultValue(value="special", source="provider")))
    assert resolve_sampling(info).effort == "special"
    assert resolve_sampling(info, effort="other").effort == "other"


@pytest.mark.parametrize("default", [0, 0.9, None])
@pytest.mark.parametrize("explicit", [0, 0.7, None])
def test_temperature_precedence(default: float | None, explicit: float | None) -> None:
    result = resolve_sampling(
        ModelCapabilities(supports_temperature=True, default_temperature=default), temperature=explicit
    )
    assert result.temperature == (explicit if explicit is not None else default)
    assert result.temperature_source == (
        "explicit" if explicit is not None else "capability_override" if default is not None else "omitted"
    )


def test_advertised_compatibility_restrictions() -> None:
    caps = ModelCapabilities(
        supports_temperature=True,
        default_temperature=0,
        supports_effort=True,
        effort_levels=["none", "high"],
        temperature_effort_levels=["none"],
    )
    info = ModelInfo(
        id="future", capabilities=caps, capability_sources={name: "provider" for name in caps.model_dump()}
    )
    assert resolve_sampling(info, effort="none").as_kwargs() == {"temperature": 0, "reasoning_effort": "none"}
    with pytest.raises(SamplingCombinationError):
        resolve_sampling(info, effort="high", temperature=0.5)
    assert resolve_sampling(info, effort="high").temperature_omission_reason == "incompatible_effort"
    # With no known default effort, the actual mode is unknown. Preserve explicit
    # temperature for the provider to validate; do not select a mode for it.
    assert resolve_sampling(info, temperature=0.5).as_kwargs() == {"temperature": 0.5}
    assert resolve_sampling(info).as_kwargs() == {}
    caps.temperature_effort_levels = []
    with pytest.raises(SamplingCombinationError):
        resolve_sampling(caps, temperature=0.5)


@pytest.mark.parametrize("value", [-0.1, 2.1, float("nan"), float("inf")])
def test_invalid_explicit_temperature(value: float) -> None:
    with pytest.raises(SamplingConfigurationError):
        resolve_sampling(ModelCapabilities(), temperature=value)


@pytest.mark.parametrize("schema", [ModelConfig, PartialModelConfig])
def test_metadata_and_explicit_override_round_trip(schema: type[ModelConfig] | type[PartialModelConfig]) -> None:
    info = ModelInfo(id="future", defaults=ModelDefaults(effort=DefaultValue(value="special", source="provider")))
    config = schema(
        model="future",
        temperature=0.7,
        model_metadata=info,
        resolved_capabilities=ModelCapabilities(supports_temperature=True),
    )
    assert schema.model_validate_json(config.model_dump_json()) == config
    assert schema.model_validate_json("{}").temperature is None
    assert schema.model_validate_json('{"temperature":0.7}').temperature == 0.7


@pytest.mark.parametrize(
    "metadata",
    [
        {"supports_effort": True, "effort_levels": []},
        {"supports_effort": True, "effort_levels": [" "]},
        {"supports_effort": True, "effort_levels": ["high", "high"]},
        {"supports_effort": True, "effort_levels": ["low"], "default_effort": "high"},
        {"supports_temperature": False, "default_temperature": 0.1},
        {"supports_temperature": True, "default_temperature": float("nan")},
    ],
)
def test_invalid_metadata(metadata: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ModelCapabilities.model_validate(metadata)


@pytest.mark.parametrize("model", ["claude-haiku-5-5", "claude-opus-5", "openai/gpt-5.2", "gpt-5-20260101", "new"])
def test_model_ids_never_imply_facts(model: str) -> None:
    assert get_capabilities(model) == ModelCapabilities()
    assert resolve_sampling(get_capabilities(model), effort="medium").effort == "medium"
    caps = get_capabilities(model)
    caps.supports_temperature = False
    assert get_capabilities(model).supports_temperature is None
