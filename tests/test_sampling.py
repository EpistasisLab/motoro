"""Public sampling policy and serialized metadata compatibility."""

import pytest
from pydantic import ValidationError

from motoro.schemas.agent import ModelConfig, PartialModelConfig
from motoro.services.model_capabilities import ModelCapabilities, get_capabilities
from motoro.services.sampling import SamplingCombinationError, UnsupportedEffortError, resolve_sampling


@pytest.mark.parametrize("temperature,effort", [(True, False), (False, True), (True, True), (False, False)])
def test_independent_controls(temperature: bool, effort: bool) -> None:
    caps = ModelCapabilities(
        supports_temperature=temperature,
        supports_effort=effort,
        effort_levels=["high", "low"] if effort else [],
    )
    result = resolve_sampling(caps)
    assert result.as_kwargs() == {
        **({"temperature": 0.5} if temperature else {}),
        **({"reasoning_effort": "low"} if effort else {}),
    }
    assert result.temperature_source == ("application_fallback" if temperature else "omitted")
    assert result.effort_source == ("application_fallback" if effort else "omitted")


@pytest.mark.parametrize("default", [0.0, 0.9, None])
@pytest.mark.parametrize("explicit", [0.0, 0.7, None])
def test_temperature_precedence(default: float | None, explicit: float | None) -> None:
    caps = ModelCapabilities(supports_temperature=True, default_temperature=default)
    result = resolve_sampling(caps, temperature=explicit)
    assert result.temperature == (explicit if explicit is not None else default if default is not None else 0.5)
    assert result.temperature_source == (
        "explicit" if explicit is not None else "advertised_default" if default is not None else "application_fallback"
    )


@pytest.mark.parametrize(
    "levels,default,explicit,expected,source",
    [
        (["high", "none", "low", "minimal"], None, None, "minimal", "application_fallback"),
        (["low", "none", "high"], None, None, "low", "application_fallback"),
        (["high", "low"], "high", None, "high", "advertised_default"),
        (["none", "low"], "none", None, "none", "advertised_default"),
        (["none", "low"], None, "none", "none", "explicit"),
        (["high", "low"], "high", "low", "low", "explicit"),
        (None, None, None, None, "omitted"),
        (["none"], None, None, None, "omitted"),
        (["provider-special"], None, None, None, "omitted"),
        (["provider-special"], "provider-special", None, "provider-special", "advertised_default"),
        (None, "none", None, "none", "advertised_default"),
        (None, "none", "none", "none", "explicit"),
    ],
)
def test_effort_precedence(
    levels: list[str] | None,
    default: str | None,
    explicit: str | None,
    expected: str | None,
    source: str,
) -> None:
    caps = ModelCapabilities(supports_effort=True, effort_levels=levels, default_effort=default)
    result = resolve_sampling(caps, effort=explicit)
    assert result.effort == expected
    assert result.effort_source == source
    assert ("reasoning_effort" in result.as_kwargs()) == (expected is not None)


@pytest.mark.parametrize("support", [True, False, None])
def test_unvalidated_explicit_effort_fails(support: bool | None) -> None:
    with pytest.raises(UnsupportedEffortError):
        resolve_sampling(ModelCapabilities(supports_effort=support), effort="low")


def test_unknown_and_unsupported_are_distinct() -> None:
    unknown = resolve_sampling(ModelCapabilities())
    unsupported = resolve_sampling(ModelCapabilities(supports_temperature=False, supports_effort=False))
    assert unknown.temperature_omission_reason == unknown.effort_omission_reason == "unknown_support"
    assert unsupported.temperature_omission_reason == unsupported.effort_omission_reason == "unsupported"
    assert resolve_sampling(ModelCapabilities(supports_effort=True)).effort_omission_reason == "unknown_levels"


@pytest.mark.parametrize("effort", ["none", "low", "high", None])
def test_combinations(effort: str | None) -> None:
    caps = get_capabilities("openai/gpt-5.2")
    if effort in (None, "none"):
        result = resolve_sampling(caps, temperature=0.7, effort=effort)
        assert result.as_kwargs() == {"temperature": 0.7, "reasoning_effort": "none"}
    else:
        with pytest.raises(SamplingCombinationError, match="incompatible.*effort"):
            resolve_sampling(caps, temperature=0.7, effort=effort)
        result = resolve_sampling(caps, effort=effort)
        assert result.as_kwargs() == {"reasoning_effort": effort}
        assert result.temperature_omission_reason == "incompatible_effort"
    assert caps.supports_temperature is True  # Configuration never alters facts.


def test_unknown_effort_cannot_establish_restricted_temperature_mode() -> None:
    caps = ModelCapabilities(supports_temperature=True, supports_effort=True, temperature_effort_levels=["none"])
    with pytest.raises(SamplingCombinationError):
        resolve_sampling(caps, temperature=0.1)
    assert resolve_sampling(caps).as_kwargs() == {}


def test_model_specific_restrictions_are_taken_from_metadata() -> None:
    caps = ModelCapabilities(
        supports_temperature=True,
        supports_effort=True,
        effort_levels=["low", "high"],
        temperature_effort_levels=["low"],
        default_temperature=0.0,
    )
    assert resolve_sampling(caps).as_kwargs() == {"temperature": 0.0, "reasoning_effort": "low"}
    with pytest.raises(SamplingCombinationError):
        resolve_sampling(caps, temperature=0.0, effort="high")
    assert resolve_sampling(caps, effort="high").temperature_omission_reason == "incompatible_effort"
    caps.temperature_effort_levels = []
    with pytest.raises(SamplingCombinationError):
        resolve_sampling(caps, temperature=0.2)


@pytest.mark.parametrize("schema", [ModelConfig, PartialModelConfig])
def test_legacy_and_unset_temperature_round_trip(schema: type[ModelConfig] | type[PartialModelConfig]) -> None:
    legacy = schema.model_validate_json(
        '{"temperature":0.7,"effort":null,"resolved_capabilities":'
        '{"supports_temperature":true,"supports_effort":false,'
        '"effort_levels":[],"default_effort":null}}'
    )
    restored = schema.model_validate_json(legacy.model_dump_json())
    assert restored.temperature == 0.7
    assert restored.resolved_capabilities is not None
    assert (
        resolve_sampling(restored.resolved_capabilities, temperature=restored.temperature).temperature_source
        == "explicit"
    )
    for raw in ("{}", '{"temperature":null}'):
        config = schema.model_validate_json(raw)
        assert config.temperature is None
        assert schema.model_validate_json(config.model_dump_json(exclude_none=True)).temperature is None


@pytest.mark.parametrize(
    "metadata",
    [
        {"supports_effort": True, "effort_levels": []},
        {"supports_effort": True, "effort_levels": [" "]},
        {"supports_effort": True, "default_effort": ""},
        {"supports_temperature": False, "default_temperature": 0.1},
        {"supports_temperature": True, "default_temperature": float("nan")},
        {
            "supports_temperature": True,
            "supports_effort": True,
            "effort_levels": ["low"],
            "temperature_effort_levels": ["none"],
        },
    ],
)
def test_invalid_metadata(metadata: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ModelCapabilities.model_validate(metadata)


@pytest.mark.parametrize("model", ["gpt-5.1", "gpt-5.2", "gpt-5.2-2025-12-11"])
def test_registry_specific_modes(model: str) -> None:
    caps = get_capabilities(model)
    assert caps.supports_temperature is True
    assert caps.default_effort == "none"
    assert caps.temperature_effort_levels == ["none"]


def test_registry_does_not_extend_rules_to_unverified_variants() -> None:
    assert get_capabilities("gpt-5.2-pro").supports_temperature is None
    assert get_capabilities("gpt-5.1-codex").effort_levels is None
    assert "minimal" in (get_capabilities("gpt-5").effort_levels or [])
