"""Resolved sampling metadata survives execution and controls every request path."""

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from motoro.schemas.agent import ModelConfig, PartialModelConfig
from motoro.services import llm_service
from motoro.services.llm_service import LLMService, UnsupportedEffortError, _sampling_kwargs
from motoro.services.model_capabilities import (
    CATALOG,
    DEFAULT_CAPABILITIES,
    TEMPERATURE_ONLY_CAPABILITIES,
    ModelCapabilities,
    get_capabilities,
)


def test_unknown_model_omits_unknown_controls() -> None:
    config = ModelConfig(model="unrecognized-model")
    assert _sampling_kwargs(config, "azure_ai/unrecognized-model") == {}


def test_unknown_model_uses_caller_default() -> None:
    assert get_capabilities("qwen/qwen3.8-27b") is DEFAULT_CAPABILITIES
    assert get_capabilities("qwen/qwen3.8-27b", default=TEMPERATURE_ONLY_CAPABILITIES) is TEMPERATURE_ONLY_CAPABILITIES


@pytest.mark.parametrize(
    "model",
    [
        "anthropic/claude-opus-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-fable-5",
        "claude-sonnet-5",
        "azure_ai/claude-opus-5-20261001",
    ],
)
def test_provider_default_does_not_replace_registered_claude(model: str) -> None:
    caps = get_capabilities(model, default=TEMPERATURE_ONLY_CAPABILITIES)
    assert caps.supports_effort is True
    assert caps.supports_temperature is False


def test_registry_precedes_effort_default() -> None:
    assert get_capabilities("openai/gpt-4o", default=DEFAULT_CAPABILITIES) is TEMPERATURE_ONLY_CAPABILITIES


@pytest.mark.parametrize("path", ["structured", "text", "tools", "select_tool"])
@pytest.mark.parametrize("effort,expected", [("medium", "medium"), (None, "xhigh"), ("none", "none")])
@pytest.mark.parametrize("with_temperature", [False, True])
async def test_resolved_openrouter_effort_bypasses_stale_litellm_metadata(
    request_stubs: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    effort: str | None,
    expected: str,
    with_temperature: bool,
) -> None:
    async def connection(config: ModelConfig, principal_id: Any) -> dict[str, Any]:
        return {"model": f"openrouter/{config.model}", "api_key": "stub", "api_base": None}

    monkeypatch.setattr(llm_service, "_resolve_connection_for_principal", connection)
    config = ModelConfig(
        provider="openrouter",
        model="qwen/qwen3.8-27b",
        effort=effort,
        temperature=0.2 if with_temperature else None,
        resolved_capabilities=ModelCapabilities(
            supports_temperature=with_temperature,
            supports_effort=True,
            effort_levels=["low", "medium", "xhigh", "none"],
            default_effort="xhigh",
        ),
    )
    await invoke(LLMService(), path, config)
    assert len(request_stubs) == 1
    assert request_stubs[0]["reasoning_effort"] == expected
    assert request_stubs[0]["allowed_openai_params"] == ["reasoning_effort"]
    assert request_stubs[0].get("temperature") == (0.2 if with_temperature else None)
    assert "drop_params" not in request_stubs[0]


def test_unresolved_openrouter_effort_does_not_bypass_validation() -> None:
    config = ModelConfig(provider="openrouter", model="unknown")
    assert "allowed_openai_params" not in _sampling_kwargs(config, "openrouter/unknown")


def test_resolved_openrouter_temperature_does_not_allow_effort() -> None:
    config = ModelConfig(provider="openrouter", resolved_capabilities=TEMPERATURE_ONLY_CAPABILITIES)
    assert _sampling_kwargs(config, "openrouter/custom") == {"temperature": 0.5}


def test_resolved_openrouter_unknown_levels_does_not_allow_effort() -> None:
    config = ModelConfig(resolved_capabilities=ModelCapabilities(supports_effort=True))
    assert _sampling_kwargs(config, "openrouter/custom") == {}


@pytest.mark.parametrize("model", ["gpt-4o", "gpt-4o-mini", "gpt-5-chat-latest", "claude-haiku-4-5-20251001"])
def test_known_temperature_only_model(model: str) -> None:
    assert _sampling_kwargs(ModelConfig(model=model, temperature=0.3), model) == {"temperature": 0.3}


def test_catalog_keeps_its_existing_controls() -> None:
    temperature_models = {"claude-haiku-4-5", "gpt-4o", "gpt-4o-mini"}
    for entry in CATALOG:
        assert entry.capabilities.supports_temperature == (entry.model in temperature_models)
        assert entry.capabilities.supports_effort == (entry.model not in temperature_models)


def test_custom_temperature_only_override() -> None:
    config = ModelConfig(model="custom", resolved_capabilities=TEMPERATURE_ONLY_CAPABILITIES, temperature=0.2)
    assert _sampling_kwargs(config, "custom") == {"temperature": 0.2}


def test_configured_effort_and_provider_specific_levels() -> None:
    assert _sampling_kwargs(ModelConfig(effort="high"), "gpt-5") == {"reasoning_effort": "high"}
    config = ModelConfig(
        effort="minimal",
        resolved_capabilities=ModelCapabilities(
            supports_temperature=False,
            supports_effort=True,
            effort_levels=["minimal", "high"],
            default_effort="high",
        ),
    )
    assert _sampling_kwargs(config, "gpt-5") == {"reasoning_effort": "minimal"}


@pytest.mark.parametrize("model,effort", [("gpt-5", "max"), ("custom", "invalid"), ("gpt-4o", "high")])
def test_unsupported_effort_rejected(model: str, effort: str) -> None:
    with pytest.raises(UnsupportedEffortError, match=f"Unsupported effort '{effort}'.*supported levels"):
        _sampling_kwargs(ModelConfig(model=model, effort=effort), model)


@pytest.mark.parametrize(
    "metadata",
    [
        {"supports_effort": True, "effort_levels": ["high"], "default_effort": "medium"},
        {"supports_effort": True, "effort_levels": ["high", "high"], "default_effort": "high"},
        {"supports_effort": False, "effort_levels": ["high"]},
        {"unknown_field": True},
    ],
)
def test_resolved_capabilities_are_validated(metadata: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ModelConfig(resolved_capabilities=metadata)


def test_partial_and_execution_config_round_trip() -> None:
    partial = PartialModelConfig(
        model="custom",
        resolved_capabilities=get_capabilities("custom"),
        fallback_models=[ModelConfig(model="legacy", resolved_capabilities=TEMPERATURE_ONLY_CAPABILITIES)],
    )
    config = ModelConfig.model_validate_json(partial.model_dump_json(exclude_none=True))
    assert config.resolved_capabilities == partial.resolved_capabilities
    assert _sampling_kwargs(config, "custom") == {}
    assert _sampling_kwargs(config.fallback_models[0], "legacy") == {"temperature": 0.5}


class Answer(BaseModel):
    text: str


@pytest.fixture
def request_stubs(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    async def connection(config: ModelConfig, principal_id: Any) -> dict[str, Any]:
        return {"model": f"openai/{config.model}", "api_key": "stub", "api_base": None}

    async def completion(**kwargs: Any) -> Any:
        calls.append(kwargs)
        if "response_model" in kwargs:
            if kwargs["response_model"] is Answer:
                return Answer(text="answer")
            return kwargs["response_model"](reasoning="done", use_tool=False)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="answer", tool_calls=[]))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )

    monkeypatch.setattr(llm_service, "_resolve_connection_for_principal", connection)
    monkeypatch.setattr(llm_service.litellm, "acompletion", completion)
    monkeypatch.setattr(llm_service.litellm, "get_supported_openai_params", lambda **kw: [])
    monkeypatch.setattr("motoro.services.pricing_service.calculate_cost", lambda **kw: (0.0, "test"))
    monkeypatch.setattr(
        llm_service.instructor,
        "from_litellm",
        lambda fn: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion))),
    )
    return calls


async def invoke(service: LLMService, path: str, config: ModelConfig) -> None:
    messages = [{"role": "user", "content": "hi"}]
    if path == "structured":
        await service.complete(config, messages, Answer)
    elif path == "text":
        await service.complete_text(config, messages)
    elif path == "tools":
        await service.complete_with_tools(config, messages, [])
    else:
        await service.select_tool(config, messages, ["echo"])


@pytest.mark.parametrize("path", ["structured", "text", "tools", "select_tool"])
@pytest.mark.parametrize(
    "case", ["unknown", "temperature", "live", "configured", "custom_temperature", "both", "neither"]
)
async def test_every_request_path_uses_resolved_sampling(
    request_stubs: list[dict[str, Any]],
    path: str,
    case: str,
) -> None:
    configs = {
        "unknown": (ModelConfig(model="unknown"), {}),
        "both": (
            ModelConfig(
                resolved_capabilities=ModelCapabilities(
                    supports_temperature=True,
                    supports_effort=True,
                    default_temperature=0.0,
                    effort_levels=["high", "none", "minimal"],
                ),
            ),
            {"temperature": 0.0, "reasoning_effort": "minimal"},
        ),
        "neither": (
            ModelConfig(resolved_capabilities=ModelCapabilities(supports_temperature=False, supports_effort=False)),
            {},
        ),
        "temperature": (ModelConfig(model="gpt-4o", temperature=0.4), {"temperature": 0.4}),
        "live": (
            ModelConfig(
                model="gpt-4o",
                resolved_capabilities=ModelCapabilities(
                    supports_temperature=False,
                    supports_effort=True,
                    effort_levels=["low", "high"],
                    default_effort="high",
                ),
            ),
            {"reasoning_effort": "high"},
        ),
        "configured": (ModelConfig(model="gpt-5", effort="low"), {"reasoning_effort": "low"}),
        "custom_temperature": (
            ModelConfig(
                model="gpt-5",
                resolved_capabilities=TEMPERATURE_ONLY_CAPABILITIES,
            ),
            {"temperature": 0.5},
        ),
    }
    config, expected = configs[case]
    await invoke(LLMService(), path, config)
    assert len(request_stubs) == 1
    assert {k: v for k, v in request_stubs[0].items() if k in {"temperature", "reasoning_effort"}} == expected
    assert "allowed_openai_params" not in request_stubs[0]


@pytest.mark.parametrize("path", ["structured", "text", "tools", "select_tool"])
@pytest.mark.parametrize("provider", ["openai", "openrouter"])
async def test_invalid_effort_never_sends_a_request_or_uses_fallback(
    request_stubs: list[dict[str, Any]],
    path: str,
    provider: str,
) -> None:
    config = ModelConfig(
        provider=provider,
        model="gpt-5",
        effort="max",
        resolved_capabilities=get_capabilities("gpt-5"),
        fallback_models=[ModelConfig(model="gpt-4o")],
    )
    with pytest.raises(UnsupportedEffortError, match="max.*gpt-5"):
        await invoke(LLMService(), path, config)
    assert request_stubs == []


@pytest.mark.parametrize("path", ["structured", "text", "tools", "select_tool"])
async def test_invalid_combination_never_sends_or_falls_back(
    request_stubs: list[dict[str, Any]],
    path: str,
) -> None:
    from motoro.services.sampling import SamplingCombinationError

    config = ModelConfig(model="gpt-5.2", temperature=0.7, effort="high", fallback_models=[ModelConfig(model="gpt-4o")])
    with pytest.raises(SamplingCombinationError, match="incompatible"):
        await invoke(LLMService(), path, config)
    assert request_stubs == []


@pytest.mark.parametrize("path", ["structured", "text", "tools", "select_tool"])
@pytest.mark.parametrize(
    "effort,expected",
    [("none", {"temperature": 0.5, "reasoning_effort": "none"}), ("high", {"reasoning_effort": "high"})],
)
async def test_restricted_modes_on_every_path(
    request_stubs: list[dict[str, Any]],
    path: str,
    effort: str,
    expected: dict[str, Any],
) -> None:
    await invoke(LLMService(), path, ModelConfig(model="gpt-5.2", effort=effort))
    assert {k: v for k, v in request_stubs[0].items() if k in {"temperature", "reasoning_effort"}} == expected


@pytest.mark.parametrize("path", ["structured", "text", "tools", "select_tool"])
@pytest.mark.parametrize("fallback_kind", ["temperature", "both", "effort", "unknown", "registry"])
async def test_fallback_uses_its_own_capabilities(
    request_stubs: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    fallback_kind: str,
) -> None:
    original = llm_service._resolve_connection_for_principal

    async def connection(config: ModelConfig, principal_id: Any) -> dict[str, Any]:
        if config.model == "primary":
            raise RuntimeError("primary unavailable")
        return await original(config, principal_id)

    monkeypatch.setattr(llm_service, "_resolve_connection_for_principal", connection)
    fallback_caps = {
        "temperature": TEMPERATURE_ONLY_CAPABILITIES,
        "both": ModelCapabilities(
            supports_temperature=True,
            supports_effort=True,
            default_temperature=0.0,
            effort_levels=["low", "high"],
            default_effort="high",
        ),
        "effort": get_capabilities("gpt-5"),
        "unknown": DEFAULT_CAPABILITIES,
        "registry": None,
    }[fallback_kind]
    config = ModelConfig(
        model="primary",
        temperature=0.9,
        resolved_capabilities=get_capabilities("primary"),
        fallback_models=[ModelConfig(model="gpt-5.2", resolved_capabilities=fallback_caps)],
    )
    await invoke(LLMService(), path, config)
    expected = {
        "temperature": {"temperature": 0.5},
        "both": {"temperature": 0.0, "reasoning_effort": "high"},
        "effort": {"reasoning_effort": "medium"},
        "unknown": {},
        "registry": {"temperature": 0.5, "reasoning_effort": "none"},
    }[fallback_kind]
    assert len(request_stubs) == 1
    assert {k: v for k, v in request_stubs[0].items() if k in {"temperature", "reasoning_effort"}} == expected


@pytest.mark.parametrize("path", ["structured", "text", "tools", "select_tool"])
async def test_fallback_configuration_errors_are_terminal(
    request_stubs: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    from motoro.services.sampling import SamplingCombinationError

    original = llm_service._resolve_connection_for_principal

    async def connection(config: ModelConfig, principal_id: Any) -> dict[str, Any]:
        if config.model == "primary":
            raise RuntimeError("unavailable")
        return await original(config, principal_id)

    monkeypatch.setattr(llm_service, "_resolve_connection_for_principal", connection)
    config = ModelConfig(
        model="primary",
        fallback_models=[
            ModelConfig(model="gpt-5.2", temperature=0.7, effort="high"),
            ModelConfig(model="gpt-4o"),
        ],
    )
    with pytest.raises(SamplingCombinationError):
        await invoke(LLMService(), path, config)
    assert request_stubs == []
