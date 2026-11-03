"""Resolved sampling metadata survives execution and controls every request path."""

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from motoro.schemas.agent import ModelConfig
from motoro.services import llm_service
from motoro.services.llm_service import LLMService, UnsupportedEffortError, _sampling_kwargs
from motoro.services.model_capabilities import (
    ModelCapabilities,
)
from motoro.services.provider_types import DefaultValue, ModelDefaults, ModelInfo
from motoro.services.sampling import SamplingCombinationError, SamplingConfigurationError


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
    monkeypatch.setattr(llm_service, "_acompletion", completion)
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
@pytest.mark.parametrize("provider", ["anthropic", "openai", "openrouter", "azure_foundry", "local", "bedrock"])
@pytest.mark.parametrize("effort", ["medium", "future-effort", "none", "max"])
async def test_explicit_unknown_settings_survive_every_path(
    request_stubs: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    provider: str,
    effort: str,
) -> None:
    prefix = {"azure_foundry": "azure_ai", "local": "openai"}.get(provider, provider)

    async def connection(config: ModelConfig, principal_id: Any) -> dict[str, Any]:
        return {"model": f"{prefix}/{config.model}", "api_key": "stub", "api_base": None}

    monkeypatch.setattr(llm_service, "_resolve_connection_for_principal", connection)
    config = ModelConfig(provider=provider, model="claude-haiku-5-5", effort=effort, temperature=0)
    await invoke(LLMService(), path, ModelConfig.model_validate_json(config.model_dump_json()))
    assert len(request_stubs) == 1
    call = request_stubs[0]
    expected = {"temperature": 0, "reasoning_effort": effort}
    if provider == "anthropic":
        expected = {"temperature": 0, "output_config": {"effort": effort}}
    elif provider == "openrouter":
        expected = {"temperature": 0, "reasoning": {"effort": effort}}
    actual = {k: call[k] for k in expected} if provider == "bedrock" else call["extra_body"]
    assert actual == expected
    assert call["drop_params"] is False


@pytest.mark.parametrize("path", ["structured", "text", "tools", "select_tool"])
@pytest.mark.parametrize("case", ["unknown", "ladder_only", "provider_default", "catalog_default", "override"])
async def test_defaults_keep_provenance_through_execution(
    request_stubs: list[dict[str, Any]],
    path: str,
    case: str,
) -> None:
    caps = ModelCapabilities(supports_effort=True, effort_levels=["high", "low"])
    info = ModelInfo(id="future", capabilities=caps, capability_sources={k: "provider" for k in caps.model_dump()})
    config = ModelConfig(model="future", model_metadata=info)
    expected = {}
    if case == "provider_default":
        info.defaults.effort = DefaultValue(value="high", source="provider")
        expected = {"reasoning_effort": "high"}
    elif case == "catalog_default":
        info.defaults.effort = DefaultValue(value="high", source="catalog")
    elif case == "override":
        config.resolved_capabilities = ModelCapabilities(supports_temperature=True, default_temperature=0)
        expected = {"temperature": 0}
    elif case == "unknown":
        config.model_metadata = None
    await invoke(LLMService(), path, ModelConfig.model_validate_json(config.model_dump_json()))
    assert len(request_stubs) == 1
    assert request_stubs[0].get("extra_body", {}) == expected


@pytest.mark.parametrize("path", ["structured", "text", "tools", "select_tool"])
@pytest.mark.parametrize("case", ["effort", "temperature", "combination", "wrong_metadata"])
async def test_known_invalid_configuration_is_terminal(
    request_stubs: list[dict[str, Any]],
    path: str,
    case: str,
) -> None:
    caps = ModelCapabilities(
        supports_temperature=True,
        supports_effort=True,
        effort_levels=["none", "high"],
        temperature_effort_levels=["none"],
    )
    config = ModelConfig(
        model="future", resolved_capabilities=caps, effort="invalid", fallback_models=[ModelConfig(model="fallback")]
    )
    error = UnsupportedEffortError
    if case == "temperature":
        config.effort = None
        config.temperature = 0.5
        config.resolved_capabilities = ModelCapabilities(supports_temperature=False)
        error = SamplingConfigurationError
    elif case == "combination":
        config.effort, config.temperature = "high", 0.5
        error = SamplingCombinationError
    elif case == "wrong_metadata":
        config.model_metadata = ModelInfo(id="other")
        error = SamplingConfigurationError
    with pytest.raises(error):
        await invoke(LLMService(), path, config)
    assert request_stubs == []


@pytest.mark.parametrize("path", ["structured", "text", "tools", "select_tool"])
async def test_provider_rejection_does_not_retry_different_settings(
    request_stubs: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    calls = []

    async def rejected(config: ModelConfig, principal_id: Any) -> dict[str, Any]:
        calls.append(config.model)
        raise llm_service.litellm.exceptions.BadRequestError(
            message="invalid effort", model=config.model, llm_provider="openai"
        )

    monkeypatch.setattr(llm_service, "_resolve_connection_for_principal", rejected)
    config = ModelConfig(model="primary", effort="future", fallback_models=[ModelConfig(model="fallback")])
    with pytest.raises(llm_service.litellm.exceptions.BadRequestError):
        await invoke(LLMService(), path, config)
    assert calls == ["primary"]


@pytest.mark.parametrize("path", ["structured", "text", "tools", "select_tool"])
async def test_unavailable_primary_uses_fallback_own_metadata(
    request_stubs: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    original = llm_service._resolve_connection_for_principal

    async def connection(config: ModelConfig, principal_id: Any) -> dict[str, Any]:
        if config.model == "primary":
            raise RuntimeError("unavailable")
        return await original(config, principal_id)

    monkeypatch.setattr(llm_service, "_resolve_connection_for_principal", connection)
    info = ModelInfo(id="fallback", defaults=ModelDefaults(effort=DefaultValue(value="special", source="provider")))
    config = ModelConfig(model="primary", fallback_models=[ModelConfig(model="fallback", model_metadata=info)])
    await invoke(LLMService(), path, config)
    assert len(request_stubs) == 1
    assert request_stubs[0]["extra_body"] == {"reasoning_effort": "special"}


def test_overrides_take_precedence_without_mutating_provider_facts() -> None:
    info = ModelInfo(id="future", defaults=ModelDefaults(effort=DefaultValue(value="special", source="provider")))
    config = ModelConfig(model="future", model_metadata=info, resolved_capabilities=ModelCapabilities())
    assert _sampling_kwargs(config, "openai/future") == {}
    assert info.defaults.effort.value == "special"
