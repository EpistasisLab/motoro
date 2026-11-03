"""Provider metadata is directly reusable by existing inference integration."""

from typing import Any

import httpx
import pytest

from motoro.schemas.agent import ModelConfig
from motoro.services import llm_service
from motoro.services.provider_types import ProviderConfig
from motoro.services.providers import discover_models, get_inference_connection, get_model_metadata


@pytest.mark.parametrize(
    "provider,expected",
    [
        ("anthropic", "chat"),
        ("openai", "chat"),
        ("openrouter", "chat"),
        ("azure_foundry", "azure_ai/chat"),
        ("local", "openai/chat"),
    ],
)
@pytest.mark.parametrize("supported", [True, False])
def test_tri_state_tool_support_uses_inference_routing(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    expected: str,
    supported: bool,
) -> None:
    calls: list[str] = []

    def model_info(model: str) -> dict[str, Any]:
        calls.append(model)
        return {}

    monkeypatch.setattr(llm_service.litellm, "get_model_info", model_info)
    monkeypatch.setattr(llm_service.litellm, "supports_function_calling", lambda model: supported)
    assert llm_service.model_tool_calling_support(provider, "chat") is supported
    assert calls == [expected]


def test_unknown_tool_support(monkeypatch: pytest.MonkeyPatch) -> None:
    def unknown(model: str) -> None:
        raise ValueError("Model not in map")

    monkeypatch.setattr(llm_service.litellm, "get_model_info", unknown)
    assert llm_service.model_tool_calling_support("azure_foundry", "prod") is None
    assert llm_service.model_tool_calling_support("openai", "") is None
    assert llm_service.model_tool_calling_support("unrecognized-provider", "chat") is None
    assert llm_service.model_supports_tool_calling(ModelConfig(provider="azure_foundry", model="prod")) is False


async def test_live_capabilities_feed_existing_inference_sampling() -> None:
    mock = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "new/model",
                        "supported_parameters": ["temperature", "reasoning_effort"],
                        "reasoning": {"supported_efforts": ["low", "high"], "default_effort": "high"},
                        "default_parameters": {"temperature": 0},
                    }
                ]
            },
        )
    )
    listing = await discover_models(ProviderConfig(provider="openrouter"), transport=mock)
    metadata = get_model_metadata(model="new/model", listing=listing)
    cfg = ModelConfig(provider="openrouter", model=metadata.id, model_metadata=metadata)
    assert llm_service._sampling_kwargs(cfg, "openrouter/new/model") == {
        "extra_body": {"temperature": 0, "reasoning": {"effort": "high"}},
        "drop_params": False,
    }
    # Application overrides affect the request only, never returned metadata.
    cfg.temperature = 0.5
    cfg.effort = "low"
    assert llm_service._sampling_kwargs(cfg, "openrouter/new/model")["extra_body"]["temperature"] == 0.5
    assert metadata.defaults.temperature.value == 0


@pytest.mark.parametrize("provider", ["openrouter", "anthropic"])
async def test_independent_provider_defaults_resolve_with_provenance(provider: str) -> None:
    entry = (
        {"id": "new/model", "default_parameters": {"temperature": 0}}
        if provider == "openrouter"
        else {
            "id": "new/model",
            "capabilities": {"temperature": {"default": 0}},
        }
    )
    mock = httpx.MockTransport(lambda request: httpx.Response(200, json={"data": [entry]}))
    listing = await discover_models(ProviderConfig(provider=provider, api_key="example-only"), transport=mock)
    metadata = get_model_metadata(model="new/model", listing=listing)
    cfg = ModelConfig(provider=provider, model=metadata.id, model_metadata=metadata)
    assert metadata.defaults.temperature.known and metadata.defaults.temperature.value == 0
    assert metadata.capabilities.supports_temperature is None
    assert metadata.capabilities.default_temperature is None
    assert llm_service._sampling_kwargs(cfg, f"{provider}/new/model")["extra_body"] == {"temperature": 0}
    cfg.temperature = 0.5
    assert llm_service._sampling_kwargs(cfg, f"{provider}/new/model")["extra_body"] == {"temperature": 0.5}
    assert cfg.temperature == 0.5
    assert get_model_metadata(model="new/model", listing=listing).defaults.temperature.value == 0


@pytest.mark.parametrize(
    "provider,base,model,key",
    [
        ("azure_foundry", "resource", "azure_ai/chat", "key"),
        ("azure_foundry", "resource/", "azure_ai/chat", "key"),
        ("azure_foundry", " resource/// ", "azure_ai/chat", "key"),
        ("azure_foundry", "resource/anthropic/", "azure_ai/chat", "key"),
        ("azure_foundry", "https://resource.services.ai.azure.com/anthropic", "azure_ai/chat", "key"),
        ("azure_foundry", "https://resource.services.ai.azure.com/anthropic/", "azure_ai/chat", "key"),
        ("local", "http://localhost:8080/v1", "openai/chat", None),
        ("openrouter", None, None, "key"),
    ],
)
def test_explicit_inference_connection(provider: str, base: str | None, model: str | None, key: str | None) -> None:
    config = ProviderConfig(provider=provider, api_base=base, api_key=key)
    connection = get_inference_connection(config, model="chat")
    assert connection["model"] == model
    assert connection["api_key"] == (key or "not-needed")
    assert connection["api_version"] is None
    if provider == "azure_foundry":
        assert connection["api_base"] == "https://resource.services.ai.azure.com"


def test_azure_trailing_slash_resource_inference_regression() -> None:
    config = ProviderConfig(provider="azure_foundry", api_base="my-resource/")
    connection = get_inference_connection(config, model="prod-chat")
    assert connection["api_base"] == "https://my-resource.services.ai.azure.com"
    assert connection["model"] == "azure_ai/prod-chat"


@pytest.mark.parametrize("provider", ["azure_foundry", "local"])
def test_missing_inference_base(provider: str) -> None:
    with pytest.raises(ValueError):
        get_inference_connection(ProviderConfig(provider=provider), model="chat")
