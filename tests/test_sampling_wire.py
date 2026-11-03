"""Exercise real LiteLLM transformations through a mocked HTTP transport."""

import json
from typing import Any

import httpx
import pytest
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from openai import AsyncOpenAI

from motoro.schemas.agent import ModelConfig
from motoro.services import llm_service
from motoro.services._sampling_transport import _BedrockSamplingHTTPHandler, acompletion
from motoro.services.sampling import SamplingConfigurationError


@pytest.mark.parametrize("provider", ["anthropic", "openai", "openrouter", "azure_foundry", "local"])
@pytest.mark.parametrize("effort", ["medium", "max", "future-effort", None])
@pytest.mark.parametrize("known_model", [False, True])
async def test_native_sampling_reaches_provider_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    effort: str | None,
    known_model: bool,
) -> None:
    bodies: list[dict[str, Any]] = []
    # Isolate SDK connection pools between MockTransport instances.
    monkeypatch.setattr(llm_service.litellm, "in_memory_llm_clients_cache", LLMClientCache())
    # These tests inspect HTTP requests, not LiteLLM's background telemetry worker.
    monkeypatch.setattr(
        GLOBAL_LOGGING_WORKER, "ensure_initialized_and_enqueue", lambda async_coroutine: async_coroutine.close()
    )

    def respond(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        if provider == "anthropic":
            return httpx.Response(
                200,
                json={
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-haiku-5-5",
                    "content": [{"type": "text", "text": "answer"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "chat_test",
                "object": "chat.completion",
                "created": 1,
                "model": "future-model",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "answer"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    # Keep LiteLLM's actual request builders; replace only HTTP transport.
    monkeypatch.setattr(
        AsyncHTTPHandler,
        "create_client",
        lambda self, **kw: httpx.AsyncClient(
            transport=httpx.MockTransport(respond),
        ),
    )

    async def connection(config: ModelConfig, principal_id: Any) -> dict[str, Any]:
        prefix = {"azure_foundry": "azure_ai", "local": "openai"}.get(provider, provider)
        return {"model": f"{prefix}/{config.model}", "api_key": "mock-key", "api_base": "https://provider.test/v1"}

    monkeypatch.setattr(llm_service, "_resolve_connection_for_principal", connection)
    monkeypatch.setattr("motoro.services.pricing_service.calculate_cost", lambda **kw: (0.0, "test"))
    # Give each test a fresh handler rather than a cached LiteLLM connection.
    handler = AsyncHTTPHandler()
    monkeypatch.setattr(llm_service.litellm, "module_level_aclient", handler)
    original_completion = llm_service.litellm.acompletion

    async def completion(**kwargs: Any) -> Any:
        kwargs.setdefault(
            "client",
            AsyncOpenAI(api_key="mock-key", base_url="https://provider.test/v1", http_client=handler.client)
            if provider in {"openai", "local"} or provider in {"openrouter", "azure_foundry"} and known_model
            else handler,
        )
        return await original_completion(**kwargs, num_retries=0)

    monkeypatch.setattr(llm_service.litellm, "acompletion", completion)
    model = "claude-haiku-5-5" if provider == "anthropic" else "future-model"
    if known_model:
        model = "claude-opus-5" if provider == "anthropic" else "gpt-5.2"
        if provider == "openrouter":
            model = "openai/gpt-5.2"
    config = ModelConfig(
        provider=provider,
        model=model,
        effort=effort,
        temperature=0.7 if effort is not None else None,
    )
    try:
        result, _ = await llm_service.LLMService().complete_text(config, [{"role": "user", "content": "hi"}])
    finally:
        await handler.client.aclose()
    assert result == "answer" and len(bodies) == 1
    body = bodies[0]
    if effort is None:
        assert not {"temperature", "reasoning", "reasoning_effort", "output_config"} & body.keys()
    else:
        assert body.get("temperature") == 0.7, body
        if provider == "anthropic":
            assert body["output_config"]["effort"] == effort
        elif provider == "openrouter":
            assert body["reasoning"]["effort"] == effort
        else:
            assert body["reasoning_effort"] == effort


@pytest.mark.parametrize(
    "body",
    [
        {"inferenceConfig": {"temperature": 0.7}, "additionalModelRequestFields": {"output_config": {"effort": "max"}}},
        {"temperature": 0.7, "reasoning_effort": "max"},
    ],
)
async def test_bedrock_signed_payload_is_sent_unchanged(body: dict[str, Any]) -> None:
    calls = []

    class Delegate:
        async def post(self, **kwargs: Any) -> None:
            calls.append(kwargs)

    guard = _BedrockSamplingHTTPHandler(Delegate(), {"temperature": 0.7, "reasoning_effort": "max"})
    original = {
        "url": "https://provider.test/model/id/converse",
        "data": json.dumps(body),
        "headers": {"Authorization": "mock-signature"},
    }
    await guard.post(**original)
    assert calls == [original]


@pytest.mark.parametrize(
    "body",
    [
        {"temperature": 0.7, "reasoning_effort": "xhigh"},
        {"temperature": 0.7, "thinking": {"type": "enabled", "budget_tokens": 1024}},
        {"temperature": 0.7, "reasoning_effort": "max", "thinking": {"budget_tokens": 1024}},
        {"reasoning_effort": "max"},
        {},
    ],
)
async def test_bedrock_sdk_must_not_drop_convert_or_rename_sampling(body: dict[str, Any]) -> None:
    class Delegate:
        async def post(self, **kwargs: Any) -> None:
            pytest.fail("A changed request must not be sent")

    guard = _BedrockSamplingHTTPHandler(Delegate(), {"temperature": 0.7, "reasoning_effort": "max"})
    with pytest.raises(SamplingConfigurationError, match="cannot preserve"):
        await guard.post(data=json.dumps(body))


async def test_sdk_wrapper_cannot_turn_sampling_error_into_retryable_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    async def completion(**kwargs: Any) -> Any:
        try:
            raise SamplingConfigurationError("adapter changed effort")
        except SamplingConfigurationError as exc:
            raise RuntimeError("SDK wrapper") from exc

    monkeypatch.setattr(llm_service.litellm, "acompletion", completion)
    with pytest.raises(SamplingConfigurationError, match="changed effort"):
        await acompletion(model="bedrock/custom", reasoning_effort="future")
