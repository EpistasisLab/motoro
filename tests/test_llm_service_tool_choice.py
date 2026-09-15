"""tool_choice is only sent where the route supports it.

litellm's ``azure_ai/`` route rejects an explicit ``tool_choice`` even
though every chat endpoint defaults to auto when tools are present —
``litellm.get_supported_openai_params("azure_ai/...")`` omits it in the
pinned litellm build, so an unconditional ``tool_choice="auto"`` in
``_complete_with_tools_once`` fails the call before a request is ever
sent.  The param is redundant anyway (auto *is* the server default), so
it is sent only when the model's param table supports it.
"""

from __future__ import annotations

from typing import Any

import litellm

from motoro.schemas.agent import ModelConfig
from motoro.services.llm_service import LLMService

TOOL = {
    "type": "function",
    "function": {
        "name": "echo",
        "description": "Echo the given text back.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
}


async def _stubbed_connection(model: str, api_base: str | None = None) -> Any:
    async def resolve(config: Any, principal_id: Any) -> dict[str, Any]:
        return {
            "model": model,
            "api_key": "k",
            "api_base": api_base,
            "api_version": None,
            "aws_region_name": None,
        }

    return resolve


class _StubLLM:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    async def __call__(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        message = type("Message", (), {"tool_calls": [], "content": ""})()
        choice = type("Choice", (), {"message": message})()
        response = type("Response", (), {"choices": [choice], "usage": None})()
        return response


async def test_tool_choice_sent_for_a_provider_that_supports_it(monkeypatch: Any) -> None:
    stub = _StubLLM()
    monkeypatch.setattr(litellm, "acompletion", stub)
    monkeypatch.setattr(
        "motoro.services.llm_service._resolve_connection_for_principal",
        await _stubbed_connection("openai/gpt-test"),
    )
    service = LLMService()
    await service.complete_with_tools(
        ModelConfig(provider="openai", model="gpt-test", temperature=0.7, max_tokens=64),
        [{"role": "user", "content": "hi"}],
        [TOOL],
    )
    assert stub.kwargs.get("tool_choice") == "auto"


async def test_tool_choice_omitted_on_the_azure_ai_route(monkeypatch: Any) -> None:
    stub = _StubLLM()
    monkeypatch.setattr(litellm, "acompletion", stub)
    monkeypatch.setattr(
        "motoro.services.llm_service._resolve_connection_for_principal",
        await _stubbed_connection(
            "azure_ai/FW-GLM-5.3", "https://example.services.ai.azure.com"
        ),
    )
    service = LLMService()
    await service.complete_with_tools(
        ModelConfig(
            provider="azure_foundry", model="FW-GLM-5.3", temperature=0.7, max_tokens=64
        ),
        [{"role": "user", "content": "hi"}],
        [TOOL],
    )
    # The azure_ai param table in a supporting litellm build lists it; in the
    # pinned build it does not.  The assertion is on Motoro's own behavior,
    # not litellm's table: tool_choice must simply never be *forced* onto a
    # route that does not declare it.  Assert exact absence-or-supported:
    supported = "tool_choice" in (
        litellm.get_supported_openai_params(model="azure_ai/FW-GLM-5.3") or []
    )
    if not supported:
        assert "tool_choice" not in stub.kwargs
