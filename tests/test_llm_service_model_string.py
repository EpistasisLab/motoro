"""litellm model-string routing — where a provider's enum value diverges from
the litellm provider prefix it actually needs (see
``services.llm_service._LITELLM_PROVIDER_PREFIX``).
"""

from __future__ import annotations

from motoro.schemas.agent import LLMProvider, ModelConfig
from motoro.services.llm_service import _build_model_string, _resolve_connection


def test_openrouter_model_string_matches_its_own_enum_value() -> None:
    """litellm has a native `openrouter/` route, so no override is needed."""
    config = ModelConfig(provider=LLMProvider.OPENROUTER, model="anthropic/claude-sonnet-5")
    assert _build_model_string(config) == "openrouter/anthropic/claude-sonnet-5"
    assert _resolve_connection(config)["model"] is None


def test_local_model_string_rides_the_generic_openai_route() -> None:
    """No dedicated "local" litellm integration -- routed through `openai/`
    plus a required `api_base` override (see services.credentials)."""
    config = ModelConfig(provider=LLMProvider.LOCAL, model="llama-3-70b-instruct")
    assert _build_model_string(config) == "openai/llama-3-70b-instruct"
    assert _resolve_connection(config)["model"] == "openai/llama-3-70b-instruct"


def test_azure_foundry_model_string_unchanged_by_the_prefix_map() -> None:
    """Pin the pre-existing behavior the new provider-prefix map generalized."""
    config = ModelConfig(provider=LLMProvider.AZURE_FOUNDRY, model="claude-sonnet-5")
    assert _build_model_string(config) == "azure_ai/claude-sonnet-5"
