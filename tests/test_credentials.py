"""Credential resolution — the reason a run can reach a provider at all.

The bug these pin: ``ModelConfig.api_key`` is ``exclude=True``, so a key set at
agent-creation time does **not** survive being persisted with the agent and
rebuilt when the run executes. Resolution therefore has to happen at call time,
from settings, and it has to run with no principal id.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_settings import SettingsConfigDict

from agentic_core import CoreSettings
from agentic_core.schemas.agent import LLMProvider, ModelConfig
from agentic_core.services.credentials import (
    disable_credential_resolution,
    env_credential_resolver,
    foundry_api_base,
    get_credential_resolver,
    resolve,
    set_credential_resolver,
)


class _Settings(CoreSettings):
    """A product's settings class, prefix and all."""

    model_config = SettingsConfigDict(env_prefix="TESTPROD_", extra="ignore", populate_by_name=True)


#: Ambient provider credentials must not leak into these tests — a developer
#: with a real ANTHROPIC_FOUNDRY_API_KEY exported would otherwise see the real
#: key where the test asserts a fixture value.
_PROVIDER_ENV = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_FOUNDRY_API_KEY",
    "ANTHROPIC_FOUNDRY_RESOURCE",
    "AWS_REGION",
)


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> Any:
    from agentic_core.config import reset_for_testing

    for name in _PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    reset_for_testing()
    set_credential_resolver(None)
    yield
    set_credential_resolver(None)
    reset_for_testing()


def _configure(**kw: Any) -> None:
    from agentic_core.config import configure

    configure(_Settings(**kw))


def test_api_key_does_not_survive_persistence() -> None:
    """The bug in one assertion: a key on ModelConfig is lost on round trip.

    ``create_agent`` persists ``model_config.model_dump()`` and ``execute_run``
    rebuilds from it, so a credential set at agent-creation time is gone by the
    time the provider is called. This is why a resolver exists.
    """
    cfg = ModelConfig(provider=LLMProvider.ANTHROPIC, model="claude-sonnet-5", api_key="secret")
    assert cfg.api_key == "secret"
    assert ModelConfig(**cfg.model_dump(mode="json")).api_key is None


def test_foundry_base_url_is_the_bare_host() -> None:
    """litellm's azure_ai route appends /anthropic/v1/messages itself.

    A base that already ends in ``/anthropic`` double-counts the segment. Foundry's
    console shows the ``/anthropic`` form, so users paste it — normalize on read.
    """
    assert foundry_api_base("my-res") == "https://my-res.services.ai.azure.com"
    assert foundry_api_base("my-res/anthropic") == "https://my-res.services.ai.azure.com"
    assert foundry_api_base("https://custom.example.com/anthropic") == "https://custom.example.com"
    assert foundry_api_base("https://custom.example.com/") == "https://custom.example.com"


async def test_credentials_read_bare_env_names_despite_product_prefix() -> None:
    """``ANTHROPIC_API_KEY``, not ``TESTPROD_ANTHROPIC_API_KEY``.

    Provider credentials use conventional names that tooling already sets;
    forcing them under a per-product prefix buys nothing.
    """
    _configure(anthropic_api_key="sk-test")
    conn = await env_credential_resolver(ModelConfig(provider=LLMProvider.ANTHROPIC, model="m"))
    assert conn is not None
    assert conn["api_key"] == "sk-test"


async def test_foundry_resolves_key_base_and_model_prefix() -> None:
    _configure(anthropic_foundry_api_key="fk", anthropic_foundry_resource="res")
    conn = await env_credential_resolver(
        ModelConfig(provider=LLMProvider.AZURE_FOUNDRY, model="claude-sonnet-5")
    )
    assert conn == {
        "model": "azure_ai/claude-sonnet-5",
        "api_key": "fk",
        "api_base": "https://res.services.ai.azure.com",
        # Claude on Foundry goes through the Anthropic passthrough, which 404s
        # if an api-version query param is attached.
        "api_version": None,
        "aws_region_name": None,
    }


async def test_foundry_key_without_resource_is_an_error() -> None:
    """The key does not encode an endpoint — say so instead of failing at 401."""
    _configure(anthropic_foundry_api_key="fk", anthropic_foundry_resource="")
    with pytest.raises(ValueError, match="ANTHROPIC_FOUNDRY_RESOURCE"):
        await env_credential_resolver(ModelConfig(provider=LLMProvider.AZURE_FOUNDRY, model="m"))


async def test_resolver_abstains_when_no_credential_configured() -> None:
    """``None`` means 'no opinion' so the ModelConfig still wins."""
    _configure()
    assert await env_credential_resolver(ModelConfig(provider=LLMProvider.ANTHROPIC, model="m")) is None


async def test_resolution_runs_without_a_principal() -> None:
    """Settings are process-wide; requiring a principal would break every run.

    The runner passes ``run.owner_id``, which is ``None`` unless a product sets
    it — so a principal-gated resolver would never fire.
    """
    _configure(anthropic_api_key="sk-test")
    default: dict[str, str | None] = {"api_key": None, "api_base": None, "model": None}
    conn = await resolve(default, ModelConfig(provider=LLMProvider.ANTHROPIC, model="m"), None)
    assert conn["api_key"] == "sk-test"


async def test_installed_resolver_overrides_the_default() -> None:
    _configure(anthropic_api_key="from-settings")

    async def custom(config: Any, principal_id: Any) -> dict[str, str | None]:
        return {"api_key": "from-product", "api_base": None, "model": None}

    set_credential_resolver(custom)
    conn = await resolve({}, ModelConfig(provider=LLMProvider.ANTHROPIC, model="m"), None)
    assert conn["api_key"] == "from-product"


async def test_resolution_can_be_disabled_entirely() -> None:
    """For a product that wants only an explicit per-call credential."""
    _configure(anthropic_api_key="from-settings")
    disable_credential_resolution()
    assert get_credential_resolver() is None
    default: dict[str, str | None] = {"api_key": None}
    assert await resolve(default, ModelConfig(provider=LLMProvider.ANTHROPIC, model="m"), None) == default
