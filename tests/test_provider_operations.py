"""Provider contract using deterministic HTTP responses, without application state."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from motoro.services.model_capabilities import get_capabilities
from motoro.services.provider_types import DefaultValue, ModelListing, ProviderConfig
from motoro.services.providers import check_connection, discover_models, get_model_defaults, get_model_metadata

KEY = "private-credential-never-surface"


def config(provider: str, **kwargs: Any) -> ProviderConfig:
    if provider == "azure_foundry":
        kwargs = {
            "api_base": "example",
            "azure_project_endpoint": "https://example.test/api/projects/project",
            **kwargs,
        }
    if provider == "local":
        kwargs = {"api_base": "http://localhost:8080/v1", **kwargs}
    return ProviderConfig(provider=provider, api_key=KEY, **kwargs)


def transport(
    status: int = 200, payload: Any = None, requests: list[httpx.Request] | None = None
) -> httpx.MockTransport:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.extensions["timeout"] == {"connect": 10.0, "read": 10.0, "write": 10.0, "pool": 10.0}
        if requests is not None:
            requests.append(request)
        return httpx.Response(status, json=payload)

    return httpx.MockTransport(respond)


@pytest.mark.parametrize("provider", ["anthropic", "openai", "openrouter", "azure_foundry", "local"])
@pytest.mark.parametrize("code", [200, 401, 403, 429, 500, 302])
async def test_connection_statuses(provider: str, code: int) -> None:
    calls: list[httpx.Request] = []
    result = await check_connection(
        config(provider), transport=transport(code, {"data": [{"id": "m"}], "value": []}, calls)
    )
    expected = "ok" if code == 200 else "failed"
    if provider == "local" and code != 200 or code == 429 and provider not in {"azure_foundry", "local"}:
        expected = "unknown"
    assert result.status == expected
    assert len(calls) == 1
    assert KEY not in result.model_dump_json()
    if code in {401, 403, 429}:
        assert result.error is not None
        assert result.error.kind == ("rate_limited" if code == 429 else "credential_rejected")
    if code == 200 and provider not in {"local", "azure_foundry"}:
        assert "doesn't verify quota or billing" in result.detail


@pytest.mark.parametrize("provider", ["anthropic", "openrouter", "azure_foundry", "local"])
@pytest.mark.parametrize("kind", ["network", "timeout"])
async def test_transport_failures_are_safe_and_distinct(
    provider: str, kind: str, caplog: pytest.LogCaptureFixture
) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        cls = httpx.ReadTimeout if kind == "timeout" else httpx.ConnectError
        raise cls(f"headers contain {KEY}", request=request)

    listing = await discover_models(config(provider), transport=httpx.MockTransport(fail))
    check = await check_connection(config(provider), transport=httpx.MockTransport(fail))
    assert listing.error is not None and listing.error.kind == kind
    assert check.error is not None and check.error.kind == kind
    assert listing.source == ("static" if provider == "anthropic" else "error")
    assert bool(listing.models) == (provider == "anthropic")
    assert KEY not in listing.model_dump_json() + check.model_dump_json() + caplog.text


@pytest.mark.parametrize("provider", ["anthropic", "openrouter", "azure_foundry", "local"])
@pytest.mark.parametrize(
    "code,kind", [(401, "credential_rejected"), (403, "credential_rejected"), (429, "rate_limited"), (503, "http")]
)
async def test_discovery_errors_and_catalog_fallback(provider: str, code: int, kind: str) -> None:
    result = await discover_models(config(provider), transport=transport(code, {"error": KEY}))
    assert result.error is not None and result.error.kind == kind and result.error.http_status == code
    assert result.source == ("static" if provider == "anthropic" else "error")
    assert result.note and KEY not in result.model_dump_json()
    if provider == "anthropic":
        assert result.models
        assert result.models[0].defaults.effort.source == "catalog"


@pytest.mark.parametrize(
    "provider,source", [("anthropic", "static"), ("openrouter", "error"), ("azure_foundry", "api"), ("local", "error")]
)
async def test_empty_listing_policy(provider: str, source: str) -> None:
    result = await discover_models(config(provider), transport=transport(payload={"data": [], "value": []}))
    assert result.source == source
    assert bool(result.models) == (provider == "anthropic")
    unknown = get_model_metadata(model="unknown", listing=result)
    assert all(value == "unknown" for value in unknown.capability_sources.values())
    assert not unknown.defaults.temperature.known and not unknown.defaults.effort.known


async def test_catalog_and_missing_config_never_contact_provider() -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        pytest.fail("Unexpected provider request")

    mock = httpx.MockTransport(fail)
    for provider in ["openai", "anthropic"]:
        result = await discover_models(ProviderConfig(provider=provider), transport=mock)
        assert result.source == "static" and result.models
    for provider in ["openai", "anthropic", "openrouter"]:
        result = await check_connection(ProviderConfig(provider=provider), transport=mock)
        assert result.status == "unknown" and result.error.kind == "credentials_missing"
    for provider in ["azure_foundry", "local"]:
        result = await discover_models(ProviderConfig(provider=provider), transport=mock)
        assert result.source == "error" and result.error.kind == "unavailable"
        check = await check_connection(ProviderConfig(provider=provider), transport=mock)
        assert check.status == "unknown"
    azure = config("azure_foundry", azure_project_endpoint=None)
    assert (await check_connection(azure, transport=mock)).status == "unknown"
    assert "Project endpoint" in (await discover_models(azure, transport=mock)).note


async def test_openai_catalog_ignores_listing_transport_even_with_key() -> None:
    calls: list[httpx.Request] = []
    listing = await discover_models(config("openai"), transport=transport(401, {}, calls))
    assert calls == [] and listing.source == "static" and listing.error is None
    assert get_model_defaults(model="gpt-5", listing=listing).effort.value == "medium"
    assert get_model_defaults(model="gpt-5", listing=listing).effort.source == "catalog"


async def test_azure_ids_underlying_model_capabilities_and_check_consistency() -> None:
    calls: list[httpx.Request] = []
    payload = {
        "value": [
            {"name": "prod", "modelName": "claude-opus-5", "capabilities": {"chat_completion": "true"}},
            {"name": "custom", "modelName": "unknown"},
        ]
    }
    listing = await discover_models(config("azure_foundry"), transport=transport(payload=payload, requests=calls))
    check = await check_connection(config("azure_foundry"), transport=transport(payload=payload))
    assert check.models == listing.models
    assert [m.id for m in listing.models] == ["custom", "prod"]
    prod = get_model_metadata(model="prod", listing=listing)
    assert prod.label == "claude-opus-5" and prod.capabilities == get_capabilities("claude-opus-5")
    assert prod.defaults.effort.source == "catalog"
    assert calls[0].url.path == "/api/projects/project/deployments"
    assert calls[0].url.params["api-version"] == "v1" and calls[0].headers["api-key"] == KEY


async def test_anthropic_pagination_and_mixed_provenance() -> None:
    calls: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.url.path == "/v1/models"
        assert request.headers["x-api-key"] == KEY
        if len(calls) == 1:
            assert request.url.params["limit"] == "100"
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "claude-opus-5",
                            "display_name": "Opus",
                            "capabilities": {"effort": {"supported": True, "high": {"supported": True}}},
                        }
                    ],
                    "has_more": True,
                    "last_id": "claude-opus-5",
                },
            )
        assert request.url.params["after_id"] == "claude-opus-5"
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "new",
                        "capabilities": {
                            "temperature": {"supported": True, "default": 0},
                            "effort": {"supported": False},
                        },
                    }
                ]
            },
        )

    listing = await discover_models(
        config("anthropic", api_base="https://example.test/v1"), transport=httpx.MockTransport(respond)
    )
    assert listing.source == "api" and len(calls) == 2
    opus, new = listing.models
    assert opus.capability_sources["supports_effort"] == "provider"
    assert opus.capability_sources["supports_temperature"] == "catalog"
    assert opus.defaults.effort.value == "high" and opus.defaults.effort.source == "catalog"
    assert new.capabilities.supports_effort is False
    assert new.defaults.temperature.value == 0 and new.defaults.temperature.source == "provider"
    assert not new.defaults.effort.known


async def test_anthropic_page_bound() -> None:
    calls: list[httpx.Request] = []
    result = await discover_models(
        config("anthropic"),
        transport=transport(payload={"data": [{"id": "new"}], "has_more": True, "last_id": "new"}, requests=calls),
    )
    assert len(calls) == 10 and result.source == "api"


async def test_public_openrouter_defaults_and_support_are_independent() -> None:
    calls: list[httpx.Request] = []
    result = await discover_models(
        ProviderConfig(provider="openrouter"),
        transport=transport(
            payload={
                "data": [
                    {
                        "id": "unknown/model",
                        "name": "Public",
                        "supported_parameters": ["temperature", "reasoning"],
                        "reasoning": {
                            "supported_efforts": ["none", "low", "high"],
                            "mandatory": True,
                            "default_effort": "high",
                        },
                        "default_parameters": {
                            "temperature": 0,
                            "stream": False,
                            "stop": [],
                            "prefix": "",
                            "missing": None,
                        },
                        "context_length": 1000,
                    }
                ]
            },
            requests=calls,
        ),
    )
    assert "Authorization" not in calls[0].headers
    model = result.models[0]
    assert model.capabilities.effort_levels == ["low", "high"]
    assert model.defaults.temperature.value == 0 and model.defaults.effort.value == "high"
    for name, expected in [("stream", False), ("stop", []), ("prefix", "")]:
        default = model.defaults.additional[name]
        assert default.known and default.value == expected and default.source == "provider"
    assert not model.defaults.additional["missing"].known
    assert "context_length" not in model.defaults.additional
    assert ModelListing.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize(
    "reasoning,levels,support",
    [
        ({"supported_efforts": None}, ["none", "minimal", "low", "medium", "high", "xhigh", "max"], True),
        ({"supported_efforts": []}, [], False),
        ({}, None, False),
    ],
)
async def test_openrouter_reasoning_semantics(reasoning: Any, levels: Any, support: bool) -> None:
    result = await discover_models(
        ProviderConfig(provider="openrouter"),
        transport=transport(payload={"data": [{"id": "new", "reasoning": reasoning}]}),
    )
    assert result.models[0].capabilities.supports_effort is support
    assert result.models[0].capabilities.effort_levels == levels


@pytest.mark.parametrize("provider", ["anthropic", "openrouter", "azure_foundry", "local"])
@pytest.mark.parametrize("payload", [[], {"data": [1], "value": [1]}, {"data": [{"id": 3}], "value": [{"name": 3}]}])
async def test_malformed_metadata_is_normalized(provider: str, payload: Any) -> None:
    result = await discover_models(config(provider), transport=transport(payload=payload))
    assert result.error is not None and result.error.kind == "invalid_response"
    assert result.source == ("static" if provider == "anthropic" else "error")


async def test_missing_metadata_keeps_curated_facts_and_unknowns() -> None:
    result = await discover_models(
        ProviderConfig(provider="openrouter"),
        transport=transport(payload={"data": [{"id": "gpt-5"}, {"id": "never-seen"}]}),
    )
    assert get_model_metadata(model="gpt-5", listing=result).capabilities == get_capabilities("gpt-5")
    assert get_model_defaults(model="gpt-5", listing=result).effort.source == "catalog"
    unknown = get_model_metadata(model="never-seen", listing=result)
    assert unknown.capabilities.supports_effort is None and not unknown.defaults.effort.known
    assert get_model_defaults(model="gpt-5", listing=ModelListing(source="api")).effort.known
    result.models[0].capabilities.supports_temperature = None
    assert get_capabilities("gpt-5").supports_temperature is False


async def test_partial_temperature_metadata_does_not_erase_independent_facts() -> None:
    for provider, entry in [
        ("anthropic", {"id": "new", "capabilities": {"temperature": {"supported": True, "default": 0}}}),
        ("openrouter", {"id": "gpt-4o", "default_parameters": {"temperature": 0}}),
    ]:
        result = await discover_models(config(provider), transport=transport(payload={"data": [entry]}))
        assert result.source == "api"
        assert result.models[0].defaults.temperature.value == 0
        assert result.models[0].defaults.temperature.source == "provider"


@pytest.mark.parametrize("provider", ["openrouter", "anthropic"])
@pytest.mark.parametrize("support", [None, False, True])
async def test_advertised_defaults_survive_partial_support_metadata(provider: str, support: bool | None) -> None:
    if provider == "openrouter":
        entry: dict[str, Any] = {"id": "new/model", "default_parameters": {"temperature": 0}}
        if support is not None:
            entry["supported_parameters"] = ["temperature"] if support else []
    else:
        temperature: dict[str, Any] = {"default": 0}
        if support is not None:
            temperature["supported"] = support
        entry = {"id": "new/model", "capabilities": {"temperature": temperature}}
    listing = await discover_models(config(provider), transport=transport(payload={"data": [entry]}))
    assert listing.source == "api" and listing.error is None
    metadata = get_model_metadata(model="new/model", listing=listing)
    defaults = get_model_defaults(model="new/model", listing=listing)
    for info in [listing.models[0], metadata]:
        assert info.capabilities.supports_temperature is support
        assert info.capabilities.default_temperature == (0 if support is True else None)
        assert info.defaults == defaults
    assert defaults.temperature.value == 0
    assert defaults.temperature.source == "provider" and defaults.temperature.known
    assert not defaults.effort.known and defaults.effort.source == "unknown"
    assert ModelListing.model_validate_json(listing.model_dump_json()) == listing


@pytest.mark.parametrize("provider", ["openrouter", "anthropic"])
async def test_advertised_effort_default_survives_missing_ladder(provider: str) -> None:
    entry = {"id": "new/model"}
    if provider == "openrouter":
        entry["reasoning"] = {"default_effort": "high"}
    else:
        entry["capabilities"] = {"effort": {"default_effort": "high"}}
    listing = await discover_models(config(provider), transport=transport(payload={"data": [entry]}))
    metadata = get_model_metadata(model="new/model", listing=listing)
    defaults = get_model_defaults(model="new/model", listing=listing)
    assert listing.source == "api" and metadata.capabilities.default_effort is None
    assert metadata.capabilities.effort_levels is None
    assert metadata.defaults == defaults
    assert defaults.effort.known and defaults.effort.value == "high" and defaults.effort.source == "provider"
    assert not defaults.temperature.known


@pytest.mark.parametrize("value", [0, False, "", [], {}, None])
async def test_partial_metadata_retains_falsey_provider_defaults(value: Any) -> None:
    listing = await discover_models(
        config("openrouter"),
        transport=transport(
            payload={
                "data": [
                    {
                        "id": "new/model",
                        "default_parameters": {"temperature": 0, "advertised": value},
                    }
                ]
            }
        ),
    )
    defaults = get_model_defaults(model="new/model", listing=listing)
    assert defaults.temperature.known and defaults.temperature.value == 0
    additional = defaults.additional["advertised"]
    assert additional.value == value and type(additional.value) is type(value)
    assert additional.known is (value is not None)
    assert additional.source == ("provider" if value is not None else "unknown")


@pytest.mark.parametrize("provider", ["openrouter", "anthropic"])
async def test_missing_and_null_provider_defaults_remain_unknown(provider: str) -> None:
    entry: dict[str, Any] = {"id": "new/model"}
    if provider == "openrouter":
        entry.update(default_parameters={"temperature": None}, reasoning={"default_effort": None})
    else:
        entry["capabilities"] = {"temperature": {"default": None}, "effort": {"default_effort": None}}
    listing = await discover_models(config(provider), transport=transport(payload={"data": [entry, {"id": "missing"}]}))
    for model in ["new/model", "missing"]:
        defaults = get_model_defaults(model=model, listing=listing)
        assert defaults.temperature.value is None and not defaults.temperature.known
        assert defaults.temperature.source == "unknown"
        assert defaults.effort.value is None and not defaults.effort.known
        assert defaults.effort.source == "unknown"


async def test_echoed_credentials_are_redacted_in_metadata() -> None:
    result = await discover_models(
        config("openrouter"),
        transport=transport(payload={"data": [{"id": "new", "name": KEY, "default_parameters": {"echo": KEY}}]}),
    )
    assert KEY not in result.model_dump_json()


@pytest.mark.parametrize("resource", ["my-resource/", " my-resource/// ", "my-resource/anthropic/"])
async def test_azure_resource_name_without_project_stays_inconclusive(resource: str) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        pytest.fail("An Azure check without a project endpoint must not make a request")

    config = ProviderConfig(provider="azure_foundry", api_base=resource, api_key=KEY)
    result = await check_connection(config, transport=httpx.MockTransport(fail))
    assert result.status == "unknown"
    assert result.endpoint == resource
    assert result.error is not None and result.error.kind == "unavailable"
    assert KEY not in config.model_dump_json() + repr(config) + result.model_dump_json()


@pytest.mark.parametrize(
    "base",
    [
        "https://user:pass@example.test/anthropic/",
        "https://example.test?key=secret",
        "https://example.test/#secret",
        f"https://example.test/{KEY}",
        "my-resource/other-path",
        "//example.test/",
        "ftp://example.test/",
        "my-resource?secret=value",
        "my-resource#secret",
        "user@my-resource/",
    ],
)
def test_azure_normalization_preserves_endpoint_validation(base: str) -> None:
    with pytest.raises(ValidationError) as exc:
        ProviderConfig(provider="azure_foundry", api_base=base, api_key=KEY)
    assert KEY not in str(exc.value)


def test_azure_project_endpoint_must_still_be_a_url() -> None:
    with pytest.raises(ValidationError):
        ProviderConfig(provider="azure_foundry", api_base="my-resource/", azure_project_endpoint="my-project/")


@pytest.mark.parametrize(
    "base",
    [
        "https://user:pass@example.test",
        "https://example.test?key=secret",
        "https://example.test/#secret",
        f"https://example.test/{KEY}",
    ],
)
def test_endpoint_credentials_are_rejected_without_echo(base: str) -> None:
    with pytest.raises(ValidationError) as exc:
        config("openai", api_base=base)
    assert KEY not in str(exc.value)


def test_config_serialization_and_falsey_defaults() -> None:
    value = config("anthropic")
    assert KEY not in repr(value) + value.model_dump_json()
    for known in [0, False, "", [], {}]:
        assert DefaultValue(value=known, source="provider").known
    assert not DefaultValue().known
    with pytest.raises(ValidationError):
        DefaultValue(value=0)


def test_independent_consumer(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    # Fresh interpreter verifies the consumer imports no app/database/runtime.
    program = (
        "import runpy,sys; runpy.run_path(sys.argv[1],run_name='__main__'); "
        "assert not any(n.startswith(('asaree', 'motoro.models', 'motoro.runner')) for n in sys.modules); "
        "assert sys.modules['motoro.config']._instance is None"
    )
    result = subprocess.run(
        [sys.executable, "-c", program, str(root / "examples/provider_metadata.py")],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert '"source":"provider"' in result.stdout
