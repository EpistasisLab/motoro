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
from motoro.services.sampling import UnsupportedEffortError, resolve_sampling

KEY = "private-credential-never-surface"


async def test_anthropic_haiku_and_future_efforts_need_no_registry() -> None:
    levels = ["low", "medium", "high", "xhigh", "max", "future-effort"]
    listing = await discover_models(
        config("anthropic"),
        transport=transport(
            payload={
                "data": [
                    {
                        "id": "claude-haiku-5-5",
                        "capabilities": {
                            "effort": {
                                "supported": True,
                                **{level: {"supported": True} for level in levels},
                                "retired-effort": {"supported": False},
                                "unknown-effort": None,
                            }
                        },
                    }
                ]
            }
        ),
    )
    info = listing.models[0]
    assert info.capabilities.effort_levels == levels
    assert info.defaults.effort.source == "unknown"
    assert resolve_sampling(info).as_kwargs() == {}
    for level in [*levels, "unknown-effort", "unlisted-effort"]:
        assert resolve_sampling(info, effort=level).effort == level
    with pytest.raises(UnsupportedEffortError):
        resolve_sampling(info, effort="retired-effort")


@pytest.mark.parametrize("raw", [None, {}, {"effort": None}, {"effort": {"supported": True}}])
async def test_incomplete_anthropic_metadata_defers_explicit_validation(raw: Any) -> None:
    listing = await discover_models(
        config("anthropic"),
        transport=transport(
            payload={
                "data": [
                    {
                        "id": "claude-haiku-5-5",
                        "capabilities": raw,
                    }
                ]
            }
        ),
    )
    assert listing.source == "api"
    info = listing.models[0]
    assert resolve_sampling(info, effort="medium", temperature=0.7).as_kwargs() == {
        "temperature": 0.7,
        "reasoning_effort": "medium",
    }
    assert resolve_sampling(info).as_kwargs() == {}


async def test_discovery_is_stateless_and_failure_does_not_reuse_previous_facts() -> None:
    good = await discover_models(
        config("anthropic"),
        transport=transport(
            payload={
                "data": [
                    {
                        "id": "future",
                        "capabilities": {"effort": {"supported": False}},
                    }
                ]
            }
        ),
    )
    assert good.models[0].capabilities.supports_effort is False
    failed = await discover_models(config("anthropic"), transport=transport(503, {}))
    info = get_model_metadata(model="future", listing=failed)
    assert info.capabilities.supports_effort is None
    assert resolve_sampling(info, effort="medium").effort == "medium"


@pytest.mark.parametrize(
    "reasoning",
    [
        {"supported_efforts": ["future", "max"], "default_effort": "future"},
        {"supported_efforts": None, "default_effort": "future"},
    ],
)
async def test_openrouter_future_names_defaults_and_invalid_values(reasoning: dict[str, Any]) -> None:
    listing = await discover_models(
        config("openrouter"),
        transport=transport(
            payload={
                "data": [
                    {
                        "id": "new/model",
                        "reasoning": reasoning,
                    }
                ]
            }
        ),
    )
    info = listing.models[0]
    assert resolve_sampling(info).effort == "future"
    assert resolve_sampling(info).effort_source == "advertised_default"
    if reasoning["supported_efforts"] is not None:
        with pytest.raises(UnsupportedEffortError):
            resolve_sampling(info, effort="invalid")
    else:
        assert resolve_sampling(info, effort="another-future-name").effort == "another-future-name"


async def test_openrouter_conditional_default_does_not_enable_reasoning() -> None:
    listing = await discover_models(
        config("openrouter"),
        transport=transport(
            payload={
                "data": [
                    {
                        "id": "new/model",
                        "reasoning": {"supported_efforts": None, "default_effort": "high", "default_enabled": False},
                    }
                ]
            }
        ),
    )
    info = listing.models[0]
    assert info.defaults.effort.value == "high"
    assert resolve_sampling(info).effort is None
    assert resolve_sampling(info, effort="high").effort == "high"


async def test_mandatory_openrouter_reasoning_rejects_explicit_none_with_unknown_ladder() -> None:
    listing = await discover_models(
        config("openrouter"),
        transport=transport(
            payload={
                "data": [
                    {
                        "id": "new/model",
                        "reasoning": {"supported_efforts": None, "mandatory": True},
                    }
                ]
            }
        ),
    )
    with pytest.raises(UnsupportedEffortError):
        resolve_sampling(listing.models[0], effort="none")


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


@pytest.mark.parametrize("provider", ["anthropic", "openai", "openrouter", "azure_foundry", "local"])
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
    assert listing.source == "error"
    assert listing.models == []
    assert KEY not in listing.model_dump_json() + check.model_dump_json() + caplog.text


@pytest.mark.parametrize("provider", ["anthropic", "openai", "openrouter", "azure_foundry", "local"])
@pytest.mark.parametrize("status", [200, 302, 503])
async def test_caller_owned_client_is_reused_with_request_protections(provider: str, status: int) -> None:
    calls: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.extensions["timeout"] == {"connect": 10.0, "read": 10.0, "write": 10.0, "pool": 10.0}
        expected_auth = None if provider in {"anthropic", "azure_foundry"} else f"Bearer {KEY}"
        assert request.headers.get("authorization") == expected_auth
        return httpx.Response(
            status,
            headers={"Location": "https://redirect.test/models"},
            json={"data": [{"id": "m"}], "value": [{"name": "deployment"}]},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond),
        timeout=0.01,
        follow_redirects=True,
        auth=httpx.BasicAuth("unrelated", "credential"),
    ) as client:
        check = await check_connection(config(provider), client=client)
        assert not client.is_closed
        listing = await discover_models(config(provider), client=client)
        assert not client.is_closed
        assert check.status == ("ok" if status == 200 else "unknown" if provider == "local" else "failed")
        assert listing.source == ("api" if status == 200 else "error")
        # No redirect followups, and the Azure/local check delegates through this client.
        assert len(calls) == 2
    assert client.is_closed


@pytest.mark.parametrize("operation", ["check", "discover"])
async def test_client_and_transport_are_mutually_exclusive(operation: str) -> None:
    mock = transport(payload={"data": []})
    async with httpx.AsyncClient(transport=mock) as client:
        call = check_connection if operation == "check" else discover_models
        with pytest.raises(ValueError, match="either client or transport"):
            await call(config("openai"), transport=mock, client=client)
        assert not client.is_closed


@pytest.mark.parametrize("provider", ["anthropic", "openai", "openrouter", "azure_foundry", "local"])
@pytest.mark.parametrize(
    "code,kind", [(401, "credential_rejected"), (403, "credential_rejected"), (429, "rate_limited"), (503, "http")]
)
async def test_discovery_errors_do_not_invent_metadata(provider: str, code: int, kind: str) -> None:
    result = await discover_models(config(provider), transport=transport(code, {"error": KEY}))
    assert result.error is not None and result.error.kind == kind and result.error.http_status == code
    assert result.source == "error" and result.models == []
    assert result.note and KEY not in result.model_dump_json()


@pytest.mark.parametrize(
    "provider,source",
    [
        ("anthropic", "error"),
        ("openai", "error"),
        ("openrouter", "error"),
        ("azure_foundry", "api"),
        ("local", "error"),
    ],
)
async def test_empty_listing_policy(provider: str, source: str) -> None:
    result = await discover_models(config(provider), transport=transport(payload={"data": [], "value": []}))
    assert result.source == source
    assert result.models == []
    unknown = get_model_metadata(model="unknown", listing=result)
    assert all(value == "unknown" for value in unknown.capability_sources.values())
    assert not unknown.defaults.temperature.known and not unknown.defaults.effort.known


async def test_missing_config_never_contacts_provider() -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        pytest.fail("Unexpected provider request")

    mock = httpx.MockTransport(fail)
    for provider in ["openai", "anthropic"]:
        result = await discover_models(ProviderConfig(provider=provider), transport=mock)
        assert result.source == "error" and result.error.kind == "credentials_missing" and result.models == []
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


async def test_openai_lists_live_ids_without_inventing_sampling_facts() -> None:
    calls: list[httpx.Request] = []
    listing = await discover_models(
        config("openai"),
        transport=transport(
            payload={
                "data": [
                    {"id": "future", "object": "model", "owned_by": "openai", "created": 123},
                    {"id": "gpt-5"},
                ]
            },
            requests=calls,
        ),
    )
    assert len(calls) == 1 and listing.source == "api" and listing.error is None
    assert [m.id for m in listing.models] == ["future", "gpt-5"]
    assert calls[0].url.path == "/v1/models" and calls[0].headers["authorization"] == f"Bearer {KEY}"
    assert get_model_defaults(model="gpt-5", listing=listing).effort.source == "unknown"


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
    assert prod.defaults.effort.source == "unknown"
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
    assert opus.capability_sources["supports_temperature"] == "unknown"
    assert opus.defaults.effort.value is None and opus.defaults.effort.source == "unknown"
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
        ({"supported_efforts": None}, None, True),
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


@pytest.mark.parametrize("provider", ["anthropic", "openai", "openrouter", "azure_foundry", "local"])
@pytest.mark.parametrize("payload", [[], {"data": [1], "value": [1]}, {"data": [{"id": 3}], "value": [{"name": 3}]}])
async def test_malformed_metadata_is_normalized(provider: str, payload: Any) -> None:
    result = await discover_models(config(provider), transport=transport(payload=payload))
    assert result.error is not None and result.error.kind == "invalid_response"
    assert result.source == "error"


@pytest.mark.parametrize("provider", ["anthropic", "openai", "openrouter", "azure_foundry", "local"])
@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"error": "bad schema"},
        {"data": None, "value": None},
        {"data": False, "value": False},
        {"data": 0, "value": 0},
        {"data": "", "value": ""},
        {"data": {}, "value": {}},
    ],
)
async def test_missing_or_falsey_listing_fields_are_invalid(provider: str, payload: Any) -> None:
    # A top-level error object and malformed collection fields are not empty lists.
    result = await discover_models(config(provider), transport=transport(payload=payload))
    assert result.error is not None and result.error.kind == "invalid_response"
    if provider == "azure_foundry":
        check = await check_connection(config(provider), transport=transport(payload=payload))
        assert check.status == "failed"
        assert check.error is not None and check.error.kind == "invalid_response"


@pytest.mark.parametrize("provider", ["openrouter", "anthropic"])
@pytest.mark.parametrize("advertised", ["missing", "high", None])
async def test_partial_metadata_does_not_supply_catalog_defaults(provider: str, advertised: str | None) -> None:
    model = "openai/gpt-5" if provider == "openrouter" else "claude-opus-5"
    effort: dict[str, Any] = (
        {"supported_efforts": ["low"]} if provider == "openrouter" else {"supported": True, "low": {"supported": True}}
    )
    if advertised != "missing":
        effort["default_effort"] = advertised
    entry = (
        {"id": model, "reasoning": effort}
        if provider == "openrouter"
        else {"id": model, "capabilities": {"effort": effort}}
    )
    listing = await discover_models(config(provider), transport=transport(payload={"data": [entry]}))
    assert listing.source == "api"
    info = get_model_metadata(model=model, listing=listing)
    default = info.defaults.effort
    assert default.value == (get_capabilities(model).default_effort if advertised == "missing" else advertised)
    assert default.source == ("unknown" if advertised in {"missing", None} else "provider")
    # The catalog default is outside the advertised ladder; inference must still omit it.
    assert info.capabilities.default_effort is None
    assert info.capabilities.effort_levels == ["low"]


async def test_openrouter_support_metadata_keeps_default_unknown() -> None:
    listing = await discover_models(
        config("openrouter"),
        transport=transport(payload={"data": [{"id": "openai/gpt-5", "supported_parameters": ["reasoning_effort"]}]}),
    )
    assert get_model_defaults(model="openai/gpt-5", listing=listing).effort.value is None
    assert get_model_defaults(model="openai/gpt-5", listing=listing).effort.source == "unknown"


@pytest.mark.parametrize("provider", ["anthropic", "openai", "openrouter", "azure_foundry", "local"])
@pytest.mark.parametrize("endpoint_field", ["api_base", "azure_project_endpoint"])
@pytest.mark.parametrize("port", ["abc", "-1", "65536"])
def test_endpoint_ports_are_validated(provider: str, endpoint_field: str, port: str) -> None:
    with pytest.raises(ValidationError):
        config(provider, **{endpoint_field: f"https://example.test:{port}/v1"})


@pytest.mark.parametrize("base", ["https://example.test:0/v1", "https://example.test:65535/v1", "http://[::1]:8080/v1"])
def test_valid_endpoint_ports(base: str) -> None:
    assert config("local", api_base=base).api_base == base


async def test_missing_metadata_keeps_fields_unknown() -> None:
    result = await discover_models(
        ProviderConfig(provider="openrouter"),
        transport=transport(payload={"data": [{"id": "gpt-5"}, {"id": "never-seen"}]}),
    )
    assert get_model_metadata(model="gpt-5", listing=result).capabilities == get_capabilities("gpt-5")
    assert get_model_defaults(model="gpt-5", listing=result).effort.source == "unknown"
    unknown = get_model_metadata(model="never-seen", listing=result)
    assert unknown.capabilities.supports_effort is None and not unknown.defaults.effort.known
    assert not get_model_defaults(model="gpt-5", listing=ModelListing(source="api")).effort.known
    result.models[0].capabilities.supports_temperature = None
    assert get_capabilities("gpt-5").supports_temperature is None


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
