"""Public, stateless provider operations. Only GET requests; no inference.

The caller owns credential resolution, authorization, rate limits and caching.
Pass a MockTransport for deterministic consumers/tests. Every request has the
baseline 10 second timeout and redirects are disabled to protect auth headers.
"""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx

from motoro.services._provider_capabilities import (
    _capabilities_from_anthropic,
    _capabilities_from_openrouter,
)
from motoro.services.credentials import Connection, foundry_api_base
from motoro.services.model_capabilities import ModelCapabilities
from motoro.services.provider_types import (
    ConnectionCheck,
    DefaultValue,
    ModelDefaults,
    ModelInfo,
    ModelListing,
    Provenance,
    ProviderConfig,
    ProviderError,
)

_BASES = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}
NO_PROJECT_ENDPOINT_NOTE = (
    "Azure deployment discovery requires a Project endpoint "
    "(https://{resource}.services.ai.azure.com/api/projects/{project}). "
    "Deployment names can still be supplied explicitly for inference."
)
_LOCAL_NOTE = "The server did not return a model list. Model names can still be supplied explicitly for inference."


@asynccontextmanager
async def _provider_client(
    client: httpx.AsyncClient | None, transport: httpx.AsyncBaseTransport | None
) -> AsyncIterator[httpx.AsyncClient]:
    if client is not None:
        yield client
    else:
        async with httpx.AsyncClient(transport=transport) as owned_client:
            yield owned_client


def _key(config: ProviderConfig) -> str:
    return config.api_key.get_secret_value() if config.api_key else ""


def _models_url(base: str) -> str:
    base = base.rstrip("/")
    if base.endswith("/models"):
        return base
    return f"{base}/models" if base.endswith("/v1") else f"{base}/v1/models"


def _headers(config: ProviderConfig) -> dict[str, str]:
    key = _key(config)
    if config.provider == "anthropic":
        return {"x-api-key": key, "anthropic-version": "2023-06-01"}
    if config.provider == "azure_foundry":
        return {"api-key": key}
    if config.provider == "local":
        return {"Authorization": f"Bearer {key or 'not-needed'}"}
    return {"Authorization": f"Bearer {key}"} if key else {}


def _error(exc: Exception) -> ProviderError:
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        kind = {401: "credential_rejected", 403: "credential_rejected", 429: "rate_limited"}.get(code, "http")
        return ProviderError(kind=kind, http_status=code)
    if isinstance(exc, httpx.TimeoutException):
        return ProviderError(kind="timeout")
    if isinstance(exc, httpx.HTTPError):
        return ProviderError(kind="network")
    return ProviderError(kind="invalid_response")


def _error_message(error: ProviderError) -> str:
    # Never expose response bodies or exception strings, which can echo keys.
    if error.http_status is not None:
        return f"The provider returned HTTP {error.http_status}."
    return {
        "timeout": "The provider request timed out.",
        "network": "Could not connect to the provider.",
        "credentials_missing": "No credential supplied for this provider.",
    }.get(error.kind, "The provider returned an invalid model listing.")


def _default(value: Any, source: Provenance) -> DefaultValue:
    return DefaultValue(value=value, source=source if value is not None else "unknown")


def _model(
    model: str,
    label: str | None = None,
    *,
    capabilities: ModelCapabilities | None = None,
    advertised: set[str] | None = None,
    additional: dict[str, Any] | None = None,
) -> ModelInfo:
    caps = capabilities if capabilities is not None else ModelCapabilities()
    sources: dict[str, Provenance] = {
        field: "provider" if value is not None and field in (advertised or set()) else "unknown"
        for field, value in caps.model_dump().items()
    }
    return ModelInfo(
        id=model,
        label=label,
        capabilities=caps,
        capability_sources=sources,
        defaults=ModelDefaults(
            temperature=_default(caps.default_temperature, sources["default_temperature"]),
            effort=_default(caps.default_effort, sources["default_effort"]),
            additional={name: _default(value, "provider") for name, value in (additional or {}).items()},
        ),
    )


def _redact_payload(value: Any, key: str) -> Any:
    if not key:
        return value
    if isinstance(value, str):
        return value.replace(key, "***")
    if isinstance(value, list):
        return [_redact_payload(item, key) for item in value]
    if isinstance(value, dict):
        return {name.replace(key, "***"): _redact_payload(item, key) for name, item in value.items()}
    return value


def get_model_metadata(*, model: str, listing: ModelListing | None = None) -> ModelInfo:
    """Use a caller's live/cached listing; absent IDs have unknown metadata.

    No I/O. An unrecognized ID has explicit unknown fields, not invented
    defaults. Listing.error/note remain on the supplied listing for the caller.
    """
    if listing is not None:
        for entry in listing.models:
            if entry.id == model:
                return entry.model_copy(deep=True)
    return _model(model)


def get_model_defaults(*, model: str, listing: ModelListing | None = None) -> ModelDefaults:
    """Provider/model facts only; does not invoke sampling preference policy."""
    return get_model_metadata(model=model, listing=listing).defaults


def get_inference_connection(config: ProviderConfig, *, model: str) -> Connection:
    """Adapt explicit credentials to the existing inference resolver contract.

    Applications retain storage and principal lookup; provider routing and
    endpoint normalization stay here. This does not retrieve metadata or infer.
    """
    base = config.api_base
    model_override = None
    key = _key(config)
    if config.provider == "azure_foundry":
        if not base:
            raise ValueError("Azure Foundry requires a resource name or base URL for inference")
        base = foundry_api_base(base)
        model_override = f"azure_ai/{model}"
    elif config.provider == "local":
        if not base:
            raise ValueError("Local inference requires a base URL")
        model_override = f"openai/{model}"
        key = key or "not-needed"
    return {
        "model": model_override,
        "api_key": key or None,
        "api_base": base,
        "api_version": None,
        "aws_region_name": None,
    }


def _parse(config: ProviderConfig, entry: dict[str, Any]) -> ModelInfo | None:
    provider = config.provider
    model = entry.get("name" if provider == "azure_foundry" else "id")
    if not model:
        return None
    if not isinstance(model, str):
        raise ValueError("Invalid model ID")
    if provider == "azure_foundry":
        label = entry.get("modelName")
        info = _model(label or model, label)
        info.id = model
        return info
    if provider in {"local", "openai"}:
        return _model(model)
    advertised: set[str] = set()
    provider_defaults: dict[str, Any] = {}
    additional: dict[str, Any] = {}
    if provider == "anthropic":
        raw = entry.get("capabilities") or {}
        effort = raw.get("effort") or {}
        temperature = raw.get("temperature") or {}
        caps = _capabilities_from_anthropic(raw, model)
        if effort:
            advertised.update({"effort_levels_complete", "unsupported_effort_levels", "effort_levels"})
        if "supported" in effort:
            advertised.update({"supports_effort", "effort_levels"})
        if "default_effort" in effort:
            advertised.add("default_effort")
            provider_defaults["effort"] = effort["default_effort"]
        if "supported" in temperature:
            advertised.add("supports_temperature")
        if "default" in temperature:
            advertised.add("default_temperature")
            provider_defaults["temperature"] = temperature["default"]
        label = entry.get("display_name")
    else:
        caps = _capabilities_from_openrouter(entry)
        reasoning = entry.get("reasoning") or {}
        if "supported_parameters" in entry:
            advertised.add("supports_temperature")
        if "supported_parameters" in entry or "reasoning" in entry:
            advertised.add("supports_effort")
        if "supported_efforts" in reasoning:
            advertised.update({"effort_levels", "effort_levels_complete"})
        if reasoning.get("mandatory"):
            advertised.add("unsupported_effort_levels")
        if "default_effort" in reasoning:
            advertised.add("default_effort")
            provider_defaults["effort"] = reasoning["default_effort"]
        defaults = entry.get("default_parameters") or {}
        if "temperature" in defaults:
            advertised.add("default_temperature")
            provider_defaults["temperature"] = defaults["temperature"]
        additional = {k: v for k, v in defaults.items() if k != "temperature"}
        if "default_enabled" in reasoning:
            additional["reasoning.default_enabled"] = reasoning["default_enabled"]
        label = entry.get("name")
    info = _model(model, label, capabilities=caps, advertised=advertised, additional=additional)
    # Preserve defaults independently of support validation. Never fill gaps.
    info.defaults.temperature = _default(
        provider_defaults.get("temperature"),
        "provider" if "temperature" in provider_defaults else "unknown",
    )
    info.defaults.effort = _default(
        provider_defaults.get("effort"),
        "provider" if "effort" in provider_defaults else "unknown",
    )
    return info


def _entries(payload: Any, field: str) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("Invalid model listing")
    entries = payload.get(field)
    if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
        raise ValueError("Invalid model listing")
    return entries


async def discover_models(
    config: ProviderConfig,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    client: httpx.AsyncClient | None = None,
) -> ModelListing:
    """List selectable models/deployments, preserving provider-specific policy.

    All listings come from providers. OpenRouter is public, Azure requires a
    project endpoint, and local discovery is best effort. Failures supply no facts.
    No implicit environment credentials, retries, caches or inference calls.
    An optional client is caller-owned and remains open. Pass client or transport,
    not both. Requests always override client timeouts, redirects and auth.
    """
    if client is not None and transport is not None:
        raise ValueError("Pass either client or transport, not both")
    provider = config.provider
    if provider in {"openai", "anthropic"} and not _key(config):
        return ModelListing(
            source="error",
            error=ProviderError(kind="credentials_missing"),
            note="No credential supplied for this provider.",
        )
    if provider == "azure_foundry":
        if not config.api_base:
            return ModelListing(
                source="error",
                note="Azure deployment discovery requires an inference resource name or base URL.",
                error=ProviderError(kind="unavailable"),
            )
        if not config.azure_project_endpoint:
            return ModelListing(source="error", note=NO_PROJECT_ENDPOINT_NOTE, error=ProviderError(kind="unavailable"))
        if not _key(config):
            return ModelListing(
                source="error",
                note="No credential supplied for this provider.",
                error=ProviderError(kind="credentials_missing"),
            )
    if provider == "local" and not config.api_base:
        return ModelListing(
            source="error",
            note="Local model discovery requires a server base URL.",
            error=ProviderError(kind="unavailable"),
        )
    base = (config.api_base or _BASES.get(provider, "")).rstrip("/")
    url = f"{config.azure_project_endpoint.rstrip('/')}/deployments" if config.azure_project_endpoint else ""
    if provider != "azure_foundry":
        url = _models_url(base) if provider == "anthropic" else f"{base}/models"
    try:
        entries: list[dict[str, Any]] = []
        async with _provider_client(client, transport) as request_client:
            params: dict[str, str | int] = {"api-version": "v1"} if provider == "azure_foundry" else {}
            if provider == "anthropic":
                params = {"limit": 100}
            for _ in range(10 if provider == "anthropic" else 1):
                response = await request_client.get(
                    url, params=params, headers=_headers(config), timeout=10.0, follow_redirects=False, auth=None
                )
                response.raise_for_status()
                # Scrub an echoed credential before interpreting *any* metadata.
                payload = _redact_payload(json.loads(response.text), _key(config))
                entries.extend(_entries(payload, "value" if provider == "azure_foundry" else "data"))
                if provider != "anthropic" or not payload.get("has_more") or not payload.get("last_id"):
                    break
                params["after_id"] = payload["last_id"]
        models = [info for entry in entries if (info := _parse(config, entry)) is not None]
    except (httpx.HTTPError, ValueError, TypeError, KeyError, AttributeError) as exc:
        error = _error(exc)
        message = _error_message(error)
        note = {
            "anthropic": f"Anthropic model discovery failed ({message}). Supply a model ID explicitly for inference.",
            "openai": f"OpenAI model discovery failed ({message}). Supply a model ID explicitly for inference.",
            "local": _LOCAL_NOTE,
            "azure_foundry": f"Could not reach the Azure Foundry project to list deployments: {message}",
            "openrouter": f"OpenRouter model discovery failed ({message}). "
            "Model IDs can still be supplied explicitly for inference.",
        }[provider]
        return ModelListing(source="error", note=note, error=error)
    if not models and provider != "azure_foundry":
        note = (
            _LOCAL_NOTE
            if provider == "local"
            else "The provider returned no models. Model IDs can still be supplied explicitly for inference."
        )
        return ModelListing(source="error", note=note, error=ProviderError(kind="empty"))
    if provider != "anthropic":
        models.sort(key=lambda m: m.id)
    return ModelListing(models=models, source="api")


async def check_connection(
    config: ProviderConfig,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    client: httpx.AsyncClient | None = None,
) -> ConnectionCheck:
    """Explicit zero-token check. Acceptance does not verify quota or billing.

    An optional client is caller-owned and remains open. Pass client or transport,
    not both. Requests always override client timeouts, redirects and auth.
    """
    if client is not None and transport is not None:
        raise ValueError("Pass either client or transport, not both")
    provider = config.provider
    if provider == "azure_foundry" and not config.azure_project_endpoint:
        return ConnectionCheck(
            status="unknown",
            endpoint=config.api_base,
            error=ProviderError(kind="unavailable"),
            detail="A Project endpoint is required to check Azure connectivity without running inference.",
        )
    if provider == "local" and not config.api_base:
        return ConnectionCheck(
            status="unknown",
            endpoint=None,
            error=ProviderError(kind="unavailable"),
            detail="A server base URL is required to check local connectivity.",
        )
    if not _key(config) and provider != "local":
        return ConnectionCheck(
            status="unknown",
            endpoint=None,
            error=ProviderError(kind="credentials_missing"),
            detail="No credential supplied for this provider.",
        )
    if provider in {"azure_foundry", "local"}:
        listing = await discover_models(config, transport=transport, client=client)
        endpoint = config.azure_project_endpoint if provider == "azure_foundry" else config.api_base
        if listing.source == "api":
            detail = (
                f"Reached the Azure resource and listed {len(listing.models)} deployment(s)."
                if provider == "azure_foundry"
                else f"Reached the server and listed {len(listing.models)} model(s)."
            )
            return ConnectionCheck(
                status="ok",
                detail=detail,
                endpoint=endpoint,
                models=listing.models if provider == "azure_foundry" else None,
            )
        return ConnectionCheck(
            status="failed" if provider == "azure_foundry" else "unknown",
            detail=listing.note or "Could not reach the Azure resource.",
            endpoint=endpoint,
            error=listing.error,
        )
    url = _models_url(config.api_base or _BASES[provider])
    try:
        async with _provider_client(client, transport) as request_client:
            response = await request_client.get(
                url, headers=_headers(config), timeout=10.0, follow_redirects=False, auth=None
            )
        if response.is_success:
            return ConnectionCheck(
                status="ok",
                endpoint=url,
                detail="Key accepted and the provider is reachable. This doesn't verify quota or billing.",
            )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        error = _error(exc)
        if error.kind == "credential_rejected":
            detail = f"The provider rejected this API key (HTTP {error.http_status})."
        elif error.kind == "rate_limited":
            detail = "The provider rate-limited this check (HTTP 429). The key authenticated; try again shortly."
        elif error.http_status is not None:
            detail = f"The provider returned HTTP {error.http_status}."
        else:
            detail = f"Could not reach the provider: {_error_message(error)}"
        return ConnectionCheck(
            status="unknown" if error.kind == "rate_limited" else "failed", detail=detail, endpoint=url, error=error
        )
    raise AssertionError("Unreachable HTTP status")
