"""Request-scoped sampling passthrough and Bedrock protocol verification.

Unlike its OpenAI-compatible routes, this handler serializes extra_body as a
nested field. Overlay sampling at the HTTP boundary so its model-map gates and
effort conversions cannot alter it. No global hooks or model-map mutations.
"""

import json
from typing import Any

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

from motoro.services.sampling import SamplingConfigurationError


class _AnthropicSamplingHTTPHandler(AsyncHTTPHandler):
    def __init__(self, delegate: AsyncHTTPHandler, sampling: dict[str, Any]) -> None:
        # The delegate owns the connection pool and its lifecycle.
        self._delegate = delegate
        self._sampling = sampling

    def __del__(self) -> None:
        # Do not run AsyncHTTPHandler's destructor on a borrowed connection pool.
        pass

    async def close(self) -> None:
        pass

    async def post(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("json") is not None:
            body = dict(kwargs["json"])
            body.update(self._sampling)
            kwargs["json"] = body
        else:
            body = json.loads(kwargs["data"])
            body.update(self._sampling)
            kwargs["data"] = json.dumps(body)
        return await self._delegate.post(*args, **kwargs)


class _BedrockSamplingHTTPHandler(_AnthropicSamplingHTTPHandler):
    """Verify the signed payload without changing it or invalidating its signature.

    Bedrock's different model protocols need the SDK adapter. If it cannot
    represent the requested effort unchanged, fail visibly before transmission.
    Never let its token-budget conversion or effort normalization pass silently.
    """

    async def post(self, *args: Any, **kwargs: Any) -> Any:
        body = kwargs.get("json")
        if body is None:
            body = json.loads(kwargs["data"])
        additional = body.get("additionalModelRequestFields") or {}
        temperature = body.get("temperature", (body.get("inferenceConfig") or {}).get("temperature"))
        efforts = [
            body.get("reasoning_effort"),
            (body.get("output_config") or {}).get("effort"),
            additional.get("reasoning_effort"),
            (additional.get("output_config") or {}).get("effort"),
            (additional.get("reasoningConfig") or {}).get("maxReasoningEffort"),
        ]
        efforts = [value for value in efforts if value is not None]
        budget = (body.get("thinking") or additional.get("thinking") or {}).get("budget_tokens")
        if (
            "temperature" in self._sampling
            and temperature != self._sampling["temperature"]
            or "reasoning_effort" in self._sampling
            and (
                not efforts
                or any(value != self._sampling["reasoning_effort"] for value in efforts)
                or budget is not None
            )
        ):
            raise SamplingConfigurationError(
                "The Bedrock SDK adapter cannot preserve the requested sampling values; request was not sent."
            )
        return await self._delegate.post(*args, **kwargs)


async def acompletion(**kwargs: Any) -> Any:
    """Preserve sampling across native and structured completions."""
    if str(kwargs.get("model", "")).startswith("anthropic/") and kwargs.get("extra_body"):
        sampling = kwargs.pop("extra_body")
        delegate = kwargs.pop("client", None) or litellm.module_level_aclient
        kwargs["client"] = _AnthropicSamplingHTTPHandler(delegate, sampling)
    elif str(kwargs.get("model", "")).startswith("bedrock/"):
        sampling = {key: kwargs[key] for key in ("temperature", "reasoning_effort") if key in kwargs}
        if sampling:
            delegate = kwargs.pop("client", None) or litellm.module_level_aclient
            kwargs["client"] = _BedrockSamplingHTTPHandler(delegate, sampling)
    try:
        return await litellm.acompletion(**kwargs)
    except Exception as exc:
        # LiteLLM can wrap an HTTP-adapter exception. Keep configuration failures
        # terminal even when that wrapper would otherwise look retryable.
        current: BaseException | None = exc
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, SamplingConfigurationError):
                raise current from exc if current is not exc else None
            current = current.__cause__ or current.__context__
        raise
