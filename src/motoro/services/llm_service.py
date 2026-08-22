"""LLM Bridge service — multi-provider structured output via litellm + Instructor."""

import asyncio
import json as _json
import time
import uuid
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import Any, TypeVar

import instructor
import litellm
import litellm.exceptions
import structlog
from instructor.core import InstructorRetryException
from opentelemetry import trace
from pydantic import BaseModel, Field, ValidationError, model_validator
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_combine,
    wait_exception,
    wait_exponential,
)

from motoro.config import settings
from motoro.observability.metrics import record_llm_call
from motoro.observability.tracing import get_tracer
from motoro.schemas.agent import LLMProvider, ModelConfig
from motoro.schemas.llm import LLMCallRecord, LLMToolCall, ToolCompletion
from motoro.services.credential_scrubber import scrub as _scrub_creds
from motoro.services.llm_errors import (
    LLMBudgetExceededError,
    LLMRateLimitError,
    normalize_llm_error,
)
from motoro.services.model_capabilities import get_capabilities

# ---------------------------------------------------------------------------
# Per-run budget check — M75 / Issue #678
# ---------------------------------------------------------------------------
#
# A ``BudgetCheck`` is an ``async`` callable that the retry layer invokes
# **before** sleeping between attempts.  It should raise
# :class:`LLMBudgetExceededError` (or any other exception) when the per-run
# cost / token budget would be exceeded by another retry attempt; that
# exception is treated as terminal and re-raised immediately, aborting any
# remaining tenacity attempts.
#
# Callers (typically the runtime / phase code) supply the callable via the
# ``budget_check`` keyword argument on ``LLMService.complete*``.  The value
# is stashed in a :class:`ContextVar` so the retry decorator (which has no
# knowledge of the call's parameters) can access it from its
# ``before_sleep`` hook.

BudgetCheck = Callable[[], Awaitable[None]]
_budget_check: ContextVar[BudgetCheck | None] = ContextVar("_budget_check", default=None)

T = TypeVar("T", bound=BaseModel)

log = structlog.get_logger()
_tracer = get_tracer("llm")

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True

# Attribute names on litellm/httpx exceptions that may echo the api_key.
_CREDENTIAL_ATTRS = frozenset({"api_key", "llm_provider", "request", "response", "message"})


def _scrub_exception(exc: BaseException) -> str:
    """Return a scrubbed string representation of *exc* suitable for OTel spans and logs.

    litellm exceptions carry attributes such as ``api_key`` and ``request`` that
    may contain the caller's raw API key.  This function redacts those before the
    value is recorded in a trace span or log record.
    """
    parts = [_scrub_creds(str(exc))]
    for attr in _CREDENTIAL_ATTRS:
        val = getattr(exc, attr, None)
        if val is not None:
            parts.append(f"{attr}={_scrub_creds(str(val))}")
    return " | ".join(parts)


def _normalize_structured_error(exc: Exception) -> Exception:
    """Unwrap Instructor's retry-exhaustion wrapper to the underlying error.

    When Instructor's structured-output repair loop is exhausted it raises
    :class:`InstructorRetryException` — which is **not** a subclass of
    :class:`pydantic.ValidationError`.  Callers that degrade gracefully on
    structured-output failure (the Reason/Plan/self-critique phases) catch
    ``ValidationError``, so an un-normalized ``InstructorRetryException`` slips
    past their handlers and hard-fails the whole run (observed on the Reason
    phase when a model returned XML-tag-wrapped output missing ``strategy``).

    Instructor passes the final underlying exception as the first positional
    argument (see ``instructor/core/retry.py``: ``raise
    InstructorRetryException(e.last_attempt._exception, ...)``).  When that is a
    ``ValidationError`` we return it so the existing handlers fire; otherwise we
    return *exc* unchanged (a genuinely different failure should still surface).
    """
    if isinstance(exc, InstructorRetryException):
        underlying = exc.args[0] if exc.args else None
        if isinstance(underlying, ValidationError):
            return underlying
    return exc


# Context var to capture usage from litellm callbacks
_last_usage: ContextVar[dict[str, Any] | None] = ContextVar("_last_usage", default=None)


def _capture_usage_callback(kwargs: Any, completion_response: Any, **cb_kwargs: Any) -> None:
    """litellm success callback that captures token usage including cache tokens (#649)."""
    from motoro.services.pricing_service import calculate_cost

    usage: dict[str, Any] = {}
    if hasattr(completion_response, "usage") and completion_response.usage:
        u = completion_response.usage
        usage["prompt_tokens"] = getattr(u, "prompt_tokens", 0)
        usage["completion_tokens"] = getattr(u, "completion_tokens", 0)
        # Issue #649: capture prompt-caching token fields from Anthropic / OpenAI
        usage["cache_read_input_tokens"] = getattr(u, "cache_read_input_tokens", 0) or 0
        usage["cache_creation_input_tokens"] = getattr(u, "cache_creation_input_tokens", 0) or 0
        # OpenAI surfaces cached tokens under prompt_tokens_details
        if not usage["cache_read_input_tokens"]:
            details = getattr(u, "prompt_tokens_details", None)
            if details is not None:
                usage["cache_read_input_tokens"] = getattr(details, "cached_tokens", 0) or 0

    model = str(getattr(completion_response, "model", "") or kwargs.get("model", ""))
    cost, source = calculate_cost(
        model=model,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        completion_response=completion_response,
    )
    usage["cost"] = cost if cost is not None else 0.0
    usage["pricing_source"] = source
    _last_usage.set(usage)


# Register the callback — guard against double-registration on module reload.
# Issue #666: the callback list is a mutable global; we register once and never
# append duplicates so concurrent test runs cannot clobber each other.
if _capture_usage_callback not in litellm.success_callback:
    litellm.success_callback = [_capture_usage_callback]


# Providers whose real litellm routing prefix differs from their own enum
# value. azure_foundry rides litellm's Azure AI Foundry inference route
# (`azure_ai/`); local (a self-hosted OpenAI-compatible server, reached via a
# required `api_base` override -- see services.credentials) rides litellm's
# generic `openai/` provider since there's no dedicated "local" litellm
# integration. Every other provider's litellm prefix IS its own enum value
# (this is exactly true for openrouter -- litellm has a native `openrouter/`
# route that takes OpenRouter's own `vendor/model` ids as-is).
_LITELLM_PROVIDER_PREFIX: dict[LLMProvider, str] = {
    LLMProvider.AZURE_FOUNDRY: "azure_ai",
    LLMProvider.LOCAL: "openai",
}


def _build_model_string(config: ModelConfig) -> str:
    """Build litellm model string from config (e.g., 'anthropic/claude-sonnet-4-20250514')."""
    prefix = _LITELLM_PROVIDER_PREFIX.get(config.provider, config.provider.value)
    return f"{prefix}/{config.model}"


def _sampling_kwargs(config: ModelConfig, model_str: str) -> dict[str, Any]:
    """Per-model sampling controls for the litellm call.

    Temperature-based models get ``temperature``. Adaptive-thinking models that reject
    temperature (Opus 4.7/4.8, Fable 5) get ``reasoning_effort`` instead — litellm maps
    it onto Anthropic's ``output_config.effort``. Capabilities are resolved from the
    single source of truth in ``motoro.services.model_capabilities``. If an effort-based
    model has no effort configured we send nothing and let the model use its default.
    """
    caps = get_capabilities(model_str)
    if caps.supports_temperature:
        return {"temperature": config.temperature}
    if caps.supports_effort and config.effort:
        return {"reasoning_effort": config.effort}
    return {}


def model_supports_tool_calling(config: ModelConfig) -> bool:
    """Whether *config*'s model can be sent function schemas.

    Only used to choose a **default** execution pattern for an agent that
    configured none (see ``PatternOrchestrator.from_pattern_config``). An
    unknown model reads as ``False`` so that default stays on the tool-free
    baseline path — the conservative direction, since a run with no pattern
    config does exactly that today. Nothing here downgrades an explicit
    ``reason_act`` config; a user who asked for the loop gets it, and a
    provider that cannot honour it fails loudly.

    litellm's model map is the oracle rather than
    ``motoro.services.model_capabilities``: that registry is hand-maintained
    because litellm mis-reports *sampling* support, but its function-calling
    flag is derived from each provider's own published metadata and covers far
    more models than a hand-kept list could.
    """
    if not config.model:
        return False
    prefix = _LITELLM_PROVIDER_PREFIX.get(config.provider)
    model_str = f"{prefix}/{config.model}" if prefix else config.model
    try:
        return bool(litellm.supports_function_calling(model_str))
    except Exception:  # noqa: BLE001 — an unrecognised model is a "no", not a crash
        return False


def _resolve_connection(config: ModelConfig) -> dict[str, str | None]:
    """Resolve API key, base URL, and model string from the *config only*.

    M112: there is no server-environment credential fallback. Only credentials
    carried explicitly on the ``ModelConfig`` (a per-agent override) are used
    here; user-scoped credentials are resolved in
    :func:`_resolve_connection_for_user`. A call that reaches this path with no
    ``api_key`` will simply have none — the provider call then fails loudly
    rather than silently borrowing a shared server key.
    """
    # azure_foundry/local still need their real litellm routing prefix (see
    # _LITELLM_PROVIDER_PREFIX) even when the key is supplied explicitly on
    # the config.
    prefix = _LITELLM_PROVIDER_PREFIX.get(config.provider)
    model_override = f"{prefix}/{config.model}" if prefix else None
    return {
        "api_key": config.api_key,
        "api_base": config.api_base,
        "model": model_override,
        "api_version": None,
        "aws_region_name": None,
    }


async def _resolve_connection_for_principal(
    config: ModelConfig,
    principal_id: uuid.UUID | None = None,
) -> dict[str, str | None]:
    """Resolve provider connection details, consulting the installed resolver.

    Resolution order:
      1. A credential carried explicitly on the ``ModelConfig``.
      2. The resolver installed via ``services.credentials.set_credential_resolver``.
      3. Nothing — the provider call then fails loudly rather than silently
         borrowing a shared server key.

    The ARES original read a ``user_llm_settings`` row here, decrypting a
    per-user key and normalising Azure Foundry bases. All of that is product
    policy about where secrets live, so it moves behind the hook; core is left
    knowing only that it needs a key and a base.
    """
    default = _resolve_connection(config)
    if config.api_key:
        return default
    from motoro.services.credentials import resolve as _resolve_via_hook

    return await _resolve_via_hook(default, config, principal_id)


_RETRYABLE_EXCEPTIONS = (
    litellm.exceptions.RateLimitError,
    litellm.exceptions.ServiceUnavailableError,
    litellm.exceptions.APIConnectionError,
    litellm.exceptions.BadGatewayError,
    litellm.exceptions.InternalServerError,
    litellm.exceptions.Timeout,
    asyncio.TimeoutError,
)


def _is_retryable(exc: BaseException) -> bool:
    """Return True for exceptions that should trigger tenacity retry.

    Also retries on APIError with a 5xx status code.
    """
    if isinstance(exc, _RETRYABLE_EXCEPTIONS):
        return True
    if isinstance(exc, litellm.exceptions.APIError):
        status = getattr(exc, "status_code", None)
        if isinstance(status, int) and status >= 500:
            return True
    return False


def _wait_for_retry_after(exc: BaseException) -> float:
    """Return the Retry-After seconds from a RateLimitError, else exponential back-off.

    Used with tenacity ``wait_exception`` to honor provider hints (#643).
    """
    if isinstance(exc, litellm.exceptions.RateLimitError):
        for attr in ("response", "litellm_response_headers"):
            headers = getattr(exc, attr, None)
            if headers is None:
                continue
            raw: object | None = None
            if hasattr(headers, "get"):
                raw = headers.get("retry-after") or headers.get("Retry-After")
            if raw is not None:
                try:
                    return min(float(str(raw)), 60.0)
                except (TypeError, ValueError):
                    pass
    # Fall back to exponential: 1s, 2s, 4s …
    return 0.0  # wait_combine will add the exponential component


# Combined wait: honor Retry-After for 429s, exponential for everything else.
_RETRY_WAIT = wait_combine(
    wait_exception(_wait_for_retry_after),
    wait_exponential(multiplier=1, min=1, max=10),
)


def _retry_predicate(exc: BaseException) -> bool:
    """Tenacity ``retry`` predicate that also honors per-run budget aborts.

    Returns ``False`` for :class:`LLMBudgetExceededError` so the retry loop
    re-raises immediately without sleeping or making another attempt.
    """
    if isinstance(exc, LLMBudgetExceededError):
        return False
    return _is_retryable(exc)


async def _before_sleep_budget_check(retry_state: Any) -> None:
    """Tenacity ``before_sleep`` hook — abort if per-run budget is exhausted.

    M75 / Issue #678.  ``tenacity stop_after_attempt(3)`` + instructor's own
    ``max_retries`` could collectively burn ~9 LLM calls per logical request
    on a flaky provider, easily blowing through the agent's hard cost cap.
    By consulting the supplied :class:`BudgetCheck` callback between
    attempts we guarantee the runtime can short-circuit further retries
    once the budget is depleted.

    The callback is expected to raise :class:`LLMBudgetExceededError` when
    the next attempt should be skipped; we re-raise that to abort the
    tenacity loop deterministically.
    """
    check = _budget_check.get()
    if check is None:
        return
    try:
        await check()
    except LLMBudgetExceededError:
        # Re-raise to terminate the retry loop.  Tenacity wraps this in
        # ``RetryError`` only when reraise=False — we use reraise=True
        # everywhere so the original exception propagates.
        raise


# ---------------------------------------------------------------------------
# Prompt caching helpers — Issue #645
# ---------------------------------------------------------------------------


def _inject_cache_control(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    model_str: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Inject Anthropic ``cache_control`` blocks on stable prefixes.

    Anthropic prompt caching requires ``cache_control: {type: "ephemeral"}``
    on the *last* message that should be cached.  We mark:

    1. The system prompt (first message with ``role == "system"``) — only if
       it is the first message and is a plain string content.
    2. The last tool definition in the ``tools`` list — only if tools are
       provided (tool schemas rarely change within a run).

    This is a no-op for non-Anthropic models so callers can always pass through.
    """
    if not model_str.startswith("anthropic/") and not model_str.startswith("azure_ai/"):
        return messages, tools

    patched_messages = list(messages)  # shallow copy — we only mutate entries we control

    # Mark system prompt for caching
    if patched_messages and patched_messages[0].get("role") == "system":
        sys_msg = dict(patched_messages[0])
        content = sys_msg.get("content", "")
        if isinstance(content, str) and content:
            sys_msg["content"] = [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
            patched_messages[0] = sys_msg

    # Mark last tool for caching
    patched_tools: list[dict[str, Any]] | None = None
    if tools:
        patched_tools = list(tools)
        last_tool = dict(patched_tools[-1])
        last_tool["cache_control"] = {"type": "ephemeral"}
        patched_tools[-1] = last_tool

    return patched_messages, patched_tools


class LLMService:
    """Abstraction over LLM providers using litellm + Instructor."""

    def __init__(self, principal_id: uuid.UUID | None = None) -> None:
        self._instructor_client = instructor.from_litellm(litellm.acompletion)
        self._principal_id = principal_id

    @property
    def principal_id(self) -> uuid.UUID | None:
        """The user this service resolves credentials for (the run-starter)."""
        return self._principal_id

    async def complete(
        self,
        config: ModelConfig,
        messages: list[dict[str, str]],
        response_model: type[T],
        max_retries: int | None = None,
        principal_id: uuid.UUID | None = None,
        budget_check: BudgetCheck | None = None,
    ) -> tuple[T, LLMCallRecord]:
        """Call LLM and return a validated Pydantic model + call record.

        Tries the primary ``config`` first, then each entry in
        ``config.fallback_models`` in order on terminal failure (#644).

        ``max_retries`` controls Instructor's internal validation-repair loop.
        Defaults to ``settings.llm_structured_max_retries`` (#651).

        ``budget_check`` is an async callable consulted between tenacity
        attempts; it should raise :class:`LLMBudgetExceededError` once the
        per-run cost cap is reached to abort further retries.  M75 / #678.
        """
        configs_to_try: list[ModelConfig] = [config] + list(config.fallback_models)
        last_exc: Exception = RuntimeError("No configs to try")
        token = _budget_check.set(budget_check)
        try:
            for idx, cfg in enumerate(configs_to_try):
                try:
                    return await self._complete(cfg, messages, response_model, max_retries, principal_id)
                except LLMBudgetExceededError:
                    # Budget exhausted — do not try fallback models.
                    raise
                except Exception as exc:
                    last_exc = exc
                    if idx < len(configs_to_try) - 1:
                        log.warning(
                            "llm.fallback",
                            failed_model=_build_model_string(cfg),
                            next_model=_build_model_string(configs_to_try[idx + 1]),
                            error=_scrub_creds(str(exc)),
                            component="llm",
                        )
            raise last_exc
        finally:
            _budget_check.reset(token)

    @retry(
        retry=retry_if_exception(_retry_predicate),
        stop=stop_after_attempt(3),
        wait=_RETRY_WAIT,
        before_sleep=_before_sleep_budget_check,
        reraise=True,
    )
    async def _complete(
        self,
        config: ModelConfig,
        messages: list[dict[str, str]],
        response_model: type[T],
        max_retries: int | None = None,
        principal_id: uuid.UUID | None = None,
    ) -> tuple[T, LLMCallRecord]:
        """Internal retry-wrapped implementation of ``complete``.

        ``max_retries`` controls Instructor's internal validation-repair loop.
        Defaults to ``settings.llm_structured_max_retries`` (#651).
        """
        # Issue #651: unified default from settings
        if max_retries is None:
            max_retries = settings.llm_structured_max_retries

        effective_principal_id = principal_id or self._principal_id
        conn = await _resolve_connection_for_principal(config, effective_principal_id)
        model_str = conn["model"] or _build_model_string(config)
        provider = config.provider.value
        _last_usage.set(None)
        start_ms = _now_ms()

        # Issue #645: inject cache_control for Anthropic prompt caching
        cached_messages, _ = _inject_cache_control(
            list(messages),
            None,
            model_str,
        )

        with _tracer.start_as_current_span(
            "llm.complete",
            attributes={"gen_ai.request.model": model_str, "gen_ai.system": provider},
        ) as span:
            failure_reason: str | None = None
            try:
                # Issue #635: apply asyncio.wait_for with configured timeout
                result = await asyncio.wait_for(
                    self._instructor_client.chat.completions.create(
                        model=model_str,
                        messages=cached_messages,
                        response_model=response_model,
                        **_sampling_kwargs(config, model_str),
                        max_tokens=config.max_tokens,
                        max_retries=max_retries,
                        api_key=conn["api_key"],
                        base_url=conn["api_base"],
                        **({"api_version": conn["api_version"]} if conn.get("api_version") else {}),
                        **({"aws_region_name": conn["aws_region_name"]} if conn.get("aws_region_name") else {}),
                    ),
                    timeout=settings.llm_call_timeout_seconds,
                )
            except (ValidationError, InstructorRetryException) as exc:
                # Issue #652: persist last invalid output text on validation failure.
                # Instructor raises InstructorRetryException (NOT a ValidationError)
                # once its repair loop is exhausted; normalize to the underlying
                # ValidationError so callers that degrade on ValidationError
                # (reason/plan/self_critique phases) actually catch it instead of
                # the run hard-failing.
                normalized = _normalize_structured_error(exc)
                span.record_exception(normalized)
                span.set_status(trace.StatusCode.ERROR, description=_scrub_creds("Instructor validation failed"))
                failure_reason = str(normalized)[:2000]  # truncate to avoid huge records
                if normalized is exc:
                    raise
                raise normalized from exc
            except Exception as exc:
                span.set_attribute("error.scrubbed", _scrub_exception(exc))
                span.set_status(trace.StatusCode.ERROR, description=_scrub_creds(str(exc)))
                raise

            latency_ms = _now_ms() - start_ms
            usage = _last_usage.get() or {}

            # Issue #637: fall back to _raw_response when callback didn't fire
            if not usage and hasattr(result, "_raw_response"):
                from motoro.services.pricing_service import calculate_cost

                raw = result._raw_response
                if hasattr(raw, "usage") and raw.usage:
                    u = raw.usage
                    usage = {
                        "prompt_tokens": getattr(u, "prompt_tokens", 0),
                        "completion_tokens": getattr(u, "completion_tokens", 0),
                        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
                        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
                    }
                    cost, source = calculate_cost(
                        model=model_str,
                        prompt_tokens=usage["prompt_tokens"],
                        completion_tokens=usage["completion_tokens"],
                        completion_response=raw,
                    )
                    usage["cost"] = cost if cost is not None else 0.0
                    usage["pricing_source"] = source

            record = LLMCallRecord(
                model=model_str,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                latency_ms=latency_ms,
                cost_estimate=usage.get("cost", 0.0),
                pricing_source=usage.get("pricing_source", "litellm"),
                cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
                cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
                failure_reason=failure_reason,
            )

            span.set_attribute("gen_ai.usage.input_tokens", record.prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", record.completion_tokens)
            span.set_attribute("cost_usd", record.cost_estimate)

            record_llm_call(
                model=model_str,
                provider=provider,
                prompt_tokens=record.prompt_tokens,
                completion_tokens=record.completion_tokens,
                cost_usd=record.cost_estimate,
                latency_seconds=latency_ms / 1000,
            )

            log.debug(
                "llm.call.completed",
                model=model_str,
                prompt_tokens=record.prompt_tokens,
                completion_tokens=record.completion_tokens,
                latency_ms=latency_ms,
                cost_usd=record.cost_estimate,
                cache_read_tokens=record.cache_read_input_tokens,
                cache_creation_tokens=record.cache_creation_input_tokens,
                component="llm",
            )

            return result, record

    async def complete_with_tools(
        self,
        config: ModelConfig,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        principal_id: uuid.UUID | None = None,
        budget_check: BudgetCheck | None = None,
        max_retries: int | None = None,
    ) -> ToolCompletion:
        """Call LLM with tool definitions using native tool use.

        Returns a :class:`ToolCompletion` carrying the assistant text, *all*
        requested tool calls, and the call record.  ``messages`` may contain
        assistant turns with ``tool_calls`` and ``role: "tool"`` result turns,
        so callers can drive a multi-turn agentic loop over the same history.

        Tries the primary ``config`` first, then each entry in
        ``config.fallback_models`` in order on terminal failure (#644).

        ``max_retries`` controls the structured-output validation retry
        loop for tool-argument parse / schema errors (M75 / #965).  When
        the LLM produces a tool call whose arguments fail to parse or
        match the declared schema, the conversation is augmented with a
        corrective message and the request is retried up to ``max_retries``
        times before surfacing the failure.  Defaults to
        ``settings.llm_structured_max_retries``.

        ``budget_check`` is an async callable consulted between tenacity
        attempts; see :meth:`complete`.  M75 / #678.
        """
        configs_to_try: list[ModelConfig] = [config] + list(config.fallback_models)
        last_exc: Exception = RuntimeError("No configs to try")
        token = _budget_check.set(budget_check)
        try:
            for idx, cfg in enumerate(configs_to_try):
                try:
                    return await self._complete_with_tools(cfg, messages, tools, principal_id, max_retries)
                except LLMBudgetExceededError:
                    raise
                except Exception as exc:
                    last_exc = exc
                    if idx < len(configs_to_try) - 1:
                        log.warning(
                            "llm.fallback",
                            failed_model=_build_model_string(cfg),
                            next_model=_build_model_string(configs_to_try[idx + 1]),
                            error=_scrub_creds(str(exc)),
                            component="llm",
                        )
            raise last_exc
        finally:
            _budget_check.reset(token)

    async def _complete_with_tools(
        self,
        config: ModelConfig,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        principal_id: uuid.UUID | None = None,
        max_retries: int | None = None,
    ) -> ToolCompletion:
        """Internal implementation of ``complete_with_tools``.

        Wraps a structured-output validation-retry loop (M75 / #965): when
        the LLM returns a tool call whose arguments fail to parse or
        match the declared schema, the conversation is augmented with a
        corrective ``assistant``/``user`` exchange and the request is
        retried up to ``max_retries`` times.  A turn is rejected as a whole
        if *any* of its tool calls fail validation, so the model always
        re-issues a coherent set rather than a partial one.  Transport-level
        errors (rate limit, 5xx, timeout) continue to be handled by the
        tenacity-wrapped ``_complete_with_tools_once``.

        The corrective exchange is plain ``assistant``/``user`` text rather
        than an echoed tool-call turn plus ``role: "tool"`` errors: the
        rejected calls were never executed, so there are no results to
        answer them with, and providers require every ``tool_call_id`` in an
        assistant turn to be resolved.
        """
        if max_retries is None:
            max_retries = settings.llm_structured_max_retries

        attempt_messages: list[dict[str, Any]] = list(messages)
        last: ToolCompletion | None = None
        for attempt in range(max_retries + 1):
            completion, retry_msg = await self._complete_with_tools_once(config, attempt_messages, tools, principal_id)
            last = completion
            # No validation problem — return immediately.
            if retry_msg is None:
                return completion
            # Validation problem — append corrective message and try again
            # unless we have exhausted attempts.
            if attempt >= max_retries:
                log.warning(
                    "llm.tool_args_validation_retries_exhausted",
                    attempts=attempt + 1,
                    failure_reason=completion.record.failure_reason,
                    component="llm",
                )
                return completion
            attempt_messages = list(attempt_messages) + [
                {
                    "role": "assistant",
                    "content": "I attempted to call one or more tools but the arguments were rejected.",
                },
                {
                    "role": "user",
                    "content": (
                        "Tool argument validation failed: "
                        f"{completion.record.failure_reason or 'unknown error'}. "
                        "Please re-issue the tool call(s) with valid arguments "
                        "that match the declared schema."
                    ),
                },
            ]
        # Defensive: loop always returns above.
        assert last is not None
        return last

    @retry(
        retry=retry_if_exception(_retry_predicate),
        stop=stop_after_attempt(3),
        wait=_RETRY_WAIT,
        before_sleep=_before_sleep_budget_check,
        reraise=True,
    )
    async def _complete_with_tools_once(
        self,
        config: ModelConfig,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        principal_id: uuid.UUID | None = None,
    ) -> tuple[ToolCompletion, str | None]:
        """One transport-retry-wrapped attempt of ``complete_with_tools``.

        Returns the completion plus a ``retry_msg`` that is non-``None`` when
        the response is structurally well-formed but one or more tool calls
        failed argument validation.  The outer ``_complete_with_tools`` loop
        uses ``retry_msg`` to decide whether to issue a corrective follow-up
        call.
        """
        effective_principal_id = principal_id or self._principal_id
        conn = await _resolve_connection_for_principal(config, effective_principal_id)
        model_str = conn["model"] or _build_model_string(config)
        provider = config.provider.value
        _last_usage.set(None)
        start_ms = _now_ms()

        # Issue #645: inject cache_control for Anthropic prompt caching
        cached_messages, cached_tools = _inject_cache_control(
            list(messages),
            tools,
            model_str,
        )

        with _tracer.start_as_current_span(
            "llm.complete_with_tools",
            attributes={
                "gen_ai.request.model": model_str,
                "gen_ai.system": provider,
                "tool_count": len(tools),
            },
        ) as span:
            try:
                # Issue #635: apply asyncio.wait_for with configured timeout
                response = await asyncio.wait_for(
                    litellm.acompletion(
                        model=model_str,
                        messages=cached_messages,
                        tools=cached_tools if cached_tools is not None else tools,
                        tool_choice="auto",
                        **_sampling_kwargs(config, model_str),
                        max_tokens=config.max_tokens,
                        api_key=conn["api_key"],
                        base_url=conn["api_base"],
                        **({"api_version": conn["api_version"]} if conn.get("api_version") else {}),
                        **({"aws_region_name": conn["aws_region_name"]} if conn.get("aws_region_name") else {}),
                    ),
                    timeout=settings.llm_call_timeout_seconds,
                )
            except Exception as exc:
                span.set_attribute("error.scrubbed", _scrub_exception(exc))
                span.set_status(trace.StatusCode.ERROR, description=_scrub_creds(str(exc)))
                raise

            latency_ms = _now_ms() - start_ms
            usage = _last_usage.get() or {}
            usage_data = getattr(response, "usage", None)

            prompt_tokens = getattr(usage_data, "prompt_tokens", None)
            if prompt_tokens is None:
                prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = getattr(usage_data, "completion_tokens", None)
            if completion_tokens is None:
                completion_tokens = usage.get("completion_tokens", 0)

            # Issue #637 parity with ``_complete_text``: the litellm success
            # callback that populates ``_last_usage`` does not always fire, in
            # which case ``cost`` is absent and the call would silently record
            # as $0.00.  Fall back to an explicit pricing lookup.
            cost = usage.get("cost")
            pricing_source = usage.get("pricing_source", "litellm")
            if cost is None:
                from motoro.services.pricing_service import calculate_cost

                calculated, pricing_source = calculate_cost(
                    model=model_str,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    completion_response=response,
                )
                cost = calculated if calculated is not None else 0.0

            record = LLMCallRecord(
                model=model_str,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                cost_estimate=cost,
                pricing_source=pricing_source,
                cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
                cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
            )

            span.set_attribute("gen_ai.usage.input_tokens", record.prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", record.completion_tokens)

            record_llm_call(
                model=model_str,
                provider=provider,
                prompt_tokens=record.prompt_tokens,
                completion_tokens=record.completion_tokens,
                cost_usd=record.cost_estimate,
                latency_seconds=latency_ms / 1000,
            )

            # Issue #667: guard against empty choices
            if not response.choices:
                log.warning(
                    "llm.empty_choices",
                    model=model_str,
                    component="llm",
                )
                return ToolCompletion(record=record), None

            message = response.choices[0].message
            # Assistant prose is kept even when tool calls are present — the two
            # arrive in the same completion and a reasoning loop needs both.
            text = message.content or ""
            raw_calls = getattr(message, "tool_calls", None) or []

            parsed: list[LLMToolCall] = []
            failures: list[str] = []
            for call in raw_calls:
                tool_name = call.function.name
                # Issue #636: handle JSON parse failures on tool arguments
                raw_args = call.function.arguments
                if raw_args:
                    try:
                        tool_args: Any = _json.loads(raw_args)
                    except (ValueError, TypeError) as parse_exc:
                        log.warning(
                            "llm.tool_args_parse_error",
                            tool_name=tool_name,
                            error=str(parse_exc),
                            raw_args=raw_args[:200],
                            component="llm",
                        )
                        failures.append(f"{tool_name}: tool_args_parse_error: {parse_exc}")
                        continue
                else:
                    tool_args = {}

                if not isinstance(tool_args, dict):
                    log.warning(
                        "llm.tool_args_parse_error",
                        tool_name=tool_name,
                        error="arguments did not decode to a JSON object",
                        component="llm",
                    )
                    failures.append(
                        f"{tool_name}: tool_args_parse_error: expected a JSON object, got {type(tool_args).__name__}"
                    )
                    continue

                # Issue #836: validate tool_args against the declared schema.
                schema_error = _validate_tool_call_args(tool_name, tool_args, tools)
                if schema_error:
                    log.warning(
                        "llm.tool_args_schema_invalid",
                        tool_name=tool_name,
                        error=schema_error,
                        component="llm",
                    )
                    failures.append(f"{tool_name}: tool_args_schema_invalid: {schema_error}")
                    continue

                parsed.append(
                    LLMToolCall(
                        id=getattr(call, "id", "") or "",
                        name=tool_name,
                        arguments=tool_args,
                    )
                )

            # M75 / #965 — reject the turn as a whole so the model re-issues a
            # coherent set of calls, and surface retry_msg for the outer loop.
            if failures:
                reason = "; ".join(failures)
                record = record.model_copy(update={"failure_reason": reason})
                return ToolCompletion(text=text, record=record), reason

            span.set_attribute("tool_call_count", len(parsed))
            if parsed:
                span.set_attribute("selected_tools", ",".join(c.name for c in parsed))
            return ToolCompletion(text=text, tool_calls=parsed, record=record), None

    async def complete_text(
        self,
        config: ModelConfig,
        messages: list[dict[str, str]],
        principal_id: uuid.UUID | None = None,
        budget_check: BudgetCheck | None = None,
    ) -> tuple[str, LLMCallRecord]:
        """Call LLM and return raw text + call record.

        Tries the primary ``config`` first, then each entry in
        ``config.fallback_models`` in order on terminal failure (#644).

        ``budget_check`` is an async callable consulted between tenacity
        attempts; see :meth:`complete`.  M75 / #678.
        """
        configs_to_try: list[ModelConfig] = [config] + list(config.fallback_models)
        last_exc: Exception = RuntimeError("No configs to try")
        token = _budget_check.set(budget_check)
        try:
            for idx, cfg in enumerate(configs_to_try):
                try:
                    return await self._complete_text(cfg, messages, principal_id)
                except LLMBudgetExceededError:
                    raise
                except Exception as exc:
                    last_exc = exc
                    if idx < len(configs_to_try) - 1:
                        log.warning(
                            "llm.fallback",
                            failed_model=_build_model_string(cfg),
                            next_model=_build_model_string(configs_to_try[idx + 1]),
                            error=_scrub_creds(str(exc)),
                            component="llm",
                        )
            raise last_exc
        finally:
            _budget_check.reset(token)

    @retry(
        retry=retry_if_exception(_retry_predicate),
        stop=stop_after_attempt(3),
        wait=_RETRY_WAIT,
        before_sleep=_before_sleep_budget_check,
        reraise=True,
    )
    async def _complete_text(
        self,
        config: ModelConfig,
        messages: list[dict[str, str]],
        principal_id: uuid.UUID | None = None,
    ) -> tuple[str, LLMCallRecord]:
        """Internal retry-wrapped implementation of ``complete_text``."""
        effective_principal_id = principal_id or self._principal_id
        conn = await _resolve_connection_for_principal(config, effective_principal_id)
        model_str = conn["model"] or _build_model_string(config)
        provider = config.provider.value
        start_ms = _now_ms()

        # Issue #645: inject cache_control for Anthropic prompt caching
        cached_messages, _ = _inject_cache_control(
            list(messages),
            None,
            model_str,
        )

        with _tracer.start_as_current_span(
            "llm.complete_text",
            attributes={"gen_ai.request.model": model_str, "gen_ai.system": provider},
        ) as span:
            try:
                # Issue #635: apply asyncio.wait_for with configured timeout
                response = await asyncio.wait_for(
                    litellm.acompletion(
                        model=model_str,
                        messages=cached_messages,
                        **_sampling_kwargs(config, model_str),
                        max_tokens=config.max_tokens,
                        api_key=conn["api_key"],
                        base_url=conn["api_base"],
                        **({"api_version": conn["api_version"]} if conn.get("api_version") else {}),
                        **({"aws_region_name": conn["aws_region_name"]} if conn.get("aws_region_name") else {}),
                    ),
                    timeout=settings.llm_call_timeout_seconds,
                )
            except Exception as exc:
                span.set_attribute("error.scrubbed", _scrub_exception(exc))
                span.set_status(trace.StatusCode.ERROR, description=_scrub_creds(str(exc)))
                raise

            latency_ms = _now_ms() - start_ms

            from motoro.services.pricing_service import calculate_cost

            # Issue #667: guard against empty choices from misconfigured providers
            if not response.choices:
                log.warning(
                    "llm.empty_choices",
                    model=model_str,
                    component="llm",
                )
                record = LLMCallRecord(
                    model=model_str,
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=latency_ms,
                    cost_estimate=0.0,
                    pricing_source="litellm",
                )
                return "", record

            usage = _last_usage.get() or {}
            text = response.choices[0].message.content or ""
            usage_data = response.usage
            prompt_tok = getattr(usage_data, "prompt_tokens", 0)
            completion_tok = getattr(usage_data, "completion_tokens", 0)
            cost, source = calculate_cost(
                model=model_str,
                prompt_tokens=prompt_tok,
                completion_tokens=completion_tok,
                completion_response=response,
            )
            cost_float = cost if cost is not None else 0.0

            record = LLMCallRecord(
                model=model_str,
                prompt_tokens=prompt_tok,
                completion_tokens=completion_tok,
                latency_ms=latency_ms,
                cost_estimate=cost_float,
                pricing_source=source,
                cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
                cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
            )

            span.set_attribute("gen_ai.usage.input_tokens", record.prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", record.completion_tokens)

            record_llm_call(
                model=model_str,
                provider=provider,
                prompt_tokens=record.prompt_tokens,
                completion_tokens=record.completion_tokens,
                cost_usd=cost_float,
                latency_seconds=latency_ms / 1000,
            )

            log.debug(
                "llm.call.completed",
                model=model_str,
                prompt_tokens=record.prompt_tokens,
                completion_tokens=record.completion_tokens,
                latency_ms=latency_ms,
                component="llm",
            )

            return text, record

    async def select_tool(
        self,
        config: ModelConfig,
        messages: list[dict[str, str]],
        tool_names: list[str],
        max_retries: int | None = None,
        principal_id: uuid.UUID | None = None,
    ) -> tuple[bool, str | None, dict[str, Any], LLMCallRecord]:
        """Ask the LLM to select a tool from an exact constrained list.

        ``max_retries`` defaults to ``settings.llm_structured_max_retries`` (#651).
        """
        valid_set = frozenset(tool_names)
        names_str = ", ".join(f'"{n}"' for n in sorted(tool_names))

        class ToolDecision(BaseModel):
            reasoning: str = Field(description="Why this tool is or is not needed")
            use_tool: bool = Field(description="True if an MCP tool must be called for this step")
            tool_name: str | None = Field(
                default=None,
                description=f"If use_tool is true, MUST be one of: {names_str}",
            )
            tool_args: dict[str, Any] = Field(
                default_factory=dict,
                description="Arguments to pass to the tool (required when use_tool is true)",
            )

            @model_validator(mode="after")
            def check_tool_name(self) -> "ToolDecision":
                if self.use_tool:
                    if not self.tool_name:
                        raise ValueError(f"tool_name is required when use_tool=true. Must be one of: {names_str}")
                    if self.tool_name not in valid_set:
                        raise ValueError(
                            f"'{self.tool_name}' is not a valid tool name. Must be exactly one of: {names_str}"
                        )
                return self

        decision, record = await self.complete(
            config=config,
            messages=messages,
            response_model=ToolDecision,
            max_retries=max_retries,
        )
        return decision.use_tool, decision.tool_name, decision.tool_args, record


def _validate_tool_call_args(
    tool_name: str,
    args: dict[str, Any],
    tools: list[dict[str, Any]],
) -> str | None:
    """Validate *args* against the declared JSON Schema for *tool_name*.

    Looks up the ``function.parameters`` schema in *tools* (OpenAI format)
    and delegates to :func:`motoro.mcp.adapters._normalize_schema_for_validation`
    + ``Draft7Validator`` for full Draft-7 validation including
    ``additionalProperties: false`` injection.

    Returns ``None`` when valid (or when no schema can be found), or an error
    string when validation fails.  Issue #836.
    """
    schema: dict[str, Any] | None = None
    for tool_def in tools:
        fn = tool_def.get("function", {})
        if fn.get("name") == tool_name:
            raw = fn.get("parameters")
            if isinstance(raw, dict):
                schema = raw
            break

    if not schema:
        return None

    from jsonschema import Draft7Validator
    from jsonschema import ValidationError as _ValidationError

    from motoro.mcp.adapters import _normalize_schema_for_validation

    effective_schema = _normalize_schema_for_validation(schema)
    try:
        Draft7Validator(effective_schema).validate(args)
    except _ValidationError as exc:
        field_path = ".".join(str(p) for p in exc.absolute_path) if exc.absolute_path else "(root)"
        if exc.validator == "additionalProperties":
            return f"Unexpected field(s) in tool arguments: {exc.message}"
        if exc.validator == "required":
            return f"Missing required field: {exc.message}"
        if exc.validator == "type":
            return f"Invalid type for '{field_path}': {exc.message}"
        return f"Validation error at '{field_path}': {exc.message}"

    return None


# Re-export for callers that need the normalized error types without a separate import
__all__ = [
    "LLMService",
    "normalize_llm_error",
    "LLMRateLimitError",
    "LLMBudgetExceededError",
    "BudgetCheck",
]


def _now_ms() -> int:
    """Current time in milliseconds."""
    return int(time.monotonic() * 1000)
