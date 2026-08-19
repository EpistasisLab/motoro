"""Normalized LLM error types.

All litellm provider exception variants are mapped to one of these five
core error classes so callers can handle them without importing litellm
directly.

Usage::

    from motoro.services.llm_errors import (
        LLMRateLimitError,
        LLMContextWindowError,
        LLMAuthError,
        LLMTimeoutError,
        LLMServerError,
        normalize_llm_error,
    )
"""

from __future__ import annotations

import asyncio
import contextlib


class LLMError(Exception):
    """Base class for all normalized LLM errors."""

    def __init__(self, message: str, *, original: Exception | None = None) -> None:
        super().__init__(message)
        self.original = original


class LLMRateLimitError(LLMError):
    """Provider rate-limit / quota exceeded (HTTP 429).

    Callers should back off and retry after the ``retry_after`` hint.
    """

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        original: Exception | None = None,
    ) -> None:
        super().__init__(message, original=original)
        #: Seconds to wait before retrying, if the provider included a hint.
        self.retry_after: float | None = retry_after


class LLMContextWindowError(LLMError):
    """Input exceeded the model's context window.

    Callers should truncate the conversation history and retry.
    """


class LLMAuthError(LLMError):
    """Authentication / authorisation failure (HTTP 401 / 403).

    The API key is invalid, expired, or lacks permissions.
    """


class LLMTimeoutError(LLMError):
    """Provider call timed out (either network or a local asyncio deadline)."""


class LLMServerError(LLMError):
    """Transient provider-side error (5xx, connection refused, etc.).

    Suitable for retry with back-off.
    """


class LLMProviderNotConfiguredError(LLMError):
    """No LLM provider could be resolved for the call.

    Raised when the registered credential resolver yields no usable provider and
    the :class:`ModelConfig` carried no credential of its own. Core does not
    decide *where* credentials live — a product installs a resolver, and whether
    that reads a per-user record, an environment variable, or a secret manager is
    the product's policy. A product serving HTTP typically maps this to 400,
    since it means the caller has not finished configuring a provider.
    """

    def __init__(
        self,
        message: str | None = None,
        *,
        purpose: str | None = None,
        original: Exception | None = None,
    ) -> None:
        #: Human-readable description of what needed the provider (e.g. "agent run",
        #: "assistant chat"), used to make the error message actionable.
        self.purpose = purpose
        if message is None:
            suffix = f" It is required for: {purpose}." if purpose else ""
            message = f"No LLM provider is configured. Add a provider and API key in Settings to continue.{suffix}"
        super().__init__(message, original=original)


class LLMBudgetExceededError(LLMError):
    """Per-run cost / token budget would be exceeded by another retry attempt.

    Raised by a caller-supplied ``budget_check`` callback that the retry layer
    invokes before sleeping between attempts.  Issue #678 / M75: tenacity
    ``stop_after_attempt(3)`` combined with ``instructor max_retries`` could
    burst up to ~9 LLM calls per logical request on a flaky provider, with no
    chance for the runtime to abort once the run's hard budget had been hit.

    The retry layer treats this exception as **terminal** — it is re-raised
    immediately without further attempts.
    """


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------


def normalize_llm_error(exc: Exception) -> LLMError:
    """Convert a litellm exception into an ``LLMError`` subclass.

    If ``exc`` is already an ``LLMError`` it is returned unchanged.  For
    unknown exception types a generic ``LLMServerError`` is returned so
    callers always receive a known type.
    """
    if isinstance(exc, LLMError):
        return exc

    # asyncio.TimeoutError — fired by asyncio.wait_for
    if isinstance(exc, asyncio.TimeoutError):
        return LLMTimeoutError(str(exc), original=exc)

    try:
        import litellm.exceptions as _le
    except ImportError:  # pragma: no cover
        return LLMServerError(str(exc), original=exc)

    # --- Rate limit / 429 ---
    if isinstance(exc, _le.RateLimitError):
        retry_after: float | None = None
        # Extract Retry-After from the response headers when available.
        # litellm stores the raw response on some exception variants.
        for attr in ("response", "litellm_response_headers"):
            headers = getattr(exc, attr, None)
            if headers is None:
                continue
            # Could be an httpx.Headers or a plain dict
            raw: object | None = None
            if hasattr(headers, "get"):
                raw = headers.get("retry-after") or headers.get("Retry-After")
            if raw is not None:
                with contextlib.suppress(TypeError, ValueError):
                    retry_after = float(str(raw))
            if retry_after is not None:
                break
        return LLMRateLimitError(str(exc), retry_after=retry_after, original=exc)

    # --- Context window ---
    if isinstance(exc, _le.ContextWindowExceededError):
        return LLMContextWindowError(str(exc), original=exc)

    # --- Auth / permission ---
    if isinstance(exc, (_le.AuthenticationError, _le.PermissionDeniedError)):
        return LLMAuthError(str(exc), original=exc)

    # --- Timeout ---
    if isinstance(exc, _le.Timeout):
        return LLMTimeoutError(str(exc), original=exc)

    # --- 5xx / connection / bad gateway ---
    if isinstance(
        exc,
        (
            _le.ServiceUnavailableError,
            _le.APIConnectionError,
            _le.BadGatewayError,
            _le.InternalServerError,
        ),
    ):
        return LLMServerError(str(exc), original=exc)

    # APIError covers remaining HTTP errors; map 5xx to server error.
    if isinstance(exc, _le.APIError):
        status = getattr(exc, "status_code", None)
        if status is not None and isinstance(status, int) and status >= 500:
            return LLMServerError(str(exc), original=exc)

    # Everything else (BadRequestError, UnprocessableEntityError, etc.) is a
    # non-retryable server error from the perspective of core callers.
    return LLMServerError(str(exc), original=exc)
