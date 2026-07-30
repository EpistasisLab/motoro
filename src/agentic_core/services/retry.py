"""Retry utility with exponential backoff for transient failures.

Used by every messaging surface (webhook delivery, A2A clients, future
inter-service callers) so backoff policy stays uniform.

#862: webhook_service previously used naive ``BACKOFF_BASE ** (attempt + 1)``
with no jitter, which aligns retry storms across clients. The shared
``exponential_backoff_delay`` helper here applies equal-jitter so retries
desynchronize naturally even when many clients fail at the same instant.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable

import structlog

log = structlog.get_logger()


def exponential_backoff_delay(
    attempt: int,
    *,
    base: float = 1.0,
    cap: float = 60.0,
) -> float:
    """Return seconds to sleep before retry ``attempt`` (0-indexed).

    Equal-jitter exponential backoff: half deterministic growth, half
    random within the band. Result is in ``[base/2, cap]``.

    Args:
        attempt: 0-indexed retry attempt. ``0`` corresponds to the first
            retry (after the initial try failed).
        base: Initial backoff in seconds for ``attempt == 0`` pre-jitter.
        cap: Hard cap in seconds.
    """
    if attempt < 0:
        raise ValueError(f"attempt must be >= 0, got {attempt}")
    exp = base * (2**attempt)
    half = exp / 2.0
    delay = half + random.uniform(0, half)
    return float(min(delay, cap))


async def retry_with_backoff[T](
    func: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: tuple[type[BaseException], ...] = (
        ConnectionError,
        OSError,
        TimeoutError,
    ),
) -> T:
    """Execute *func* with exponential backoff on transient failures.

    Args:
        func: An async callable that takes no arguments and returns a result.
        max_retries: Maximum number of retry attempts after the first failure.
        base_delay: Initial delay in seconds before the first retry.
        max_delay: Maximum delay cap in seconds.
        retryable_exceptions: Exception types that trigger a retry.

    Returns:
        The return value of *func* on success.

    Raises:
        The last exception if all retries are exhausted.
    """
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except retryable_exceptions as exc:
            if attempt == max_retries:
                raise
            delay = exponential_backoff_delay(attempt, base=base_delay, cap=max_delay)
            log.warning(
                "retry_attempt",
                attempt=attempt + 1,
                max_retries=max_retries,
                delay=round(delay, 2),
                error=str(exc),
            )
            await asyncio.sleep(delay)

    # Unreachable, but satisfies type checkers
    raise RuntimeError("retry_with_backoff loop exited unexpectedly")  # pragma: no cover
