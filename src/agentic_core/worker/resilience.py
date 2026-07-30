"""Connection resilience utilities for long-running workers."""

from __future__ import annotations

from typing import Any

import structlog
from sqlalchemy.exc import OperationalError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

log = structlog.get_logger(component="connection_resilience")


db_retry = retry(
    retry=retry_if_exception_type(OperationalError),
    stop=stop_after_attempt(3),
    wait=wait_fixed(1),
    before_sleep=lambda retry_state: log.warning(
        "db_retry",
        attempt=retry_state.attempt_number,
        error=str(retry_state.outcome.exception()) if retry_state.outcome else "unknown",
    ),
    reraise=True,
)


async def check_redis_health(ctx: dict[str, Any]) -> None:
    """Periodic Redis ping to detect disconnection early."""
    redis_conn = ctx.get("redis")
    if redis_conn is None:
        return

    try:
        await redis_conn.ping()
    except Exception:
        log.error("redis_health_check_failed")
