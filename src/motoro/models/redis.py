"""Redis client factory — async singleton, mirrors models/database.py."""

from __future__ import annotations

import redis.asyncio as aioredis

from motoro.config import settings

_redis: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    """Get or create the async Redis client (lazy singleton).

    decode_responses=True means all values are returned as str, not bytes.
    """
    global _redis  # noqa: PLW0603
    if _redis is None:
        _redis = aioredis.from_url(  # type: ignore[no-untyped-call]
            settings.redis_url,
            decode_responses=True,
        )
    return _redis
