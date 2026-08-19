"""Redis-backed working memory manager for intra-run state.

Supports two eviction strategies:
- sliding_window: retain the last N entries; oldest are dropped when full.
- summarization: when token budget is exceeded, an LLM compresses older
  entries into a single summary entry, freeing budget for new entries.

Concurrency model (issue #739)
------------------------------
Both eviction strategies are read-modify-write against the same Redis
hash.  Two concurrent ``add_entry`` calls for the same ``run_id`` could
both observe the overflow condition before either deleted any entries,
which in the summarization path meant *two* expensive LLM calls and
*two* summary rows added to the same window.  ``_apply_strategy`` is
now guarded by a Redis SET-NX lock per ``run_id``; the strategy block
becomes a no-op for callers that lose the race.  The lock is released
in a ``finally`` so a crash inside the strategy still frees it (with
a short TTL as a backstop).
"""

from __future__ import annotations

import asyncio
import enum
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    import redis.asyncio as aioredis

    from motoro.schemas.agent import ModelConfig
    from motoro.services.llm_service import LLMService

logger = logging.getLogger(__name__)

# Redis-lock parameters (issue #739).  The TTL is the backstop in case
# the holder crashes before releasing; in normal operation
# ``_apply_strategy`` releases the lock in its ``finally`` block.
_APPLY_STRATEGY_LOCK_TTL_SECONDS = 30
# Brief retry budget if the lock is held when we try to acquire — the
# common case is "another writer just compressed; nothing for us to do."
# We sleep briefly and retry once, then give up (the entry has already
# been written; the strategy is best-effort).
_APPLY_STRATEGY_LOCK_RETRIES = 2
_APPLY_STRATEGY_LOCK_RETRY_SLEEP = 0.05


class WorkingMemoryEntryType(enum.StrEnum):
    """Category of a working memory entry."""

    CONVERSATION_TURN = "conversation_turn"
    TOOL_RESULT = "tool_result"
    INTERMEDIATE_REASONING = "intermediate_reasoning"
    LOOP_VARIABLE = "loop_variable"
    SUMMARY = "summary"


class WorkingMemoryEntry(BaseModel):
    """A single entry in working memory."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: WorkingMemoryEntryType
    content: str
    token_estimate: int = 0
    timestamp: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkingMemoryStrategy(enum.StrEnum):
    """Context window management strategy."""

    SLIDING_WINDOW = "sliding_window"
    SUMMARIZATION = "summarization"


class WorkingMemoryConfig(BaseModel):
    """Per-agent working memory configuration."""

    strategy: WorkingMemoryStrategy = WorkingMemoryStrategy.SLIDING_WINDOW
    window_size: int = Field(default=20, ge=1, description="Max entries for sliding window")
    token_budget: int = Field(default=4000, ge=1, description="Token budget for summarization")


# Issue #727: load tiktoken encoder once at module level so it is available
# before any asyncio event loop starts (avoids file-I/O interleaving in tests).
_TIKTOKEN_ENCODER: Any = None
try:
    import tiktoken as _tiktoken

    _TIKTOKEN_ENCODER = _tiktoken.get_encoding("cl100k_base")
except Exception:  # noqa: BLE001
    pass


def _estimate_tokens(text: str) -> int:
    """Issue #727: token count via tiktoken when available, else char/4 heuristic."""
    if _TIKTOKEN_ENCODER is not None:
        try:
            return max(1, len(_TIKTOKEN_ENCODER.encode(text)))
        except Exception:  # noqa: BLE001
            pass
    return max(1, len(text) // 4)


class WorkingMemoryManager:
    """Per-run working memory backed by a Redis hash.

    Redis key: ``working_memory:{run_id}``
    Each field in the hash is an entry UUID mapped to a JSON-serialized
    WorkingMemoryEntry.  TTL is set on the key to auto-expire after the run.
    """

    def __init__(
        self,
        run_id: uuid.UUID,
        config: WorkingMemoryConfig,
        redis_client: aioredis.Redis,
        llm_service: LLMService | None = None,
        model_config: ModelConfig | None = None,
        ttl_seconds: int = 86400,
    ) -> None:
        self._run_id = run_id
        self._config = config
        self._redis = redis_client
        self._llm = llm_service
        self._model_config = model_config
        self._ttl = ttl_seconds
        self._key = f"working_memory:{run_id}"
        # Issue #739: per-run lock key, distinct from the data hash key so
        # SET-NX on the lock cannot collide with HSET on the data.
        self._lock_key = f"working_memory_lock:{run_id}"
        # Lock token used to verify ownership at release time so we never
        # delete a lock the current process didn't acquire.
        self._lock_token = uuid.uuid4().hex

    async def add_entry(
        self,
        content: str,
        entry_type: WorkingMemoryEntryType = WorkingMemoryEntryType.CONVERSATION_TURN,
        metadata: dict[str, Any] | None = None,
    ) -> WorkingMemoryEntry:
        """Add an entry to working memory and apply the eviction strategy.

        Args:
            content: The text content to store.
            entry_type: Category of this entry.
            metadata: Optional key/value metadata.

        Returns:
            The newly created WorkingMemoryEntry.
        """
        entry = WorkingMemoryEntry(
            type=entry_type,
            content=content,
            token_estimate=_estimate_tokens(content),
            metadata=metadata or {},
        )
        await self._redis.hset(self._key, entry.id, entry.model_dump_json())  # type: ignore[misc]
        await self._redis.expire(self._key, self._ttl)
        await self._apply_strategy()
        return entry

    async def get_entries(self) -> list[WorkingMemoryEntry]:
        """Return all entries sorted by timestamp (oldest first)."""
        raw: dict[str, str] = await self._redis.hgetall(self._key)  # type: ignore[misc]
        entries = [WorkingMemoryEntry.model_validate_json(v) for v in raw.values()]
        entries.sort(key=lambda e: e.timestamp)
        return entries

    async def count_tokens(self) -> int:
        """Approximate total tokens currently in working memory."""
        entries = await self.get_entries()
        return sum(e.token_estimate for e in entries)

    async def get_context_string(self) -> str:
        """Format working memory as a readable string for LLM prompts."""
        entries = await self.get_entries()
        if not entries:
            return ""
        return "\n".join(f"[{e.type}] {e.content}" for e in entries)

    async def cleanup(self) -> None:
        """Delete the working memory key from Redis."""
        await self._redis.delete(self._key)

    async def _acquire_lock(self) -> bool:
        """Try to acquire the per-run strategy lock.

        Implements a short retry loop so callers that lose the initial
        race wait briefly (covering the typical case where the holder is
        in the middle of a fast sliding-window trim) and then give up —
        the entry has already been written, so dropping the strategy
        pass for this writer is safe.

        Returns:
            True if the lock is now held by this caller, False otherwise.
        """
        for attempt in range(_APPLY_STRATEGY_LOCK_RETRIES + 1):
            acquired = await self._redis.set(
                self._lock_key,
                self._lock_token,
                nx=True,
                ex=_APPLY_STRATEGY_LOCK_TTL_SECONDS,
            )
            if acquired:
                return True
            if attempt < _APPLY_STRATEGY_LOCK_RETRIES:
                await asyncio.sleep(_APPLY_STRATEGY_LOCK_RETRY_SLEEP)
        return False

    async def _release_lock(self) -> None:
        """Release the lock — but only if we still hold it.

        Uses a tiny CAS so we never delete a lock another process took
        over (e.g. after a TTL expiry on a slow strategy run).  Any
        Redis error here is swallowed — the lock will time out on its
        own.
        """
        try:
            current = await self._redis.get(self._lock_key)
            # Redis returns bytes by default; decode for compare.
            if isinstance(current, bytes):
                current = current.decode("utf-8", errors="replace")
            if current == self._lock_token:
                await self._redis.delete(self._lock_key)
        except Exception:  # noqa: BLE001 — best-effort release; lock TTL is the backstop
            logger.debug(
                "Failed to release working-memory lock for run %s; relying on TTL",
                self._run_id,
                exc_info=True,
            )

    async def _apply_strategy(self) -> None:
        """Apply the configured eviction strategy after adding a new entry.

        Guarded by a Redis SET-NX lock (issue #739) so two concurrent
        ``add_entry`` calls never both observe the overflow condition
        before either acts.  Writers that lose the race simply skip
        their strategy pass — the entry they wrote remains in place and
        the lock-holder's pass already evicted on their behalf.
        """
        acquired = await self._acquire_lock()
        if not acquired:
            # Another writer is (or just was) compressing/trimming the
            # window — leave it to them.  Our entry is already in Redis;
            # the next call will re-evaluate.
            return

        try:
            entries = await self.get_entries()

            if self._config.strategy == WorkingMemoryStrategy.SLIDING_WINDOW:
                if len(entries) > self._config.window_size:
                    overflow = entries[: len(entries) - self._config.window_size]
                    for old in overflow:
                        await self._redis.hdel(self._key, old.id)  # type: ignore[misc]

            elif self._config.strategy == WorkingMemoryStrategy.SUMMARIZATION:
                total = sum(e.token_estimate for e in entries)
                if total > self._config.token_budget and self._llm and self._model_config:
                    keep = max(1, len(entries) // 2)
                    to_compress = entries[:-keep]
                    if to_compress:
                        summary_text = await self._summarize(to_compress)
                        for old in to_compress:
                            await self._redis.hdel(self._key, old.id)  # type: ignore[misc]
                        # Issue #752: persist provenance on the summary so
                        # it is auditable — operators can answer "what got
                        # compressed?" without re-running the LLM.  We
                        # store both the count (cheap aggregate) and the
                        # source entry IDs (full trace).
                        summary = WorkingMemoryEntry(
                            type=WorkingMemoryEntryType.SUMMARY,
                            content=summary_text,
                            token_estimate=_estimate_tokens(summary_text),
                            metadata={
                                "original_count": len(to_compress),
                                "source_entry_ids": [e.id for e in to_compress],
                            },
                        )
                        await self._redis.hset(  # type: ignore[misc]
                            self._key, summary.id, summary.model_dump_json()
                        )
        finally:
            await self._release_lock()

    async def _summarize(self, entries: list[WorkingMemoryEntry]) -> str:
        """Summarize a list of entries into a compact string via LLM."""
        assert self._llm is not None
        assert self._model_config is not None
        content = "\n".join(f"[{e.type}] {e.content}" for e in entries)
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "Summarize the following working memory entries concisely, "
                    "preserving key facts, decisions, tool results, and outcomes."
                ),
            },
            {"role": "user", "content": content},
        ]
        text, _ = await self._llm.complete_text(self._model_config, messages)
        return text
