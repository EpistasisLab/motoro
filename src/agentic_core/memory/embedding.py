"""Embedding generation — shared utility for episodic and semantic memory.

Two backends:

- **Local** (default): a ``sentence-transformers`` model
  (e.g. ``BAAI/bge-base-en-v1.5``), selected by a ``sentence-transformers/``
  prefix on the model name. No API key needed — the model runs in-process.
- **Remote**: any litellm-supported embedding provider, when the model name does
  not carry that prefix.

Remote credential resolution reads settings directly rather than a per-user
table. ARES resolves a remote embedding key from the *acting user's* same-vendor
LLM setting (``UserLLMSetting``, decrypted per call) — the same per-user
credential coupling already excluded from core's LLM bridge (see
``services.credentials``), for the same reason: core has no users table to
join against. An explicit ``api_key`` argument still takes precedence, so a
product with its own per-user credential store can pass one in.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import OrderedDict
from typing import Any

from agentic_core.config import settings
from agentic_core.observability.tracing import get_tracer

_tracer = get_tracer("embedding")

logger = logging.getLogger(__name__)

# Sentinel distinguishing "key not resolved yet" from "resolved to no key".
_UNSET: Any = object()


def _embedding_provider(model: str) -> str | None:
    """Map a remote embedding model name to the settings field that supplies its key.

    Only OpenAI-family embedding models (``text-embedding-*``) are mapped today;
    other remote vendors would need their own settings field and are treated as
    unresolvable (no key) rather than silently borrowing an unrelated one.
    """
    if model.startswith("text-embedding") or model.startswith("openai/"):
        return "openai"
    return None


# Lazy-loaded sentence-transformers model (shared across instances using the
# same model name so we only pay the load cost once).
_local_models: dict[str, Any] = {}

# LRU cap for the embedding cache.
_CACHE_MAX_SIZE = 2048

# Remote embedding call timeout (seconds).
_REMOTE_EMBED_TIMEOUT = 60


def _get_local_model(model_name: str) -> Any:
    """Return (and cache) a SentenceTransformer instance.

    Always cache by the bare HuggingFace model name (without the
    ``sentence-transformers/`` prefix) so ``sentence-transformers/X`` and ``X``
    resolve to the same loaded model and are never loaded twice.
    """
    bare_name = model_name.removeprefix("sentence-transformers/")
    if bare_name not in _local_models:
        from sentence_transformers import SentenceTransformer

        _local_models[bare_name] = SentenceTransformer(bare_name)
    return _local_models[bare_name]


class EmbeddingService:
    """Generates text embeddings, with in-process LRU caching.

    The cache is keyed by (model, tenant, agent, text-hash) so identical texts
    with the same model are only embedded once per process lifetime, and
    embeddings from different tenants/agents never collide inside a shared
    process. Bounded by ``_CACHE_MAX_SIZE`` entries (LRU eviction).
    """

    def __init__(
        self,
        model: str | None = None,
        dimensions: int | None = None,
        api_key: str | None = None,
        tenant_id: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        self._model = model or settings.embedding_model
        self._dimensions = dimensions or settings.embedding_dimensions
        self._api_key = api_key
        self._resolved_key: Any = _UNSET
        self._tenant_id = tenant_id or ""
        self._agent_id = agent_id or ""
        self._is_local = self._model.startswith("sentence-transformers/")
        self._cache: OrderedDict[str, list[float]] = OrderedDict()

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        with _tracer.start_as_current_span(
            "embedding.embed",
            attributes={"embedding.model": self._model, "embedding.text_length": len(text)},
        ) as span:
            cache_key = self._cache_key(text)
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)
                span.set_attribute("embedding.cache_hit", True)
                return self._cache[cache_key]

            span.set_attribute("embedding.cache_hit", False)
            vector = self._embed_local(text) if self._is_local else await self._embed_remote(text)
            self._put_cache(cache_key, vector)
            return vector

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts, using cache where possible.

        Uncached texts are batched into remote calls, chunked to at most
        ``settings.embedding_batch_max`` per call. Identical texts are
        deduplicated so each unique string is embedded once per call, then the
        result is fanned back out to every position that needs it.
        """
        with _tracer.start_as_current_span(
            "embedding.embed_batch",
            attributes={"embedding.model": self._model, "embedding.batch_size": len(texts)},
        ) as span:
            if not texts:
                return []

            results: list[list[float] | None] = [None] * len(texts)
            key_to_positions: dict[str, list[int]] = {}
            for i, text in enumerate(texts):
                cache_key = self._cache_key(text)
                if cache_key in self._cache:
                    self._cache.move_to_end(cache_key)
                    results[i] = self._cache[cache_key]
                else:
                    key_to_positions.setdefault(cache_key, []).append(i)

            uncached_keys = list(key_to_positions.keys())
            uncached_texts = [texts[key_to_positions[k][0]] for k in uncached_keys]

            span.set_attribute("embedding.cache_hits", len(texts) - len(uncached_texts))
            span.set_attribute("embedding.cache_misses", len(uncached_texts))

            if uncached_texts:
                vectors = (
                    self._embed_local_batch(uncached_texts)
                    if self._is_local
                    else await self._embed_remote_batch(uncached_texts)
                )
                for j, key in enumerate(uncached_keys):
                    self._put_cache(key, vectors[j])
                    for idx in key_to_positions[key]:
                        results[idx] = vectors[j]

            return [v for v in results if v is not None]

    # ------------------------------------------------------------------
    # Local (sentence-transformers) backend
    # ------------------------------------------------------------------

    def _embed_local(self, text: str) -> list[float]:
        model = _get_local_model(self._model)
        vector = model.encode(text, normalize_embeddings=True)
        result: list[float] = vector.tolist()
        return result

    def _embed_local_batch(self, texts: list[str]) -> list[list[float]]:
        model = _get_local_model(self._model)
        vectors = model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    # ------------------------------------------------------------------
    # Remote (litellm) backend
    # ------------------------------------------------------------------

    async def _resolve_api_key(self) -> str | None:
        """Resolve the remote embedding API key.

        Precedence: an explicit ``api_key`` passed to the constructor, else the
        matching field on settings (``CoreSettings.openai_api_key`` for the
        OpenAI family today), else ``None`` — the provider call then fails
        loudly rather than borrowing an unrelated key. Resolved once and cached
        for the life of the service.
        """
        if self._resolved_key is not _UNSET:
            return self._resolved_key  # type: ignore[no-any-return]
        key = self._api_key
        if key is None:
            provider = _embedding_provider(self._model)
            if provider == "openai":
                key = settings.openai_api_key or None
        self._resolved_key = key
        return key

    async def _embed_remote(self, text: str) -> list[float]:
        """Embed a single text via litellm with timeout + retry."""
        import litellm

        from agentic_core.services.credential_scrubber import scrub as _scrub

        api_key = await self._resolve_api_key()
        for attempt in range(3):
            try:
                response = await asyncio.wait_for(
                    litellm.aembedding(model=self._model, input=[text], api_key=api_key),
                    timeout=_REMOTE_EMBED_TIMEOUT,
                )
                return response.data[0].embedding  # type: ignore[no-any-return]
            except (TimeoutError, Exception) as exc:
                if attempt == 2:
                    safe_msg = _scrub(str(exc))
                    raise RuntimeError(f"Embedding request failed: {safe_msg}") from None
                await asyncio.sleep(2**attempt)
                logger.warning("embedding.remote_retry attempt=%d error=%s", attempt + 1, exc)
        raise RuntimeError("Unreachable")  # pragma: no cover

    async def _embed_remote_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts via litellm, chunked by settings.embedding_batch_max."""
        import litellm

        from agentic_core.services.credential_scrubber import scrub as _scrub

        max_batch = settings.embedding_batch_max
        all_embeddings: list[list[float]] = []
        api_key = await self._resolve_api_key()

        for chunk_start in range(0, len(texts), max_batch):
            chunk = texts[chunk_start : chunk_start + max_batch]
            for attempt in range(3):
                try:
                    response = await asyncio.wait_for(
                        litellm.aembedding(model=self._model, input=chunk, api_key=api_key),
                        timeout=_REMOTE_EMBED_TIMEOUT,
                    )
                    all_embeddings.extend(d.embedding for d in response.data)
                    break
                except (TimeoutError, Exception) as exc:
                    if attempt == 2:
                        safe_msg = _scrub(str(exc))
                        raise RuntimeError(f"Embedding batch request failed: {safe_msg}") from None
                    await asyncio.sleep(2**attempt)
                    logger.warning(
                        "embedding.remote_batch_retry chunk_start=%d chunk_size=%d attempt=%d error=%s",
                        chunk_start,
                        len(chunk),
                        attempt + 1,
                        exc,
                    )

        return all_embeddings

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def dimensions(self) -> int:
        """Return the configured embedding dimension."""
        return self._dimensions

    @property
    def model(self) -> str:
        """Return the configured embedding model identifier."""
        return self._model

    @property
    def version(self) -> str:
        """Return the configured embedding version tag.

        Per-row tag stored alongside vectors so search can filter by "vectors
        produced by this exact build of this model".
        """
        return settings.embedding_version

    def _cache_key(self, text: str) -> str:
        """Build a cache key for the (model, tenant, agent, text) tuple.

        Uses the full 64-char SHA-256 hex digest to avoid collisions.
        """
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        return f"{self._model}|{self._tenant_id}|{self._agent_id}|{text_hash}"

    def _put_cache(self, key: str, value: list[float]) -> None:
        """Insert into the LRU cache, evicting the oldest entry if full."""
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > _CACHE_MAX_SIZE:
            self._cache.popitem(last=False)
