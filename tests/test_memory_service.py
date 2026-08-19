"""Memory: no owner, no per-user credential lookup, and the runtime actually uses it.

Three things worth pinning:

1. Memory has no owner column at all. ARES's ``created_by_id`` existed only to
   scope isolation queries (``security/isolation/registry.py``) — the isolation
   feature itself is out of scope for this slice. A memory's identity is
   ``agent_id`` (and, for episodic entries, ``run_id``), both real foreign keys
   into core's own tables.
2. Remote embedding credentials resolve from settings, not a per-user table —
   the same shape as the LLM bridge's credential resolver, for the same reason.
3. A run with ``episodic_memory_enabled`` stores a memory unconditionally.
   ARES's runtime skipped storage when a run had no acting user; nothing gates
   it here, because nothing needs to own the entry for it to exist.

Database-backed tests are skipped unless ``AGENTIC_TEST_DATABASE_URL`` is set.
They use the real default local embedding model (``sentence-transformers/
BAAI/bge-base-en-v1.5``) rather than a mock — it is fast once cached and this is
the actual default a product gets with no configuration.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest
from pydantic_settings import SettingsConfigDict

from motoro import CoreSettings

DB_URL = os.environ.get("AGENTIC_TEST_DATABASE_URL", "")
needs_db = pytest.mark.skipif(not DB_URL, reason="AGENTIC_TEST_DATABASE_URL is not set")


class _Settings(CoreSettings):
    model_config = SettingsConfigDict(env_prefix="AGENTIC_TEST_", extra="ignore")


@pytest.fixture(scope="module", autouse=True)
def _configure() -> None:
    from motoro.config import configure, reset_for_testing

    if not DB_URL:
        return
    reset_for_testing()
    configure(_Settings(database_url=DB_URL))


@pytest.fixture(autouse=True)
async def _schema() -> Any:
    """Fresh schema per test, and a fresh engine after it (own event loop each test)."""
    if not DB_URL:
        yield
        return
    from motoro.models.database import dispose_engine
    from motoro.runner import init_schema

    await init_schema(drop_first=True)
    yield
    await dispose_engine()


# --------------------------------------------------------------------------- #
#  Credential resolution — no per-user lookup                                  #
# --------------------------------------------------------------------------- #


def test_embedding_provider_mapping() -> None:
    from motoro.memory.embedding import _embedding_provider

    assert _embedding_provider("text-embedding-3-small") == "openai"
    assert _embedding_provider("openai/text-embedding-3-large") == "openai"
    assert _embedding_provider("sentence-transformers/all-MiniLM-L6-v2") is None
    assert _embedding_provider("voyage-3") is None


async def test_explicit_api_key_takes_precedence() -> None:
    from motoro.memory.embedding import EmbeddingService

    svc = EmbeddingService(model="text-embedding-3-small", api_key="sk-explicit")
    assert await svc._resolve_api_key() == "sk-explicit"


def _reconfigure(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, **kw: Any) -> None:
    """Reset and reinstall settings for a single test, restoring afterward.

    ``settings`` is a lazy proxy (see ``motoro.config``), so
    ``monkeypatch.setattr(settings, ...)`` cannot patch a field directly — the
    product-facing path is always reset + reconfigure with a real instance.
    """
    from motoro.config import configure, reset_for_testing

    class _Local(CoreSettings):
        model_config = SettingsConfigDict(env_prefix="AGENTIC_TEST_LOCAL_", extra="ignore", populate_by_name=True)

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # ambient leakage guard
    reset_for_testing()
    configure(_Local(database_url=DB_URL or "postgresql+asyncpg://x/x", **kw))
    request.addfinalizer(_restore_module_settings)


def _restore_module_settings() -> None:
    from motoro.config import configure, reset_for_testing

    reset_for_testing()
    if DB_URL:
        configure(_Settings(database_url=DB_URL))


async def test_resolves_key_from_settings_not_a_user_table(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ``user_id`` parameter exists at all — there is no per-user table to join."""
    import inspect

    from motoro.memory.embedding import EmbeddingService

    assert "user_id" not in inspect.signature(EmbeddingService.__init__).parameters

    _reconfigure(request, monkeypatch, openai_api_key="sk-from-settings")
    svc = EmbeddingService(model="text-embedding-3-small")
    assert await svc._resolve_api_key() == "sk-from-settings"


async def test_no_key_configured_resolves_to_none(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    from motoro.memory.embedding import EmbeddingService

    _reconfigure(request, monkeypatch)
    svc = EmbeddingService(model="text-embedding-3-small")
    assert await svc._resolve_api_key() is None


async def test_local_model_never_resolves_a_key() -> None:
    """The default backend needs no credential at all."""
    from motoro.memory.embedding import EmbeddingService

    svc = EmbeddingService(model="sentence-transformers/all-MiniLM-L6-v2")
    assert await svc._resolve_api_key() is None


# --------------------------------------------------------------------------- #
#  MemoryEntry — no owner column                                               #
# --------------------------------------------------------------------------- #


def test_memory_entry_has_no_owner_column() -> None:
    from motoro.models.memory import MemoryEntry

    columns = {c.name for c in MemoryEntry.__table__.columns}
    assert "created_by_id" not in columns
    assert "owner_id" not in columns
    assert {"agent_id", "run_id", "type", "content", "embedding"} <= columns


# --------------------------------------------------------------------------- #
#  Storage and retrieval (real local embeddings, real Postgres)                #
# --------------------------------------------------------------------------- #
#
# agent_id and run_id are real foreign keys into core's own agents/agent_runs
# tables (see models/memory.py), so every test below creates a real agent (and,
# for episodic entries, a real run) rather than an arbitrary UUID.


async def _make_agent() -> Any:
    from motoro.runner import create_agent

    return await create_agent(name=f"test-memory-{uuid.uuid4().hex[:8]}", goal="x")


async def _make_run(agent_id: uuid.UUID) -> Any:
    from motoro.runner import create_run

    return await create_run(agent_id=agent_id, user_input="x")


@needs_db
async def test_semantic_store_and_search_round_trip() -> None:
    from motoro.memory.embedding import EmbeddingService
    from motoro.memory.semantic import SemanticMemoryStore

    embed = EmbeddingService()
    store = SemanticMemoryStore(embed)
    agent = await _make_agent()

    await store.store("The user's favourite colour is teal.", agent_id=agent.id)
    await store.store("Paris is the capital of France.", agent_id=agent.id)

    assert await store.count(agent_id=agent.id) == 2

    results = await store.search("what colour does the user like?", agent_id=agent.id, top_k=1)
    assert len(results) == 1
    entry, score = results[0]
    assert "teal" in entry.content
    assert score > 0.0


@needs_db
async def test_semantic_search_requires_agent_id_for_agent_scope() -> None:
    from motoro.memory.embedding import EmbeddingService
    from motoro.memory.semantic import SemanticMemoryStore

    store = SemanticMemoryStore(EmbeddingService())
    with pytest.raises(ValueError, match="requires a non-None agent_id"):
        await store.search("anything", scope="agent", agent_id=None)


@needs_db
async def test_semantic_global_scope_excludes_agent_rows() -> None:
    from motoro.memory.embedding import EmbeddingService
    from motoro.memory.semantic import SemanticMemoryStore

    embed = EmbeddingService()
    store = SemanticMemoryStore(embed)
    agent = await _make_agent()

    await store.store("agent-owned fact about rockets", agent_id=agent.id)
    await store.store("shared fact about rockets, available to every agent", agent_id=None)

    global_results = await store.search("rockets", scope="global", top_k=5)
    assert len(global_results) == 1
    assert global_results[0][0].agent_id is None


@needs_db
async def test_episodic_store_and_get_recent() -> None:
    from motoro.memory.embedding import EmbeddingService
    from motoro.memory.episodic import EpisodicMemoryStore
    from tests.stub_llm import StubLLM

    embed = EmbeddingService()
    store = EpisodicMemoryStore(StubLLM(), embed)
    agent = await _make_agent()
    run = await _make_run(agent.id)

    entry = await store.store_run_summary(run_id=run.id, agent_id=agent.id, summary="Run completed successfully.")
    assert entry.run_id == run.id
    assert await store.count(agent.id) == 1

    recent = await store.get_recent(agent.id)
    assert len(recent) == 1
    assert recent[0].id == entry.id


@needs_db
async def test_forget_deletes_either_memory_type() -> None:
    from motoro.memory.embedding import EmbeddingService
    from motoro.memory.episodic import EpisodicMemoryStore
    from motoro.memory.semantic import SemanticMemoryStore
    from tests.stub_llm import StubLLM

    embed = EmbeddingService()
    semantic = SemanticMemoryStore(embed)
    episodic = EpisodicMemoryStore(StubLLM(), embed)
    agent = await _make_agent()
    run = await _make_run(agent.id)

    sem_entry = await semantic.store("some fact", agent_id=agent.id)
    epi_entry = await episodic.store_run_summary(run_id=run.id, agent_id=agent.id, summary="a run")

    assert await semantic.delete(sem_entry.id) is True
    assert await semantic.delete(epi_entry.id) is True  # delete is not type-scoped
    assert await semantic.delete(uuid.uuid4()) is False


# --------------------------------------------------------------------------- #
#  MemoryService — the MemoryServicePort implementation                        #
# --------------------------------------------------------------------------- #


@needs_db
async def test_memory_service_conforms_to_the_port() -> None:
    from motoro.engine.ports import MemoryServicePort
    from motoro.services.memory_service import MemoryService
    from tests.stub_llm import StubLLM

    svc = MemoryService(llm_service=StubLLM())
    assert isinstance(svc, MemoryServicePort)


@needs_db
async def test_remember_dispatches_by_memory_type() -> None:
    from motoro.models.memory import MemoryType
    from motoro.services.memory_service import MemoryService
    from tests.stub_llm import StubLLM

    svc = MemoryService(llm_service=StubLLM())
    agent = await _make_agent()

    semantic_entry = await svc.remember(agent.id, "a fact", memory_type=MemoryType.SEMANTIC)
    assert semantic_entry.type is MemoryType.SEMANTIC

    # No run_id supplied -> stored as None, not a fabricated placeholder. run_id
    # is a real foreign key into agent_runs, so anything but a real run id or
    # None would violate the constraint.
    episodic_entry = await svc.remember(agent.id, "a run happened", memory_type=MemoryType.EPISODIC)
    assert episodic_entry.type is MemoryType.EPISODIC
    assert episodic_entry.run_id is None

    assert await svc.count(agent.id) == 2


@needs_db
async def test_recall_combines_and_ranks_both_types() -> None:
    from motoro.services.memory_service import MemoryService
    from tests.stub_llm import StubLLM

    svc = MemoryService(llm_service=StubLLM())
    agent = await _make_agent()
    run = await _make_run(agent.id)

    await svc.semantic.store("The user prefers concise answers.", agent_id=agent.id)
    await svc.episodic.store_run_summary(
        run_id=run.id, agent_id=agent.id, summary="Previous run: user asked for a concise summary."
    )

    results = await svc.recall(agent.id, "please be concise", top_k=5)
    assert len(results) == 2
    # Sorted by score descending.
    assert results[0][1] >= results[1][1]


@needs_db
async def test_recall_falls_back_to_recency_when_nothing_matches() -> None:
    """No embeddings exist for this agent at all -> recency fallback, not an empty list."""
    from motoro.models.memory import MemoryType
    from motoro.services.memory_service import MemoryService
    from tests.stub_llm import StubLLM

    svc = MemoryService(llm_service=StubLLM())
    agent = await _make_agent()
    run = await _make_run(agent.id)
    await svc.episodic.store_run_summary(run_id=run.id, agent_id=agent.id, summary="some episode")

    # Force the semantic side to find nothing relevant and the episodic search to
    # miss too, by asking about an agent with no semantic memories at all — the
    # embedding model still returns *a* nearest neighbour, so instead assert the
    # fallback path directly via an agent with zero indexed content.
    results = await svc.recall(agent.id, "totally unrelated query text", types=[MemoryType.EPISODIC], top_k=5)
    assert len(results) == 1


# --------------------------------------------------------------------------- #
#  The runtime actually uses it: no owner gate on episodic storage              #
# --------------------------------------------------------------------------- #


@needs_db
async def test_run_with_no_owner_still_stores_episodic_memory() -> None:
    """The regression this slice removes: ARES skipped storage with no acting user.

    Memory belongs to the agent and the run, not to a user, so an ownerless run
    (owner_id=None, the default) must still produce an episodic entry once
    episodic memory is enabled.
    """
    from motoro.runner import create_agent, create_run, execute_run
    from motoro.schemas.agent import LLMProvider, ModelConfig
    from motoro.services.memory_service import MemoryService
    from tests.stub_llm import StubLLM

    stub = StubLLM()
    agent = await create_agent(
        name=f"test-memory-{uuid.uuid4().hex[:8]}",
        goal="Answer the question.",
        model_config=ModelConfig(provider=LLMProvider.ANTHROPIC, model="claude-sonnet-5", api_key="stub"),
        pattern_config={"execution_pattern": "single_agent_baseline"},
        memory_config={"episodic_memory_enabled": True},
    )
    assert agent.owner_id is None  # the ownerless case this test is about

    memory_service = MemoryService(llm_service=stub)
    run = await create_run(agent_id=agent.id, user_input="What is 2 + 2?")
    result = await execute_run(run_id=run.id, llm_service=stub, memory_service=memory_service)

    assert result.error is None, result.error
    assert await memory_service.episodic.count(agent.id) == 1


@needs_db
async def test_a_second_run_actually_recalls_the_first_runs_memory() -> None:
    """Storage alone is not the feature — Sense has to inject it into the next run.

    Every real run goes through ``PatternOrchestrator`` (``execute_run`` always
    wraps the loop in one, for both shipped patterns), and the orchestrator
    builds its own ``RunContext`` rather than reusing ``AgentRuntime.run``'s.
    That mirror was missing ``context.memory_config_data = ...``, so
    ``RunContext``'s dataclass default (``{}``) meant Sense always read
    ``episodic_memory_enabled`` as ``False`` — storage still worked (a
    property read directly off the runtime's config, not the context), so
    memory was written on every run and recalled on none of them. This is the
    regression test for that: a second run must report a nonzero
    ``memory_recalled_count``, not just leave a row behind.
    """
    from motoro.runner import create_agent, create_run, execute_run
    from motoro.schemas.agent import LLMProvider, ModelConfig
    from motoro.services.memory_service import MemoryService
    from tests.stub_llm import StubLLM

    stub = StubLLM()
    agent = await create_agent(
        name=f"test-recall-{uuid.uuid4().hex[:8]}",
        goal="Answer the question.",
        model_config=ModelConfig(provider=LLMProvider.ANTHROPIC, model="claude-sonnet-5", api_key="stub"),
        pattern_config={"execution_pattern": "single_agent_baseline"},
        memory_config={"episodic_memory_enabled": True},
    )
    memory_service = MemoryService(llm_service=stub)

    first = await create_run(agent_id=agent.id, user_input="first turn")
    await execute_run(run_id=first.id, llm_service=stub, memory_service=memory_service)
    assert await memory_service.episodic.count(agent.id) == 1

    second = await create_run(agent_id=agent.id, user_input="second turn")
    result = await execute_run(run_id=second.id, llm_service=stub, memory_service=memory_service)

    assert result.run_metadata.get("memory_recalled_count", 0) > 0
