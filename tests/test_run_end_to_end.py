"""A full SRPA run against a real Postgres, with the LLM stubbed.

This is the test that says "core works". It creates the schema, an agent and a
run through the public runner, executes the whole Sense→Reason→Plan→Act loop under
each of the two migrated patterns, and asserts the steps were persisted.

The LLM is a stub, deliberately: a test that needs a provider key is a test nobody
runs. The four methods stubbed here are the entire surface the phases use, which
is itself worth pinning — if a future slice widens it, this fails.

Skipped unless ``AGENTIC_TEST_DATABASE_URL`` is set, e.g.::

    AGENTIC_TEST_DATABASE_URL='postgresql+asyncpg://ares:pw@localhost:5452/agentic_core_test' \
        .venv/bin/pytest tests/test_run_end_to_end.py -v
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest
from pydantic_settings import SettingsConfigDict

from agentic_core import CoreSettings
from agentic_core.schemas.agent import LLMProvider, ModelConfig
from tests.stub_llm import StubLLM

DB_URL = os.environ.get("AGENTIC_TEST_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(not DB_URL, reason="AGENTIC_TEST_DATABASE_URL is not set")


class _Settings(CoreSettings):
    model_config = SettingsConfigDict(env_prefix="AGENTIC_TEST_", extra="ignore")


@pytest.fixture(scope="module", autouse=True)
def _configure() -> None:
    from agentic_core.config import configure, reset_for_testing

    reset_for_testing()
    configure(_Settings(database_url=DB_URL))


@pytest.fixture(autouse=True)
async def _schema() -> Any:
    """Fresh schema per test, and a fresh engine after it.

    The engine is disposed rather than left cached because asyncpg connections
    belong to the loop that opened them, and each test here gets its own loop.
    """
    from agentic_core.models.database import dispose_engine
    from agentic_core.runner import init_schema

    await init_schema(drop_first=True)
    yield
    await dispose_engine()


@pytest.mark.parametrize("pattern", ["single_agent_baseline", "reason_act"])
async def test_full_run(pattern: str) -> None:
    """A run executes end to end and persists its steps."""
    from sqlalchemy import func, select

    from agentic_core.models.database import system_session
    from agentic_core.models.run import RunStatus, RunStep
    from agentic_core.runner import create_agent, create_run, execute_run

    stub = StubLLM()
    async with system_session(reason="test_full_run") as db:
        agent = await create_agent(
            db,
            name=f"test-{pattern}-{uuid.uuid4().hex[:8]}",
            goal="Answer the question.",
            model_config=ModelConfig(provider=LLMProvider.ANTHROPIC, model="claude-sonnet-5", api_key="stub"),
            pattern_config={"execution_pattern": pattern},
        )
        run = await create_run(db, agent_id=agent.id, user_input="What is 2 + 2?")
        assert run.status is RunStatus.PENDING

        result = await execute_run(db, run_id=run.id, llm_service=stub)

        assert result.error is None, result.error
        assert RunStatus(result.status) is RunStatus.COMPLETED
        assert result.output, "the run produced no output"
        assert stub.calls, "the LLM was never called — the loop did not run"

        # Steps are persisted, not merely returned.
        n = (
            await db.execute(select(func.count()).select_from(RunStep).where(RunStep.run_id == run.id))
        ).scalar()
        assert n and n > 0, "no RunStep rows were written"

        # And the outcome landed on the run row.
        await db.refresh(run)
        assert RunStatus(run.status) is RunStatus.COMPLETED
        assert run.token_usage["prompt_tokens"] > 0
        assert run.completed_at is not None


async def test_registry_holds_exactly_the_migrated_patterns() -> None:
    """Only the two patterns we chose are registered.

    Discovery imports every module in ``engine/patterns/builtin``, so this fails
    the moment a plugin is added without a decision to add it.
    """
    from agentic_core.engine.patterns.registry import PluginRegistry

    PluginRegistry.discover(raise_on_error=True)
    assert sorted(PluginRegistry.all()) == ["reason_act", "single_agent_baseline"]
