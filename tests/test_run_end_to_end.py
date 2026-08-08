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
from tests.stub_llm import FINAL_ANSWER, StubLLM

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
    from agentic_core.models.run import RunStatus
    from agentic_core.runner import create_agent, create_run, execute_run, get_run, get_run_steps

    stub = StubLLM()
    # No session anywhere: core owns its database and manages its own connections.
    agent = await create_agent(
        name=f"test-{pattern}-{uuid.uuid4().hex[:8]}",
        goal="Answer the question.",
        model_config=ModelConfig(provider=LLMProvider.ANTHROPIC, model="claude-sonnet-5", api_key="stub"),
        pattern_config={"execution_pattern": pattern},
    )
    run = await create_run(agent_id=agent.id, user_input="What is 2 + 2?")
    assert run.status is RunStatus.PENDING

    result = await execute_run(run_id=run.id, llm_service=stub)

    assert result.error is None, result.error
    assert RunStatus(result.status) is RunStatus.COMPLETED
    assert result.output, "the run produced no output"
    assert stub.calls, "the LLM was never called — the loop did not run"

    # Steps are persisted, not merely returned — read back through core's API.
    assert await get_run_steps(run.id), "no RunStep rows were written"

    # And the outcome landed on the run row.
    stored = await get_run(run.id)
    assert stored is not None
    assert RunStatus(stored.status) is RunStatus.COMPLETED
    assert stored.token_usage["prompt_tokens"] > 0
    assert stored.completed_at is not None


async def test_output_contract_produces_an_envelope_with_a_payload() -> None:
    """An agent with an output_contract gets its run.output wrapped in an
    envelope carrying a payload — the exact mechanism spinal_surgery's
    DC/FTE/FS/MLM/Critic agents depend on for structured handoffs."""
    from agentic_core.runner import create_agent, create_run, execute_run
    from agentic_core.schemas.output import parse_envelope

    stub = StubLLM()
    agent = await create_agent(
        name=f"test-output-contract-{uuid.uuid4().hex[:8]}",
        goal="Review the input and return a verdict.",
        model_config=ModelConfig(provider=LLMProvider.ANTHROPIC, model="claude-sonnet-5", api_key="stub"),
        pattern_config={"execution_pattern": "reason_act"},
        output_contract={
            "name": "TestVerdict",
            "fields": [
                {"name": "approved", "type": "bool"},
                {"name": "feedback", "type": "str", "default": ""},
            ],
        },
    )
    run = await create_run(agent_id=agent.id, user_input="Review this.")

    result = await execute_run(run_id=run.id, llm_service=stub)

    assert result.error is None, result.error
    envelope = parse_envelope(result.output)
    assert envelope is not None, f"run.output was not wrapped in an envelope: {result.output!r}"
    assert envelope.payload == {"approved": None, "feedback": ""}
    assert envelope.result == FINAL_ANSWER
    assert "complete:TestVerdict" in stub.calls

    # The persisted row agrees with what execute_run returned — not just the
    # in-memory result object.
    from agentic_core.runner import get_run

    stored = await get_run(run.id)
    assert stored is not None
    assert stored.output == result.output


async def test_fail_run_forces_a_non_terminal_run_to_failed() -> None:
    """fail_run closes out a run that execute_run never got to finish —
    the case a stale-run sweep needs: the process running execute_run died,
    so nothing else will ever write a terminal status onto this row."""
    from agentic_core.models.run import RunStatus
    from agentic_core.runner import create_agent, create_run, fail_run, get_run

    agent = await create_agent(
        name=f"test-fail-run-{uuid.uuid4().hex[:8]}",
        goal="Answer the question.",
        model_config=ModelConfig(provider=LLMProvider.ANTHROPIC, model="claude-sonnet-5", api_key="stub"),
    )
    run = await create_run(agent_id=agent.id, user_input="What is 2 + 2?")
    assert run.status is RunStatus.PENDING

    failed = await fail_run(run.id, error="worker lost — no heartbeat for >= 300s")

    assert failed is not None
    assert failed.status is RunStatus.FAILED
    assert failed.error == "worker lost — no heartbeat for >= 300s"
    assert failed.completed_at is not None

    stored = await get_run(run.id)
    assert stored is not None
    assert stored.status is RunStatus.FAILED
    assert stored.error == failed.error


async def test_fail_run_is_a_noop_on_an_already_terminal_run() -> None:
    """A detector racing against a slow-but-live execute_run commit must not
    clobber the status/error/completed_at execute_run already wrote."""
    from agentic_core.models.run import RunStatus
    from agentic_core.runner import create_agent, create_run, execute_run, fail_run, get_run

    stub = StubLLM()
    agent = await create_agent(
        name=f"test-fail-run-noop-{uuid.uuid4().hex[:8]}",
        goal="Answer the question.",
        model_config=ModelConfig(provider=LLMProvider.ANTHROPIC, model="claude-sonnet-5", api_key="stub"),
    )
    run = await create_run(agent_id=agent.id, user_input="What is 2 + 2?")
    await execute_run(run_id=run.id, llm_service=stub)

    completed = await get_run(run.id)
    assert completed is not None
    assert completed.status is RunStatus.COMPLETED

    result = await fail_run(run.id, error="should never be written")

    assert result is not None
    assert result.status is RunStatus.COMPLETED
    assert result.error != "should never be written"
    assert result.completed_at == completed.completed_at


async def test_fail_run_on_an_unknown_run_returns_none() -> None:
    from agentic_core.runner import fail_run

    assert await fail_run(uuid.uuid4(), error="no such run") is None


async def test_registry_holds_exactly_the_migrated_patterns() -> None:
    """Only the two patterns we chose are registered.

    Discovery imports every module in ``engine/patterns/builtin``, so this fails
    the moment a plugin is added without a decision to add it.
    """
    from agentic_core.engine.patterns.registry import PluginRegistry

    PluginRegistry.discover(raise_on_error=True)
    assert sorted(PluginRegistry.all()) == ["reason_act", "single_agent_baseline"]
