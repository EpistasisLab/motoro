"""A full SRPA run against a real Postgres, with the LLM stubbed.

This is the test that says "core works". It creates the schema, an agent and a
run through the public runner, executes the whole Sense→Reason→Plan→Act loop under
each of the two migrated patterns, and asserts the steps were persisted.

The LLM is a stub, deliberately: a test that needs a provider key is a test nobody
runs. The four methods stubbed here are the entire surface the phases use, which
is itself worth pinning — if a future slice widens it, this fails.

Skipped unless ``MOTORO_TEST_DATABASE_URL`` is set, e.g.::

    MOTORO_TEST_DATABASE_URL='postgresql+asyncpg://ares:pw@localhost:5452/motoro_test' \
        .venv/bin/pytest tests/test_run_end_to_end.py -v
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest
from pydantic_settings import SettingsConfigDict

from motoro import CoreSettings
from motoro.schemas.agent import LLMProvider, ModelConfig
from tests.stub_llm import FINAL_ANSWER, StubLLM

DB_URL = os.environ.get("MOTORO_TEST_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(not DB_URL, reason="MOTORO_TEST_DATABASE_URL is not set")


class _Settings(CoreSettings):
    model_config = SettingsConfigDict(env_prefix="MOTORO_TEST_", extra="ignore")


@pytest.fixture(scope="module", autouse=True)
def _configure() -> None:
    from motoro.config import configure, reset_for_testing

    reset_for_testing()
    configure(_Settings(database_url=DB_URL))


@pytest.fixture(autouse=True)
async def _schema() -> Any:
    """Fresh schema per test, and a fresh engine after it.

    The engine is disposed rather than left cached because asyncpg connections
    belong to the loop that opened them, and each test here gets its own loop.
    """
    from motoro.models.database import dispose_engine
    from motoro.runner import init_schema

    await init_schema(drop_first=True)
    yield
    await dispose_engine()


@pytest.mark.parametrize("pattern", ["single_agent_baseline", "reason_act"])
async def test_full_run(pattern: str) -> None:
    """A run executes end to end and persists its steps."""
    from motoro.models.run import RunStatus
    from motoro.runner import create_agent, create_run, execute_run, get_run, get_run_steps

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
    from motoro.runner import create_agent, create_run, execute_run
    from motoro.schemas.output import parse_envelope

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
    from motoro.runner import get_run

    stored = await get_run(run.id)
    assert stored is not None
    assert stored.output == result.output


async def test_pinned_variants_execute_with_persisted_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real persistence, skill resolution and SRPA execution on one shared agent."""
    from motoro.engine.patterns.orchestrator import PatternOrchestrator
    from motoro.runner import create_agent, create_run, effective_agent_config, execute_run, get_run, update_agent
    from motoro.schemas.output import parse_envelope
    from motoro.services.skill_service import create_skill

    owner_id = uuid.uuid4()
    agent = await create_agent(name="shared", goal="original", owner_id=owner_id)
    captured: list[Any] = []
    factory = PatternOrchestrator.from_pattern_config

    def record(runtime: Any, config: Any, **kwargs: Any) -> Any:
        orchestrator = factory(runtime, config, **kwargs)
        captured.append((runtime._config, orchestrator))
        return orchestrator

    monkeypatch.setattr(PatternOrchestrator, "from_pattern_config", record)
    variants = []
    for iterations, tokens in [(3, 300), (6, 600)]:
        skill = await create_skill(
            name=f"variant-{iterations}",
            description="Variant instructions",
            body=f"Use {iterations}",
            owner_id=owner_id,
        )
        run = await create_run(
            agent_id=agent.id,
            user_input="question",
            config_snapshot={
                "goal": f"goal-{iterations}",
                "system_prompt": f"prompt-{iterations}",
                "model_config": {"provider": "openai", "model": "gpt-4o", "max_tokens": tokens},
                "pattern_config": {
                    "execution_pattern": "reason_act",
                    "pattern_params": {"reason_act": {"max_iterations": iterations}},
                },
                "skill_config": {"skill_ids": [str(skill.id)]},
                "output_contract": {"name": f"Verdict{iterations}", "fields": [{"name": "value", "type": "int"}]},
                "budget_limit_usd": iterations,
                "max_run_duration_seconds": iterations * 10,
            },
        )
        variants.append((run, skill, iterations, tokens))
    await update_agent(
        agent.id,
        goal="changed",
        system_prompt="changed",
        model_config=ModelConfig(max_tokens=1),
        pattern_config={"execution_pattern": "single_agent_baseline"},
        skill_config={"skill_ids": []},
        output_contract={"fields": [{"name": "changed", "type": "bool"}]},
        budget_limit_usd=999,
        max_run_duration_seconds=999,
    )
    for run, skill, iterations, tokens in variants:
        stub = StubLLM()
        result = await execute_run(run_id=run.id, llm_service=stub)
        assert result.status == "completed", result.error
        config, orchestrator = captured[-1]
        assert config.system_prompt == f"prompt-{iterations}"
        assert config.goal == f"goal-{iterations}"
        assert config.model_config.max_tokens == tokens
        assert config.skills[0]["name"] == skill.name
        assert config.skills[0]["body"] == skill.body
        assert orchestrator._plugins[0].max_iterations == iterations
        assert f"complete:Verdict{iterations}" in stub.calls
        assert parse_envelope(result.output).payload == {"value": None}
        stored = await get_run(run.id)
        assert stored is not None
        assert stored.config_snapshot["system_prompt"] == f"prompt-{iterations}"
        assert stored.run_metadata["reason_act_state"]["max_iterations"] == iterations
        assert stored.pattern_overrides is None
        effective = effective_agent_config(stored)
        assert effective.budget_limit_usd == iterations
        assert effective.max_run_duration_seconds == iterations * 10


@pytest.mark.parametrize("snapshot", [None, {}, {"goal": ""}])
async def test_empty_goal_agent_runs_with_and_without_a_snapshot(snapshot: dict[str, Any] | None) -> None:
    from motoro.runner import create_agent, create_run, execute_run, get_run

    agent = await create_agent(name="canvas-agent", goal="")
    run = await create_run(agent_id=agent.id, user_input="Answer the question", config_snapshot=snapshot)
    result = await execute_run(run_id=run.id, llm_service=StubLLM())
    assert result.status == "completed", result.error
    stored = await get_run(run.id)
    assert stored is not None
    assert stored.config_snapshot["goal"] == ""


async def test_fail_run_forces_a_non_terminal_run_to_failed() -> None:
    """fail_run closes out a run that execute_run never got to finish —
    the case a stale-run sweep needs: the process running execute_run died,
    so nothing else will ever write a terminal status onto this row."""
    from motoro.models.run import RunStatus
    from motoro.runner import create_agent, create_run, fail_run, get_run

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
    from motoro.models.run import RunStatus
    from motoro.runner import create_agent, create_run, execute_run, fail_run, get_run

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
    from motoro.runner import fail_run

    assert await fail_run(uuid.uuid4(), error="no such run") is None


async def test_list_runs_metadata_filter_matches_exactly() -> None:
    """list_runs(metadata=...) filters server-side on run_metadata, not just
    agent_id/status/owner_id -- the gap that let a product (ASAREE's
    spinal_pipeline notebook) pull *every* run for an agent and filter by its
    own experiment_id/treatment_group/replicate client-side in Python, which
    silently truncated once the agent had more total runs than list_runs's
    default limit. A dict with two matching keys must exclude a run that
    only satisfies one of them, and a mismatched value on an existing key
    must exclude a run entirely, not just "not select" it."""
    from motoro.runner import create_agent, create_run, execute_run, list_runs

    stub = StubLLM()
    agent = await create_agent(
        name=f"test-metadata-filter-{uuid.uuid4().hex[:8]}",
        goal="Answer the question.",
        model_config=ModelConfig(provider=LLMProvider.ANTHROPIC, model="claude-sonnet-5", api_key="stub"),
    )
    target = await create_run(agent_id=agent.id, user_input="q1", metadata={"experiment_id": "exp-1", "replicate": 3})
    await execute_run(run_id=target.id, llm_service=stub)
    other_replicate = await create_run(
        agent_id=agent.id, user_input="q2", metadata={"experiment_id": "exp-1", "replicate": 4}
    )
    await execute_run(run_id=other_replicate.id, llm_service=stub)
    other_experiment = await create_run(
        agent_id=agent.id, user_input="q3", metadata={"experiment_id": "exp-2", "replicate": 3}
    )
    await execute_run(run_id=other_experiment.id, llm_service=stub)

    matched = await list_runs(agent_id=agent.id, metadata={"experiment_id": "exp-1", "replicate": 3})

    assert [r.id for r in matched] == [target.id]


async def test_list_runs_metadata_filter_beyond_limit_still_finds_the_match() -> None:
    """The whole point: a match older than `limit` newest runs is still
    found, because the filter runs in SQL before the limit is applied --
    not after pulling the `limit` most-recent rows and filtering in Python."""
    from motoro.runner import create_agent, create_run, execute_run, list_runs

    stub = StubLLM()
    agent = await create_agent(
        name=f"test-metadata-filter-limit-{uuid.uuid4().hex[:8]}",
        goal="Answer the question.",
        model_config=ModelConfig(provider=LLMProvider.ANTHROPIC, model="claude-sonnet-5", api_key="stub"),
    )
    target = await create_run(agent_id=agent.id, user_input="oldest", metadata={"experiment_id": "exp-old"})
    await execute_run(run_id=target.id, llm_service=stub)
    for i in range(5):
        r = await create_run(agent_id=agent.id, user_input=f"filler-{i}", metadata={"experiment_id": "exp-new"})
        await execute_run(run_id=r.id, llm_service=stub)

    # limit=1: without server-side filtering this would only see the single
    # newest run (an "exp-new" one) and never find the older "exp-old" match.
    matched = await list_runs(agent_id=agent.id, metadata={"experiment_id": "exp-old"}, limit=1)

    assert [r.id for r in matched] == [target.id]


async def test_registry_holds_exactly_the_migrated_patterns() -> None:
    """Only the two patterns we chose are registered.

    Discovery imports every module in ``engine/patterns/builtin``, so this fails
    the moment a plugin is added without a decision to add it.
    """
    from motoro.engine.patterns.registry import PluginRegistry

    PluginRegistry.discover(raise_on_error=True)
    assert sorted(PluginRegistry.all()) == ["reason_act", "single_agent_baseline"]


async def test_list_run_summaries_matches_metadata_values_and_ids_in_keyset_pages() -> None:
    """Bulk accounting reads: any of several metadata values or explicit ids,
    owner-scoped, typed JSON comparison, paged by id, scalar columns only."""
    from motoro.runner import create_agent, create_run, list_run_summaries

    owner, stranger = uuid.uuid4(), uuid.uuid4()
    agent = await create_agent(
        name=f"test-run-summaries-{uuid.uuid4().hex[:8]}",
        goal="Answer the question.",
        model_config=ModelConfig(provider=LLMProvider.ANTHROPIC, model="claude-sonnet-5", api_key="stub"),
        owner_id=owner,
    )
    tagged = [
        await create_run(agent_id=agent.id, user_input="q", metadata={"batch": f"b-{i}"}, owner_id=owner)
        for i in range(3)
    ]
    referenced = await create_run(agent_id=agent.id, user_input="q", metadata={}, owner_id=owner)
    await create_run(agent_id=agent.id, user_input="q", metadata={"batch": 1}, owner_id=owner)  # number, not "1"
    await create_run(agent_id=agent.id, user_input="q", metadata={"batch": "b-0"}, owner_id=stranger)

    expected = sorted([*(run.id for run in tagged), referenced.id])
    rows = await list_run_summaries(
        owner_id=owner, metadata_key="batch", metadata_values=["b-0", "b-1", "b-2", "1"], run_ids=[referenced.id]
    )
    assert [row["id"] for row in rows] == expected
    assert "input" not in rows[0] and rows[0]["run_metadata"] is not None

    first = await list_run_summaries(
        owner_id=owner, metadata_key="batch", metadata_values=["b-0", "b-1", "b-2"], limit=2
    )
    rest = await list_run_summaries(
        owner_id=owner, metadata_key="batch", metadata_values=["b-0", "b-1", "b-2"], after_id=first[-1]["id"]
    )
    assert [row["id"] for row in first + rest] == sorted(run.id for run in tagged)
    assert await list_run_summaries(owner_id=owner) == []
    with pytest.raises(ValueError):
        await list_run_summaries(owner_id=owner, run_ids=[uuid.uuid4()] * 201)


async def test_list_run_step_summaries_pages_previews_and_respects_owner() -> None:
    from motoro.runner import create_agent, create_run, execute_run, get_run_steps, list_run_step_summaries

    owner = uuid.uuid4()
    agent = await create_agent(
        name=f"test-step-summaries-{uuid.uuid4().hex[:8]}",
        goal="Answer the question.",
        model_config=ModelConfig(provider=LLMProvider.ANTHROPIC, model="claude-sonnet-5", api_key="stub"),
        owner_id=owner,
    )
    run = await create_run(agent_id=agent.id, user_input="q", owner_id=owner)
    await execute_run(run_id=run.id, llm_service=StubLLM())
    steps = await get_run_steps(run.id)
    assert len(steps) >= 2

    page = await list_run_step_summaries(run.id, owner_id=owner, limit=1, preview_chars=5)
    assert [item["sequence"] for item in page["items"]] == [steps[0].sequence]
    assert page["next_after"] == steps[0].sequence
    assert all(len(item["preview"] or "") <= 5 for item in page["items"])
    rest = await list_run_step_summaries(run.id, owner_id=owner, after_sequence=page["next_after"], limit=100)
    assert [item["sequence"] for item in rest["items"]] == [step.sequence for step in steps[1:]]
    assert rest["next_after"] is None
    assert (await list_run_step_summaries(run.id, owner_id=uuid.uuid4()))["items"] == []
