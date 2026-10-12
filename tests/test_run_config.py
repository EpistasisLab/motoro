"""Per-run isolation at the public runner boundary, without a database server."""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from copy import deepcopy
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from motoro import runner
from motoro.engine.context import RunContext
from motoro.engine.patterns.builtin.reason_act import ReasonActPlugin
from motoro.engine.patterns.orchestrator import PatternOrchestrator
from motoro.engine.runtime import AgentRunResult
from motoro.models.agent import Agent
from motoro.models.run import AgentRun
from motoro.schemas.agent import AgentUpdate, ModelConfig
from motoro.schemas.output import parse_envelope
from motoro.schemas.run_config import AgentSnapshotUpdate
from motoro.services.run_config import effective_agent_config


@pytest.fixture
def storage(monkeypatch: pytest.MonkeyPatch) -> Any:
    agent = Agent(
        id=uuid.uuid4(),
        owner_id=uuid.uuid4(),
        name="shared",
        description="original description",
        goal="original goal",
        system_prompt="original prompt",
        model_config_data={"model": "gpt-4o", "provider": "openai", "max_tokens": 900},
        memory_config_data={"episodic_memory_enabled": True},
        pattern_config={"execution_pattern": "reason_act", "pattern_params": {"reason_act": {"max_iterations": 9}}},
        skill_config={"skill_ids": [str(uuid.uuid4())]},
        output_contract={"fields": [{"name": "original", "type": "int"}]},
        budget_limit_usd=5.0,
        max_run_duration_seconds=90,
    )
    runs: dict[uuid.UUID, AgentRun] = {}
    db = MagicMock()

    async def execute(statement: Any) -> Any:
        result = MagicMock()
        entity = statement.column_descriptions[0]["entity"]
        row = agent if entity is Agent else runs[statement.compile().params["id_1"]]
        result.scalar_one.return_value = row
        result.scalar_one_or_none.return_value = row
        return result

    def add(row: Agent | AgentRun) -> None:
        nonlocal agent
        row.id = uuid.uuid4()
        if isinstance(row, Agent):
            agent = row
        else:
            row.agent = agent
            runs[row.id] = row

    db.execute = AsyncMock(side_effect=execute)
    db.add.side_effect = add
    db.commit = AsyncMock()
    db.refresh = AsyncMock()

    @asynccontextmanager
    async def session(_reason: str) -> Any:
        yield db

    monkeypatch.setattr(runner, "_session", session)
    return agent, runs, db


@pytest.fixture
def execution(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Capture actual runtime/plugin settings; run real output finalization."""
    captured: dict[uuid.UUID, Any] = {}
    factory = PatternOrchestrator.from_pattern_config

    def build(runtime: Any, pattern_config: Any, **kwargs: Any) -> Any:
        orchestrator = factory(runtime, pattern_config, hook_timeout=300.0, fail_on_type_mismatch=False, **kwargs)

        async def run(*, run_id: uuid.UUID, **_kwargs: Any) -> AgentRunResult:
            captured[run_id] = (runtime._config, orchestrator)
            await asyncio.sleep(0)  # exercise overlapping runs sharing one Agent
            return AgentRunResult(status="completed", output='Answer\n```json\n{"a": 1, "b": 2, "original": 3}\n```')

        orchestrator.run = run
        return orchestrator

    async def resolve(config: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return [{"name": skill_id, "body": "instructions"} for skill_id in (config or {}).get("skill_ids", [])]

    resolver = AsyncMock(side_effect=resolve)
    monkeypatch.setattr(PatternOrchestrator, "from_pattern_config", build)
    monkeypatch.setattr("motoro.services.skill_service.resolve_skills", resolver)
    return captured, resolver


async def test_two_pinned_runs_survive_shared_agent_mutation(storage: Any, execution: Any) -> None:
    agent, _runs, _db = storage
    captured, resolver = execution
    runs = []
    expected = []
    for index, (iterations, tokens, field) in enumerate([(3, 300, "a"), (6, 600, "b")]):
        skill_id = str(uuid.uuid4())
        snapshot = {
            "name": f"variant-{index}",
            "goal": f"goal-{index}",
            "description": f"description-{index}",
            "system_prompt": f"prompt-{index}",
            "pattern_config": {
                "execution_pattern": "reason_act",
                "pattern_params": {"reason_act": {"max_iterations": iterations}},
            },
            "model_config": {"provider": "openai", "model": "gpt-4o", "max_tokens": tokens},
            "skill_config": {"skill_ids": [skill_id]},
            "output_contract": {"fields": [{"name": field, "type": "int"}]},
            "memory_config": {"episodic_memory_enabled": bool(index)},
            "budget_limit_usd": index + 1.0,
            "max_run_duration_seconds": (index + 1) * 30,
        }
        expected.append(deepcopy(snapshot))
        runs.append(
            await runner.create_run(
                agent_id=agent.id,
                user_input="question",
                config_snapshot=snapshot,
                # A pinned pattern wins, while model overrides still work.
                pattern_overrides={"execution_pattern": "single_agent_baseline"},
                model_config_overrides={"temperature": 0.2},
            )
        )
        snapshot["pattern_config"]["pattern_params"]["reason_act"]["max_iterations"] = 1
        snapshot["output_contract"]["fields"][0]["name"] = "caller-mutated"

    agent.name = "mutated"
    agent.goal = "mutated"
    agent.description = "mutated"
    agent.system_prompt = "mutated"
    agent.model_config_data = {"max_tokens": 1}
    agent.pattern_config = {"execution_pattern": "single_agent_baseline"}
    agent.skill_config = None
    agent.output_contract = None
    agent.memory_config_data = {}
    agent.budget_limit_usd = 999
    agent.max_run_duration_seconds = 999

    results = await asyncio.gather(*(runner.execute_run(run_id=run.id, llm_service=MagicMock()) for run in runs))
    for run, result, snapshot in zip(runs, results, expected, strict=True):
        config, orchestrator = captured[run.id]
        assert config.name == snapshot["name"]
        assert config.goal == snapshot["goal"]
        assert config.system_prompt == snapshot["system_prompt"]
        assert config.model_config.max_tokens == snapshot["model_config"]["max_tokens"]
        assert config.model_config.temperature == 0.2
        assert run.config_snapshot["model_config"]["temperature"] == 0.7
        assert (
            orchestrator._plugins[0].max_iterations
            == snapshot["pattern_config"]["pattern_params"]["reason_act"]["max_iterations"]
        )
        assert config.skills[0]["name"] == snapshot["skill_config"]["skill_ids"][0]
        assert config.memory_config_data == snapshot["memory_config"]
        envelope = parse_envelope(result.output)
        assert envelope is not None
        field = snapshot["output_contract"]["fields"][0]["name"]
        assert envelope.payload == {field: 1 if field == "a" else 2}
        effective = effective_agent_config(run)
        assert effective.description == snapshot["description"]
        assert effective.budget_limit_usd == snapshot["budget_limit_usd"]
        assert effective.max_run_duration_seconds == snapshot["max_run_duration_seconds"]
    assert all(call.kwargs["owner_id"] == agent.owner_id for call in resolver.call_args_list)


@pytest.mark.parametrize("snapshot", [None, {}, {"goal": ""}, AgentSnapshotUpdate(goal="")])
async def test_empty_goal_agent_created_through_runner_executes(
    storage: Any, execution: Any, snapshot: dict[str, Any] | AgentUpdate | None
) -> None:
    agent = await runner.create_agent(name="canvas-agent", goal="")
    run = await runner.create_run(agent_id=agent.id, user_input="question", config_snapshot=snapshot)
    result = await runner.execute_run(run_id=run.id, llm_service=MagicMock())
    assert result.status == "completed"
    assert execution[0][run.id][0].goal == ""
    assert run.config_snapshot["goal"] == ""


@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("field,value", [("name", ""), ("budget_limit_usd", -1), ("max_run_duration_seconds", 0)])
async def test_execution_accepts_stored_values_without_request_only_constraints(
    storage: Any, execution: Any, field: str, value: Any, pinned: bool
) -> None:
    agent, _runs, _db = storage
    setattr(agent, field, value)
    run = await runner.create_run(
        agent_id=agent.id, user_input="question", config_snapshot={field: value} if pinned else None
    )
    effective = effective_agent_config(run)
    assert getattr(effective, field) == value
    assert (await runner.execute_run(run_id=run.id, llm_service=MagicMock())).status == "completed"


@pytest.mark.parametrize("overrides", [False, True])
async def test_unpinned_run_reads_agent_at_execution_and_records_audit(
    storage: Any, execution: Any, overrides: bool
) -> None:
    agent, _runs, _db = storage
    captured, _resolver = execution
    run = await runner.create_run(
        agent_id=agent.id,
        user_input="question",
        pattern_overrides={"execution_pattern": "reason_act", "pattern_params": {"reason_act": {"max_iterations": 2}}}
        if overrides
        else None,
        model_config_overrides={"max_tokens": 123} if overrides else None,
    )
    assert run.config_snapshot is None
    agent.system_prompt = "latest prompt"
    await runner.execute_run(run_id=run.id, llm_service=MagicMock())
    config, orchestrator = captured[run.id]
    assert config.system_prompt == "latest prompt"
    assert config.goal == agent.goal
    assert config.model_config.max_tokens == (123 if overrides else 900)
    assert orchestrator._plugins[0].max_iterations == (2 if overrides else 9)
    assert config.skills[0]["name"] == agent.skill_config["skill_ids"][0]
    assert run.config_snapshot["system_prompt"] == "latest prompt"
    assert run.config_snapshot["model_config"]["max_tokens"] == (123 if overrides else 900)
    assert parse_envelope(run.output).payload == {"original": 3}


async def test_partial_snapshot_freezes_omitted_fields_and_explicit_nulls(storage: Any, execution: Any) -> None:
    agent, _runs, _db = storage
    run = await runner.create_run(
        agent_id=agent.id,
        user_input="question",
        config_snapshot=AgentUpdate(
            system_prompt="",
            pattern_config=None,
            skill_config=None,
            output_contract=None,
            budget_limit_usd=None,
            max_run_duration_seconds=None,
        ),
    )
    agent.goal = "changed"
    effective = effective_agent_config(run)
    assert effective.goal == "original goal"
    assert effective.system_prompt == ""
    assert effective.description == "original description"
    assert effective.pattern_config is None
    assert effective.skill_config is None
    assert effective.output_contract is None
    assert effective.budget_limit_usd is None
    assert effective.max_run_duration_seconds is None
    await runner.execute_run(run_id=run.id, llm_service=MagicMock())
    config, _orchestrator = execution[0][run.id]
    assert config.system_prompt == "You are shared. original description"
    assert config.skills == []
    assert parse_envelope(run.output).payload is None


@pytest.mark.parametrize(
    "snapshot",
    [
        {"goal": None},
        {"name": None},
        {"goal": 123},
        {"name": 123},
        {"description": 123},
        {"model_config": {"max_tokens": 0}},
        {"skill_config": {"skill_ids": ["not-a-uuid"]}},
        {"budget_limit_usd": "not-a-number"},
        {"max_run_duration_seconds": "not-an-integer"},
        {"pattern_config": {"execution_pattern": "does-not-exist"}},
        {
            "pattern_config": {
                "execution_pattern": "reason_act",
                "pattern_params": {"reason_act": {"max_iterations": 0}},
            }
        },
    ],
)
async def test_invalid_snapshot_never_creates_run(storage: Any, snapshot: dict[str, Any]) -> None:
    agent, runs, _db = storage
    with pytest.raises((ValidationError, ValueError)):
        await runner.create_run(agent_id=agent.id, user_input="question", config_snapshot=snapshot)
    assert not runs


async def test_snapshot_does_not_store_model_credentials(storage: Any) -> None:
    agent, _runs, _db = storage
    run = await runner.create_run(
        agent_id=agent.id,
        user_input="question",
        config_snapshot={"model_config": {"api_key": "secret", "fallback_models": [{"api_key": "fallback-secret"}]}},
    )
    assert "api_key" not in run.config_snapshot["model_config"]
    assert "api_key" not in run.config_snapshot["model_config"]["fallback_models"][0]


async def test_model_overrides_apply_on_top_without_replacing_pinned_base(storage: Any) -> None:
    agent, _runs, _db = storage
    run = await runner.create_run(
        agent_id=agent.id,
        user_input="question",
        config_snapshot={"model_config": {"max_tokens": 300}},
        model_config_overrides={"max_tokens": 400},
    )
    assert run.config_snapshot["model_config"]["max_tokens"] == 300
    assert effective_agent_config(run).model_config_data.max_tokens == 400
    run.model_config_overrides = None
    assert effective_agent_config(run).model_config_data.max_tokens == 300


def test_existing_partial_snapshot_inherits_only_absent_fields(storage: Any) -> None:
    agent, _runs, _db = storage
    run = AgentRun(agent=agent, config_snapshot={"system_prompt": "pinned", "output_contract": None})
    effective = effective_agent_config(run)
    assert effective.system_prompt == "pinned"
    assert effective.goal == agent.goal
    assert effective.output_contract is None


async def test_reason_act_state_preserves_input_config_and_caller_metadata(storage: Any) -> None:
    agent, runs, db = storage
    run = await runner.create_run(
        agent_id=agent.id,
        user_input="question",
        pattern_overrides={"execution_pattern": "reason_act"},
        metadata={"workspace_id": "workspace", "experiment": "cell"},
    )
    context = RunContext(
        run_id=run.id, agent_goal="goal", system_prompt="prompt", model_config=ModelConfig(), user_input="question"
    )
    context.metadata["_runtime"] = MagicMock(_db=db)
    plugin = ReasonActPlugin()
    plugin.configure({"max_iterations": 3})
    await plugin._persist_state(context)
    stored = runs[run.id]
    assert stored.pattern_overrides == {"execution_pattern": "reason_act"}
    assert stored.run_metadata["experiment"] == "cell"
    assert stored.run_metadata["workspace_id"] == "workspace"
    assert stored.run_metadata["reason_act_state"]["max_iterations"] == 3
