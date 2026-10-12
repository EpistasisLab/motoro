"""Resolve execution settings without changing a shared Agent row."""

from copy import deepcopy
from typing import Any

from motoro.models.run import AgentRun
from motoro.schemas.agent import AgentCreate, ModelConfig, PartialModelConfig

# Public snapshot keys use the same aliases as AgentCreate/AgentUpdate.
_AGENT_FIELDS = {
    "name": "name",
    "goal": "goal",
    "description": "description",
    "system_prompt": "system_prompt",
    "model_config": "model_config_data",
    "pattern_config": "pattern_config",
    "skill_config": "skill_config",
    "output_contract": "output_contract",
    "memory_config": "memory_config_data",
    "budget_limit_usd": "budget_limit_usd",
    "max_run_duration_seconds": "max_run_duration_seconds",
}


def effective_agent_config(run: AgentRun, *, apply_run_overrides: bool = True) -> AgentCreate:
    """Pinned fields win, including explicit nulls; absent fields inherit.

    Legacy pattern overrides apply only when the snapshot has no pattern key.
    Model overrides always apply on top. Identity and ownership remain on the
    Agent/AgentRun rows. Budget and duration are carried for product enforcement;
    core does not enforce these limits itself.

    Creation uses ``apply_run_overrides=False`` to capture the pinned base
    settings separately from the run's legacy override channels.
    """
    snapshot = run.config_snapshot or {}
    data = {
        key: deepcopy(snapshot[key] if key in snapshot else getattr(run.agent, attr))
        for key, attr in _AGENT_FIELDS.items()
    }
    if apply_run_overrides and "pattern_config" not in snapshot and run.pattern_overrides:
        data["pattern_config"] = deepcopy(run.pattern_overrides)
    overrides = (run.model_config_overrides or {}) if apply_run_overrides else {}
    model_data = {**(data["model_config"] or {}), **overrides}
    data["model_config"] = PartialModelConfig.model_validate(
        ModelConfig.model_validate(model_data), from_attributes=True
    )
    data["memory_config"] = data["memory_config"] or {}
    return AgentCreate.model_validate(data)


def snapshot_agent_config(config: AgentCreate) -> dict[str, Any]:
    """Serialize execution fields only; schema serialization excludes secrets."""
    data = config.model_dump(mode="json", by_alias=True)
    return {key: data[key] for key in _AGENT_FIELDS}
