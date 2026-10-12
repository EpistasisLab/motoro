"""Run settings distinguish persisted execution data from create requests."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from motoro.schemas.agent import AgentUpdate, MemoryConfig, ModelConfig, SkillConfig
from motoro.schemas.pattern import PatternConfig


class AgentExecutionConfig(BaseModel):
    """Resolved settings for execution and audit snapshots.

    String and limit fields reflect stored Agent values, which the public
    runner allows without the AgentCreate request constraints. Structured
    model, memory, skill and pattern settings retain their typed schemas.
    """

    model_config = ConfigDict(populate_by_name=True)

    name: str = ""
    goal: str = ""
    description: str = ""
    system_prompt: str = ""
    model_config_data: ModelConfig = Field(default_factory=ModelConfig, alias="model_config")
    memory_config: MemoryConfig = Field(default_factory=MemoryConfig)
    pattern_config: PatternConfig | None = None
    skill_config: SkillConfig | None = None
    output_contract: dict[str, Any] | None = None
    budget_limit_usd: float | None = None
    max_run_duration_seconds: int | None = None


class AgentSnapshotUpdate(AgentUpdate):
    """Partial snapshot input, accepting strings and limits allowed by storage.

    Other input validation remains the same as AgentUpdate. Explicit nulls
    for required strings are rejected when the full execution config is built.
    """

    name: str | None = None
    goal: str | None = None
    budget_limit_usd: float | None = None
    max_run_duration_seconds: int | None = None
