"""Pydantic schemas for Agent API."""

import enum
import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from motoro.schemas.pattern import PatternConfig


class LLMProvider(enum.StrEnum):
    """Supported LLM providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    AZURE_FOUNDRY = "azure_foundry"
    BEDROCK = "bedrock"
    LOCAL = "local"


class MemoryConfig(BaseModel):
    """Agent memory configuration."""

    model_config = ConfigDict(extra="allow")

    episodic_memory_enabled: bool = False


class ModelConfig(BaseModel):
    """LLM model configuration."""

    provider: LLMProvider = LLMProvider.ANTHROPIC
    model: str = "claude-sonnet-5"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = Field(
        default=None,
        description=(
            "Reasoning effort for adaptive-thinking models that reject temperature "
            "(Opus 4.7/4.8, Fable 5). Mapped to Anthropic's output_config.effort. "
            "Ignored for models that use temperature; see motoro.services.model_capabilities."
        ),
    )
    max_tokens: int = Field(default=4096, gt=0, le=200000)
    api_base: str | None = Field(default=None, description="Custom API base URL (e.g., Foundry)")
    api_key: str | None = Field(
        default=None, exclude=True, description="Custom API key (never serialized in responses)"
    )
    fallback_models: list["ModelConfig"] = Field(
        default_factory=list,
        description=(
            "Ordered list of fallback model configurations to try on terminal errors. "
            "Tried in order after the primary model fails with a non-retryable error."
        ),
    )


class PartialModelConfig(BaseModel):
    """Boundary schema where ``provider``/``model`` may be omitted to mean "inherit".

    Stored agent configs and per-run overrides use this so "the user explicitly
    chose provider X" is distinguishable from "the user said nothing, use my
    default provider" (M112). Leaving ``provider``/``model`` as ``None`` means
    *inherit the user's default LLM setting at run time*.

    The engine never sees this type: ``resolve_model_config_for_user`` always
    turns it into a concrete :class:`ModelConfig` (with provider/model filled)
    before execution. All other fields mirror :class:`ModelConfig` so an explicit
    temperature / effort / max_tokens still flows through.
    """

    provider: LLMProvider | None = None
    model: str | None = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = Field(
        default=None,
        description="Reasoning effort for adaptive-thinking models; see ModelConfig.effort.",
    )
    max_tokens: int = Field(default=4096, gt=0, le=200000)
    api_base: str | None = Field(default=None, description="Custom API base URL (e.g., Foundry)")
    api_key: str | None = Field(
        default=None, exclude=True, description="Custom API key (never serialized in responses)"
    )
    fallback_models: list["ModelConfig"] = Field(default_factory=list)


class AgentCreate(BaseModel):
    """Request schema for creating an agent."""

    name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(default="")
    goal: str = Field(..., min_length=1)
    system_prompt: str = Field(default="")
    model_config_data: PartialModelConfig = Field(default_factory=PartialModelConfig, alias="model_config")
    tool_config: dict[str, object] = Field(default_factory=dict)
    memory_config: MemoryConfig = Field(default_factory=MemoryConfig)
    budget_limit_usd: float | None = Field(
        default=None,
        description="Optional cost budget in USD; a warning is logged when exceeded",
        ge=0.0,
    )
    max_run_duration_seconds: int | None = Field(
        default=None,
        description="Optional per-agent timeout in seconds; overrides global worker_job_timeout",
        gt=0,
    )
    pattern_config: PatternConfig | None = Field(
        default=None,
        description="Architectural patterns to activate for this agent.",
    )
    output_contract: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional field-spec ({'name', 'fields': [...]}) used to extract a "
            "structured payload into the run output envelope. None = envelope only."
        ),
    )
    auto_eval_enabled: bool = Field(default=True, description="Auto-evaluate runs for this agent")
    auto_eval_model: str | None = Field(default=None, description="Override eval model for this agent")

    model_config = ConfigDict(populate_by_name=True)


class AgentUpdate(BaseModel):
    """Request schema for updating an agent. All fields optional."""

    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = None
    goal: str | None = Field(default=None, min_length=1)
    system_prompt: str | None = None
    model_config_data: PartialModelConfig | None = Field(default=None, alias="model_config")
    tool_config: dict[str, object] | None = None
    memory_config: MemoryConfig | None = None
    budget_limit_usd: float | None = Field(default=None, ge=0.0)
    max_run_duration_seconds: int | None = Field(default=None, gt=0)
    pattern_config: PatternConfig | None = Field(
        default=None,
        description="Update the agent's active architectural patterns.",
    )
    output_contract: dict[str, Any] | None = Field(
        default=None,
        description="Update the agent's output contract (field-spec for payload extraction).",
    )
    auto_eval_enabled: bool | None = None
    auto_eval_model: str | None = None

    model_config = ConfigDict(populate_by_name=True)


class AgentResponse(BaseModel):
    """Response schema for a single agent."""

    id: uuid.UUID
    name: str
    description: str
    goal: str
    system_prompt: str
    model_config_data: PartialModelConfig = Field(serialization_alias="model_config")
    tool_config_data: dict[str, object] = Field(serialization_alias="tool_config")
    memory_config_data: MemoryConfig = Field(serialization_alias="memory_config")
    is_system: bool = False
    budget_limit_usd: float | None = None
    max_run_duration_seconds: int | None = None
    pattern_config: PatternConfig | None = None
    output_contract: dict[str, Any] | None = None
    auto_eval_enabled: bool = True
    auto_eval_model: str | None = None
    source_plan_id: uuid.UUID | None = None
    source_plan_title: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class AgentListResponse(BaseModel):
    """Response schema for paginated agent list."""

    items: list[AgentResponse]
    total: int
    offset: int
    limit: int
