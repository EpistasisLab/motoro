"""Pydantic schemas for architectural patterns and agent pattern configuration."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pattern registry schemas (Issue #57)
# ---------------------------------------------------------------------------


class PatternResponse(BaseModel):
    """Full detail view of a single architectural pattern."""

    id: uuid.UUID
    slug: str
    name: str
    category: str
    description: str
    phase: str
    configuration_schema: dict[str, Any]
    requires_multi_agent: bool
    dependencies: list[str]
    version: str
    is_implemented: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class PatternListResponse(BaseModel):
    """Paginated list of architectural patterns."""

    items: list[PatternResponse]
    total: int
    offset: int
    limit: int


# ---------------------------------------------------------------------------
# Agent pattern configuration schemas (Issue #58)
# ---------------------------------------------------------------------------


class PatternConfig(BaseModel):
    """Selects which architectural patterns are active for an agent and their params."""

    execution_pattern: str | None = Field(
        default=None,
        description="Slug of the execution pattern (e.g. 'react', 'reflexion'). At most one.",
    )
    safety_patterns: list[str] = Field(
        default_factory=list,
        description="Slugs of active safety patterns (e.g. ['watchdog_timeout_supervisor']).",
    )
    coordination_pattern: str | None = Field(
        default=None,
        description="Slug of the coordination pattern (e.g. 'supervisor_architecture').",
    )
    knowledge_patterns: list[str] = Field(
        default_factory=list,
        description="Slugs of active knowledge patterns (e.g. ['simple_rag']).",
    )
    quality_patterns: list[str] = Field(
        default_factory=list,
        description="Slugs of active quality patterns (e.g. ['fractal_cot']).",
    )
    routing_pattern: str | None = Field(
        default=None,
        description="Slug of the active routing pattern (e.g. 'agent_router').",
    )
    resolution_patterns: list[str] = Field(
        default_factory=list,
        description="Slugs of active resolution patterns (e.g. ['consensus']).",
    )
    pattern_params: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Per-pattern parameter overrides keyed by pattern slug. Each value is a dict of parameter name → value."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_coordination_role(cls, data: Any) -> Any:
        """Backward compat: move top-level coordination_role into pattern_params."""
        if isinstance(data, dict):
            role = data.pop("coordination_role", None)
            if role is not None:
                logger.warning(
                    "Deprecated: coordination_role should be in "
                    "pattern_params[coordination_pattern]['role'], not top-level"
                )
                coord = data.get("coordination_pattern")
                if coord:
                    params = data.setdefault("pattern_params", {})
                    coord_params = params.setdefault(coord, {})
                    coord_params.setdefault("role", role)
        return data

    @property
    def coordination_role(self) -> str | None:
        """Read the coordination role from pattern_params for the active coordination pattern."""
        if self.coordination_pattern:
            params = self.pattern_params.get(self.coordination_pattern, {})
            role = params.get("role")
            if isinstance(role, str):
                return role
        return None

    def all_active_slugs(self) -> list[str]:
        """Return every slug currently referenced in this config."""
        slugs: list[str] = []
        if self.execution_pattern:
            slugs.append(self.execution_pattern)
        slugs.extend(self.safety_patterns)
        if self.coordination_pattern:
            slugs.append(self.coordination_pattern)
        slugs.extend(self.knowledge_patterns)
        slugs.extend(self.quality_patterns)
        if self.routing_pattern:
            slugs.append(self.routing_pattern)
        slugs.extend(self.resolution_patterns)
        return slugs


class PatternConfigValidationError(BaseModel):
    """A single validation error from PatternConfig validation."""

    field: str
    message: str


class PatternConfigValidationResult(BaseModel):
    """Result of validating a PatternConfig against the pattern registry."""

    valid: bool
    errors: list[PatternConfigValidationError] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Composition validation schemas (Issue #61)
# ---------------------------------------------------------------------------


class ResolvedPatternInfo(BaseModel):
    """Lightweight resolved view of a pattern — used in agent pattern config responses."""

    slug: str
    name: str
    category: str
    description: str
    is_implemented: bool

    model_config = {"from_attributes": True}


class AgentPatternConfigResponse(BaseModel):
    """Response for GET /api/agents/{id}/patterns.

    Returns the agent's current PatternConfig alongside resolved details
    (name, category, description) for every active slug.
    """

    config: PatternConfig
    resolved: dict[str, ResolvedPatternInfo] = Field(
        default_factory=dict,
        description="Slug → pattern details for every slug referenced in config.",
    )


class PatternConfigPatch(BaseModel):
    """Body for PATCH /api/agents/{id}/patterns.

    All fields are optional. Scalars replace the current value when present
    (send ``None`` to unset). List operations add or remove individual slugs
    without replacing the full list.
    """

    # Scalar field overrides — only applied when the field is explicitly included
    execution_pattern: str | None = Field(
        default=None,
        description="New execution pattern slug, or null to unset.",
    )
    set_execution_pattern: bool = Field(
        default=False,
        description=(
            "Set to true when sending execution_pattern=null to explicitly unset, "
            "rather than treating null as 'no change'."
        ),
    )
    coordination_pattern: str | None = Field(
        default=None, description="New coordination pattern slug, or null to unset."
    )
    set_coordination_pattern: bool = Field(default=False)
    routing_pattern: str | None = Field(default=None, description="New routing pattern slug, or null to unset.")
    set_routing_pattern: bool = Field(default=False)

    # List add/remove operations
    add_safety_patterns: list[str] = Field(default_factory=list)
    remove_safety_patterns: list[str] = Field(default_factory=list)
    add_knowledge_patterns: list[str] = Field(default_factory=list)
    remove_knowledge_patterns: list[str] = Field(default_factory=list)
    add_quality_patterns: list[str] = Field(default_factory=list)
    remove_quality_patterns: list[str] = Field(default_factory=list)
    add_resolution_patterns: list[str] = Field(default_factory=list)
    remove_resolution_patterns: list[str] = Field(default_factory=list)

    # Pattern parameter updates
    update_pattern_params: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Slug-keyed params to merge into existing pattern_params.",
    )
    remove_pattern_params: list[str] = Field(
        default_factory=list,
        description="Slugs whose params should be removed.",
    )

    def apply_to(self, current: PatternConfig) -> PatternConfig:
        """Return a new PatternConfig with this patch applied."""

        def _list_op(existing: list[str], add: list[str], remove: list[str]) -> list[str]:
            seen: set[str] = set()
            result: list[str] = []
            for item in existing + add:
                if item not in seen and item not in remove:
                    seen.add(item)
                    result.append(item)
            return result

        new_params = dict(current.pattern_params)
        for key in self.remove_pattern_params:
            new_params.pop(key, None)
        new_params.update(self.update_pattern_params)

        return PatternConfig(
            execution_pattern=(
                self.execution_pattern
                if (self.execution_pattern is not None or self.set_execution_pattern)
                else current.execution_pattern
            ),
            safety_patterns=_list_op(current.safety_patterns, self.add_safety_patterns, self.remove_safety_patterns),
            coordination_pattern=(
                self.coordination_pattern
                if (self.coordination_pattern is not None or self.set_coordination_pattern)
                else current.coordination_pattern
            ),
            knowledge_patterns=_list_op(
                current.knowledge_patterns,
                self.add_knowledge_patterns,
                self.remove_knowledge_patterns,
            ),
            quality_patterns=_list_op(
                current.quality_patterns,
                self.add_quality_patterns,
                self.remove_quality_patterns,
            ),
            routing_pattern=(
                self.routing_pattern
                if (self.routing_pattern is not None or self.set_routing_pattern)
                else current.routing_pattern
            ),
            resolution_patterns=_list_op(
                current.resolution_patterns,
                self.add_resolution_patterns,
                self.remove_resolution_patterns,
            ),
            pattern_params=new_params,
        )


class CompositionValidationResponse(BaseModel):
    """Response from POST /api/patterns/validate."""

    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    resolved_hooks: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Hook point → ordered list of plugin slugs that run there.",
    )
