"""Validated sampling facts and explicit caller overrides; no model registry."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelCapabilities(BaseModel):
    """What a model accepts, independent of configuration. None means unknown.

    Discovery wraps these values in ModelInfo to retain provenance. Passing this
    object directly to sampling is an explicit caller override. An absent
    combination allowlist imposes no restriction; it does not prove compatibility.
    """

    model_config = ConfigDict(extra="forbid", revalidate_instances="always")

    supports_temperature: bool | None = None
    supports_effort: bool | None = None
    default_temperature: float | None = Field(default=None, ge=0.0, le=2.0, allow_inf_nan=False)
    effort_levels: list[str] | None = Field(default=None, description="Accepted levels; None means unknown.")
    effort_levels_complete: bool = Field(
        default=True,
        description="Whether effort_levels is exhaustive; false allows unlisted values.",
    )
    unsupported_effort_levels: list[str] | None = Field(
        default=None,
        description="Explicit negative facts, useful when the accepted ladder is incomplete.",
    )
    default_effort: str | None = Field(
        default=None,
        description="Advertised provider default; None means unadvertised, 'none' disables reasoning.",
    )
    temperature_effort_levels: list[str] | None = Field(
        default=None,
        description="When supplied, temperature requires resolved effort in this allowlist. "
        "None imposes no additional restriction; [] forbids temperature in every effort mode.",
    )

    @model_validator(mode="after")
    def validate_effort_metadata(self) -> ModelCapabilities:
        for name, levels in (
            ("effort_levels", self.effort_levels),
            ("temperature_effort_levels", self.temperature_effort_levels),
            ("unsupported_effort_levels", self.unsupported_effort_levels),
        ):
            if levels is not None:
                if any(not level.strip() or level != level.strip() for level in levels):
                    raise ValueError(f"{name} must contain nonblank, unpadded values")
                if len(set(levels)) != len(levels):
                    raise ValueError(f"{name} must contain unique values")
        if self.default_effort is not None:
            if not self.default_effort.strip() or self.default_effort != self.default_effort.strip():
                raise ValueError("default_effort must be nonblank and unpadded")
            if (
                self.effort_levels_complete
                and self.effort_levels is not None
                and self.default_effort not in self.effort_levels
            ):
                raise ValueError("default_effort must be one of effort_levels")
        if self.supports_effort is False and (self.effort_levels or self.default_effort is not None):
            raise ValueError("Effort metadata contradicts unsupported effort")
        if self.unsupported_effort_levels:
            if set(self.unsupported_effort_levels) & set(self.effort_levels or []):
                raise ValueError("Accepted and unsupported efforts must not overlap")
            if self.default_effort in self.unsupported_effort_levels:
                raise ValueError("default_effort must not be unsupported")
        if self.supports_effort is True and self.effort_levels == []:
            raise ValueError("Use effort_levels=None for unknown accepted levels")
        if self.default_temperature is not None and self.supports_temperature is False:
            raise ValueError("default_temperature contradicts unsupported temperature")
        if self.temperature_effort_levels is not None:
            if self.supports_temperature is False or self.supports_effort is False:
                raise ValueError("Combination restrictions contradict unsupported controls")
            if (
                self.effort_levels_complete
                and self.effort_levels is not None
                and not set(self.temperature_effort_levels) <= set(self.effort_levels)
            ):
                raise ValueError("temperature_effort_levels must be a subset of effort_levels")
        return self


class ModelCatalogEntry(BaseModel):
    """Deprecated compatibility type; discovery returns provider ModelInfo values."""

    provider: str
    model: str
    label: str
    capabilities: ModelCapabilities


# Caller overrides and old imports remain supported, without model-specific facts.
TEMPERATURE_ONLY_CAPABILITIES = ModelCapabilities(
    supports_temperature=True,
    supports_effort=False,
    effort_levels=[],
)
DEFAULT_CAPABILITIES = ModelCapabilities()
CATALOG: list[ModelCatalogEntry] = []


def get_capabilities(model: str, *, default: ModelCapabilities = DEFAULT_CAPABILITIES) -> ModelCapabilities:
    """Compatibility helper: IDs imply no facts; use provider discovery.

    An explicit caller default remains an application override. Return a copy
    so callers cannot mutate shared unknown metadata.
    """
    return default.model_copy(deep=True)


def supports_temperature(model: str) -> bool | None:
    """Unknown without provider metadata; retained for existing imports."""
    return None


def supports_effort(model: str) -> bool | None:
    """Unknown without provider metadata; retained for existing imports."""
    return None
