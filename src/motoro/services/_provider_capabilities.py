"""Interpret provider responses without model-name heuristics or static maps."""

from typing import Any

from motoro.services.model_capabilities import ModelCapabilities


def _capabilities_from_anthropic(caps: dict[str, Any], model: str = "") -> ModelCapabilities:
    """Read capability nodes dynamically, including future effort names.

    The published schema exposes effort support/levels, but no sampling defaults
    or temperature facts. Optional fields below are used only if actually sent
    by the provider; their absence must never imply a default.
    """
    effort = caps.get("effort") or {}
    support = effort.get("supported")
    nodes = {name: node for name, node in effort.items() if isinstance(node, dict) and "supported" in node}
    levels = [name for name, node in nodes.items() if node["supported"] is True]
    # The tree advertises independent positive/negative facts, without an
    # exhaustive-list marker. Missing or null future nodes remain unknown.
    levels_or_unknown = levels if support is not False and levels else [] if support is False else None
    default = effort.get("default_effort")
    if support is not True or (levels_or_unknown is not None and default not in levels_or_unknown):
        default = None
    temperature = caps.get("temperature") or {}
    return ModelCapabilities(
        supports_effort=support,
        effort_levels=levels_or_unknown,
        effort_levels_complete=False,
        unsupported_effort_levels=[name for name, node in nodes.items() if node["supported"] is False] or None,
        default_effort=default,
        supports_temperature=temperature.get("supported"),
        default_temperature=temperature.get("default") if temperature.get("supported") is True else None,
    )


def _capabilities_from_openrouter(entry: dict[str, Any]) -> ModelCapabilities:
    """Read independent parameter support, reasoning options, and defaults."""
    parameters = entry.get("supported_parameters")
    reasoning = entry.get("reasoning")
    support = None
    levels = None
    if isinstance(reasoning, dict):
        if "supported_efforts" in reasoning:
            levels = reasoning["supported_efforts"]
            # null means all gateway efforts, without hardcoding that vocabulary.
            support = levels != []
        elif reasoning.get("default_effort") is not None:
            support = None  # A default alone does not establish a full ladder.
        else:
            support = False  # Present reasoning node without an effort selector.
        if reasoning.get("mandatory") and levels is not None:
            levels = [level for level in levels if level != "none"]
            if not levels:
                support = False
    elif isinstance(parameters, list):
        support = "reasoning_effort" in parameters if "reasoning" not in parameters else None
    reasoning = reasoning or {}
    default = reasoning.get("default_effort")
    if (
        support is not True
        or (levels is not None and default not in levels)
        or reasoning.get("default_enabled") is False
    ):
        default = None
    temperature_support = "temperature" in parameters if isinstance(parameters, list) else None
    return ModelCapabilities(
        supports_temperature=temperature_support,
        default_temperature=(entry.get("default_parameters") or {}).get("temperature")
        if temperature_support is True
        else None,
        supports_effort=support,
        effort_levels=levels,
        default_effort=default,
        unsupported_effort_levels=["none"] if reasoning.get("mandatory") else None,
    )
