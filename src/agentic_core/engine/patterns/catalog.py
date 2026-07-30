"""Pattern metadata read from the registry rather than the database.

Core needs pattern metadata for two things, and both happen in-process where the
registry is already loaded:

1. **Validation** — reject a bad ``pattern_config`` when an agent is created,
   rather than at run time when the model has already been billed.
2. **Parameter defaults** — fill ``pattern_params`` from each pattern's
   ``configuration_schema`` before ``plugin.configure()`` runs.

ARES does both by querying ``architectural_patterns``. That has a failure mode
worth avoiding: with an unseeded table, *every* agent creation fails with
"Unknown pattern slug", so correctness depends on a data migration having run.
Reading the registry, both work the instant a plugin is importable.

The table is still written — see :func:`agentic_core.services.pattern_catalog.
sync_pattern_catalog` — but as a projection of these values for products that
need to query the catalog. Nothing here reads it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentic_core.engine.patterns.registry import PluginRegistry

if TYPE_CHECKING:
    from agentic_core.engine.patterns.base import PatternPlugin
    from agentic_core.schemas.pattern import (
        PatternConfig,
        PatternConfigValidationResult,
    )


class PatternConfigError(ValueError):
    """A ``pattern_config`` that references or configures patterns incorrectly.

    A ``ValueError`` subclass so a product that does not care about the
    distinction can catch the broad type, and one that maps errors to responses
    (a 422, say) can catch this specifically.
    """

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.errors = errors or []


def display_name_for(plugin_cls: type[PatternPlugin]) -> str:
    """The plugin's display name, or one derived from its slug."""
    return plugin_cls.display_name or plugin_cls.slug.replace("_", " ").title()


def registry_metadata(slugs: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """Build the metadata mapping ``validate_composition`` expects.

    This is the in-process replacement for ARES's ``get_patterns_by_slugs``: same
    shape, no session. Unknown slugs are simply absent, which is how the caller
    detects them.
    """
    PluginRegistry.discover()
    available = PluginRegistry.all()
    wanted = list(available) if slugs is None else slugs
    return {
        slug: {
            "category": str(available[slug].category),
            "dependencies": list(available[slug].dependencies),
            "requires_multi_agent": available[slug].requires_multi_agent,
        }
        for slug in wanted
        if slug in available
    }


def schema_defaults(schema: dict[str, Any]) -> dict[str, Any]:
    """Extract ``{property: default}`` from a JSON Schema object.

    Properties listed as ``required`` are skipped: they have no implicit default,
    and supplying one would paper over config a caller was obliged to provide.
    """
    if not isinstance(schema, dict):
        return {}
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    return {
        name: prop["default"]
        for name, prop in properties.items()
        if name not in required and isinstance(prop, dict) and "default" in prop
    }


def merge_schema_defaults(plugin_cls: type[PatternPlugin], params: dict[str, Any]) -> dict[str, Any]:
    """Return *params* with the plugin's schema defaults filled in underneath.

    Caller-supplied values always win. A plugin with no
    ``configuration_schema`` gets its params unchanged, so its own
    ``params.get(key, fallback)`` still applies — which is what makes this
    additive rather than a breaking change for patterns that have not declared a
    schema yet.
    """
    defaults = schema_defaults(plugin_cls.configuration_schema)
    if not defaults:
        return dict(params)
    merged = defaults | dict(params)
    return merged


def validate_pattern_config(config: PatternConfig | dict[str, Any] | None) -> PatternConfigValidationResult:
    """Validate a pattern config against the registry.

    Checks, in the order a caller most wants to hear about them:

    1. Every referenced slug is a registered pattern.
    2. Per-pattern params satisfy the plugin's own ``validate_config``.
    3. Declared ``dependencies`` are satisfiable — either co-active, or
       auto-activatable into an empty singleton slot the way
       ``PatternOrchestrator.from_pattern_config`` would.
    4. Patterns requiring a multi-agent role have one in ``pattern_params``.
    5. ``pattern_params`` keys all name known slugs, so a typo in a params key
       is not silently ignored.
    """
    from agentic_core.engine.patterns import composition
    from agentic_core.schemas.pattern import (
        PatternConfig,
        PatternConfigValidationError,
        PatternConfigValidationResult,
    )

    if config is None:
        return PatternConfigValidationResult(valid=True)
    cfg = config if isinstance(config, PatternConfig) else PatternConfig.model_validate(config)

    active_slugs = cfg.all_active_slugs()
    if not active_slugs and not cfg.pattern_params:
        return PatternConfigValidationResult(valid=True)

    PluginRegistry.discover()
    available = PluginRegistry.all()
    errors: list[PatternConfigValidationError] = []

    # 1. Unknown slugs. Stop here if any are found — every later check reads
    #    metadata off a plugin class, so they would all report the same absence.
    for slug in list(dict.fromkeys([*active_slugs, *cfg.pattern_params])):
        if slug not in available:
            known = ", ".join(sorted(available)) or "none registered"
            errors.append(
                PatternConfigValidationError(
                    field=_field_for_slug(cfg, slug),
                    message=f"Unknown pattern slug '{slug}'. Registered patterns: {known}.",
                )
            )
    if errors:
        return PatternConfigValidationResult(valid=False, errors=errors)

    # 2. Per-pattern params, via each plugin's own validator.
    for slug in active_slugs:
        plugin_cls = available[slug]
        params = merge_schema_defaults(plugin_cls, cfg.pattern_params.get(slug, {}))
        for message in plugin_cls().validate_config(params):
            errors.append(PatternConfigValidationError(field=f"pattern_params.{slug}", message=message))

    # 3. Dependencies and conflicts, through the same composition engine the
    #    orchestrator uses — so validation and execution cannot disagree.
    #
    #    Pass metadata for *every* registered pattern, not just the active ones.
    #    The dependency checks skip a dependency in the same singleton category as
    #    its dependent (``reason_act`` needs ``single_agent_baseline``, and both
    #    are ``execution``, so it cannot and need not be co-active) — and that
    #    skip can only fire if the dependency's own category is known. Fetching
    #    the full set is free here; ARES pays a query per slug and so passes only
    #    the active ones, which is why its equivalent check reports a dependency
    #    it should skip.
    report = composition.validate_composition(active_slugs, registry_metadata())
    errors.extend(PatternConfigValidationError(field="pattern_config", message=message) for message in report.errors)

    # 4. A multi-agent pattern with no role assigned would activate and do
    #    nothing useful.
    for slug in active_slugs:
        if available[slug].requires_multi_agent and not cfg.pattern_params.get(slug, {}).get("role"):
            errors.append(
                PatternConfigValidationError(
                    field=f"pattern_params.{slug}.role",
                    message=f"Pattern '{slug}' requires a multi-agent role in pattern_params['{slug}']['role'].",
                )
            )

    # 5. Params for a pattern that is not active are a silent no-op otherwise.
    for slug in cfg.pattern_params:
        if slug not in set(active_slugs):
            errors.append(
                PatternConfigValidationError(
                    field=f"pattern_params.{slug}",
                    message=f"pattern_params names '{slug}', which is not active in this config.",
                )
            )

    return PatternConfigValidationResult(valid=not errors, errors=errors)


def _field_for_slug(cfg: PatternConfig, slug: str) -> str:
    """Name the config field a slug came from, so an error points somewhere."""
    singletons = ("execution_pattern", "coordination_pattern", "routing_pattern")
    for field in singletons:
        if getattr(cfg, field, None) == slug:
            return field
    lists = ("safety_patterns", "knowledge_patterns", "quality_patterns", "resolution_patterns")
    for field in lists:
        if slug in getattr(cfg, field, []):
            return field
    return f"pattern_params.{slug}"
