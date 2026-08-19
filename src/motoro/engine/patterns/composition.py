"""Composition engine — resolves hook ordering, detects conflicts, validates deps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from motoro.engine.patterns.base import HookPoint, PatternPlugin
from motoro.engine.patterns.registry import PluginRegistry
from motoro.models.pattern import PatternCategory

# Priority order: lower number = runs first.
_CATEGORY_PRIORITY: dict[str, int] = {
    PatternCategory.SAFETY: 0,
    PatternCategory.EXECUTION: 1,
    PatternCategory.KNOWLEDGE: 2,
    PatternCategory.QUALITY: 3,
    PatternCategory.ROUTING: 4,
    PatternCategory.RESOLUTION: 5,
    PatternCategory.COORDINATION: 6,
}


@dataclass
class CompositionReport:
    """Result of validating a set of pattern slugs against composition rules."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    resolved_hooks: dict[str, list[str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------

# Categories where at most one pattern may be active.
# ``coordination`` is included (#1037): two coordination plugins almost always
# fight over who owns the loop — e.g. supervisor_architecture and
# supervision_tree_with_guarded_capabilities both claim hierarchy semantics; swarm_architecture and
# contract_net both bind their own act-phase routing.  Relying on every
# coordination plugin to declare every other one in ``conflicts_with`` is
# fragile and was empirically incomplete.
_SINGLETON_CATEGORIES: set[str] = {"execution", "routing", "coordination"}


def detect_conflicts(
    plugins: list[PatternPlugin],
) -> list[str]:
    """Return a list of human-readable conflict errors.

    Rules:
    - At most one execution pattern.
    - At most one routing pattern.
    - At most one coordination pattern (#1037).
    - Plugin-declared ``conflicts_with`` slugs are checked.
    """
    errors: list[str] = []

    # Singleton category check
    by_category: dict[str, list[str]] = {}
    for p in plugins:
        by_category.setdefault(p.category, []).append(p.slug)

    for cat in _SINGLETON_CATEGORIES:
        slugs = by_category.get(cat, [])
        if len(slugs) > 1:
            errors.append(f"At most one {cat} pattern allowed, but found {len(slugs)}: " + ", ".join(sorted(slugs)))

    # Per-plugin declared conflicts
    active_slugs = {p.slug for p in plugins}
    for p in plugins:
        for conflict_slug in p.conflicts_with:
            if conflict_slug in active_slugs:
                errors.append(f"Pattern '{p.slug}' conflicts with '{conflict_slug}' (both are active).")

    return errors


# ---------------------------------------------------------------------------
# Dependency resolution (topological sort with cycle detection)
# ---------------------------------------------------------------------------


def collect_missing_dependencies(
    active_slugs: set[str],
    dep_graph: dict[str, list[str]],
    category_map: dict[str, str] | None = None,
) -> set[str]:
    """Return the set of dependency slugs that are required but not in *active_slugs*.

    Performs a transitive walk of *dep_graph*, skipping same-category
    singleton dependencies (e.g. ``reason_act → single_agent_baseline`` when both
    are ``execution`` category).  The caller can union the result with
    *active_slugs* to auto-resolve all missing deps.

    Issue #1043 — seed dependencies are not loaded into ``pattern_params``
    defaults; this helper lets callers auto-activate them instead of
    returning an opaque "missing dep" error.
    """
    missing: set[str] = set()
    _collect_transitive(active_slugs, active_slugs, dep_graph, missing, set(), category_map)
    return missing


def _collect_transitive(
    roots: set[str],
    active: set[str],
    graph: dict[str, list[str]],
    missing: set[str],
    visited: set[str],
    category_map: dict[str, str] | None = None,
) -> None:
    for slug in roots:
        if slug in visited:
            continue
        visited.add(slug)
        for dep in graph.get(slug, []):
            if dep in active:
                _collect_transitive({dep}, active, graph, missing, visited, category_map)
                continue
            # Skip same-category singleton deps (e.g. reason_act → single_agent_baseline)
            if category_map:
                slug_cat = category_map.get(slug)
                dep_cat = category_map.get(dep)
                if slug_cat and dep_cat and slug_cat == dep_cat:
                    continue
            missing.add(dep)
            # Recurse into transitive deps of the newly-found missing dep
            _collect_transitive({dep}, active | missing, graph, missing, visited, category_map)


def resolve_dependencies(
    active_slugs: set[str],
    dep_graph: dict[str, list[str]],
    category_map: dict[str, str] | None = None,
) -> list[str]:
    """Return a list of errors: missing dependencies and circular dependencies.

    *dep_graph* maps slug → list of dependency slugs (from pattern registry).
    *category_map* maps slug → category string (used to skip same-category
    singleton dependencies, e.g., reason_act depending on single_agent_baseline).
    """
    errors: list[str] = []

    # Missing direct and transitive dependencies
    for slug in active_slugs:
        _check_transitive(slug, active_slugs, dep_graph, errors, set(), category_map)

    # Cycle detection via DFS
    cycle = _detect_cycle(active_slugs, dep_graph)
    if cycle:
        errors.append(f"Circular dependency detected: {' → '.join(cycle)}")

    return errors


def _check_transitive(
    slug: str,
    active: set[str],
    graph: dict[str, list[str]],
    errors: list[str],
    visited: set[str],
    category_map: dict[str, str] | None = None,
) -> None:
    if slug in visited:
        return
    visited.add(slug)
    for dep in graph.get(slug, []):
        if dep not in active:
            # Skip same-category singleton deps (e.g., reason_act → single_agent_baseline)
            if category_map:
                slug_cat = category_map.get(slug)
                dep_cat = category_map.get(dep)
                if slug_cat and dep_cat and slug_cat == dep_cat:
                    continue
            errors.append(f"Pattern '{slug}' requires '{dep}', which is not active.")
        else:
            _check_transitive(dep, active, graph, errors, visited, category_map)


def _detect_cycle(slugs: set[str], graph: dict[str, list[str]]) -> list[str] | None:
    """Return the cycle path if one exists, else None."""
    white, grey, black = 0, 1, 2
    colour: dict[str, int] = {s: white for s in slugs}
    parent: dict[str, str | None] = {s: None for s in slugs}

    def _dfs(node: str) -> list[str] | None:
        colour[node] = grey
        for dep in graph.get(node, []):
            if dep not in colour:
                continue
            if colour[dep] == grey:
                # Reconstruct cycle
                cycle = [dep, node]
                p = parent[node]
                while p is not None and p != dep:
                    cycle.append(p)
                    p = parent[p]
                cycle.append(dep)
                cycle.reverse()
                return cycle
            if colour[dep] == white:
                parent[dep] = node
                result = _dfs(dep)
                if result is not None:
                    return result
        colour[node] = black
        return None

    for s in slugs:
        if colour[s] == white:
            result = _dfs(s)
            if result is not None:
                return result
    return None


# ---------------------------------------------------------------------------
# Hook ordering
# ---------------------------------------------------------------------------


def build_ordered_hooks(
    plugins: list[PatternPlugin],
) -> dict[str, list[str]]:
    """Build the execution-order mapping: HookPoint name → [plugin slugs].

    Plugins are sorted by category priority; within the same priority,
    they appear in the order they were supplied (registration order).
    """
    sorted_plugins = sorted(plugins, key=lambda p: _CATEGORY_PRIORITY.get(p.category, 99))

    hook_order: dict[str, list[str]] = {}
    for point in HookPoint:
        slugs: list[str] = []
        for plugin in sorted_plugins:
            hooks = plugin.get_hooks()
            if point in hooks and hooks[point]:
                slugs.append(plugin.slug)
        if slugs:
            hook_order[point.value] = slugs
    return hook_order


# ---------------------------------------------------------------------------
# Full composition validation
# ---------------------------------------------------------------------------


def validate_composition(
    slugs: list[str],
    registry_meta: dict[str, dict[str, Any]],
) -> CompositionReport:
    """Validate a set of pattern slugs and produce a full composition report.

    *registry_meta* maps slug → pattern metadata dict (from the DB), each
    containing at least ``category``, ``dependencies``, ``requires_multi_agent``.

    Does **not** require a DB session — operates on pre-fetched metadata.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Instantiate plugins where available; fall back to metadata for unregistered
    PluginRegistry.discover()
    plugins: list[PatternPlugin] = []
    active_set = set(slugs)

    for slug in slugs:
        meta = registry_meta.get(slug)
        if meta is None:
            errors.append(f"Unknown pattern slug '{slug}'.")
            continue

        plugin_cls = PluginRegistry.get(slug)
        if plugin_cls is not None:
            instance = plugin_cls()
            instance.configure({})
            plugins.append(instance)

    # Conflict detection
    errors.extend(detect_conflicts(plugins))

    # Dependency resolution
    dep_graph: dict[str, list[str]] = {}
    category_map: dict[str, str] = {}
    for slug in slugs:
        meta = registry_meta.get(slug)
        if meta is not None:
            dep_graph[slug] = meta.get("dependencies", [])
            category_map[slug] = meta.get("category", "")
            # Also map dependency slugs to their categories if available
            for dep in meta.get("dependencies", []):
                dep_meta = registry_meta.get(dep)
                if dep_meta:
                    category_map[dep] = dep_meta.get("category", "")
    errors.extend(resolve_dependencies(active_set, dep_graph, category_map))

    # Hook ordering
    resolved_hooks = build_ordered_hooks(plugins) if not errors else {}

    # Performance warnings (informational)
    if len(plugins) > 5:
        warnings.append(
            f"{len(plugins)} patterns active — hook execution may add measurable latency to each SRPA iteration."
        )

    return CompositionReport(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        resolved_hooks=resolved_hooks,
    )
