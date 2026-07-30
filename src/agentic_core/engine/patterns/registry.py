"""Plugin registry — discovers and retrieves PatternPlugin classes by slug."""

from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from agentic_core.engine.patterns.base import PatternPlugin

log = structlog.get_logger()


class PluginRegistry:
    """Singleton registry mapping pattern slugs to plugin classes."""

    _plugins: dict[str, type[PatternPlugin]] = {}

    @classmethod
    def register(cls, plugin_class: type[PatternPlugin]) -> type[PatternPlugin]:
        """Register a plugin class.  Can be used as a decorator::

        @PluginRegistry.register
        class MyPlugin(PatternPlugin):
            slug = "my_pattern"
            ...
        """
        slug = plugin_class.slug
        if slug in cls._plugins:
            log.warning(
                "pattern.registry.overwrite",
                slug=slug,
                existing=cls._plugins[slug].__name__,
                new=plugin_class.__name__,
            )
        cls._plugins[slug] = plugin_class
        return plugin_class

    @classmethod
    def get(cls, slug: str) -> type[PatternPlugin] | None:
        """Return the plugin class for *slug*, or ``None``."""
        return cls._plugins.get(slug)

    @classmethod
    def all(cls) -> dict[str, type[PatternPlugin]]:
        """Return a copy of the full slug → class mapping."""
        return dict(cls._plugins)

    @classmethod
    def discover(cls, *, raise_on_error: bool = False) -> None:
        """Auto-import every module in ``engine.patterns.builtin``.

        Each module is expected to register its plugin via the
        ``@PluginRegistry.register`` decorator at import time.

        Args:
            raise_on_error: When ``True``, re-raise the first import error
                instead of swallowing it.  Pass ``raise_on_error=True`` in
                tests to surface syntax errors or broken imports immediately.
                Production callers use the default (``False``) so a single
                broken plugin does not disable all discovery.
        """
        import agentic_core.engine.patterns.builtin as _builtin_pkg

        package_path = _builtin_pkg.__path__
        # ``pkgutil.iter_modules`` returns filesystem-order results, which can
        # vary between worker restarts and across container image builds.
        # That non-determinism flows into ``cls._plugins`` insertion order,
        # which in turn determines tie-breaking when hooks share the same
        # (category, priority) bucket.  Sort by module name so hook order is
        # stable across processes (#1038).
        # Index [1] is the module name in both pkgutil's ``ModuleInfo`` named
        # tuple and any plain 3-tuple a test fixture might substitute.
        modules = sorted(pkgutil.iter_modules(package_path), key=lambda m: m[1])
        for _importer, modname, _ispkg in modules:
            full_name = f"agentic_core.engine.patterns.builtin.{modname}"
            try:
                importlib.import_module(full_name)
            except Exception:
                log.exception("pattern.registry.discover_error", module=full_name)
                if raise_on_error:
                    raise

    @classmethod
    def reset(cls) -> None:
        """Clear the registry.  Intended for tests only."""
        cls._plugins.clear()
