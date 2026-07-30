"""Pattern engine — plugin system for architectural agent patterns."""

from agentic_core.engine.patterns.base import HookAction, HookPoint, PatternPlugin
from agentic_core.engine.patterns.registry import PluginRegistry

__all__ = [
    "HookAction",
    "HookPoint",
    "PatternPlugin",
    "PluginRegistry",
]
