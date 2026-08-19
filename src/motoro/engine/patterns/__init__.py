"""Pattern engine — plugin system for architectural agent patterns."""

from motoro.engine.patterns.base import HookAction, HookPoint, PatternPlugin
from motoro.engine.patterns.registry import PluginRegistry

__all__ = [
    "HookAction",
    "HookPoint",
    "PatternPlugin",
    "PluginRegistry",
]
