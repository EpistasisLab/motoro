"""Core abstractions for the pattern plugin system.

Defines:
- HookPoint — enum of all hook insertion points in the SRPA loop
- HookAction — enum of control-flow actions a hook can return
- HookCallable — type alias for async hook functions
- PatternPlugin — ABC that every pattern implements
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any, ClassVar

from pydantic import BaseModel

from agentic_core.engine.context import RunContext
from agentic_core.models.pattern import PatternCategory, PatternPhase

# ---------------------------------------------------------------------------
# Hook system types
# ---------------------------------------------------------------------------


class HookPoint(StrEnum):
    """Insertion points in the agentic loop where hooks can execute."""

    PRE_SENSE = "pre_sense"
    POST_SENSE = "post_sense"
    PRE_REASON = "pre_reason"
    POST_REASON = "post_reason"
    PRE_PLAN = "pre_plan"
    POST_PLAN = "post_plan"
    PRE_ACT = "pre_act"
    POST_ACT = "post_act"
    LOOP_CONTROL = "loop_control"
    ON_COMPLETION = "on_completion"
    ON_ERROR = "on_error"


class HookAction(StrEnum):
    """Control-flow actions a hook can return.

    When a hook needs to carry additional data (abort reason, review request,
    retry reason), it should write to ``context.metadata`` before returning
    the action.  Conventional keys:

    * ``abort_error`` — error message for ABORT
    * ``retry_reason`` — reason string for RETRY_PHASE
    * ``review_request`` — dict payload for PAUSE
    """

    SKIP_PHASE = "skip_phase"
    RETRY_PHASE = "retry_phase"
    ABORT = "abort"
    PAUSE = "pause"


# Hook signature: receives the RunContext and an optional phase output
# (None for pre-phase hooks, the phase's BaseModel output for post-phase hooks).
# Returns None (continue), a modified BaseModel output, or a HookAction.
HookCallable = Callable[
    [RunContext, BaseModel | None],
    Awaitable[BaseModel | HookAction | None],
]


# ---------------------------------------------------------------------------
# Pattern plugin ABC
# ---------------------------------------------------------------------------


class PatternPlugin(ABC):
    """Abstract base class that every architectural pattern implements.

    Subclass this, set the class-level ``slug`` and ``category``,
    implement ``configure`` and ``get_hooks``, and register the class
    via :class:`PluginRegistry`.
    """

    # -- Class-level identity (set on each concrete subclass) ---------------
    slug: ClassVar[str]
    category: ClassVar[PatternCategory]

    # -- Catalog metadata --------------------------------------------------
    # The plugin class is the source of truth for what a pattern *is*. ARES
    # keeps this in an `architectural_patterns` table seeded by a data
    # migration, which makes code and catalog two sources of truth that drift:
    # renaming ``solo_agent_loop`` to ``single_agent_baseline`` needed migration
    # 0009 to chase the change through the dependency arrays in data, and the
    # ``is_implemented`` column is hand-maintained with nothing reconciling it
    # against which plugins actually exist.
    #
    # Declared here instead, so validation and parameter defaults read them
    # in-process with no database round trip and nothing to keep in sync. The
    # table becomes a projection: ``sync_pattern_catalog()`` writes these values
    # into it for products that need to query the catalog (a UI listing, an
    # advisor's knowledge base), and never the other way around.

    #: Human-readable name. Empty means "derive from the slug".
    display_name: ClassVar[str] = ""

    #: What the pattern does, in prose. Surfaced to products, not used at run time.
    description: ClassVar[str] = ""

    #: Complexity tier. Unrelated to SRPA phases — see ``replaces_phases`` for those.
    complexity_phase: ClassVar[PatternPhase] = PatternPhase.BASIC

    #: Slugs this pattern needs co-active. Already honoured by
    #: ``PatternOrchestrator.from_pattern_config``, which auto-activates a missing
    #: dependency when it can safely fill an empty singleton slot.
    dependencies: ClassVar[list[str]] = []

    #: Whether the pattern is meaningless without a multi-agent role assignment.
    requires_multi_agent: ClassVar[bool] = False

    #: JSON Schema for ``pattern_params[slug]``. Every property should carry a
    #: ``default``: those defaults are merged into the params a plugin receives,
    #: so ``configure`` gets a fully populated dict and its own
    #: ``params.get(key, fallback)`` literals become a redundant second copy.
    configuration_schema: ClassVar[dict[str, Any]] = {}

    #: Pattern version, for products tracking catalog changes.
    version: ClassVar[str] = "1.0.0"

    # Optional: slugs of patterns this pattern conflicts with
    conflicts_with: ClassVar[list[str]] = []

    # Optional: phase names that this plugin always replaces (i.e. always returns
    # SKIP_PHASE for the pre-hook).  Declared here so composition checks can
    # detect incompatible combinations — two plugins that both replace "plan"
    # cannot run together.  Example: ReasonActPlugin replaces ["plan"] because
    # its PRE_PLAN hook always returns HookAction.SKIP_PHASE.
    #
    # Phase names must be lowercase members of the default SRPA sequence:
    # "sense", "reason", "plan", "act".
    replaces_phases: ClassVar[list[str]] = []

    # Optional: minimum hook-execution timeout (seconds) this plugin needs.
    # Patterns whose hooks fan out to multiple LLM calls (e.g., Tree-of-
    # Thought generates ``breadth`` reasoning + ``breadth`` evaluation calls
    # inside one ``pre_reason``) should set this so the orchestrator doesn't
    # cancel them with the default 30s timeout. ``None`` means "use the
    # orchestrator default."
    recommended_hook_timeout: ClassVar[float | None] = None

    # -- Configuration ------------------------------------------------------

    @abstractmethod
    def configure(self, params: dict[str, Any]) -> None:
        """Apply runtime parameters from the agent's ``pattern_params[slug]``.

        Called once at run start, before ``on_activate``.
        """

    @abstractmethod
    def get_hooks(self) -> dict[HookPoint, list[HookCallable]]:
        """Return the hooks this plugin registers at each insertion point.

        Called once after ``configure`` to build the hook pipeline.
        """

    # -- Optional lifecycle methods -----------------------------------------

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        """Validate pattern-specific parameters.

        Returns a list of error strings (empty = valid).
        Default implementation accepts any params.
        """
        return []

    async def on_activate(self, context: RunContext) -> None:  # noqa: B027
        """Called when the pattern is activated at run start.

        Override to initialise per-run state in ``context.metadata``.
        """

    async def on_deactivate(self, context: RunContext) -> None:  # noqa: B027
        """Called when the pattern is deactivated at run end.

        Override to clean up resources or persist state.
        """
