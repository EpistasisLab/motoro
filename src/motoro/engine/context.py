"""Run context — working memory for a single agent run."""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel

from motoro.schemas.agent import ModelConfig

if TYPE_CHECKING:
    from motoro.memory.working import WorkingMemoryManager

_log = structlog.get_logger()
_logger = logging.getLogger(__name__)

# Issue #897: 200-char limit silently drops tool results from the transcript.
# Raise to 2000 so reviewers can see what tools returned while still bounding
# memory use.  The constant is module-level so tests can patch it if needed.
_SUMMARIZE_FIELD_LIMIT = 2000


@dataclass
class RunContext:
    """Working memory for the current agent run.

    Passed to each phase; phases read from and write to it.
    Instantiated fresh per run — not shared across runs.
    """

    # Agent configuration
    agent_goal: str
    system_prompt: str
    model_config: ModelConfig
    user_input: str
    agent_id: uuid.UUID | None = None
    # The agent's own name, carried alongside agent_id (also not part of the
    # resume snapshot below — both are static config, re-derivable from the
    # agent row rather than needing to round-trip through a paused run).
    agent_name: str | None = None
    run_id: uuid.UUID | None = None
    # The user who started this run; LLM callers scope credential resolution to
    # them (M112). Populated by the runtime from its scoped LLMService.
    owner_id: uuid.UUID | None = None
    # Ambient workspace identity for this run (issue #1455). Injected out-of-band
    # into every MCP tool call's request ``_meta`` so the model never threads
    # ``workspace_id``/``dataset_id`` through tool arguments. Sourced from the
    # run's ``run_metadata["workspace_id"]``; ``None`` for runs with no workspace.
    workspace_id: str | None = None

    # Phase outputs (populated as phases execute)
    phase_outputs: dict[str, BaseModel] = field(default_factory=dict)

    # Conversation history for multi-iteration loops
    conversation_history: list[dict[str, Any]] = field(default_factory=list)

    # Available MCP tools (populated by Sense, consumed by Plan/Act)
    available_tools: list[dict[str, Any]] = field(default_factory=list)

    # Memory context (populated by Sense from memory service)
    memories: list[dict[str, Any]] = field(default_factory=list)

    # Agent Skills carried by this run: [{"name", "description", "body"}, ...],
    # resolved from the agent's skill_config before the first phase. Part of the
    # snapshot below, unlike agent_id/agent_name: a skill can be edited or
    # deleted while a run is paused, and a resumed run must continue against the
    # instructions it was actually following, not whatever the row says now.
    skills: list[dict[str, Any]] = field(default_factory=list)

    # Optional Redis-backed working memory (injected by runtime when configured)
    working_memory_manager: WorkingMemoryManager | None = field(default=None)

    # Token tracking across all phases in this run
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    # Internal Decimal accumulator for cost (issue #941) — keeps precision
    # across many small additions; the public ``total_cost`` attribute is
    # synced as a float for backwards compatibility.
    _total_cost_decimal: Decimal = field(default_factory=lambda: Decimal("0"))
    total_cost: float = 0.0

    # Loop control
    iteration: int = 0
    max_iterations: int = 10
    max_history: int = 50

    # Pattern engine state (populated by PatternOrchestrator)
    current_phase: str = ""
    pattern_config: dict[str, Any] | None = field(default=None)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Agent memory configuration (injected by runtime before first phase)
    memory_config_data: dict[str, Any] = field(default_factory=dict)

    # Async lock guarding atomic updates to token / cost accumulators
    # (issue #691). Excluded from snapshots — recreated on each run.
    _usage_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def add_llm_usage(self, prompt_tokens: int, completion_tokens: int, cost: float) -> None:
        """Accumulate token usage from an LLM call.

        Uses an ``asyncio.Lock`` so that concurrent phase hooks
        (e.g. ReasonAct, Tree-of-Thoughts) cannot lose updates when they
        race on the same ``RunContext`` (issue #691).

        Cost is accumulated as a ``Decimal`` internally to avoid drift
        from float summation (issue #941); ``total_cost`` remains a float
        for downstream JSON/API compatibility.
        """
        async with self._usage_lock:
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self._total_cost_decimal += Decimal(str(cost))
            self.total_cost = float(self._total_cost_decimal)

    def update_phase_output(self, phase: str, output: BaseModel) -> None:
        """Replace the stored output for *phase* without touching history.

        Used by ``PatternOrchestrator._run_hooks`` (#1039) when a hook returns
        a modified phase output.  Hook-driven modifications must not append
        new entries to ``conversation_history``, because that would let a
        chain of N output-modifying hooks evict N legitimate phase summaries
        via the FIFO ``max_history`` trim — silent transcript drift.  The
        history entry produced by the phase execution itself is the
        authoritative record; hooks only swap the output object that later
        phases read back via ``phase_outputs[phase]``.
        """
        self.phase_outputs[phase] = output

    def record_phase_output(self, phase: str, output: BaseModel) -> None:
        """Record a phase's output for use by subsequent phases.

        Stores a summary string instead of the full model dump to limit
        memory growth.  When ``max_history`` is exceeded the oldest entries
        are evicted (FIFO).

        Issue #783: eviction removes history entries but must not evict the
        corresponding ``phase_outputs`` entry — a later phase (e.g. Reason)
        still needs ``phase_outputs["sense"]`` even after the history entry
        for that phase scrolled out of the window.  Only the history summary
        is trimmed; the output objects themselves are kept indefinitely.
        """
        self.phase_outputs[phase] = output
        # Lazy serialization: store a compact summary instead of full dump
        summary = _summarize_output(output)
        self.conversation_history.append(
            {
                "phase": phase,
                "iteration": self.iteration,
                "output_summary": summary,
            }
        )
        # Evict oldest *history* entries when history exceeds the cap.
        # Issue #783: phase_outputs dict is intentionally NOT trimmed here —
        # subsequent phases (Reason, Plan, Act) reference outputs by key and
        # would silently lose Sense output if we removed it.
        if len(self.conversation_history) > self.max_history:
            excess = len(self.conversation_history) - self.max_history
            self.conversation_history = self.conversation_history[excess:]

    def to_snapshot(self) -> dict[str, Any]:
        """Serialize the RunContext to a JSON-safe dict for pause/resume.

        Phase outputs are stored with their fully qualified class name
        so they can be reconstructed on resume.

        Issue #774: ``metadata`` may contain un-serialisable values stashed by
        plugins (DB sessions, futures, asyncio tasks).  We filter ``_runtime``
        and any value that fails ``json.dumps`` so the snapshot is always
        JSON-safe.  Non-serialisable values are dropped with a warning rather
        than crashing the pause path.
        """
        serialized_outputs: dict[str, dict[str, Any]] = {}
        for phase_name, output in self.phase_outputs.items():
            cls = type(output)
            serialized_outputs[phase_name] = {
                "_class": f"{cls.__module__}.{cls.__qualname__}",
                "_data": output.model_dump(),
            }

        # Issue #774: scrub metadata to JSON-safe primitives.
        safe_metadata: dict[str, Any] = {}
        for k, v in self.metadata.items():
            if k == "_runtime":
                continue
            try:
                json.dumps(v)
                safe_metadata[k] = v
            except (TypeError, ValueError):
                _log.warning(
                    "context.snapshot.metadata_skip",
                    key=k,
                    value_type=type(v).__name__,
                    component="context",
                )

        return {
            "agent_goal": self.agent_goal,
            "system_prompt": self.system_prompt,
            "model_config": self.model_config.model_dump(),
            "user_input": self.user_input,
            "owner_id": str(self.owner_id) if self.owner_id else None,
            "workspace_id": self.workspace_id,
            "phase_outputs": serialized_outputs,
            "conversation_history": self.conversation_history,
            "available_tools": self.available_tools,
            "memories": self.memories,
            "skills": self.skills,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_cost": self.total_cost,
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "max_history": self.max_history,
            "current_phase": self.current_phase,
            "pattern_config": self.pattern_config,
            "metadata": safe_metadata,
        }

    @classmethod
    def from_snapshot(cls, data: dict[str, Any]) -> RunContext:
        """Deserialize a RunContext from a snapshot dict.

        Reconstructs Pydantic phase outputs from stored class references.
        """
        # Reconstruct phase_outputs
        phase_outputs: dict[str, BaseModel] = {}
        for phase_name, entry in data.get("phase_outputs", {}).items():
            class_path = entry["_class"]
            model_data = entry["_data"]
            model_cls = _resolve_class(class_path)
            phase_outputs[phase_name] = model_cls.model_validate(model_data)

        return cls(
            agent_goal=data["agent_goal"],
            system_prompt=data["system_prompt"],
            model_config=ModelConfig.model_validate(data["model_config"]),
            user_input=data["user_input"],
            owner_id=(uuid.UUID(data["owner_id"]) if data.get("owner_id") else None),
            workspace_id=data.get("workspace_id"),
            phase_outputs=phase_outputs,
            conversation_history=data.get("conversation_history", []),
            available_tools=data.get("available_tools", []),
            memories=data.get("memories", []),
            skills=data.get("skills", []),
            total_prompt_tokens=data.get("total_prompt_tokens", 0),
            total_completion_tokens=data.get("total_completion_tokens", 0),
            total_cost=data.get("total_cost", 0.0),
            iteration=data.get("iteration", 0),
            max_iterations=data.get("max_iterations", 10),
            max_history=data.get("max_history", 50),
            current_phase=data.get("current_phase", ""),
            pattern_config=data.get("pattern_config"),
            metadata=data.get("metadata", {}),
            _total_cost_decimal=Decimal(str(data.get("total_cost", 0.0))),
        )


# Issue #897: 200-char limit silently drops tool results from the transcript.
# Raise to 2000 so reviewers can see what tools returned while still bounding
# memory use.  The constant is module-level so tests can patch it if needed.
_SUMMARIZE_FIELD_LIMIT = 2000


def _summarize_output(output: BaseModel, field_limit: int = _SUMMARIZE_FIELD_LIMIT) -> str:
    """Return a compact string summary of a Pydantic model.

    Truncates long field values to keep memory bounded.  ``field_limit``
    controls the per-field character cap (default :data:`_SUMMARIZE_FIELD_LIMIT`).
    """
    parts: list[str] = []
    for field_name in type(output).model_fields:
        val = getattr(output, field_name, None)
        s = str(val)
        if len(s) > field_limit:
            s = s[:field_limit] + "..."
        parts.append(f"{field_name}={s}")
    return f"{type(output).__name__}({', '.join(parts)})"


def _resolve_class(class_path: str) -> type[BaseModel]:
    """Import and return a Pydantic model class from a dotted path."""
    module_path, _, class_name = class_path.rpartition(".")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    if not (isinstance(cls, type) and issubclass(cls, BaseModel)):
        raise TypeError(f"{class_path} is not a Pydantic BaseModel subclass")
    return cls
