"""Ports the runtime depends on but does not own.

Two collaborators the SRPA loop reaches for are deliberately *not* in core yet:
long-term memory, and the resource limits a supervisor imposes on a child agent.
Both are optional at runtime — the loop already tolerates their absence — so
rather than dragging their implementations across, core states the shape it needs
and lets a slice or a product supply it.

That keeps real weight out of core. A concrete ``MemoryService`` pulls in episodic
and semantic memory and, with them, pgvector and an embedding model; the runtime
only ever calls four methods on it.
"""

from __future__ import annotations

import uuid
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EpisodicMemoryPort(Protocol):
    """The episodic surface the runtime uses to persist a finished run."""

    async def generate_run_summary(self, *args: Any, **kwargs: Any) -> Any:
        """Summarize a completed run for later recall."""
        ...

    async def store_run_summary(self, *args: Any, **kwargs: Any) -> Any:
        """Persist a run summary."""
        ...


@runtime_checkable
class MemoryServicePort(Protocol):
    """The whole memory surface the SRPA loop touches — four methods.

    ``Sense`` calls :meth:`count` and :meth:`recall` to fold prior knowledge into
    the phase context; the runtime uses :attr:`episodic` to write a run summary on
    completion. Anything else a real memory service offers is invisible here.
    """

    async def count(self, agent_id: uuid.UUID) -> int:
        """Number of memories available to *agent_id*."""
        ...

    async def recall(self, *args: Any, **kwargs: Any) -> Any:
        """Retrieve memories relevant to the current context."""
        ...

    @property
    def episodic(self) -> EpisodicMemoryPort:
        """Episodic sub-service used for run summaries."""
        ...


class ResourceLimitResult(Protocol):
    """Mapping returned by a resource-limit check.

    Truthy ``exceeded`` aborts the run; every other truthy key is reported as the
    name of a limit that was hit.
    """

    def get(self, key: str, /) -> Any: ...

    def items(self) -> Any: ...


#: Signature of a resource-limit check: ``(db, agent_id, token_usage, tool_calls)``
#: returning a mapping with an ``exceeded`` key.
#:
#: ARES implements this in ``agent_relationship_service.check_resource_limits``,
#: enforcing the budget a supervisor set on a child agent. Core has no agent
#: hierarchy yet, so the default is None and the check is skipped — which is what
#: the runtime already did whenever the call failed.
_resource_limit_checker: Any = None


def set_resource_limit_checker(fn: Any) -> None:
    """Install the resource-limit check the runtime consults each iteration.

    Passing ``None`` restores the default of not checking.
    """
    global _resource_limit_checker
    _resource_limit_checker = fn


def get_resource_limit_checker() -> Any:
    """Return the installed check, or ``None`` if limits are not enforced."""
    return _resource_limit_checker
