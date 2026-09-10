"""Ports the runtime depends on but does not own.

Three collaborators the SRPA loop reaches for are deliberately *not* in core yet:
long-term memory, the resource limits a supervisor imposes on a child agent, and
delivery of a message from one agent to another. All are optional at runtime —
the loop already tolerates their absence — so rather than dragging their
implementations across, core states the shape it needs and lets a slice or a
product supply it.

That keeps real weight out of core. A concrete ``MemoryService`` pulls in episodic
and semantic memory and, with them, pgvector and an embedding model; the runtime
only ever calls four methods on it. An agent messenger is the same bargain for
multi-agent work: core has no notion of which agents may talk to each other, and
acquiring one would mean acquiring a topology, a transcript store and a budget
model it has no use for on a single-agent run.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from motoro.engine.context import RunContext


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


@dataclass
class AgentReply:
    """The outcome of one peer consultation, shaped after A2A's ``TaskState``.

    A string state rather than a success flag on purpose: ``rejected`` (the
    caller may not talk to this peer, or a budget is spent) and
    ``input_required`` (the peer asked a clarifying question instead of
    answering) are distinct, actionable outcomes, and a boolean flattens both
    into "it failed" — which reads to a model as a dead end rather than as
    something to respond to.
    """

    #: completed | input_required | rejected | failed | canceled
    state: str
    #: A2A Parts — ``{"kind": "text", "text": ...}`` today; data/file parts need
    #: no change here.
    parts: list[dict[str, Any]] = field(default_factory=list)
    task_id: str | None = None
    error: str | None = None

    @property
    def text(self) -> str:
        """The reply's text parts joined — what Act feeds back to the model."""
        return "\n".join(str(part.get("text", "")) for part in self.parts if part.get("kind") == "text").strip()


@runtime_checkable
class AgentMessengerPort(Protocol):
    """The whole agent-to-agent surface the SRPA loop touches — one method.

    Core never decides who may talk to whom. It projects the peers the caller
    declared in :attr:`RunContext.available_agents` into callable function
    schemas, and hands any resulting call straight back to the implementer,
    which owns authorization, transcript, message identity, ordering and budget.
    A reply is an ordinary call result: the caller's turn continues with it in
    context, exactly as it would with a tool result.

    Implementations must **not** raise to refuse. A refused recipient or an
    exhausted budget is a :class:`AgentReply` with ``state="rejected"``, so the
    calling agent can absorb it and still produce a real answer. An exception
    here means genuine infrastructure failure.
    """

    async def send(
        self,
        *,
        from_agent_id: str,
        to_agent_id: str,
        parts: list[dict[str, Any]],
        context: RunContext,
    ) -> AgentReply:
        """Deliver a message to a peer and return its reply."""
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
