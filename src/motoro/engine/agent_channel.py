"""Projection of peer agents into what a model can read and call.

One module because there are two dispatch loops, not one: ``ActPhase`` and the
``reason_act`` pattern each bind their own native tool payload, and the default
execution pattern is ``reason_act``. A projection written twice drifts, and the
symptom of drift here is an agent that can see a peer but never calls it.

Peers are projected as ordinary function schemas because function calling is the
only channel through which a model can *choose* an action. They are not tools:
nothing here touches the MCP registry, no call is recorded as a
``ToolCallRecord``, and dispatch goes to
:class:`~motoro.engine.ports.AgentMessengerPort`. See
:attr:`~motoro.engine.context.RunContext.available_agents`.
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from motoro.engine.context import RunContext

log = structlog.get_logger()

#: Every projected peer function is ``ask_<slug>``. Reserved: an MCP tool whose
#: sanitized name starts with this could otherwise be mistaken for a peer, so
#: dispatch checks the peer name map (built only from declared peers) first and
#: a colliding tool name simply never matches an entry in it.
RESERVED_PREFIX = "ask_"

# Provider constraint on function names: ``^[a-zA-Z0-9_-]{1,64}$``.
_NAME_MAX = 64
_HASH_LEN = 8


def _slug(agent: dict[str, Any]) -> str:
    """A stable, provider-legal function name for one peer.

    Derived from the display name so the model reads ``ask_critic`` rather than
    ``ask_node_mtt2o0xz_52jfnuu1``, but disambiguated by a hash of the
    ``agent_id`` -- two peers can share a canvas label, and the id is what
    identity actually hangs off.
    """
    label = str(agent.get("name") or agent.get("agent_id") or "agent")
    agent_id = str(agent.get("agent_id", ""))
    base = re.sub(r"[^A-Za-z0-9_-]", "_", label).strip("_").lower() or "agent"
    digest = hashlib.sha256(agent_id.encode("utf-8")).hexdigest()[:_HASH_LEN]
    budget = _NAME_MAX - len(RESERVED_PREFIX) - (_HASH_LEN + 1)
    return f"{RESERVED_PREFIX}{base[:budget]}_{digest}"


def build_agent_name_map(available_agents: list[dict[str, Any]]) -> dict[str, str]:
    """Map projected function names back to ``agent_id``.

    The model echoes the function name, not the id; dispatch must translate
    before handing anything to the messenger. Mirrors
    :func:`~motoro.mcp.adapters.build_openai_tool_name_map` and exists for the
    same reason.
    """
    return {_slug(agent): str(agent.get("agent_id", "")) for agent in available_agents if agent.get("agent_id")}


def agents_to_openai_format(available_agents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project each peer as its own single-argument function.

    One function per peer rather than one ``ask_agent(agent_id, question)`` with
    an enum: the peer's own description becomes the function description, which
    is the single strongest signal the model has for choosing well, and an enum
    would collapse every peer's purpose into one shared string. Peer counts per
    agent are small, so the extra schemas cost little.
    """
    schemas: list[dict[str, Any]] = []
    for agent in available_agents:
        if not agent.get("agent_id"):
            continue
        name = str(agent.get("name") or agent["agent_id"])
        description = str(agent.get("description") or "").strip()
        skills = _skill_names(agent)
        summary = f"Ask {name}, a peer agent you are connected to. {description}".strip()
        if skills:
            summary = f"{summary} Skills: {', '.join(skills)}."
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": _slug(agent),
                    "description": summary[:1024],
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": (
                                    "What you want from this agent. It sees only this text, "
                                    "not your prompt or your reasoning, so include the context "
                                    "it needs to answer."
                                ),
                            }
                        },
                        "required": ["question"],
                    },
                },
            }
        )
    return schemas


def format_agents_for_prompt(available_agents: list[dict[str, Any]]) -> str:
    """Render the peer roster as a prompt section.

    Kept visibly separate from the tool section: a collaborator that reasons and
    can push back is not a tool that returns a value, and blurring the two makes
    a model treat a peer as a lookup.
    """
    if not available_agents:
        return ""
    lines: list[str] = []
    for agent in available_agents:
        if not agent.get("agent_id"):
            continue
        name = str(agent.get("name") or agent["agent_id"])
        description = str(agent.get("description") or "").strip()
        line = f"- {name}: {description}" if description else f"- {name}"
        skills = _skill_names(agent)
        if skills:
            line = f"{line} (skills: {', '.join(skills)})"
        lines.append(line[:250])
    if not lines:
        return ""
    return (
        "Connected agents you can consult:\n"
        + "\n".join(lines)
        + "\n\nEach is a separate agent with its own reasoning, not a tool. Consult one "
        "when its perspective would materially improve your answer; otherwise just answer. "
        "It sees only what you send it, and its reply comes back to you to use as you see fit."
    )


#: Reply states that carry something the caller can actually use. A rejection
#: is a legitimate answer from the peer's side but produced no content, so it is
#: reported as an unsuccessful step rather than silently reading as an answer.
_ANSWERED = frozenset({"completed", "input_required"})


async def consult_peer(
    *,
    to_agent_id: str,
    question: str,
    context: RunContext,
) -> tuple[str, bool]:
    """Send one question to a peer and render its reply as step text.

    Returns ``(result_text, success)``. Never raises: a peer that fails, refuses
    or is unreachable is a fact the calling model should read and work around,
    not an exception that ends the caller's own run. The messenger signals
    refusal through :attr:`~motoro.engine.ports.AgentReply.state`, so an
    exception here means something genuinely broke.
    """
    messenger = context.agent_messenger
    if messenger is None:
        # The peer roster reached the model but the transport did not. Say so
        # plainly rather than fabricating an answer in the peer's voice.
        log.warning("agent_channel.no_messenger", to_agent_id=to_agent_id, component="agent_channel")
        return ("No agent messenger is configured for this run, so the consultation was not sent.", False)

    try:
        reply = await messenger.send(
            from_agent_id=str(context.agent_id or ""),
            to_agent_id=to_agent_id,
            parts=[{"kind": "text", "text": question}],
            context=context,
        )
    except Exception as exc:
        log.warning(
            "agent_channel.send_failed",
            to_agent_id=to_agent_id,
            error=f"{type(exc).__name__}: {exc}",
            component="agent_channel",
        )
        return (f"The consultation failed: {type(exc).__name__}: {exc}", False)

    text = reply.text.strip()
    success = reply.state in _ANSWERED
    log.debug(
        "agent_channel.reply",
        to_agent_id=to_agent_id,
        state=reply.state,
        task_id=reply.task_id,
        component="agent_channel",
    )
    if reply.state == "completed":
        return (text or "The agent replied with nothing.", True)

    detail = text or reply.error or ""
    prefix = f"The agent's reply is {reply.state}"
    return (f"{prefix}: {detail}" if detail else f"{prefix}.", success)


def _skill_names(agent: dict[str, Any]) -> list[str]:
    skills = agent.get("skills")
    if not isinstance(skills, list):
        return []
    return [str(s.get("name")) for s in skills if isinstance(s, dict) and s.get("name")]


__all__ = [
    "RESERVED_PREFIX",
    "agents_to_openai_format",
    "build_agent_name_map",
    "consult_peer",
    "format_agents_for_prompt",
]
