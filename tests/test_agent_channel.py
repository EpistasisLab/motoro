"""The peer-agent channel: projection, dispatch, and the tool boundary.

The invariant under test throughout is that a peer is *not* a tool. It reaches
the model through the same function-calling payload — that is the only channel a
model can choose an action through — but it resolves through its own name map,
dispatches through :class:`AgentMessengerPort`, and produces a ``StepResult``
with no ``tool_call`` attached. Every assertion about ``tool_call is None`` below
is guarding that boundary, not being pedantic about a field.
"""

from __future__ import annotations

import uuid
from typing import Any, cast

from motoro.engine.act import ActPhase
from motoro.engine.agent_channel import (
    RESERVED_PREFIX,
    agents_to_openai_format,
    build_agent_name_map,
    consult_peer,
    format_agents_for_prompt,
)
from motoro.engine.context import RunContext
from motoro.engine.ports import AgentReply
from motoro.schemas.agent import ModelConfig
from motoro.schemas.llm import ActOutput, LLMToolCall, PlanOutput, PlanStep, SenseOutput, ToolCompletion
from motoro.services.llm_service import LLMService
from tests.stub_llm import StubLLM, call_record

CRITIC_ID = "11111111-1111-1111-1111-111111111111"
PLANNER_ID = "22222222-2222-2222-2222-222222222222"

CRITIC = {
    "agent_id": CRITIC_ID,
    "name": "Critic",
    "description": "Reviews a draft answer and names its weakest claim.",
    "skills": [{"name": "critique"}, {"name": "fact-check"}],
}
PLANNER = {
    "agent_id": PLANNER_ID,
    "name": "Planner",
    "description": "Breaks a goal into ordered steps.",
}


def _context(**kwargs: Any) -> RunContext:
    return RunContext(
        agent_goal="answer well",
        system_prompt="you are an agent",
        model_config=ModelConfig(model="gpt-4o"),
        user_input="what is the weakest part of this plan?",
        agent_id=uuid.UUID(PLANNER_ID),
        **kwargs,
    )


class _RecordingMessenger:
    """Captures what was sent and answers with a scripted reply."""

    def __init__(self, reply: AgentReply | None = None, raises: Exception | None = None) -> None:
        self.reply = reply or AgentReply(state="completed", parts=[{"kind": "text", "text": "the third claim"}])
        self.raises = raises
        self.sent: list[dict[str, Any]] = []

    async def send(
        self,
        *,
        from_agent_id: str,
        to_agent_id: str,
        parts: list[dict[str, Any]],
        context: RunContext,
    ) -> AgentReply:
        self.sent.append({"from": from_agent_id, "to": to_agent_id, "parts": parts})
        if self.raises is not None:
            raise self.raises
        return self.reply


# ----------------------------------------------------------------------
# Projection
# ----------------------------------------------------------------------


def test_projected_names_are_provider_legal_and_readable() -> None:
    (name,) = build_agent_name_map([CRITIC])
    assert name.startswith(f"{RESERVED_PREFIX}critic_")
    assert len(name) <= 64
    assert all(c.isalnum() or c in "_-" for c in name)


def test_peers_sharing_a_display_name_still_get_distinct_functions() -> None:
    """A canvas label is not an identity — two nodes can carry the same one."""
    twin = {**CRITIC, "agent_id": "33333333-3333-3333-3333-333333333333"}
    name_map = build_agent_name_map([CRITIC, twin])
    assert len(name_map) == 2
    assert set(name_map.values()) == {CRITIC_ID, twin["agent_id"]}


def test_name_map_round_trips_to_agent_ids() -> None:
    name_map = build_agent_name_map([CRITIC, PLANNER])
    schemas = agents_to_openai_format([CRITIC, PLANNER])
    assert [s["function"]["name"] for s in schemas] == list(name_map)
    assert list(name_map.values()) == [CRITIC_ID, PLANNER_ID]


def test_each_peer_is_its_own_function_with_one_question_argument() -> None:
    """Not one ``ask_agent(agent_id, ...)``: the peer's own description is the
    strongest signal the model has for choosing, and an enum would collapse
    every peer's purpose into one shared string."""
    schemas = agents_to_openai_format([CRITIC, PLANNER])
    assert len(schemas) == 2
    critic = schemas[0]["function"]
    assert critic["parameters"]["required"] == ["question"]
    assert set(critic["parameters"]["properties"]) == {"question"}
    assert "weakest claim" in critic["description"]
    assert "critique" in critic["description"]


def test_a_peer_without_an_id_is_never_projected() -> None:
    assert agents_to_openai_format([{"name": "ghost"}]) == []
    assert build_agent_name_map([{"name": "ghost"}]) == {}
    assert "ghost" not in format_agents_for_prompt([{"name": "ghost"}])


def test_prompt_section_frames_peers_as_agents_not_tools() -> None:
    section = format_agents_for_prompt([CRITIC])
    assert "Critic" in section
    assert "not a tool" in section
    assert format_agents_for_prompt([]) == ""


# ----------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------


async def test_consult_peer_sends_the_question_and_returns_the_reply() -> None:
    messenger = _RecordingMessenger()
    context = _context(available_agents=[CRITIC], agent_messenger=messenger)

    text, ok = await consult_peer(to_agent_id=CRITIC_ID, question="is step 3 sound?", context=context)

    assert (text, ok) == ("the third claim", True)
    assert messenger.sent == [
        {
            "from": PLANNER_ID,
            "to": CRITIC_ID,
            "parts": [{"kind": "text", "text": "is step 3 sound?"}],
        }
    ]


async def test_a_refusal_is_a_reply_not_an_error() -> None:
    """The port contract: implementations signal refusal through ``state``,
    never by raising. A refused consultation is reported to the caller as an
    unsuccessful step it can read and work around."""
    messenger = _RecordingMessenger(AgentReply(state="rejected", error="consultation budget exhausted"))
    context = _context(available_agents=[CRITIC], agent_messenger=messenger)

    text, ok = await consult_peer(to_agent_id=CRITIC_ID, question="again?", context=context)

    assert ok is False
    assert "rejected" in text
    assert "budget exhausted" in text


async def test_a_broken_messenger_does_not_end_the_callers_run() -> None:
    messenger = _RecordingMessenger(raises=RuntimeError("peer unreachable"))
    context = _context(available_agents=[CRITIC], agent_messenger=messenger)

    text, ok = await consult_peer(to_agent_id=CRITIC_ID, question="hello", context=context)

    assert ok is False
    assert "peer unreachable" in text


async def test_a_roster_without_a_messenger_says_so_rather_than_inventing_a_reply() -> None:
    context = _context(available_agents=[CRITIC])

    text, ok = await consult_peer(to_agent_id=CRITIC_ID, question="hello", context=context)

    assert ok is False
    assert "not sent" in text


# ----------------------------------------------------------------------
# Act
# ----------------------------------------------------------------------


class _ToolCallingLLM(StubLLM):
    """Answers the first ``complete_with_tools`` with a scripted call."""

    def __init__(self, calls: list[LLMToolCall]) -> None:
        super().__init__()
        self._scripted = calls
        self.bound_tools: list[dict[str, Any]] = []

    async def complete_with_tools(self, *args: Any, **kwargs: Any) -> ToolCompletion:
        self.calls.append("complete_with_tools")
        self.bound_tools = list(kwargs.get("tools") or [])
        calls, self._scripted = self._scripted, []
        return ToolCompletion(text="I should check with the critic.", tool_calls=calls, record=call_record())


async def _run_act(context: RunContext, llm: StubLLM, step: PlanStep) -> ActOutput:
    context.record_phase_output(
        "sense",
        SenseOutput(
            user_input=context.user_input,
            agent_goal=context.agent_goal,
            system_prompt=context.system_prompt,
            available_agents=context.available_agents,
        ),
    )
    context.record_phase_output("plan", PlanOutput(steps=[step], is_complete=False))
    context.metadata["suppress_final_synthesis"] = True
    result = await ActPhase(cast("LLMService", llm)).execute(context)
    assert isinstance(result.output, ActOutput)
    return result.output


async def test_a_peer_only_agent_still_gets_a_callable_payload() -> None:
    """With no MCP tools at all, the old guard fell through to plain text
    generation — leaving an agent able to read its roster and unable to call
    anyone on it."""
    (peer_fn,) = build_agent_name_map([CRITIC])
    llm = _ToolCallingLLM([LLMToolCall(id="c1", name=peer_fn, arguments={"question": "is step 3 sound?"})])
    messenger = _RecordingMessenger()
    context = _context(available_agents=[CRITIC], agent_messenger=messenger)

    output = await _run_act(context, llm, PlanStep(action="review", description="Get a second opinion."))

    assert "complete_with_tools" in llm.calls
    assert [t["function"]["name"] for t in llm.bound_tools] == [peer_fn]
    assert messenger.sent[0]["to"] == CRITIC_ID
    reply = [r for r in output.results if r.result == "the third claim"]
    assert len(reply) == 1
    # The boundary: a consultation is not MCP telemetry.
    assert reply[0].tool_call is None


async def test_a_planned_consultation_is_dispatched_not_re_chosen() -> None:
    """Plan addresses a peer through the same ``tool_name`` slot as a tool, and
    a caller that already named the peer owns that decision."""
    (peer_fn,) = build_agent_name_map([CRITIC])
    llm = StubLLM()
    messenger = _RecordingMessenger()
    context = _context(available_agents=[CRITIC], agent_messenger=messenger)

    output = await _run_act(
        context,
        llm,
        PlanStep(
            action="consult critic",
            description="Ask the critic.",
            tool_name=peer_fn,
            tool_args={"question": "is step 3 sound?"},
        ),
    )

    assert llm.calls == []  # dispatched directly — no second selection call
    assert messenger.sent[0]["parts"] == [{"kind": "text", "text": "is step 3 sound?"}]
    assert [r.result for r in output.results] == ["the third claim"]
    assert output.results[0].tool_call is None


async def test_a_planned_consultation_falls_back_to_the_step_description() -> None:
    """A plan can name the peer and omit the argument."""
    (peer_fn,) = build_agent_name_map([CRITIC])
    messenger = _RecordingMessenger()
    context = _context(available_agents=[CRITIC], agent_messenger=messenger)

    await _run_act(
        context,
        StubLLM(),
        PlanStep(action="consult critic", description="Ask the critic about step 3.", tool_name=peer_fn),
    )

    assert messenger.sent[0]["parts"][0]["text"] == "Ask the critic about step 3."


async def test_a_failed_consultation_is_reported_as_an_unsuccessful_step() -> None:
    (peer_fn,) = build_agent_name_map([CRITIC])
    messenger = _RecordingMessenger(raises=RuntimeError("peer unreachable"))
    context = _context(available_agents=[CRITIC], agent_messenger=messenger)

    output = await _run_act(
        context,
        StubLLM(),
        PlanStep(action="consult critic", description="Ask.", tool_name=peer_fn, tool_args={"question": "hi"}),
    )

    assert output.results[0].success is False
    assert output.results[0].tool_call is None


async def test_no_peers_leaves_act_exactly_as_it_was() -> None:
    llm = StubLLM()
    context = _context()

    output = await _run_act(context, llm, PlanStep(action="respond", description="Answer directly."))

    assert llm.calls == ["complete_text"]
    assert output.results[0].tool_call is None


# ----------------------------------------------------------------------
# reason_act
# ----------------------------------------------------------------------


class _StubRuntime:
    """Enough runtime for ``_pre_act``; ``_db`` of None skips step persistence."""

    def __init__(self, llm_service: Any) -> None:
        self._llm_service = llm_service
        self._db = None


async def test_reason_act_binds_peers_and_plans_the_consultation_it_chose() -> None:
    """The default execution pattern runs its own tool-call loop and never
    touches ActPhase, so the projection has to reach it too — a peer visible in
    one loop and not the other is the drift ``agent_channel`` exists to prevent.
    """
    from motoro.engine.patterns.builtin.reason_act import ReasonActPlugin

    (peer_fn,) = build_agent_name_map([CRITIC])
    llm = _ToolCallingLLM([LLMToolCall(id="c1", name=peer_fn, arguments={"question": "is step 3 sound?"})])
    context = _context(available_agents=[CRITIC], agent_messenger=_RecordingMessenger())
    context.record_phase_output(
        "sense",
        SenseOutput(
            user_input=context.user_input,
            agent_goal=context.agent_goal,
            system_prompt=context.system_prompt,
            available_agents=context.available_agents,
        ),
    )
    context.metadata["_runtime"] = _StubRuntime(llm)

    plugin = ReasonActPlugin()
    plugin.configure({})
    await plugin.on_activate(context)
    assert await plugin._pre_act(context, None) is None  # continue to Act

    bound = [t["function"]["name"] for t in llm.bound_tools]
    assert peer_fn in bound
    # The roster is framed in the stable prefix, not left to the schema alone.
    prefix = " ".join(str(m.get("content", "")) for m in context.metadata["reason_act_messages"])
    assert "not a tool" in prefix

    plan = context.phase_outputs["plan"]
    assert isinstance(plan, PlanOutput)
    # Named, not dispatched here: Act resolves the ``ask_*`` name to an agent id.
    assert [s.tool_name for s in plan.steps] == [peer_fn]
    assert plan.steps[0].tool_args == {"question": "is step 3 sound?"}
