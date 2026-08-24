"""Prompt templates for the ReasonAct pattern.

The loop runs on native tool calling, so the prompt does not describe a
Thought/Action text format or enumerate tools in prose — the schemas are bound
to the request and the provider enforces the call shape. What remains is the
behavioural contract: reason before acting, and declare completion explicitly.
"""

from __future__ import annotations

from typing import Any

from motoro.schemas.llm import SenseOutput

FINAL_ANSWER_TOOL = "final_answer"

# Placeholder for a tool that returned nothing. Providers reject a ``role:
# "tool"`` turn with empty content, and a silent drop would leave the assistant
# turn's tool_call unanswered — also a provider error.
EMPTY_RESULT_PLACEHOLDER = "(the tool returned no output)"

REASON_ACT_SYSTEM_PROMPT = """\
You are an AI agent operating in Reason-Act mode. You work in a loop: reason \
about what you know, call tools to learn more or to act, observe the results, \
then reason again.

## Rules
- Before each set of tool calls, state your reasoning in your message text. That \
reasoning is recorded as part of your trajectory.
- Call as many tools per turn as the step genuinely needs. Tools you request \
together are executed together, so batch calls that do not depend on each \
other's results.
- If a tool call fails, reason about why and try a different approach rather \
than repeating the same call.
- When you have enough information to answer, call the `final_answer` tool with \
your conclusive answer. Do not simply stop — the answer must be declared.
- If you need clarification from the user before you can proceed, call \
`final_answer` explaining exactly what you need.
"""


def resolve_final_answer_name(bound_names: set[str]) -> str:
    """Pick a terminator name that no bound tool already claims.

    ``final_answer`` is a plausible name for a real MCP tool. If one existed,
    the loop would read that tool's call as a termination signal and never
    execute it, so the collision has to be resolved rather than assumed away.
    """
    name = FINAL_ANSWER_TOOL
    suffix = 0
    while name in bound_names:
        suffix += 1
        name = f"{FINAL_ANSWER_TOOL}_{suffix}"
    return name


def build_final_answer_tool(name: str = FINAL_ANSWER_TOOL) -> dict[str, Any]:
    """The pseudo-tool that terminates the loop.

    Termination is declared rather than inferred. A turn that emits no tool
    calls also ends the loop, but binding an explicit tool means the model has a
    way to signal completion that carries its answer as a payload, and makes the
    terminating turn legible in the trajectory.
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": (
                "Conclude the task. Call this when you have enough information "
                "to answer, or to explain what clarification you need."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The conclusive answer, following any required output format.",
                    }
                },
                "required": ["answer"],
            },
        },
    }


def build_initial_messages(
    sense_output: SenseOutput,
    agent_system_prompt: str = "",
    skill_index: str = "",
) -> list[dict[str, Any]]:
    """Build the opening message history for a ReasonAct run.

    Everything here is a stable prefix: the history only ever grows by appending
    assistant and tool turns, rather than being rebuilt each iteration, so
    provider prompt caching holds across the loop — up until
    ``window_messages`` starts dropping turns past ``scratchpad_window``, from
    which point the sent prefix shifts every turn and the cache is reset.

    *skill_index* is the always-loaded half of Agent Skills (see
    :func:`motoro.engine.skills.render_skill_index`) — names and descriptions
    only. It sits in this prefix precisely because it is stable: the index is
    identical on every turn, so it is cached with the rest of the prefix, while
    the bodies it points at arrive later as ordinary tool results and are never
    paid for unless the model asks.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": REASON_ACT_SYSTEM_PROMPT},
    ]

    if agent_system_prompt:
        messages.append({"role": "system", "content": f"Agent instructions: {agent_system_prompt}"})

    if skill_index:
        messages.append({"role": "system", "content": skill_index})

    context_parts: list[str] = [f"Goal: {sense_output.agent_goal}"]

    if sense_output.memories:
        memory_texts = [str(m.get("content", "")) for m in sense_output.memories]
        context_parts.append("Relevant memories:\n" + "\n".join(f"- {t}" for t in memory_texts))

    messages.append({"role": "system", "content": "\n\n".join(context_parts)})
    messages.append({"role": "user", "content": sense_output.user_input})

    return messages


def window_messages(messages: list[dict[str, Any]], window: int) -> list[dict[str, Any]]:
    """Keep the leading system/user prefix plus the last *window* assistant turns.

    Truncation drops history, so it is only applied when the caller asks for it.
    Tool-result turns are kept with the assistant turn that requested them —
    an orphaned ``role: "tool"`` message is a provider error, not just noise.
    A ``window`` of 0 keeps the prefix only, which is what
    ``include_scratchpad: false`` means: no memory of previous turns.
    """
    prefix_end = 0
    for i, msg in enumerate(messages):
        if msg.get("role") in ("system", "user"):
            prefix_end = i + 1
        else:
            break

    prefix = messages[:prefix_end]
    rest = messages[prefix_end:]

    if window <= 0:
        return prefix

    assistant_indices = [i for i, m in enumerate(rest) if m.get("role") == "assistant"]
    if len(assistant_indices) <= window:
        return messages

    start = assistant_indices[-window]
    return prefix + rest[start:]


def format_tool_result(result: str, success: bool, observation_format: str, max_chars: int) -> str:
    """Render a tool result for its ``role: "tool"`` turn.

    ``summarized`` head-truncates long results — ids and keys a later turn needs
    sit near the start. ``raw`` passes the result through untouched.
    """
    text = result or ""
    if not text:
        text = "The tool call failed with no output." if not success else EMPTY_RESULT_PLACEHOLDER
    if observation_format == "summarized" and max_chars and len(text) > max_chars:
        text = text[:max_chars] + f"\n…[truncated {len(text) - max_chars} chars]"
    return text
