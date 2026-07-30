"""Pydantic schemas for LLM calls and agent phase outputs."""

import json
import re
from typing import Any

from pydantic import BaseModel, Field, model_validator


class LLMCallRecord(BaseModel):
    """Record of a single LLM API call."""

    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    cost_estimate: float = 0.0
    pricing_source: str = "litellm"
    # Prompt-caching token accounting (Anthropic / OpenAI cached-input pricing).
    # Both default to 0 so existing records remain valid without migration.
    cache_read_input_tokens: int = Field(
        default=0,
        description="Tokens served from the provider's prompt cache (Anthropic / OpenAI).",
    )
    cache_creation_input_tokens: int = Field(
        default=0,
        description="Tokens written into the provider's prompt cache on this call.",
    )
    # Structured-output failure attribution (#652)
    failure_reason: str | None = Field(
        default=None,
        description="Last invalid raw text when Instructor validation exhausts retries.",
    )


class LLMToolCall(BaseModel):
    """A single native tool call requested by the model."""

    id: str = Field(default="", description="Provider-assigned call id, echoed back on the tool result turn")
    name: str = Field(description="Tool name as declared in the bound tool schemas")
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCompletion(BaseModel):
    """Result of a native tool-calling completion.

    ``text`` and ``tool_calls`` are *not* mutually exclusive: providers return
    assistant prose alongside tool calls in the same completion. Callers that
    implement a reasoning loop rely on that pairing — the prose is the model's
    reasoning for the calls it made in the same turn.
    """

    text: str = Field(default="", description="Assistant prose accompanying the tool calls, if any")
    tool_calls: list[LLMToolCall] = Field(default_factory=list)
    record: LLMCallRecord

    @property
    def has_tool_calls(self) -> bool:
        """True when the model requested at least one tool call."""
        return bool(self.tool_calls)


# --- Sense Phase ---


class SenseOutput(BaseModel):
    """Output of the Sense phase — structured perception."""

    user_input: str
    agent_goal: str
    system_prompt: str
    conversation_history: list[dict[str, object]] = Field(default_factory=list)
    available_tools: list[dict[str, object]] = Field(default_factory=list)
    memories: list[dict[str, object]] = Field(default_factory=list)


# --- Reason Phase ---


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort recovery of a JSON object from tag/fence-wrapped LLM text.

    Models sometimes wrap the JSON in a single XML/pseudo-tag (``<ReasonOutput>
    …</ReasonOutput>``) or a markdown code fence, or surround it with prose.
    Strip those and return the first JSON object found, or ``None``.
    """
    t = text.strip()
    # Unwrap a single enclosing <tag>…</tag> block.
    t = re.sub(r"^<([A-Za-z_][\w-]*)>\s*(.*?)\s*</\1>\s*$", r"\2", t, flags=re.S).strip()
    # Strip a leading markdown code fence (``` or ```json).
    if t.startswith("```"):
        t = t.strip("`").strip()
        if t[:4].lower() == "json":
            t = t[4:].strip()
    candidates = [t]
    braces = re.search(r"\{.*\}", t, re.S)
    if braces:
        candidates.append(braces.group(0))
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _salvage_strategy(data: dict[str, Any]) -> str:
    """Recover a non-empty strategy from adjacent reasoning fields.

    When a model emits ``strategy`` outside the JSON object (e.g. in a sibling
    XML tag), Instructor builds a dict that is missing it. Rather than discard
    the model's actual reasoning, reuse its own ``reasoning_trace`` or
    ``key_observations`` content. Returns ``""`` when nothing usable is present
    (the caller then lets validation fail and the Reason phase degrades — the
    llm_service normalization safety net).
    """
    trace = str(data.get("reasoning_trace") or "").strip()
    if trace:
        return trace[:500]
    observations = data.get("key_observations")
    if isinstance(observations, list):
        joined = "; ".join(str(o).strip() for o in observations if str(o).strip())
        if joined:
            return joined[:500]
    return ""


class ReasonOutput(BaseModel):
    """Output of the Reason phase — strategic analysis."""

    strategy: str = Field(description="High-level strategy for achieving the goal")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Calibrated confidence in the strategy. "
            "Most requests should score 0.6-0.85. "
            "Above 0.90 only if there is truly zero ambiguity."
        ),
    )
    key_observations: list[str] = Field(default_factory=list, description="Key observations from the input")
    requires_tools: bool = Field(default=False, description="Whether the strategy requires tool usage")
    reasoning_trace: str = Field(default="", description="Chain-of-thought reasoning for evaluation")

    @model_validator(mode="before")
    @classmethod
    def _tolerate_malformed_output(cls, data: Any) -> Any:
        """Salvage tag/fence-wrapped or partially-malformed model output.

        Models occasionally wrap the JSON in XML tags or a code fence, or emit
        ``strategy`` outside the JSON object, so the parsed dict is missing the
        required field. Recover the model's own content where possible instead
        of hard-failing; when nothing is salvageable, validation still fails and
        the Reason phase falls back to a degraded output.
        """
        if isinstance(data, str):
            data = _extract_json_object(data) or {"reasoning_trace": data.strip()}
        if not isinstance(data, dict):
            return data
        if not str(data.get("strategy") or "").strip():
            salvaged = _salvage_strategy(data)
            if salvaged:
                data = {**data, "strategy": salvaged}
        return data


# --- Plan Phase ---


class PlanStep(BaseModel):
    """A single executable step in the agent's plan."""

    action: str = Field(description="What to do in this step")
    description: str = Field(description="Detailed description of the step")
    tool_name: str | None = Field(default=None, description="MCP tool to invoke, if any")
    tool_args: dict[str, object] | None = Field(default=None, description="Arguments for the tool call")
    expected_outcome: str = Field(default="", description="What this step should produce")


class PlanOutput(BaseModel):
    """Output of the Plan phase — ordered executable steps."""

    steps: list[PlanStep] = Field(description="Ordered list of steps to execute")
    is_complete: bool = Field(
        default=False,
        description="True if the goal is already satisfied and no action is needed",
    )
    completion_reason: str | None = Field(
        default=None, description="Why the goal is already complete, if is_complete=True"
    )


# --- Act Phase ---


class ToolCallRecord(BaseModel):
    """Record of a tool invocation during the Act phase."""

    server: str = ""
    tool: str
    arguments: dict[str, object] = Field(default_factory=dict)
    result: str = ""
    latency_ms: int = 0
    success: bool = True
    error_type: str | None = Field(
        default=None,
        description="Error type: validation, timeout, connection, server_error, not_found",
    )


def flatten_tool_call_records(tool_call: object) -> list[dict[str, Any]]:
    """Flatten a persisted ``RunStep.tool_call`` value into individual calls.

    A step that invoked one tool stores a bare :class:`ToolCallRecord` dump; a
    step that invoked several stores ``{"calls": [...], "success": bool}``.
    Consumers that only understand the bare shape count a three-call step as
    one call, and read its name from a ``name`` key that does not exist —
    the field is ``tool``. Returns ``[]`` for anything that is neither shape.
    """
    if not isinstance(tool_call, dict):
        return []
    calls = tool_call.get("calls")
    if isinstance(calls, list):
        return [c for c in calls if isinstance(c, dict)]
    return [tool_call]


class StepResult(BaseModel):
    """Result of executing a single plan step."""

    step_index: int
    action: str
    result: str
    success: bool = True
    tool_call: ToolCallRecord | None = None


class ActOutput(BaseModel):
    """Output of the Act phase — execution results."""

    results: list[StepResult] = Field(description="Results for each executed step")
    final_response: str = Field(description="Aggregated final response to the user")
    should_continue: bool = Field(default=False, description="Whether the loop should run another iteration")
