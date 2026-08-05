"""Build the run output envelope and (optionally) extract its domain payload.

The universal envelope (:mod:`agentic_core.schemas.output`) is assembled from
data already in the finished run — no LLM call. When the agent declares an
``output_contract``, we additionally run one *extraction* pass that coerces the
agent's free-text output into the contracted fields.

Two rules keep the "arbitrary goal" caveat away:

1. **Absence is a valid value.** Every contracted field is optional, so the
   model can report "not present" instead of fabricating one.
2. **Extraction, not generation.** The prompt forbids inference — a field is
   filled only when the text explicitly supports it. If extraction fails
   entirely, the payload degrades to ``None`` with a caveat; the run is never
   killed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

import structlog
from pydantic import BaseModel, Field, ValidationError, create_model

from agentic_core.models.run import RunStatus
from agentic_core.schemas.llm import flatten_tool_call_records
from agentic_core.schemas.output import (
    STATUS_COMPLETE,
    OutputArtifact,
    OutputEnvelope,
)

if TYPE_CHECKING:
    from agentic_core.engine.runtime import AgentRunResult
    from agentic_core.models.agent import Agent
    from agentic_core.schemas.agent import ModelConfig
    from agentic_core.services.llm_service import LLMService

log = structlog.get_logger()

# Contract field ``type`` -> Python type. Unknown types fall back to ``str``.
_TYPE_MAP: dict[str, type] = {
    "str": str,
    "string": str,
    "int": int,
    "integer": int,
    "float": float,
    "number": float,
    "bool": bool,
    "boolean": bool,
    "list": list,
    "array": list,
    "dict": dict,
    "object": dict,
}

_EXTRACT_SYSTEM = (
    "You convert an agent's final output into a structured object. Fill a field "
    "ONLY if the text explicitly and clearly supports a value; otherwise leave it "
    "null (or its stated default). Do NOT infer, guess, or invent values, and do "
    "NOT decide anything yourself — you are extracting what the text already "
    "states, nothing more."
)

# Ready-made contracts for common roles so callers need not hand-author field
# specs. A contract is ``{"name": str, "fields": [{"name", "type", "default"?,
# "description"?}, ...]}``; every field is treated as optional at extraction.
ROLE_CONTRACTS: dict[str, dict[str, Any]] = {
    "critic": {
        "name": "CriticVerdict",
        "fields": [
            {
                "name": "approved",
                "type": "bool",
                "description": (
                    "true if the reviewed output passes every criterion, false if it "
                    "should be revised, null if the text does not state an overall verdict."
                ),
            },
            {
                "name": "feedback",
                "type": "str",
                "default": "",
                "description": "Actionable revision instructions when not approved; empty when approved.",
            },
        ],
    },
}


def _model_from_contract(contract: dict[str, Any]) -> type[BaseModel]:
    """Build a Pydantic model from a contract's field specs.

    Every field is optional (``T | None``) so that "not present in the text" is
    always a valid, non-fabricated answer.
    """
    fields: dict[str, Any] = {}
    for spec in contract.get("fields", []):
        name = spec["name"]
        py_type = _TYPE_MAP.get(str(spec.get("type", "str")).lower(), str)
        default = spec.get("default", None)
        fields[name] = (py_type | None, Field(default=default, description=spec.get("description", "")))
    name = str(contract.get("name") or "OutputPayload")
    return create_model(name, **fields)


async def extract_payload(
    llm: LLMService,
    model_config: ModelConfig,
    contract: dict[str, Any],
    result_text: str,
    *,
    principal_id: UUID | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Coerce ``result_text`` into the contracted payload; degrade on failure.

    Returns ``(payload, caveats)``. ``payload`` is ``None`` (with an explanatory
    caveat) if the contract is invalid or extraction cannot produce a valid
    object — never raises.
    """
    try:
        model = _model_from_contract(contract)
    except Exception as exc:  # malformed contract
        return None, [f"invalid output_contract: {str(exc)[:200]}"]

    messages = [
        {"role": "system", "content": _EXTRACT_SYSTEM},
        {"role": "user", "content": f"Agent output to extract from:\n\n{result_text}"},
    ]
    try:
        obj, _record = await llm.complete(
            config=model_config,
            messages=messages,
            response_model=model,
            principal_id=principal_id,
        )
    except ValidationError as exc:
        # llm_service normalizes InstructorRetryException -> ValidationError, so
        # this covers retry exhaustion too.
        log.warning("output_contract.extract_failed", error=str(exc)[:300])
        return None, [f"payload extraction failed: {str(exc)[:200]}"]
    except Exception as exc:  # provider/transport error — degrade, don't kill the run
        log.warning("output_contract.extract_error", error=f"{type(exc).__name__}: {str(exc)[:200]}")
        return None, [f"payload extraction error: {type(exc).__name__}"]

    return obj.model_dump(mode="json"), []


def _artifacts_from_result(result: AgentRunResult) -> list[OutputArtifact]:
    """Best-effort artifact list from the run's tool calls (never raises).

    ``RunStep.tool_call`` is a JSON dict — a ``ToolCallRecord`` dump for a single
    call, or ``{"calls": [...]}`` when a step invoked several — so the tool name
    is read by key, not attribute.
    """
    artifacts: list[OutputArtifact] = []
    for step in getattr(result, "steps", []) or []:
        for call in flatten_tool_call_records(getattr(step, "tool_call", None)):
            tool = call.get("tool")
            if tool:
                artifacts.append(OutputArtifact(kind="tool_result", ref=str(tool)))
    return artifacts


def build_envelope(
    result: AgentRunResult,
    *,
    status: str = STATUS_COMPLETE,
    payload: dict[str, Any] | None = None,
    caveats: list[str] | None = None,
) -> OutputEnvelope:
    """Assemble the universal envelope from finished-run data (no LLM call)."""
    return OutputEnvelope(
        status=status,
        result=result.output or "",
        artifacts=_artifacts_from_result(result),
        payload=payload,
        caveats=caveats or [],
    )


async def finalize_output(
    *,
    llm: LLMService,
    agent: Agent,
    model_config: ModelConfig,
    result: AgentRunResult,
    principal_id: UUID | None = None,
) -> str:
    """Return the string to persist as ``run.output``.

    Completed runs are wrapped in an envelope (with a payload when the agent has
    an ``output_contract``); non-terminal/failed runs keep their raw output so
    resume and error handling are unaffected.
    """
    if str(result.status) != RunStatus.COMPLETED.value:
        return result.output

    payload: dict[str, Any] | None = None
    caveats: list[str] = []
    contract = getattr(agent, "output_contract", None)
    if contract:
        payload, caveats = await extract_payload(
            llm, model_config, contract, result.output or "", principal_id=principal_id
        )

    return build_envelope(result, status=STATUS_COMPLETE, payload=payload, caveats=caveats).to_json()
