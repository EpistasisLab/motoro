"""Build the run output envelope and (optionally) read its domain payload.

The universal envelope (:mod:`motoro.schemas.output`) is assembled from
data already in the finished run — no LLM call. When the agent declares an
``output_contract``, the contracted fields are read out of the finished text by
one of two routes, in this order:

1. **Inline** (:func:`parse_payload_inline`) — the caller has told the agent to
   end its reply with a fenced JSON object holding those values, so reading them
   is a ``json.loads`` and a schema validation. Free.
2. **Extraction** (:func:`extract_payload`) — a second LLM pass that coerces the
   free text into the contracted fields. This is the fallback for a reply that
   arrived without a usable block, and for callers that never asked for one.

Route 1 exists because route 2 used to be the only one: every contracted run
paid for a second model call over an answer that had already been written, and
(before the caller began naming the fields up front) the extractor was reading
prose with no particular reason to contain them. Asking for the values and
reading them back are now the same request.

Two rules keep the "arbitrary goal" caveat away, and hold on both routes:

1. **Absence is a valid value.** Every contracted field is optional, so the
   model can report "not present" instead of fabricating one.
2. **Extraction, not generation.** The prompt forbids inference — a field is
   filled only when the text explicitly supports it. If extraction fails
   entirely, the payload degrades to ``None`` with a caveat; the run is never
   killed.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any
from uuid import UUID

import structlog
from pydantic import BaseModel, Field, ValidationError, create_model

from motoro.models.run import RunStatus
from motoro.schemas.llm import flatten_tool_call_records
from motoro.schemas.output import (
    STATUS_COMPLETE,
    OutputArtifact,
    OutputEnvelope,
)

if TYPE_CHECKING:
    from motoro.engine.runtime import AgentRunResult
    from motoro.models.agent import Agent
    from motoro.schemas.agent import ModelConfig
    from motoro.services.llm_service import LLMService

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


# A fenced code block, optionally tagged ```json. Non-greedy so several blocks
# in one reply stay separate; the last usable one wins (see below).
_JSON_BLOCK_RE = re.compile(r"```[ \t]*(?:json)?[ \t]*\r?\n(.*?)\r?\n?[ \t]*```", re.DOTALL | re.IGNORECASE)


def parse_payload_inline(contract: dict[str, Any], result_text: str) -> tuple[dict[str, Any] | None, str]:
    """Read the contracted payload out of a fenced JSON block in *result_text*.

    Returns ``(payload, prose)``. ``payload`` is ``None`` when no block in the
    text is usable, in which case ``prose`` is *result_text* unchanged and the
    caller should fall back to :func:`extract_payload`. Never raises.

    The block is **removed** from the prose that goes on to the envelope's
    ``result``. It is machine punctuation the caller asked for, not part of the
    answer: whoever reads this run next wants the reply, and gets the values
    through ``payload`` instead. Leaving it in would put a JSON dump into the
    next agent's prompt, which is the noise the payload exists to avoid.

    Three guards stop this from claiming JSON it has no business claiming, all
    of them deliberately biased towards falling through to the model:

    * The parsed value must be a JSON **object**. A list or a bare string is not
      a payload.
    * It must share **at least one key** with the contract. An agent that
      happened to end its answer with an unrelated config sample has not
      answered the contract, and silently reporting that sample as the payload
      would be worse than paying for the extraction pass.
    * It must **validate** against the contract's model. A block with the right
      keys and the wrong types is a malformed answer, and the extractor gets a
      chance to do better with it.

    The **last** usable block wins: a reply that shows an example block mid-
    answer and states its real values at the end means the second one.
    """
    try:
        model = _model_from_contract(contract)
    except Exception:  # malformed contract -- let extract_payload report it
        return None, result_text
    declared = {str(spec["name"]) for spec in contract.get("fields", []) if spec.get("name")}
    if not declared:
        return None, result_text

    for match in reversed(list(_JSON_BLOCK_RE.finditer(result_text))):
        try:
            raw = json.loads(match.group(1))
        except ValueError:
            continue
        if not isinstance(raw, dict) or not declared & raw.keys():
            continue
        try:
            obj = model.model_validate(raw)
        except ValidationError:
            continue
        head = result_text[: match.start()].rstrip()
        tail = result_text[match.end() :].lstrip()
        prose = f"{head}\n\n{tail}".strip() if head and tail else (head or tail)
        return obj.model_dump(mode="json"), prose
    return None, result_text


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
    result_text: str | None = None,
) -> OutputEnvelope:
    """Assemble the universal envelope from finished-run data (no LLM call).

    *result_text* overrides ``result.output`` for the envelope's ``result``,
    which is how :func:`parse_payload_inline`'s prose (the reply minus the JSON
    block it read) reaches the envelope without mutating the run result itself.
    """
    return OutputEnvelope(
        status=status,
        result=(result.output or "") if result_text is None else result_text,
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

    Inline first, model second -- see this module's docstring. The fallback is
    kept rather than made an error the way a strict parser would: a caller that
    never asked the agent for a JSON block still gets its payload, just at the
    old price, so turning a contract on can never *stop* working.
    """
    if str(result.status) != RunStatus.COMPLETED.value:
        return result.output

    payload: dict[str, Any] | None = None
    caveats: list[str] = []
    result_text = result.output or ""
    contract = getattr(agent, "output_contract", None)
    if contract:
        payload, result_text = parse_payload_inline(contract, result_text)
        if payload is None:
            payload, caveats = await extract_payload(
                llm, model_config, contract, result_text, principal_id=principal_id
            )
            if payload is not None:
                # Worth saying out loud: this run cost a second model call that
                # a reply carrying the block would not have. It is the one
                # signal that the instruction did not land.
                caveats = [*caveats, "payload read by a second model call: the answer carried no usable JSON block"]

    return build_envelope(
        result, status=STATUS_COMPLETE, payload=payload, caveats=caveats, result_text=result_text
    ).to_json()
