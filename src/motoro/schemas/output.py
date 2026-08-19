"""The output envelope: the structured contract every completed run carries.

Every agent produces free text (``engine/act.py`` concatenates plan-step
results into ``final_response``). That makes runs hard to compare and forces
downstream consumers to parse prose. The **envelope** wraps that free text in a
uniform, machine-readable structure on every completed run:

* the universal fields (``status``, ``result``, ``artifacts``, ...) are always
  truthfully fillable from data already in the run — no LLM call needed;
* the optional ``payload`` holds a role/task-specific structured object, filled
  only when the agent declares an ``output_contract`` (see
  :mod:`motoro.services.output_contract`).

``result`` always carries the human-readable text the agent produced (prose *or*
a JSON string), so :func:`output_text` can recover it for any consumer that
still wants plain text.
"""

from __future__ import annotations

import json
from typing import Any, Final, Literal

from pydantic import BaseModel, Field

# Collision-proof discriminator: a stored output is an envelope iff its top-level
# JSON object carries this exact ``kind``. Bump the version suffix on a breaking
# change to the envelope shape.
ENVELOPE_KIND: Final = "motoro.output.envelope/v1"

# Discriminators written by earlier versions, still accepted on read. This one
# is not a stylistic leftover from the agentic-core -> Motoro rename: the kind is
# *persisted* in ``agent_runs.output``, so dropping it would make every envelope
# written before the rename fail :func:`parse_envelope` and start rendering as
# raw JSON everywhere ``output_text`` is used. Old runs are not rewritten — the
# stored bytes stay truthful about what produced them — so this tuple is the only
# thing keeping them readable. Never remove an entry; add to it when bumping
# ENVELOPE_KIND.
LEGACY_ENVELOPE_KINDS: Final = ("agentic_core.output.envelope/v1",)

# Envelope ``status`` values. Kept as a small closed vocabulary rather than an
# enum so it stays trivially JSON-serializable and forward-compatible.
STATUS_COMPLETE = "complete"
STATUS_NEEDS_REVISION = "needs_revision"
STATUS_BLOCKED = "blocked"


class OutputArtifact(BaseModel):
    """A thing the run produced (a dataset, model, file, or tool result)."""

    kind: str = ""
    ref: str = ""
    description: str = ""


class OutputEnvelope(BaseModel):
    """Uniform structured wrapper stored as a run's ``output``."""

    kind: Literal["motoro.output.envelope/v1"] = ENVELOPE_KIND
    status: str = STATUS_COMPLETE
    result: str = ""
    summary: str = ""
    artifacts: list[OutputArtifact] = Field(default_factory=list)
    # Opt-in role/task payload extracted against the agent's output_contract.
    payload: dict[str, Any] | None = None
    confidence: float | None = None
    caveats: list[str] = Field(default_factory=list)

    def to_json(self) -> str:
        return self.model_dump_json()


def parse_envelope(raw: str | None) -> OutputEnvelope | None:
    """Return the envelope encoded in ``raw``, or ``None`` if it is not one.

    Detection is strict — the top-level object must carry :data:`ENVELOPE_KIND`
    or one of :data:`LEGACY_ENVELOPE_KINDS` — so an agent's own JSON output is
    never mistaken for an envelope. A legacy ``kind`` is normalized to the
    current one here, at the storage boundary, so nothing downstream has to know
    more than one name for the same envelope.
    """
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("kind") in LEGACY_ENVELOPE_KINDS:
        data = {**data, "kind": ENVELOPE_KIND}
    elif data.get("kind") != ENVELOPE_KIND:
        return None
    try:
        return OutputEnvelope.model_validate(data)
    except Exception:  # malformed envelope -> treat as non-envelope, never raise
        return None


def output_text(raw: str | None) -> str:
    """Human-readable text for a run output, enveloped or not.

    Idempotent and safe on prose: returns ``envelope.result`` when ``raw`` is an
    envelope, otherwise ``raw`` unchanged. Every consumer that wants plain text
    should read a run's output through this, so the envelope rollout cannot leave
    anything rendering raw JSON.
    """
    envelope = parse_envelope(raw)
    if envelope is not None:
        return envelope.result
    return raw or ""
