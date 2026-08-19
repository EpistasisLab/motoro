"""The envelope discriminator is persisted, so it has to survive a rename.

``kind`` lives inside ``agent_runs.output`` for every run ever completed. When
the project was renamed agentic-core -> Motoro, ``ENVELOPE_KIND`` changed with
it — and a strict equality check against the new value alone would have made
every pre-rename envelope parse as "not an envelope", so ``output_text`` would
have returned the raw JSON blob instead of ``result`` for all of them. These
tests pin the read-both/write-new behaviour that prevents that.
"""

from __future__ import annotations

import json

from motoro.schemas.output import (
    ENVELOPE_KIND,
    LEGACY_ENVELOPE_KINDS,
    OutputEnvelope,
    output_text,
    parse_envelope,
)

LEGACY_KIND = "agentic_core.output.envelope/v1"


def test_legacy_kind_is_still_accepted() -> None:
    assert LEGACY_KIND in LEGACY_ENVELOPE_KINDS


def test_parses_a_pre_rename_envelope() -> None:
    raw = json.dumps({"kind": LEGACY_KIND, "status": "complete", "result": "the answer"})
    envelope = parse_envelope(raw)
    assert envelope is not None
    assert envelope.result == "the answer"
    # Normalized at the parse boundary: downstream sees exactly one name.
    assert envelope.kind == ENVELOPE_KIND


def test_output_text_recovers_result_from_a_pre_rename_envelope() -> None:
    raw = json.dumps({"kind": LEGACY_KIND, "status": "complete", "result": "the answer"})
    assert output_text(raw) == "the answer"


def test_current_kind_still_round_trips() -> None:
    raw = OutputEnvelope(result="fresh").to_json()
    assert json.loads(raw)["kind"] == ENVELOPE_KIND
    assert output_text(raw) == "fresh"


def test_an_unrelated_json_output_is_not_an_envelope() -> None:
    assert parse_envelope(json.dumps({"kind": "something.else/v1", "result": "x"})) is None
    assert parse_envelope(json.dumps({"result": "x"})) is None
    # Prose passes through untouched rather than raising.
    assert output_text("just text") == "just text"
