"""Reading a contracted payload out of the answer itself, without a model call.

The caller tells the agent to end its reply with a fenced JSON object holding
the contracted values; `parse_payload_inline` reads that block and strips it
back off. What these tests pin down is mostly the *refusals* -- every case where
the block is absent, unrelated or malformed has to fall through to the
extraction pass rather than report a confident wrong payload, because a silently
wrong payload is worse than the second model call it was avoiding.
"""

from __future__ import annotations

from motoro.services.output_contract import parse_payload_inline

CONTRACT = {
    "name": "Profile",
    "fields": [
        {"name": "n_rows", "type": "int"},
        {"name": "n_cols", "type": "int"},
    ],
}


def test_a_trailing_block_is_read_and_stripped() -> None:
    payload, prose = parse_payload_inline(
        CONTRACT,
        'The dataset is wide and shallow.\n\n```json\n{"n_rows": 4300, "n_cols": 57}\n```',
    )
    assert payload == {"n_rows": 4300, "n_cols": 57}
    assert prose == "The dataset is wide and shallow."


def test_an_untagged_fence_counts() -> None:
    """```json is what we ask for; a bare ``` is what models routinely send."""
    payload, _ = parse_payload_inline(CONTRACT, '```\n{"n_rows": 1, "n_cols": 2}\n```')
    assert payload == {"n_rows": 1, "n_cols": 2}


def test_a_missing_field_is_null_not_a_failure() -> None:
    """Absence is a valid value -- the whole contract is optional, so a partial
    block is a partial answer, not a malformed one."""
    payload, _ = parse_payload_inline(CONTRACT, '```json\n{"n_rows": 4300}\n```')
    assert payload == {"n_rows": 4300, "n_cols": None}


def test_the_last_usable_block_wins() -> None:
    """A reply that demonstrates the format mid-answer and then fills it in
    means the second one."""
    payload, prose = parse_payload_inline(
        CONTRACT,
        'Format:\n```json\n{"n_rows": null, "n_cols": null}\n```\nHere it is:\n'
        '```json\n{"n_rows": 9, "n_cols": 3}\n```',
    )
    assert payload == {"n_rows": 9, "n_cols": 3}
    # Only the block it read is removed; the earlier one is still the agent's
    # own prose and none of our business.
    assert "Format:" in prose and '"n_rows": null' in prose


def test_prose_on_both_sides_of_the_block_is_rejoined() -> None:
    payload, prose = parse_payload_inline(
        CONTRACT, 'Before.\n\n```json\n{"n_rows": 1}\n```\n\nAfter.'
    )
    assert payload == {"n_rows": 1, "n_cols": None}
    assert prose == "Before.\n\nAfter."


# ----------------------------------------------------------------------
# Falling through to the extraction pass
# ----------------------------------------------------------------------


def test_no_block_at_all_falls_through_untouched() -> None:
    text = "The dataset has 4,300 rows and 57 columns."
    assert parse_payload_inline(CONTRACT, text) == (None, text)


def test_an_unrelated_json_block_is_not_claimed_as_the_payload() -> None:
    """An agent that ends on a config sample has not answered the contract.
    Reporting that sample would be a confident lie; the extractor gets a turn."""
    text = 'Use this config:\n```json\n{"host": "localhost", "port": 5432}\n```'
    assert parse_payload_inline(CONTRACT, text) == (None, text)


def test_a_json_array_is_not_a_payload() -> None:
    text = '```json\n[1, 2, 3]\n```'
    assert parse_payload_inline(CONTRACT, text) == (None, text)


def test_malformed_json_falls_through() -> None:
    text = '```json\n{"n_rows": 4300,\n```'
    assert parse_payload_inline(CONTRACT, text) == (None, text)


def test_right_keys_wrong_types_falls_through() -> None:
    """The extractor may well do better with "four thousand" than we can."""
    text = '```json\n{"n_rows": "four thousand", "n_cols": 57}\n```'
    assert parse_payload_inline(CONTRACT, text) == (None, text)


def test_a_contract_with_no_fields_declares_nothing_to_find() -> None:
    text = '```json\n{"n_rows": 1}\n```'
    assert parse_payload_inline({"name": "Empty", "fields": []}, text) == (None, text)


def test_extra_keys_are_dropped_rather_than_disqualifying_the_block() -> None:
    payload, _ = parse_payload_inline(
        CONTRACT, '```json\n{"n_rows": 4300, "n_cols": 57, "notes": "wide"}\n```'
    )
    assert payload == {"n_rows": 4300, "n_cols": 57}
