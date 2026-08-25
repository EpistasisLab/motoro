"""motoro.mcp_servers.okf — no MCP transport needed: ``@mcp.tool()``
returns the plain function unchanged, so every tool is called directly.

Covers: path jailing, progressive-disclosure discovery, the attested-
computation write guard (the actual enforcement of the OKF spec's "MAY only
supply values, MUST NOT edit the computation" rule), actor-string resolution
from ambient _meta, log.md attribution, and reference-file read/write.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import uuid
from pathlib import Path

import pytest

from motoro.mcp_servers import okf

_OKF_SERVER_MODULE = Path(__file__).parent.parent / "src" / "motoro" / "mcp_servers" / "okf.py"
_OKF_COMMAND = f"{sys.executable} {_OKF_SERVER_MODULE}"
_LIVE_TIMEOUT = 15.0


async def _with_timeout(coro):
    return await asyncio.wait_for(coro, timeout=_LIVE_TIMEOUT)


@pytest.fixture(autouse=True)
def _bundle(tmp_path, monkeypatch):
    monkeypatch.setenv(okf._BUNDLE_ROOT_ENV, str(tmp_path))
    return tmp_path


def _load(result: str) -> dict:
    return json.loads(result)


class _FakeMeta:
    def __init__(self, model_extra):
        self.model_extra = model_extra


class _FakeRequestContext:
    def __init__(self, model_extra):
        self.meta = _FakeMeta(model_extra)


class _FakeCtx:
    def __init__(self, model_extra):
        self.request_context = _FakeRequestContext(model_extra)


def test_create_and_get_concept_round_trip():
    created = _load(okf.create_concept("model", "XGBoost v1", "models/xgb-v1", description="d", tags="a, b"))
    assert created["created"] is True
    assert created["generated"]["by"] == "process:unknown"

    got = _load(okf.get_concept("models/xgb-v1"))
    assert got["type"] == "model"
    assert got["title"] == "XGBoost v1"
    assert got["tags"] == ["a", "b"]
    assert got["id"] == "models/xgb-v1"


def test_create_concept_refuses_an_existing_path():
    okf.create_concept("model", "v1", "models/x")
    result = _load(okf.create_concept("model", "v1 again", "models/x"))
    assert "error" in result
    assert "already exists" in result["error"]


def test_get_concept_rejects_path_traversal():
    result = _load(okf.get_concept("../../../etc/passwd"))
    assert "error" in result


def test_list_concepts_filters_by_type_and_tags():
    okf.create_concept("model", "A", "a", tags="x,y")
    okf.create_concept("dataset", "B", "b", tags="y,z")
    okf.create_concept("model", "C", "c", tags="z")

    only_models = _load(okf.list_concepts(type="model"))
    assert {c["id"] for c in only_models["concepts"]} == {"a", "c"}

    only_y = _load(okf.list_concepts(tags="y"))
    assert {c["id"] for c in only_y["concepts"]} == {"a", "b"}


def test_list_concepts_skips_malformed_files_gracefully(_bundle):
    (_bundle / "broken.md").write_text("no frontmatter here at all")
    okf.create_concept("model", "Fine", "fine")
    result = _load(okf.list_concepts())
    assert result["count"] == 1
    assert result["concepts"][0]["id"] == "fine"


def test_reserved_root_files_are_not_concepts(_bundle):
    (_bundle / "index.md").write_text("---\ntype: index\n---\nnot a concept")
    (_bundle / "log.md").write_text("# Log\n")
    okf.create_concept("model", "Real", "real")
    result = _load(okf.list_concepts())
    assert result["count"] == 1
    assert result["concepts"][0]["id"] == "real"


def test_search_concepts_matches_title_description_and_body():
    okf.create_concept("model", "Discharge Risk", "risk", description="predicts discharge", body="uses XGBoost")
    okf.create_concept("model", "Unrelated", "unrelated")

    by_title = _load(okf.search_concepts("discharge"))
    assert [c["id"] for c in by_title["concepts"]] == ["risk"]

    by_body = _load(okf.search_concepts("xgboost"))
    assert [c["id"] for c in by_body["concepts"]] == ["risk"]

    none = _load(okf.search_concepts("nonexistent-term"))
    assert none["concepts"] == []


def test_update_concept_bumps_generated_and_merges_fields():
    okf.create_concept("model", "A", "a", description="old")
    updated = _load(okf.update_concept("a", fields={"description": "new"}))
    assert updated["updated"] is True
    got = _load(okf.get_concept("a"))
    assert got["description"] == "new"
    assert got["title"] == "A"  # untouched fields survive the merge


def test_update_concept_body_replace_and_append():
    okf.create_concept("model", "A", "a", body="original")
    okf.update_concept("a", append_body=" more")
    assert _load(okf.get_concept("a"))["body"] == "original more"

    okf.update_concept("a", body="replaced")
    assert _load(okf.get_concept("a"))["body"] == "replaced"


def test_update_concept_refuses_to_touch_an_attested_computations_definition():
    okf.create_concept("computation", "Calc", "calc")
    # First write to attester/parameters succeeds -- nothing attested yet.
    first = _load(
        okf.update_concept("calc", fields={"attester": "sql-equality", "parameters": [{"name": "threshold"}]})
    )
    assert first["updated"] is True

    # Now that it's an attested computation, touching those fields again is refused.
    refused = _load(okf.update_concept("calc", fields={"parameters": [{"name": "other"}]}))
    assert "error" in refused
    assert "attested computation" in refused["error"]

    # An unrelated field is still editable.
    ok = _load(okf.update_concept("calc", fields={"description": "fine"}))
    assert ok["updated"] is True


def test_supply_computation_value_validates_declared_parameters():
    okf.create_concept("computation", "Calc", "calc")
    okf.update_concept("calc", fields={"attester": "sql-equality", "parameters": [{"name": "threshold"}]})

    ok = _load(okf.supply_computation_value("calc", "threshold", 0.5))
    assert ok["values"] == {"threshold": 0.5}

    rejected = _load(okf.supply_computation_value("calc", "not_a_real_param", 1))
    assert "error" in rejected
    assert "not a declared parameter" in rejected["error"]

    # Supplying a value never touches parameters/attester themselves.
    got = _load(okf.get_concept("calc"))
    assert got["parameters"] == [{"name": "threshold"}]
    assert got["attester"] == "sql-equality"


def test_mark_verified_does_not_touch_generated_or_body():
    okf.create_concept("model", "A", "a", body="body text")
    before = _load(okf.get_concept("a"))

    okf.mark_verified("a", note="looks right")
    after = _load(okf.get_concept("a"))

    assert after["generated"] == before["generated"]
    assert after["body"] == before["body"]
    assert after["verified"]["by"] == "process:unknown"

    # A second verification accumulates as a list, not a silent overwrite.
    okf.mark_verified("a")
    twice = _load(okf.get_concept("a"))
    assert isinstance(twice["verified"], list)
    assert len(twice["verified"]) == 2


def test_a_yaml_native_timestamp_does_not_break_a_read(_bundle):
    # An unquoted `at:` is valid YAML that safe_load resolves to a datetime, so
    # a hand-authored bundle used to fail every read with "Object of type
    # datetime is not JSON serializable" -- the tool raised while serializing,
    # not while parsing, so nothing came back at all.
    (_bundle / "hand-written.md").write_text(
        "---\n"
        "type: Metric\n"
        "title: Activation Rate\n"
        "generated:\n"
        "  by: process:okf-test-generator\n"
        "  at: 2026-08-25T16:00:00Z\n"
        "verified:\n"
        "  - by: human:someone\n"
        "    at: 2026-08-26\n"
        "---\n"
        "# Definition\n"
    )
    listed = _load(okf.list_concepts())
    assert [c["id"] for c in listed["concepts"]] == ["hand-written"]
    # ISO 8601, matching the string form this server writes itself, rather than
    # YAML's space-separated str(datetime).
    assert listed["concepts"][0]["generated"]["at"].startswith("2026-08-25T16:00:00")
    assert listed["concepts"][0]["verified"][0]["at"] == "2026-08-26"

    # Every read, not just the one that was reported: search and get share the
    # same encoder.
    assert [c["id"] for c in _load(okf.search_concepts("activation"))["concepts"]] == ["hand-written"]
    assert _load(okf.get_concept("hand-written"))["generated"]["at"].startswith("2026-08-25T16:00:00")

    # And a write path that echoes frontmatter back still answers.
    assert _load(okf.mark_verified("hand-written"))["verified"][0]["at"] == "2026-08-26"


def test_get_concept_resolves_outgoing_links():
    okf.create_concept("model", "A", "a", body="see [the dataset](../datasets/b.md) for details")
    got = _load(okf.get_concept("a"))
    assert got["links"] == ["../datasets/b.md"]


def test_reference_read_write_round_trip_text_and_binary():
    write = _load(okf.write_reference("references/scripts/x.py", "print('hi')"))
    assert write["written"] is True
    read = _load(okf.get_reference("references/scripts/x.py"))
    assert read == {"encoding": "text", "content": "print('hi')"}

    import base64

    # 0xFF is never a valid UTF-8 leading byte -- genuinely undecodable,
    # unlike low control bytes (which are valid, if odd, UTF-8).
    raw = b"\xff\xfe\x00\x01binary"
    payload = base64.b64encode(raw).decode("ascii")
    okf.write_reference("references/blobs/x.bin", payload, encoding="base64")
    read_bin = _load(okf.get_reference("references/blobs/x.bin"))
    assert read_bin["encoding"] == "base64"
    assert base64.b64decode(read_bin["content"]) == raw


def test_reference_paths_are_confined_to_references_dir():
    result = _load(okf.write_reference("not-references/x.py", "content"))
    assert "error" in result
    result2 = _load(okf.get_reference("../outside.py"))
    assert "error" in result2


def test_actor_from_ctx_prefers_agent_name_then_owner_then_unknown():
    with_agent = _FakeCtx({okf.META_KEY_AGENT_NAME: "SF-DC", okf.META_KEY_MODEL: "claude-sonnet-5"})
    assert okf._actor_from_ctx(with_agent) == "SF-DC/claude-sonnet-5"

    owner_only = _FakeCtx({okf.META_KEY_OWNER_ID: "11111111-1111-1111-1111-111111111111"})
    assert okf._actor_from_ctx(owner_only) == "process:11111111-1111-1111-1111-111111111111"

    assert okf._actor_from_ctx(None) == "process:unknown"
    assert okf._actor_from_ctx(_FakeCtx({})) == "process:unknown"


def test_create_concept_stamps_actor_from_ctx():
    ctx = _FakeCtx({okf.META_KEY_AGENT_NAME: "SF-DC", okf.META_KEY_MODEL: "claude-sonnet-5"})
    created = _load(okf.create_concept("model", "A", "a", ctx=ctx))
    assert created["generated"]["by"] == "SF-DC/claude-sonnet-5"


def test_log_md_records_creation_and_update_entries(_bundle):
    okf.create_concept("model", "A", "a")
    okf.update_concept("a", fields={"description": "x"})
    log_text = (_bundle / "log.md").read_text()
    assert "**Creation** `a`" in log_text
    assert "**Update** `a`" in log_text


def test_concurrent_updates_to_the_same_concept_are_serialized(_bundle):
    okf.create_concept("model", "A", "a", body="")

    results: list[str] = []

    def _writer(tag: str) -> None:
        okf.update_concept("a", append_body=tag)
        results.append(tag)

    threads = [threading.Thread(target=_writer, args=(str(i),)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    body = _load(okf.get_concept("a"))["body"]
    # Every writer's tag landed exactly once -- a lost update (two writers
    # reading the same pre-write body and clobbering each other) would show
    # up as a missing tag here.
    for i in range(8):
        assert body.count(str(i)) == 1, f"tag {i} missing or duplicated: {body!r}"


def test_bundle_root_env_var_required(monkeypatch):
    monkeypatch.delenv(okf._BUNDLE_ROOT_ENV, raising=False)
    result = _load(okf.list_concepts())
    assert "error" in result
    assert okf._BUNDLE_ROOT_ENV in result["error"]


async def test_live_transport_delivers_ambient_meta_to_actor_string(tmp_path):
    """Everything above calls the tool functions directly -- real, but it
    never proves the wire path: that a real MCPClient.call_tool(meta=...)
    against a real subprocess actually arrives as ctx.request_context.meta,
    the way _actor_from_ctx assumes. Same regression class as
    test_call_tool_delivers_meta_verbatim (test_mcp_service.py) -- a
    consumer's own reading of an ambient meta key can silently drift from
    what the sender actually delivers without either side erroring."""
    from motoro.mcp.registry import MCPServerRegistry

    registry = MCPServerRegistry()
    name = f"okf-{uuid.uuid4().hex[:8]}"
    await _with_timeout(
        registry.register(
            name=name,
            command=_OKF_COMMAND,
            server_env={okf._BUNDLE_ROOT_ENV: str(tmp_path)},
        )
    )
    try:
        client = registry.servers[name].client
        sent_meta = {okf.META_KEY_AGENT_NAME: "SF-DC", okf.META_KEY_MODEL: "claude-opus-5"}
        result = await _with_timeout(
            client.call_tool(
                "create_concept",
                {"type": "model", "title": "A", "path": "a"},
                meta=sent_meta,
            )
        )
        assert not result.is_error, result.content
        created = json.loads(result.content)
        assert created["generated"]["by"] == "SF-DC/claude-opus-5"
    finally:
        await registry.disconnect_all()
