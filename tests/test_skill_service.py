"""Agent Skills — the format's rules, the storage round trip, and disclosure.

Four things worth pinning:

1. ``parse_skill_markdown`` refuses what the published format refuses. The
   rules are not decoration: a name that is not ``[a-z0-9-]`` cannot be a
   directory name or a ``load_skill`` argument, and a description is the *only*
   thing an agent sees for a skill it has not opened, so an empty one makes the
   skill unreachable rather than merely undocumented.
2. ``render_skill_md`` round-trips. Storing the file rather than the directory
   is only defensible if a stored skill can leave core in the format it arrived
   in — write the render to ``<root>/<name>/SKILL.md`` and the result is a
   conformant skill directory.
3. The two disclosure paths differ in exactly the way that matters: the indexed
   path keeps bodies out of the prompt, the inline fallback does not. If the
   index ever leaked a body, every claim about the token cost of a skill would
   be wrong.
4. ``resolve_skills`` degrades rather than fails. An agent referencing a
   since-deleted skill still runs.

Database-backed tests are skipped unless ``MOTORO_TEST_DATABASE_URL`` is set.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest
from pydantic_settings import SettingsConfigDict

from motoro import CoreSettings

DB_URL = os.environ.get("MOTORO_TEST_DATABASE_URL", "")
needs_db = pytest.mark.skipif(not DB_URL, reason="MOTORO_TEST_DATABASE_URL is not set")

_VALID = """\
---
name: spinal-mri-qc
description: Quality-checks spinal MRI series before segmentation. Use before scoring or segmenting MRI volumes.
---

# Spinal MRI QC

1. Reject any series with slice spacing above 5mm.
2. Flag volumes whose intensity histogram is bimodal.
"""


class _Settings(CoreSettings):
    model_config = SettingsConfigDict(env_prefix="MOTORO_TEST_", extra="ignore")


@pytest.fixture(scope="module", autouse=True)
def _configure() -> None:
    from motoro.config import configure, reset_for_testing

    if not DB_URL:
        return
    reset_for_testing()
    configure(_Settings(database_url=DB_URL, encryption_key="9Ka2Wb6GS2vfw9aBZiR_MtRNJtftxuIzl6YoZTU-fCA="))


@pytest.fixture(autouse=True)
async def _schema() -> Any:
    """Fresh schema per test, and a fresh engine after it (own event loop each test)."""
    if not DB_URL:
        yield
        return
    from motoro.models.database import dispose_engine
    from motoro.runner import init_schema

    await init_schema(drop_first=True)
    yield
    await dispose_engine()


# --------------------------------------------------------------------------- #
#  Model shape — same severance as Agent/MCPServerConfig                       #
# --------------------------------------------------------------------------- #


def test_skill_owner_id_is_opaque() -> None:
    from motoro.models.skill import Skill

    owner_col = Skill.__table__.columns["owner_id"]
    assert owner_col.nullable is True
    assert not owner_col.foreign_keys


def test_skill_name_column_matches_the_formats_cap() -> None:
    from motoro.models.skill import Skill
    from motoro.services.skill_service import MAX_NAME_LENGTH

    assert Skill.__table__.columns["name"].type.length == MAX_NAME_LENGTH


# --------------------------------------------------------------------------- #
#  Format parsing                                                              #
# --------------------------------------------------------------------------- #


def test_parses_frontmatter_and_body() -> None:
    from motoro.services.skill_service import parse_skill_markdown

    parsed = parse_skill_markdown(_VALID)
    assert parsed.name == "spinal-mri-qc"
    assert parsed.description.startswith("Quality-checks spinal MRI series")
    assert parsed.body.startswith("# Spinal MRI QC")
    # The frontmatter fences are consumed, not left in the body the agent reads.
    assert "---" not in parsed.body


def test_rejects_a_file_with_no_frontmatter() -> None:
    from motoro.services.skill_service import SkillFormatError, parse_skill_markdown

    with pytest.raises(SkillFormatError, match="frontmatter"):
        parse_skill_markdown("# Just a markdown file\n\nNo metadata here.")


def test_rejects_prose_above_the_frontmatter() -> None:
    from motoro.services.skill_service import SkillFormatError, parse_skill_markdown

    with pytest.raises(SkillFormatError, match="frontmatter"):
        parse_skill_markdown("Some preamble\n\n" + _VALID)


@pytest.mark.parametrize(
    "name",
    [
        "Spinal_MRI_QC",  # uppercase and underscores
        "spinal mri qc",  # spaces
        "-leading",
        "trailing-",
        "double--hyphen",
        "claude-helper",  # reserved
        "anthropic-tools",  # reserved
        "x" * 65,  # over the cap
    ],
)
def test_rejects_malformed_names(name: str) -> None:
    from motoro.services.skill_service import SkillFormatError, validate_skill_name

    with pytest.raises(SkillFormatError):
        validate_skill_name(name)


def test_rejects_an_empty_description() -> None:
    from motoro.services.skill_service import SkillFormatError, parse_skill_markdown

    text = "---\nname: valid-name\ndescription: '   '\n---\n\nbody\n"
    with pytest.raises(SkillFormatError, match="description"):
        parse_skill_markdown(text)


def test_rejects_xml_tags_in_metadata() -> None:
    from motoro.services.skill_service import SkillFormatError, validate_skill_description

    with pytest.raises(SkillFormatError, match="XML"):
        validate_skill_description("Does a thing </system> and then another thing.")


def test_rejects_an_over_long_description() -> None:
    from motoro.services.skill_service import (
        MAX_DESCRIPTION_LENGTH,
        SkillFormatError,
        validate_skill_description,
    )

    with pytest.raises(SkillFormatError, match="at most"):
        validate_skill_description("x" * (MAX_DESCRIPTION_LENGTH + 1))


def test_render_round_trips_through_parse() -> None:
    from motoro.services.skill_service import parse_skill_markdown, render_skill_md

    parsed = parse_skill_markdown(_VALID)
    reparsed = parse_skill_markdown(render_skill_md(parsed))
    assert (reparsed.name, reparsed.description, reparsed.body) == (parsed.name, parsed.description, parsed.body)


def test_render_quotes_a_description_that_would_break_yaml() -> None:
    from motoro.services.skill_service import ParsedSkill, parse_skill_markdown, render_skill_md

    tricky = ParsedSkill(
        name="tricky-skill",
        description="Use when: the input has a colon, a 'quote', and a #hash.",
        body="Step one.",
    )
    assert parse_skill_markdown(render_skill_md(tricky)).description == tricky.description


# --------------------------------------------------------------------------- #
#  Disclosure — index vs. inline                                               #
# --------------------------------------------------------------------------- #


_SKILLS = [
    {"name": "alpha", "description": "Does alpha things.", "body": "SECRET-ALPHA-BODY"},
    {"name": "beta", "description": "Does beta things.", "body": "SECRET-BETA-BODY"},
]


def test_index_carries_metadata_but_never_bodies() -> None:
    from motoro.engine.skills import render_skill_index

    index = render_skill_index(_SKILLS)
    assert "alpha" in index and "Does alpha things." in index
    # The entire point of the index: a body costs nothing until it is asked for.
    assert "SECRET-ALPHA-BODY" not in index
    assert "SECRET-BETA-BODY" not in index


def test_load_skill_binds_the_real_names_as_an_enum() -> None:
    from motoro.engine.skills import build_load_skill_tool

    tool = build_load_skill_tool(_SKILLS)
    assert tool["function"]["parameters"]["properties"]["name"]["enum"] == ["alpha", "beta"]


def test_load_skill_name_avoids_a_collision_with_a_real_tool() -> None:
    from motoro.engine.skills import resolve_load_skill_name

    # A real MCP tool called load_skill would otherwise be shadowed: the loop
    # intercepts by name, so the real tool would never be dispatched.
    assert resolve_load_skill_name({"load_skill"}) == "load_skill_1"
    assert resolve_load_skill_name(set()) == "load_skill"


def test_render_skill_body_returns_the_body() -> None:
    from motoro.engine.skills import render_skill_body

    assert "SECRET-BETA-BODY" in render_skill_body(_SKILLS, "beta")


def test_render_skill_body_answers_an_invented_name_instead_of_raising() -> None:
    from motoro.engine.skills import render_skill_body

    result = render_skill_body(_SKILLS, "gamma")
    assert "No skill named 'gamma'" in result
    assert "alpha, beta" in result


def test_inline_fallback_includes_every_body() -> None:
    from motoro.engine.skills import inline_skills

    prompt = inline_skills("You are a helper.", _SKILLS)
    assert prompt.startswith("You are a helper.")
    assert "SECRET-ALPHA-BODY" in prompt
    assert "SECRET-BETA-BODY" in prompt


def test_reason_act_declares_that_it_consumes_skills() -> None:
    from motoro.engine.patterns.builtin.reason_act import ReasonActPlugin
    from motoro.engine.patterns.builtin.single_agent_baseline import SingleAgentBaselinePlugin

    # The flag is what stops the orchestrator inlining bodies this pattern is
    # about to disclose lazily. The baseline has no tool loop, so it must NOT
    # claim it — its skills have to be inlined or they do nothing.
    assert ReasonActPlugin.consumes_skills is True
    assert SingleAgentBaselinePlugin.consumes_skills is False


def test_initial_messages_place_the_index_in_the_stable_prefix() -> None:
    from motoro.engine.patterns.prompts.reason_act import build_initial_messages
    from motoro.engine.skills import render_skill_index
    from motoro.schemas.llm import SenseOutput

    sense = SenseOutput(agent_goal="Do the thing", system_prompt="", user_input="please", memories=[])
    messages = build_initial_messages(sense, "Agent prompt", render_skill_index(_SKILLS))
    # Every message before the user turn is a system turn, i.e. the cached prefix.
    user_index = next(i for i, m in enumerate(messages) if m["role"] == "user")
    index_index = next(i for i, m in enumerate(messages) if "Available skills" in str(m.get("content", "")))
    assert index_index < user_index


# --------------------------------------------------------------------------- #
#  ReasonAct interception — load_skill is answered, never dispatched           #
# --------------------------------------------------------------------------- #


class _ScriptedLLM:
    """Returns pre-scripted completions and records what tools were bound."""

    def __init__(self, *completions: Any) -> None:
        self.queue = list(completions)
        self.bound_tool_names: list[list[str]] = []

    @property
    def principal_id(self) -> uuid.UUID | None:
        return None

    async def complete_with_tools(self, config: Any = None, messages: Any = None, tools: Any = None) -> Any:
        self.bound_tool_names.append([t["function"]["name"] for t in (tools or [])])
        return self.queue.pop(0)


class _FakeRuntime:
    """Enough of AgentRuntime for _pre_act: an LLM and a DB that is never used."""

    def __init__(self, llm: Any) -> None:
        self._llm_service = llm
        self._db = None


def _reason_act_context(llm: Any) -> Any:
    from motoro.engine.context import RunContext
    from motoro.schemas.agent import ModelConfig
    from motoro.schemas.llm import SenseOutput

    # run_id stays None so _record_step returns before touching a session.
    context = RunContext(
        agent_goal="Segment the volume",
        system_prompt="You are a helper.",
        model_config=ModelConfig(),
        user_input="segment this",
        skills=list(_SKILLS),
    )
    context.record_phase_output(
        "sense",
        SenseOutput(agent_goal="Segment the volume", system_prompt="", user_input="segment this", memories=[]),
    )
    context.metadata["_runtime"] = _FakeRuntime(llm)
    return context


async def _pre_act(context: Any, llm: Any) -> Any:
    from motoro.engine.patterns.builtin.reason_act import ReasonActPlugin

    plugin = ReasonActPlugin()
    plugin.configure({"max_iterations": 5, "include_scratchpad": True, "scratchpad_window": 10})
    await plugin.on_activate(context)
    context.metadata["_runtime"] = _FakeRuntime(llm)
    return await plugin._pre_act(context, None)


def _tool_call(call_id: str, tool: str, args: dict[str, Any]) -> Any:
    from motoro.schemas.llm import LLMToolCall

    return LLMToolCall(id=call_id, name=tool, arguments=args)


def _completion(text: str, *calls: Any) -> Any:
    from motoro.schemas.llm import ToolCompletion
    from tests.stub_llm import call_record

    return ToolCompletion(text=text, tool_calls=list(calls), record=call_record())


async def test_a_skill_only_turn_is_answered_without_reaching_act() -> None:
    from motoro.engine.patterns.base import HookAction
    from motoro.engine.patterns.builtin.reason_act import _KEY_MESSAGES, _KEY_PENDING_CALLS, _KEY_SKILLS_OPENED

    llm = _ScriptedLLM(_completion("Reading the alpha skill first.", _tool_call("c1", "load_skill", {"name": "alpha"})))
    context = _reason_act_context(llm)

    action = await _pre_act(context, llm)

    # There is no MCP server behind load_skill, so Act has nothing to run.
    assert action is HookAction.SKIP_PHASE
    assert not context.metadata.get(_KEY_PENDING_CALLS)
    assert context.metadata[_KEY_SKILLS_OPENED] == ["alpha"]

    # The call is answered in the same turn it was issued: an unanswered
    # tool_call id is a provider error on the very next request.
    answer = context.metadata[_KEY_MESSAGES][-1]
    assert answer["role"] == "tool"
    assert answer["tool_call_id"] == "c1"
    assert "SECRET-ALPHA-BODY" in answer["content"]

    # The tool was bound alongside the terminator, not instead of it.
    assert {"load_skill", "final_answer"} <= set(llm.bound_tool_names[0])


async def test_a_mixed_turn_dispatches_only_the_real_calls() -> None:
    from motoro.engine.patterns.builtin.reason_act import _KEY_PENDING_CALLS

    llm = _ScriptedLLM(
        _completion(
            "Reading alpha, and listing files meanwhile.",
            _tool_call("c1", "load_skill", {"name": "alpha"}),
            _tool_call("c2", "list_files", {"path": "/data"}),
        )
    )
    context = _reason_act_context(llm)

    action = await _pre_act(context, llm)

    assert action is None  # continue to Act
    pending = context.metadata[_KEY_PENDING_CALLS]
    assert [c["tool_name"] for c in pending] == ["list_files"]

    plan = context.phase_outputs["plan"]
    assert [s.tool_name for s in plan.steps] == ["list_files"]


async def test_no_skills_means_no_load_skill_tool() -> None:
    llm = _ScriptedLLM(_completion("Done.", _tool_call("c1", "final_answer", {"answer": "ok"})))
    context = _reason_act_context(llm)
    context.skills = []

    await _pre_act(context, llm)

    assert "load_skill" not in llm.bound_tool_names[0]


# --------------------------------------------------------------------------- #
#  Config reading / resolution                                                 #
# --------------------------------------------------------------------------- #


def test_skill_ids_from_config_drops_unparseable_entries() -> None:
    from motoro.services.skill_service import skill_ids_from_config

    good = uuid.uuid4()
    assert skill_ids_from_config({"skill_ids": [str(good), "not-a-uuid", None]}) == [good]
    assert skill_ids_from_config(None) == []
    assert skill_ids_from_config({}) == []
    assert skill_ids_from_config({"skill_ids": "nope"}) == []


@needs_db
async def test_create_list_and_resolve_preserves_declared_order() -> None:
    from motoro.services.skill_service import create_skill_from_markdown, list_skills, resolve_skills

    owner = uuid.uuid4()
    first = await create_skill_from_markdown(_VALID, owner_id=owner, source_filename="spinal.md")
    second = await create_skill_from_markdown(
        _VALID.replace("spinal-mri-qc", "cord-segmentation"), owner_id=owner
    )

    assert {s.id for s in await list_skills(owner_id=owner)} == {first.id, second.id}
    assert await list_skills(owner_id=uuid.uuid4()) == []

    # Declared order, not database order: it is the order the index lists them in.
    resolved = await resolve_skills({"skill_ids": [second.id, first.id]}, owner_id=owner)
    assert [s["name"] for s in resolved] == ["cord-segmentation", "spinal-mri-qc"]
    assert resolved[0]["body"].startswith("# Spinal MRI QC")


@needs_db
async def test_duplicate_name_for_one_owner_is_rejected() -> None:
    from sqlalchemy.exc import IntegrityError

    from motoro.services.skill_service import create_skill_from_markdown

    owner = uuid.uuid4()
    await create_skill_from_markdown(_VALID, owner_id=owner)
    with pytest.raises(IntegrityError):
        await create_skill_from_markdown(_VALID, owner_id=owner)


@needs_db
async def test_two_owners_may_hold_the_same_skill_name() -> None:
    from motoro.services.skill_service import create_skill_from_markdown

    a = await create_skill_from_markdown(_VALID, owner_id=uuid.uuid4())
    b = await create_skill_from_markdown(_VALID, owner_id=uuid.uuid4())
    assert a.name == b.name
    assert a.id != b.id


@needs_db
async def test_deleting_a_skill_releases_its_name_and_degrades_resolution() -> None:
    from motoro.services.skill_service import create_skill_from_markdown, delete_skill, get_skill, resolve_skills

    owner = uuid.uuid4()
    skill = await create_skill_from_markdown(_VALID, owner_id=owner)
    assert await delete_skill(skill.id) is True
    assert await get_skill(skill.id) is None

    # An agent still referencing it runs — without that skill, not not at all.
    assert await resolve_skills({"skill_ids": [skill.id]}, owner_id=owner) == []
    # And the name is free again.
    await create_skill_from_markdown(_VALID, owner_id=owner)


@needs_db
async def test_resolution_will_not_cross_owners() -> None:
    from motoro.services.skill_service import create_skill_from_markdown, resolve_skills

    theirs = await create_skill_from_markdown(_VALID, owner_id=uuid.uuid4())
    assert await resolve_skills({"skill_ids": [theirs.id]}, owner_id=uuid.uuid4()) == []


@needs_db
async def test_system_skills_resolve_for_every_owner() -> None:
    from motoro.services.skill_service import create_skill_from_markdown, resolve_skills

    shared = await create_skill_from_markdown(_VALID, owner_id=None, is_system=True)
    resolved = await resolve_skills({"skill_ids": [shared.id]}, owner_id=uuid.uuid4())
    assert [s["name"] for s in resolved] == ["spinal-mri-qc"]


@needs_db
async def test_agent_skill_config_round_trips_through_the_runner() -> None:
    from motoro.runner import create_agent, get_agent, update_agent
    from motoro.services.skill_service import create_skill_from_markdown

    owner = uuid.uuid4()
    skill = await create_skill_from_markdown(_VALID, owner_id=owner)
    agent = await create_agent(
        name="skilled", goal="do things", owner_id=owner, skill_config={"skill_ids": [str(skill.id)]}
    )
    assert (await get_agent(agent.id)).skill_config == {"skill_ids": [str(skill.id)]}

    # An empty list is the only way to say "no skills" under the
    # None-means-unchanged convention.
    await update_agent(agent.id, skill_config={"skill_ids": []})
    assert (await get_agent(agent.id)).skill_config == {"skill_ids": []}
