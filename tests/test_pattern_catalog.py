"""The plugin class is the source of truth for pattern metadata.

Three things are worth pinning:

1. Validation and defaults read the *registry*, so they work against an empty
   ``architectural_patterns`` table. ARES reads the table, which means an unseeded
   one rejects every agent — the failure mode this design exists to avoid.
2. ``sync_pattern_catalog`` projects the registry into that table, one way, and is
   idempotent.
3. The catalog and the code cannot drift, because there is only one copy.

The database tests are skipped unless ``MOTORO_TEST_DATABASE_URL`` is set.
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


class _Settings(CoreSettings):
    model_config = SettingsConfigDict(env_prefix="MOTORO_TEST_", extra="ignore")


@pytest.fixture(scope="module", autouse=True)
def _configure() -> None:
    """Point core at the test database. Harmless for the registry-only tests."""
    from motoro.config import configure, reset_for_testing

    if not DB_URL:
        return
    reset_for_testing()
    configure(_Settings(database_url=DB_URL))


@pytest.fixture(autouse=True)
async def _dispose_engine() -> Any:
    """asyncpg connections belong to the loop that opened them, one loop per test."""
    yield
    if DB_URL:
        from motoro.models.database import dispose_engine

        await dispose_engine()


# --------------------------------------------------------------------------- #
#  Metadata off the plugin classes                                             #
# --------------------------------------------------------------------------- #


def test_every_registered_plugin_declares_catalog_metadata() -> None:
    """A pattern with no name or description would sync a useless catalog row."""
    from motoro.engine.patterns.catalog import display_name_for
    from motoro.engine.patterns.registry import PluginRegistry

    PluginRegistry.discover()
    plugins = PluginRegistry.all()
    assert plugins, "no plugins discovered"

    missing: list[str] = []
    for slug, cls in plugins.items():
        if not display_name_for(cls):
            missing.append(f"{slug}: no display_name and slug yields nothing")
        if not cls.description.strip():
            missing.append(f"{slug}: no description")
        if not cls.configuration_schema and cls.__name__ != "PatternPlugin":
            missing.append(f"{slug}: no configuration_schema")
    assert not missing, "\n".join(missing)


def test_declared_dependencies_name_real_patterns() -> None:
    """A dependency on a slug nothing implements makes a pattern unusable.

    ARES needed migration 0009 to fix exactly this in data: ``solo_agent_loop``
    was renamed to ``single_agent_baseline`` and the dependency arrays in the
    seeded rows kept pointing at the old slug. Declared on the class, a rename
    that misses this is a test failure rather than a stale row.
    """
    from motoro.engine.patterns.registry import PluginRegistry

    PluginRegistry.discover()
    plugins = PluginRegistry.all()
    dangling = [f"{slug} -> {dep}" for slug, cls in plugins.items() for dep in cls.dependencies if dep not in plugins]
    assert not dangling, f"dependencies naming unregistered patterns: {dangling}"


def test_configuration_schema_defaults_match_configure_fallbacks() -> None:
    """The schema default and the ``params.get`` fallback must be the same number.

    Two copies of every default is the drift this design removes: the schema is
    what a product renders and what gets merged into params, while the fallback is
    what runs if the merge did not happen. They must agree.
    """
    from motoro.engine.patterns.catalog import schema_defaults
    from motoro.engine.patterns.registry import PluginRegistry

    PluginRegistry.discover()

    expected = {
        "single_agent_baseline": {"max_iterations": 10, "stop_on_first_success": True},
        "reason_act": {
            "max_iterations": 15,
            "include_scratchpad": True,
            "scratchpad_window": 10,
            "observation_format": "raw",
        },
    }
    for slug, fallbacks in expected.items():
        cls = PluginRegistry.get(slug)
        assert cls is not None, f"{slug} is not registered"
        assert schema_defaults(cls.configuration_schema) == fallbacks


def test_schema_defaults_are_applied_to_plugin_params() -> None:
    """A plugin configured with nothing still sees its schema defaults."""
    from motoro.engine.patterns.catalog import merge_schema_defaults
    from motoro.engine.patterns.registry import PluginRegistry

    PluginRegistry.discover()
    cls = PluginRegistry.get("reason_act")
    assert cls is not None

    assert merge_schema_defaults(cls, {})["scratchpad_window"] == 10
    # A caller's value wins over the default.
    assert merge_schema_defaults(cls, {"scratchpad_window": 3})["scratchpad_window"] == 3
    # And unrelated defaults still land.
    assert merge_schema_defaults(cls, {"scratchpad_window": 3})["max_iterations"] == 15


def test_schema_defaults_skip_required_properties() -> None:
    """A required property has no implicit default; supplying one hides the error."""
    from motoro.engine.patterns.catalog import schema_defaults

    schema = {
        "properties": {"role": {"type": "string", "default": "worker"}, "n": {"type": "integer", "default": 1}},
        "required": ["role"],
    }
    assert schema_defaults(schema) == {"n": 1}


# --------------------------------------------------------------------------- #
#  Validation                                                                  #
# --------------------------------------------------------------------------- #


def test_validation_accepts_the_shipped_patterns() -> None:
    from motoro.engine.patterns.catalog import validate_pattern_config

    for slug in ("single_agent_baseline", "reason_act"):
        result = validate_pattern_config({"execution_pattern": slug})
        assert result.valid, f"{slug} rejected: {[e.message for e in result.errors]}"


def test_validation_rejects_an_unknown_slug_and_names_the_field() -> None:
    from motoro.engine.patterns.catalog import validate_pattern_config

    result = validate_pattern_config({"execution_pattern": "reasson_act"})
    assert not result.valid
    assert result.errors[0].field == "execution_pattern"
    assert "reasson_act" in result.errors[0].message
    # The message lists what *is* available, since a typo is the likely cause.
    assert "reason_act" in result.errors[0].message


def test_validation_rejects_bad_params_via_the_plugins_own_validator() -> None:
    from motoro.engine.patterns.catalog import validate_pattern_config

    result = validate_pattern_config(
        {"execution_pattern": "reason_act", "pattern_params": {"reason_act": {"max_iterations": -1}}}
    )
    assert not result.valid
    assert any("max_iterations" in e.message for e in result.errors)


def test_validation_rejects_params_for_an_inactive_pattern() -> None:
    """Otherwise a params block for the wrong slug is silently ignored."""
    from motoro.engine.patterns.catalog import validate_pattern_config

    result = validate_pattern_config(
        {"execution_pattern": "reason_act", "pattern_params": {"single_agent_baseline": {"max_iterations": 5}}}
    )
    assert not result.valid
    assert any("not active" in e.message for e in result.errors)


def test_validation_allows_a_same_category_singleton_dependency() -> None:
    """``reason_act`` depends on ``single_agent_baseline``; both are execution.

    They cannot be co-active — one execution pattern at a time — so the
    dependency must be skipped rather than reported. This is the check that fails
    if the composition helpers are called without category metadata for the
    *dependency* as well as the dependent.
    """
    from motoro.engine.patterns.catalog import validate_pattern_config

    result = validate_pattern_config({"execution_pattern": "reason_act"})
    assert result.valid, [e.message for e in result.errors]


def test_empty_config_is_valid() -> None:
    from motoro.engine.patterns.catalog import validate_pattern_config

    assert validate_pattern_config(None).valid
    assert validate_pattern_config({}).valid


@needs_db
async def test_create_agent_rejects_an_unknown_pattern() -> None:
    """The gate is at creation, before a run exists and a model is billed."""
    from motoro.engine.patterns.catalog import PatternConfigError
    from motoro.runner import create_agent, get_agent_by_name

    name = f"test-badpattern-{uuid.uuid4().hex[:8]}"
    with pytest.raises(PatternConfigError, match="Unknown pattern slug"):
        await create_agent(name=name, goal="x", pattern_config={"execution_pattern": "nope"})

    # And nothing was persisted on the way to raising.
    assert await get_agent_by_name(name) is None


@needs_db
async def test_create_agent_accepts_a_valid_pattern() -> None:
    from motoro.runner import create_agent

    agent = await create_agent(
        name=f"test-goodpattern-{uuid.uuid4().hex[:8]}",
        goal="x",
        pattern_config={"execution_pattern": "single_agent_baseline"},
    )
    assert agent.id is not None


@needs_db
async def test_agent_names_are_unique_per_owner_not_installation() -> None:
    """uq_agents_owner_name_active scopes the constraint to (owner_id, lower(name))."""
    from sqlalchemy.exc import IntegrityError

    from motoro.runner import create_agent, get_agent_by_name

    name = f"test-perowner-{uuid.uuid4().hex[:8]}"
    owner_a, owner_b = uuid.uuid4(), uuid.uuid4()

    a = await create_agent(name=name, goal="x", owner_id=owner_a)
    b = await create_agent(name=name, goal="x", owner_id=owner_b)
    assert a.id != b.id

    with pytest.raises(IntegrityError):
        await create_agent(name=name, goal="x", owner_id=owner_a)

    found_a = await get_agent_by_name(name, owner_id=owner_a)
    found_b = await get_agent_by_name(name, owner_id=owner_b)
    assert found_a is not None and found_a.id == a.id
    assert found_b is not None and found_b.id == b.id


# --------------------------------------------------------------------------- #
#  The projection                                                              #
# --------------------------------------------------------------------------- #


def test_catalog_rows_come_from_the_plugin_classes() -> None:
    from motoro.services.pattern_catalog import catalog_rows

    rows = {r["slug"]: r for r in catalog_rows()}
    assert set(rows) == {"single_agent_baseline", "reason_act"}

    react = rows["reason_act"]
    assert react["name"] == "ReAct"
    assert react["dependencies"] == ["single_agent_baseline"]
    assert str(react["phase"]) == "introspective"
    assert react["is_implemented"] is True
    assert react["configuration_schema"]["properties"]["max_iterations"]["default"] == 15


@needs_db
async def test_sync_is_idempotent_and_corrects_a_drifted_row() -> None:
    """A second sync inserts nothing, and a stale row is brought back in line.

    ``ON CONFLICT DO UPDATE`` rather than ``DO NOTHING`` is the point: ARES seeded
    with ``DO NOTHING`` and then needed migrations 0009, 0010 and 0025 to amend
    rows after the fact.
    """
    from sqlalchemy import select, update

    from motoro.models.database import system_session
    from motoro.models.pattern import ArchitecturalPattern
    from motoro.runner import init_schema
    from motoro.services.pattern_catalog import sync_pattern_catalog

    await init_schema(drop_first=True)

    first = await sync_pattern_catalog()
    assert first["inserted"] == 2
    assert first["updated"] == 0

    second = await sync_pattern_catalog()
    assert second["inserted"] == 0
    assert second["updated"] == 2

    # Drift the row the way a hand edit or a stale seed would.
    async with system_session(reason="test_drift") as db:
        await db.execute(
            update(ArchitecturalPattern)
            .where(ArchitecturalPattern.slug == "reason_act")
            .values(name="WRONG", dependencies=["solo_agent_loop"])
        )
        await db.commit()

    await sync_pattern_catalog()

    async with system_session(reason="test_read") as db:
        row = (
            await db.execute(select(ArchitecturalPattern).where(ArchitecturalPattern.slug == "reason_act"))
        ).scalar_one()
        assert row.name == "ReAct"
        assert row.dependencies == ["single_agent_baseline"]


@needs_db
async def test_core_runs_against_an_empty_catalog() -> None:
    """The whole point: core never reads the table, so an empty one changes nothing.

    Under ARES's design this is the state in which every ``create_agent`` fails.
    """
    from sqlalchemy import delete

    from motoro.models.database import system_session
    from motoro.models.pattern import ArchitecturalPattern
    from motoro.runner import create_agent, init_schema

    await init_schema(drop_first=True)
    async with system_session(reason="test_empty_catalog") as db:
        await db.execute(delete(ArchitecturalPattern))
        await db.commit()

    agent = await create_agent(
        name=f"test-emptycatalog-{uuid.uuid4().hex[:8]}",
        goal="x",
        pattern_config={"execution_pattern": "reason_act", "pattern_params": {"reason_act": {"max_iterations": 3}}},
    )
    assert agent.id is not None


@needs_db
async def test_list_catalog_reads_the_projection() -> None:
    """The one API a product needs for this table — no session on its side."""
    from motoro.runner import init_schema
    from motoro.services.pattern_catalog import list_catalog, sync_pattern_catalog

    await init_schema(drop_first=True)
    await sync_pattern_catalog()

    rows = await list_catalog()
    assert {r.slug for r in rows} == {"reason_act", "single_agent_baseline"}
    # Usable after core closed its session (expire_on_commit=False).
    assert all(r.name for r in rows)

    assert len(await list_catalog(implemented_only=True)) == 2
