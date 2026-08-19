"""Core owns its schema, so the migration chain is part of the contract.

Two things worth pinning:

1. ``upgrade()`` and ``runner.init_schema()`` (``create_all``) must produce the
   *same* schema. They are two paths to one contract, and silent divergence would
   mean tests pass against a schema production never has.
2. Core's chain must use its own version table and touch only core's tables, so a
   product's chain can run alongside it without either stamping over the other.

Skipped unless ``MOTORO_TEST_DATABASE_URL`` is set. These tests create and drop
their own scratch databases on that server rather than using that database, since
comparing two schemas needs two of them.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest
from pydantic_settings import SettingsConfigDict

from motoro import CoreSettings

DB_URL = os.environ.get("MOTORO_TEST_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(not DB_URL, reason="MOTORO_TEST_DATABASE_URL is not set")

COLUMNS_SQL = """
SELECT table_name || '.' || column_name || ':' || data_type || ':' || is_nullable
FROM information_schema.columns
WHERE table_schema = 'public' AND table_name NOT LIKE 'alembic%'
ORDER BY 1
"""

INDEXES_SQL = """
SELECT tablename || ' ' || indexdef
FROM pg_indexes
WHERE schemaname = 'public' AND tablename NOT LIKE 'alembic%'
ORDER BY 1
"""


class _Settings(CoreSettings):
    model_config = SettingsConfigDict(extra="ignore")


def _url_for(db_name: str) -> str:
    base, _, _ = DB_URL.rpartition("/")
    return f"{base}/{db_name}"


async def _admin(sql: str) -> None:
    """Run a CREATE/DROP DATABASE, which cannot happen inside a transaction."""
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(_url_for("postgres"), isolation_level="AUTOCOMMIT")
    try:
        from sqlalchemy import text

        async with engine.connect() as conn:
            await conn.execute(text(sql))
    finally:
        await engine.dispose()


async def _introspect(db_name: str, sql: str) -> list[str]:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(_url_for(db_name))
    try:
        async with engine.connect() as conn:
            return [r[0] for r in (await conn.execute(text(sql))).all()]
    finally:
        await engine.dispose()


@pytest.fixture
async def scratch_dbs() -> Any:
    """Two empty databases: one for create_all, one for the migration chain."""
    names = ("motoro_t_createall", "motoro_t_migrated")
    for n in names:
        await _admin(f'DROP DATABASE IF EXISTS "{n}"')
        await _admin(f'CREATE DATABASE "{n}"')
    yield names
    for n in names:
        await _admin(f'DROP DATABASE IF EXISTS "{n}"')


async def test_migrated_schema_matches_create_all(scratch_dbs: tuple[str, str]) -> None:
    """The two provisioning paths agree, column for column and index for index."""
    from motoro.config import configure, reset_for_testing
    from motoro.migrations import upgrade_async
    from motoro.models.database import dispose_engine
    from motoro.runner import init_schema

    ca_db, mig_db = scratch_dbs

    reset_for_testing()
    configure(_Settings(database_url=_url_for(ca_db)))
    await init_schema()
    await dispose_engine()

    await upgrade_async(_url_for(mig_db))

    ca_cols, mig_cols = await _introspect(ca_db, COLUMNS_SQL), await _introspect(mig_db, COLUMNS_SQL)
    assert ca_cols, "create_all produced no columns"
    assert ca_cols == mig_cols, (
        "create_all and the migration chain disagree.\n"
        f"only in create_all: {sorted(set(ca_cols) - set(mig_cols))}\n"
        f"only in migrated:   {sorted(set(mig_cols) - set(ca_cols))}"
    )

    ca_idx, mig_idx = await _introspect(ca_db, INDEXES_SQL), await _introspect(mig_db, INDEXES_SQL)
    assert ca_idx == mig_idx, (
        f"index mismatch.\nonly in create_all: {sorted(set(ca_idx) - set(mig_idx))}\n"
        f"only in migrated:   {sorted(set(mig_idx) - set(ca_idx))}"
    )

    reset_for_testing()


async def test_chain_uses_its_own_version_table(scratch_dbs: tuple[str, str]) -> None:
    """Core stamps ``alembic_version_motoro``, leaving ``alembic_version``
    free for the product's chain."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from motoro.migrations import VERSION_TABLE, upgrade_async

    _, mig_db = scratch_dbs
    await upgrade_async(_url_for(mig_db))

    engine = create_async_engine(_url_for(mig_db))
    try:
        async with engine.connect() as conn:
            assert await conn.scalar(text("SELECT to_regclass(:t)"), {"t": VERSION_TABLE}) is not None
            # The product's default table name must be untouched.
            assert await conn.scalar(text("SELECT to_regclass('alembic_version')")) is None
    finally:
        await engine.dispose()


async def test_upgrade_is_idempotent(scratch_dbs: tuple[str, str]) -> None:
    """Running upgrade twice is a no-op, so a product may call it on every boot."""
    from motoro.migrations import current_revision, upgrade_async

    _, mig_db = scratch_dbs
    url = _url_for(mig_db)

    assert await current_revision(url) is None
    await upgrade_async(url)
    first = await current_revision(url)
    assert first is not None
    await upgrade_async(url)
    assert await current_revision(url) == first


async def test_chain_owns_only_core_tables() -> None:
    """The autogenerate filter is confined to core's own tables.

    Core and a product share one ``Base.metadata``; without this filter core's
    autogenerate would see every product table as something to drop.

    Lives in the package rather than in ``env.py`` because ``env.py`` reads
    ``alembic.context.config`` at module scope and is therefore only importable
    while Alembic is actually running.
    """
    from motoro.migrations import CORE_TABLES

    assert set(CORE_TABLES) == {
        "agents",
        "agent_runs",
        "run_steps",
        "architectural_patterns",
        "llm_pricing_overrides",
        "memory_entries",
        "mcp_server_configs",
    }


def test_migration_scripts_are_present() -> None:
    """``env.py`` and at least one revision ship inside the package.

    They live under ``src/motoro/migrations`` rather than at the repo root
    precisely so a pip-installed core can still be migrated; if they stop being
    packaged, ``upgrade()`` has nothing to run.
    """
    from motoro.migrations import MIGRATIONS_DIR

    assert (MIGRATIONS_DIR / "env.py").is_file()
    assert (MIGRATIONS_DIR / "script.py.mako").is_file()
    revisions = list((MIGRATIONS_DIR / "versions").glob("*.py"))
    assert revisions, "no revisions found"


def test_env_module_refuses_to_nest_event_loops() -> None:
    """``upgrade()`` is sync on purpose; ``upgrade_async`` exists for async callers."""
    import inspect

    from motoro.migrations import upgrade, upgrade_async

    assert not inspect.iscoroutinefunction(upgrade)
    assert inspect.iscoroutinefunction(upgrade_async)


def test_asyncio_import_is_used() -> None:
    """Guard against the asyncio import being pruned by a linter."""
    assert asyncio is not None


TYPES_SQL = """
SELECT t.typname
FROM pg_type t
JOIN pg_namespace n ON n.oid = t.typnamespace
WHERE n.nspname = 'public' AND t.typtype = 'e'
ORDER BY 1
"""


async def test_downgrade_then_upgrade_reproduces_the_schema(scratch_dbs: tuple[str, str]) -> None:
    """A full down-and-up cycle lands on the identical schema.

    This is the idempotency that actually bites. ``upgrade`` twice is guarded by
    the version table and trivially safe; a *downgrade* that fails to reverse
    everything is not. The baseline originally dropped its tables but not its
    enum types, so the next upgrade died on ``type "pattern_category" already
    exists`` — a database that could be migrated once and never again.
    """
    from motoro.migrations import current_revision, downgrade, upgrade_async

    _, db = scratch_dbs
    url = _url_for(db)

    await upgrade_async(url)
    head = await current_revision(url)
    columns = await _introspect(db, COLUMNS_SQL)
    indexes = await _introspect(db, INDEXES_SQL)
    assert columns and indexes

    await asyncio.to_thread(downgrade, url, "base")
    assert await current_revision(url) is None
    assert await _introspect(db, COLUMNS_SQL) == [], "downgrade left tables behind"

    await upgrade_async(url)
    assert await current_revision(url) == head
    assert await _introspect(db, COLUMNS_SQL) == columns
    assert await _introspect(db, INDEXES_SQL) == indexes


async def test_downgrade_leaves_no_orphan_enum_types(scratch_dbs: tuple[str, str]) -> None:
    """Types are schema too. A left-behind enum blocks the next upgrade."""
    from motoro.migrations import downgrade, upgrade_async

    _, db = scratch_dbs
    url = _url_for(db)

    await upgrade_async(url)
    assert await _introspect(db, TYPES_SQL), "the chain should have created enum types"

    await asyncio.to_thread(downgrade, url, "base")
    leftover = await _introspect(db, TYPES_SQL)
    assert leftover == [], f"downgrade left enum types behind: {leftover}"


def test_every_revision_drops_the_enums_it_creates() -> None:
    """Static guard, so the next revision cannot reintroduce the same bug.

    The round-trip tests above catch this empirically, but only for revisions that
    happen to be exercised. This reads every revision file and pairs each enum
    created in ``upgrade()`` with a ``DROP TYPE`` in ``downgrade()``.
    """
    import re
    from pathlib import Path

    import motoro.migrations as m

    versions = Path(m.__file__).parent / "versions"
    offenders: list[str] = []
    for path in sorted(versions.glob("*.py")):
        src = path.read_text()
        up, _, down = src.partition("def downgrade(")
        created = set(re.findall(r"sa\.Enum\([^)]*?name=['\"](\w+)['\"]", up, re.S))
        # Literal drops: `op.execute("DROP TYPE IF EXISTS run_status")`.
        dropped = set(re.findall(r"DROP TYPE (?:IF EXISTS )?(\w+)", down))
        # Plus loop drops, which is how the baseline does it. Scoped to the
        # iterable of a loop whose body actually drops a type — crediting every
        # quoted string in `downgrade` would let a table name vouch for an enum.
        for iterable, body in re.findall(r"for \w+ in ([\(\[].*?[\)\]]):\n(.*?)(?=\n    \w|\Z)", down, re.S):
            if "DROP TYPE" in body:
                dropped |= set(re.findall(r"['\"](\w+)['\"]", iterable))
        for name in sorted(created - dropped):
            offenders.append(f"{path.name}: creates enum '{name}' but downgrade never drops it")
    assert not offenders, "\n".join(offenders)


def test_no_revision_uses_non_transactional_ddl() -> None:
    """Every revision must be atomic, which is what makes a retry safe.

    Postgres has transactional DDL and Alembic wraps each revision in a
    transaction, so a revision that fails rolls back whole — including its version
    stamp — leaving the database re-runnable. ``CONCURRENTLY`` opts out of that
    guarantee: it cannot run in a transaction, so a failure mid-way leaves an
    invalid index behind and the revision half-applied.

    If core ever genuinely needs a concurrent index, this test should be replaced
    with a documented exception, not deleted quietly.
    """
    from pathlib import Path

    import motoro.migrations as m

    versions = Path(m.__file__).parent / "versions"
    offenders = [
        f"{p.name}: uses CONCURRENTLY" for p in sorted(versions.glob("*.py")) if "CONCURRENTLY" in p.read_text().upper()
    ]
    assert not offenders, "\n".join(offenders)
