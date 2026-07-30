"""Core's schema, and how a product applies it.

Core's five tables are a **requirement**, not a convenience: nothing in the
runtime works without ``agents``, ``agent_runs``, ``run_steps``,
``architectural_patterns`` and ``llm_pricing_overrides``. So core owns them, and
owns their migration chain. A product does not hand-write migrations for core's
tables; it runs core's chain and then its own.

The database *instance* is a different question and remains the product's: one
Postgres per deployment, holding core's tables alongside that product's. Core does
not care whose server it is, only that its schema is present and current.

Two chains, two version tables::

    alembic_version_agentic_core    core's revisions   (this chain)
    alembic_version                 the product's revisions

They are independent, so neither stamps over the other, and core's autogenerate is
filtered to core's own tables so it never proposes dropping a product's.

A product applies core's schema before its own::

    from agentic_core.migrations import upgrade
    upgrade()                              # core's tables, at head
    # ...then the product's own `alembic upgrade head`

Or from async code::

    from agentic_core.migrations import upgrade_async
    await upgrade_async()

Ordering matters: core first. Product tables routinely carry foreign keys into
``agents`` and ``agent_runs``; the reverse is forbidden.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alembic.config import Config

#: Directory holding this chain's ``env.py`` and ``versions/``. Inside the
#: installed package, so it ships in the wheel and a pip-installed core can still
#: be migrated.
MIGRATIONS_DIR = Path(__file__).resolve().parent

#: Core's revisions are tracked separately from a product's.
VERSION_TABLE = "alembic_version_agentic_core"

#: The tables core owns. A product's tables share ``Base.metadata`` but are not
#: core's to create, alter, or drop.
#:
#: Lives here rather than in ``env.py`` because it is a fact about core, and
#: ``env.py`` is only importable while Alembic is actually running — it reads
#: ``alembic.context.config``, which does not exist outside an invocation.
CORE_TABLES = frozenset(
    {
        "agents",
        "agent_runs",
        "run_steps",
        "architectural_patterns",
        "llm_pricing_overrides",
    }
)


def include_object(obj: object, name: str | None, type_: str, reflected: bool, compare_to: object) -> bool:
    """Confine autogenerate to core's tables.

    Core and a product share one ``Base.metadata``. Without this filter, core's
    autogenerate would treat every product table as something to drop.
    """
    if type_ == "table":
        return name in CORE_TABLES
    parent = getattr(obj, "table", None)
    if parent is not None:
        return parent.name in CORE_TABLES
    return True


def make_config(url: str | None = None) -> Config:
    """Build an Alembic config pointed at core's chain.

    *url* defaults to ``CoreSettings.database_url``, so a product that has already
    called ``configure()`` need pass nothing.
    """
    from alembic.config import Config

    from agentic_core.config import settings

    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", url or settings.database_url)
    return cfg


def upgrade(url: str | None = None, revision: str = "head") -> None:
    """Bring core's schema up to *revision*. Safe to call repeatedly.

    Synchronous, because that is what Alembic's command API is and because
    migrations are a startup step rather than request-path work. ``env.py`` opens
    and disposes its own async engine internally. From inside a running event
    loop, use :func:`upgrade_async`.
    """
    from alembic import command

    command.upgrade(make_config(url), revision)


def downgrade(url: str | None = None, revision: str = "-1") -> None:
    """Step core's schema back to *revision*. Mainly a testing affordance."""
    from alembic import command

    command.downgrade(make_config(url), revision)


def stamp(url: str | None = None, revision: str = "head") -> None:
    """Mark the schema as being at *revision* without running anything.

    The migration path for a database whose tables were created by
    ``runner.init_schema`` (``create_all``): stamp it at head so subsequent
    revisions apply cleanly instead of trying to recreate what is already there.
    """
    from alembic import command

    command.stamp(make_config(url), revision)


async def upgrade_async(url: str | None = None, revision: str = "head") -> None:
    """:func:`upgrade`, callable from async code.

    Run in a worker thread because ``env.py`` drives its own event loop, and
    nesting one inside a running loop is an error.
    """
    await asyncio.to_thread(upgrade, url, revision)


async def current_revision(url: str | None = None) -> str | None:
    """The revision core's schema is stamped at, or ``None`` if unmigrated."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from agentic_core.config import settings

    engine = create_async_engine(url or settings.database_url)
    try:
        async with engine.connect() as conn:
            if await conn.scalar(text("SELECT to_regclass(:t)"), {"t": VERSION_TABLE}) is None:
                return None
            return await conn.scalar(text(f"SELECT version_num FROM {VERSION_TABLE} LIMIT 1"))  # noqa: S608
    finally:
        await engine.dispose()


__all__ = [
    "CORE_TABLES",
    "MIGRATIONS_DIR",
    "VERSION_TABLE",
    "current_revision",
    "downgrade",
    "include_object",
    "make_config",
    "stamp",
    "upgrade",
    "upgrade_async",
]
