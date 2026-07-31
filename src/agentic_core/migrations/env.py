"""Alembic environment for core's own migration chain.

Lives *inside* the installed package rather than at the repo root, because a
product that ``pip install``s agentic-core has to be able to run this chain. A
script directory outside the wheel would not ship.

Two properties make this chain safe to run alongside a product's own:

* ``version_table`` is ``alembic_version_agentic_core``, not ``alembic_version``,
  so the two chains track their revisions independently and neither stamps over
  the other.
* ``include_object`` restricts autogenerate to core's own tables. Core and a
  product share one ``Base.metadata``, so without this filter core's autogenerate
  would see every product table as something to drop.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig
from typing import Any

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

# Importing for the metadata side effect — a model not imported here is a table
# this chain cannot see.
import agentic_core.models.agent  # noqa: F401
import agentic_core.models.mcp_server  # noqa: F401
import agentic_core.models.memory  # noqa: F401
import agentic_core.models.pattern  # noqa: F401
import agentic_core.models.pricing  # noqa: F401
import agentic_core.models.run  # noqa: F401
from agentic_core.config import settings
from agentic_core.migrations import VERSION_TABLE, include_object
from agentic_core.models.base import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _url() -> str:
    return config.get_main_option("sqlalchemy.url") or settings.database_url


def run_migrations_offline() -> None:
    context.configure(
        url=_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table=VERSION_TABLE,
        include_object=include_object,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def _do_run(connection: Any) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        version_table=VERSION_TABLE,
        include_object=include_object,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _url()
    engine = async_engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    async with engine.connect() as connection:
        await connection.run_sync(_do_run)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    # A caller already inside a running loop (e.g. an async startup hook) should
    # use ``agentic_core.migrations.upgrade`` instead of invoking alembic directly.
    asyncio.run(run_migrations_online())
