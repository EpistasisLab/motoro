"""point persisted OKF server rows at motoro.* after the agentic-core rename

``mcp_server_configs`` holds a *live* registration: ``command`` is the argv a
stdio MCP server is spawned with, and it is read back verbatim by
``hydrate_registry`` on every boot. The bundled OKF server's command names core's
module path (``python -m agentic_core.mcp_servers.okf``), so renaming the package
strands every existing row -- the module no longer exists, the subprocess fails
to start, and the server silently drops out of the registry with no other symptom.
Product code cannot fix this on its own: ASAREE's registration helper is
first-boot-only (it returns early when a row with that name already exists), so
nothing ever rewrites the stale command.

Core renamed the module, so core repairs the rows that point at it. The command
rewrite is a targeted substring replacement rather than a rebuild of the whole
argv, because the rest of it (``uv run --directory <product repo root>``) is the
product's, not core's, and must survive untouched.

The server's registered *name* moves too, ``agentic-core-okf`` -> ``motoro-okf``,
matching the FastMCP name core now serves it under. That name is a foreign key in
spirit: protocols and agent tool configs may reference a server by name, so the
row is renamed in place rather than replaced, keeping its id and every reference
to it intact. ``mcp_server_configs.name`` is globally unique (see f64307429723),
hence the NOT EXISTS guard -- if a ``motoro-okf`` row somehow already exists, the
legacy row is left alone for a human to reconcile instead of failing the deploy.

Revision ID: b7e2a91c4f30
Revises: f64307429723
Create Date: 2026-08-19 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "b7e2a91c4f30"
down_revision: str | None = "f64307429723"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_OLD_MODULE = "agentic_core.mcp_servers.okf"
_NEW_MODULE = "motoro.mcp_servers.okf"
_OLD_NAME = "agentic-core-okf"
_NEW_NAME = "motoro-okf"


def _rewrite(old_module: str, new_module: str, old_name: str, new_name: str) -> None:
    op.execute(
        f"""
        UPDATE mcp_server_configs
           SET command = replace(command, '{old_module}', '{new_module}')
         WHERE command LIKE '%{old_module}%'
        """  # noqa: S608
    )
    op.execute(
        f"""
        UPDATE mcp_server_configs
           SET name = '{new_name}'
         WHERE name = '{old_name}'
           AND NOT EXISTS (SELECT 1 FROM mcp_server_configs WHERE name = '{new_name}')
        """  # noqa: S608
    )


def upgrade() -> None:
    _rewrite(_OLD_MODULE, _NEW_MODULE, _OLD_NAME, _NEW_NAME)


def downgrade() -> None:
    _rewrite(_NEW_MODULE, _OLD_MODULE, _NEW_NAME, _OLD_NAME)
