"""index run metadata containment and run step order

Revision ID: 7d3b1e5a9c20
Revises: 6a4d9f2c7e10
Create Date: 2026-10-13 00:00:00.000000

Products attribute runs by stamping their own keys into ``run_metadata`` and
page long traces by sequence. Both were full scans per owner/run, so a product
with tens of thousands of runs had to add its own indexes to core's tables.
The GIN index is key-agnostic (jsonb_path_ops over the whole document), so any
product's keys are served without core knowing them.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "7d3b1e5a9c20"
down_revision: str | None = "6a4d9f2c7e10"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "ix_agent_runs_run_metadata",
        "agent_runs",
        [sa.text("(run_metadata::jsonb) jsonb_path_ops")],
        postgresql_using="gin",
        if_not_exists=True,
    )
    op.create_index("ix_run_steps_run_sequence", "run_steps", ["run_id", "sequence"], if_not_exists=True)


def downgrade() -> None:
    op.drop_index("ix_run_steps_run_sequence", table_name="run_steps", if_exists=True)
    op.drop_index("ix_agent_runs_run_metadata", table_name="agent_runs", if_exists=True)
