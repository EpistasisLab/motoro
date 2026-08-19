"""MCP Server configuration ORM model — persisted registration for a live connection.

Two severances from ARES's ``MCPServerConfig``:

- ``created_by_id`` (``NOT NULL FK -> users``) becomes ``owner_id``: nullable,
  no foreign key, no relationship — the same opaque attribution tag as
  ``Agent.owner_id`` and ``AgentRun.owner_id``, for the same reason (``users``
  is a product table core cannot reference).
- ``source_plan_id`` (``FK -> plan_records``) is dropped entirely rather than
  made opaque. It records that a server was proposed by ARES's Plan Builder — a
  product feature, not a fact about the server. An opaque UUID would still be
  provenance about a product concept core has no business encoding at all.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Enum, String, Text, func
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column

from motoro.models.base import Base, generate_uuid


class MCPTransport(enum.StrEnum):
    """MCP server transport type."""

    STDIO = "stdio"
    HTTP = "http"
    SSE = "sse"


class MCPServerStatus(enum.StrEnum):
    """MCP server connection status."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class MCPServerConfig(Base):
    """Persisted configuration for an MCP server connection."""

    __tablename__ = "mcp_server_configs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    transport: Mapped[MCPTransport] = mapped_column(
        Enum(
            MCPTransport,
            name="mcp_transport",
            create_constraint=True,
            native_enum=True,
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
    )
    command: Mapped[str | None] = mapped_column(Text, nullable=True)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    capabilities: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    status: Mapped[MCPServerStatus] = mapped_column(
        Enum(
            MCPServerStatus,
            name="mcp_server_status",
            create_constraint=True,
            native_enum=True,
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
        default=MCPServerStatus.DISCONNECTED,
    )
    headers_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_system: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false", default=False)
    # Opaque attribution tag — see the module docstring and Agent.owner_id.
    owner_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
