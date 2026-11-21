"""Public registration joins caller transactions without extra writes or connections."""
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from motoro.services import mcp_service, skill_service


def session():
    added = []
    return SimpleNamespace(add=added.append, added=added, flush=AsyncMock(), refresh=AsyncMock(),
        commit=AsyncMock(), rollback=AsyncMock())


async def test_bundle_registration_uses_native_validation_and_caller_transaction():
    db = session()
    identifier, owner = uuid.uuid4(), uuid.uuid4()
    skill = await skill_service.create_skill_from_bundle([
        ("SKILL.md", "---\nname: portable-skill\ndescription: Register safely\n---\nInstructions"),
        ("references/context.md", b"Exact\r\n"), ("assets/icon.bin", b"\xff\x00"),
    ], owner_id=owner, identifier=identifier, db=db)
    assert skill is db.added[0] and skill.id == identifier and skill.owner_id == owner
    assert [(file.path, file.encoding, file.position) for file in skill.files] == [
        ("references/context.md", "utf-8", 0), ("assets/icon.bin", "base64", 1)]
    db.flush.assert_awaited_once()
    db.commit.assert_not_awaited()
    db.rollback.assert_not_awaited()
    with pytest.raises(skill_service.SkillFormatError):
        await skill_service.create_skill_from_bundle([("SKILL.md", "Missing frontmatter")], db=db)
    assert len(db.added) == 1


async def test_caller_controls_rollback_after_skill_storage_failure():
    db = session()
    db.flush.side_effect = RuntimeError("storage unavailable")
    with pytest.raises(RuntimeError, match="storage unavailable"):
        await skill_service.create_skill(name="valid-skill", description="A valid description", db=db)
    db.commit.assert_not_awaited()
    db.rollback.assert_not_awaited()


async def test_deferred_mcp_registration_uses_native_namespace_rules_without_connecting(monkeypatch):
    db = session()
    lock = AsyncMock()
    monkeypatch.setattr(mcp_service, "_lock_server_name", lock)
    monkeypatch.setattr(mcp_service, "_name_conflicts", AsyncMock(return_value=False))
    registry = SimpleNamespace(register=AsyncMock())
    identifier, owner = uuid.uuid4(), uuid.uuid4()
    server = await mcp_service.register_server(name="portable-server", transport="http", db=db,
        identifier=identifier, owner_id=owner, connect=False, registry=registry)
    assert server.id == identifier and server.owner_id == owner and server is db.added[0]
    lock.assert_awaited_once_with(db, "portable-server")
    registry.register.assert_not_awaited()
    db.commit.assert_not_awaited()
    db.rollback.assert_not_awaited()
    monkeypatch.setattr(mcp_service, "_name_conflicts", AsyncMock(return_value=True))
    with pytest.raises(mcp_service.MCPServerNameConflictError):
        await mcp_service.register_server(name="portable-server", transport="http", db=db, connect=False)
    assert len(db.added) == 1


async def test_caller_transaction_cannot_start_a_connection_before_commit():
    with pytest.raises(ValueError, match="defer connection"):
        await mcp_service.register_server(name="server", transport="http", db=session())
