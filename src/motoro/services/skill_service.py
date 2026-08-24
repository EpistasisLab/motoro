"""Agent Skills — parsing, storage, and resolution of ``SKILL.md`` documents.

A skill is a markdown file with YAML frontmatter: a ``name`` and a
``description`` the agent always sees, over a body it reads only once it
decides the skill applies. That two-level split is the whole point of the
format — the description is a cheap, always-loaded pointer, and the body is
the expensive part that stays out of the context window until it is wanted.
The engine honours the split (see :mod:`motoro.engine.skills`); this module is
the half that turns a file into a row and a row back into a file.

Why the *file* and not the directory the published format describes: see
:mod:`motoro.models.skill`. Short version — a skill with no bundled resources
is exactly one ``SKILL.md``, and bundled scripts presume a shell no Motoro
agent has.

Frontmatter keys other than ``name``/``description`` (``license``,
``allowed-tools``, ``metadata``) are parsed and discarded rather than stored.
They are not unsupported by accident: ``allowed-tools`` names Claude Code's own
tool namespace, which has no meaning against an MCP registry, and honouring it
here would mean claiming an enforcement core does not perform.

Each function opens and closes its own session, like every other public entry
point in core (see ``runner.py``'s module docstring) — with one documented
exception on :func:`resolve_skills`, which a caller already holding a session
may pass it to.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import yaml
from sqlalchemy import or_, select

from motoro.models.skill import Skill

if TYPE_CHECKING:
    from collections.abc import Sequence
    from contextlib import AbstractAsyncContextManager

    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024

# The format's own name rule: lowercase letters, digits and hyphens. It is not
# cosmetic — the name is what a directory would be called on disk and what the
# model types into ``load_skill``, so anything needing quoting or escaping in
# either position is rejected up front.
_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

# Frontmatter is delimited by `---` on its own line. A leading BOM/blank line is
# tolerated; anything else before the opening fence is not, because a file with
# prose above its frontmatter is a file whose frontmatter was never intended.
_FRONTMATTER_RE = re.compile(r"\A﻿?\s*---[ \t]*\r?\n(?P<yaml>.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)

# Reserved words the format forbids in a skill name, so a user-authored skill
# cannot present itself as a first-party one.
_RESERVED_NAME_SUBSTRINGS = ("anthropic", "claude")

# XML tags are excluded from both metadata fields: the metadata block is
# assembled into a system prompt alongside whatever tag-delimited structure a
# product uses, and a skill that can open a tag there can forge that structure.
_XML_TAG_RE = re.compile(r"<[^>]+>")


class SkillFormatError(ValueError):
    """A ``SKILL.md`` document does not conform to the Agent Skills format."""


@dataclass(frozen=True)
class ParsedSkill:
    """The three fields a ``SKILL.md`` yields, already validated."""

    name: str
    description: str
    body: str


def _session(reason: str) -> AbstractAsyncContextManager[AsyncSession]:
    from motoro.models.database import system_session

    return system_session(reason=f"skill_service: {reason}")


# --------------------------------------------------------------------------- #
#  Parsing / validation                                                        #
# --------------------------------------------------------------------------- #


def validate_skill_name(name: str) -> str:
    """Return *name* if it satisfies the format's name rule, else raise."""
    if not name:
        raise SkillFormatError("Skill frontmatter must include a non-empty 'name'.")
    if len(name) > MAX_NAME_LENGTH:
        raise SkillFormatError(f"Skill name must be at most {MAX_NAME_LENGTH} characters (got {len(name)}).")
    if not _NAME_RE.match(name):
        raise SkillFormatError(
            f"Skill name '{name}' is invalid: use lowercase letters, digits and hyphens "
            "(e.g. 'spinal-mri-qc'), with no leading, trailing or doubled hyphens."
        )
    lowered = name.lower()
    for reserved in _RESERVED_NAME_SUBSTRINGS:
        if reserved in lowered:
            raise SkillFormatError(f"Skill name may not contain '{reserved}'.")
    if _XML_TAG_RE.search(name):
        raise SkillFormatError("Skill name may not contain XML tags.")
    return name


def validate_skill_description(description: str) -> str:
    """Return *description* if it satisfies the format's description rule, else raise."""
    if not description.strip():
        raise SkillFormatError(
            "Skill frontmatter must include a non-empty 'description'. It is the only "
            "part of the skill the agent sees before opening it, so it must say both "
            "what the skill does and when to use it."
        )
    if len(description) > MAX_DESCRIPTION_LENGTH:
        raise SkillFormatError(
            f"Skill description must be at most {MAX_DESCRIPTION_LENGTH} characters (got {len(description)})."
        )
    if _XML_TAG_RE.search(description):
        raise SkillFormatError("Skill description may not contain XML tags.")
    return description.strip()


def parse_skill_markdown(text: str) -> ParsedSkill:
    """Parse a ``SKILL.md`` document into validated name/description/body.

    Raises :class:`SkillFormatError` for anything the format rejects — a missing
    or unparseable frontmatter block, a missing or malformed ``name``, a missing
    or over-long ``description``. Failing here means a bad upload is refused at
    the boundary rather than surfacing as a confusing prompt at run time.
    """
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        raise SkillFormatError(
            "Skill file must begin with a YAML frontmatter block delimited by '---' "
            "lines, containing at least 'name' and 'description'."
        )

    try:
        loaded = yaml.safe_load(match.group("yaml")) or {}
    except yaml.YAMLError as exc:
        raise SkillFormatError(f"Skill frontmatter is not valid YAML: {exc}") from exc

    if not isinstance(loaded, dict):
        raise SkillFormatError("Skill frontmatter must be a YAML mapping of keys to values.")

    name = validate_skill_name(str(loaded.get("name") or "").strip())
    description = validate_skill_description(str(loaded.get("description") or ""))
    body = text[match.end() :].strip()
    return ParsedSkill(name=name, description=description, body=body)


def render_skill_md(skill: Skill | ParsedSkill) -> str:
    """Render a stored skill back to a ``SKILL.md`` document.

    The round trip is what keeps the single-file model spec-conformant: write
    this to ``<root>/<name>/SKILL.md`` and the result is a valid skill
    directory, so a stored skill can always leave core in the format it arrived
    in. ``yaml.safe_dump`` does the quoting, so a description containing a
    colon or a quote survives the trip.
    """
    front = yaml.safe_dump(
        {"name": skill.name, "description": skill.description},
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    ).strip()
    body = (skill.body or "").strip()
    return f"---\n{front}\n---\n\n{body}\n" if body else f"---\n{front}\n---\n"


# --------------------------------------------------------------------------- #
#  CRUD                                                                        #
# --------------------------------------------------------------------------- #


async def create_skill(
    *,
    name: str,
    description: str,
    body: str = "",
    owner_id: uuid.UUID | None = None,
    is_system: bool = False,
    source_filename: str | None = None,
) -> Skill:
    """Persist a skill from already-separated fields.

    Both metadata fields are validated here as well as in
    :func:`parse_skill_markdown`, so a caller that builds a skill in a form
    (rather than uploading a file) cannot bypass the format's rules.
    """
    skill = Skill(
        name=validate_skill_name(name.strip()),
        description=validate_skill_description(description),
        body=body.strip(),
        owner_id=owner_id,
        is_system=is_system,
        source_filename=source_filename,
    )
    async with _session("create_skill") as db:
        db.add(skill)
        await db.commit()
        await db.refresh(skill)
    return skill


async def create_skill_from_markdown(
    text: str,
    *,
    owner_id: uuid.UUID | None = None,
    is_system: bool = False,
    source_filename: str | None = None,
) -> Skill:
    """Parse a ``SKILL.md`` document and persist it."""
    parsed = parse_skill_markdown(text)
    return await create_skill(
        name=parsed.name,
        description=parsed.description,
        body=parsed.body,
        owner_id=owner_id,
        is_system=is_system,
        source_filename=source_filename,
    )


async def get_skill(skill_id: uuid.UUID) -> Skill | None:
    """Fetch a live skill by id, or ``None``."""
    async with _session("get_skill") as db:
        return (
            await db.execute(select(Skill).where(Skill.id == skill_id, Skill.deleted_at.is_(None)))
        ).scalar_one_or_none()


async def list_skills(*, owner_id: uuid.UUID | None = None, limit: int = 100) -> Sequence[Skill]:
    """List live skills, newest first.

    A plain filter, not enforcement — core has no viewer to scope against; see
    ``mcp_service.list_servers``. When *owner_id* is given, ``is_system`` skills
    are included alongside the owner's own, since a platform-provided skill is
    available to everyone by definition.
    """
    stmt = select(Skill).where(Skill.deleted_at.is_(None))
    if owner_id is not None:
        stmt = stmt.where(or_(Skill.owner_id == owner_id, Skill.is_system.is_(True)))
    stmt = stmt.order_by(Skill.created_at.desc()).limit(limit)
    async with _session("list_skills") as db:
        return (await db.execute(stmt)).scalars().all()


async def update_skill(
    skill_id: uuid.UUID,
    *,
    name: str | None = None,
    description: str | None = None,
    body: str | None = None,
) -> Skill | None:
    """Update a live skill. ``None`` means "leave unchanged" for every field.

    Returns ``None`` if *skill_id* does not name a live skill.
    """
    async with _session("update_skill") as db:
        skill = (
            await db.execute(select(Skill).where(Skill.id == skill_id, Skill.deleted_at.is_(None)))
        ).scalar_one_or_none()
        if skill is None:
            return None
        if name is not None:
            skill.name = validate_skill_name(name.strip())
        if description is not None:
            skill.description = validate_skill_description(description)
        if body is not None:
            skill.body = body.strip()
        await db.commit()
        await db.refresh(skill)
        return skill


async def update_skill_from_markdown(skill_id: uuid.UUID, text: str) -> Skill | None:
    """Replace a live skill's contents from a re-uploaded ``SKILL.md``."""
    parsed = parse_skill_markdown(text)
    return await update_skill(skill_id, name=parsed.name, description=parsed.description, body=parsed.body)


async def delete_skill(skill_id: uuid.UUID) -> bool:
    """Soft-delete a skill, releasing its name. Returns False if it did not exist.

    Soft rather than hard because agents reference skills by id: a hard delete
    would leave a dangling id in some agent's ``skill_config`` with no way to
    say what used to be there. :func:`resolve_skills` skips a deleted skill and
    logs it, so an agent still runs — just without that skill.
    """
    from datetime import UTC, datetime

    async with _session("delete_skill") as db:
        skill = (
            await db.execute(select(Skill).where(Skill.id == skill_id, Skill.deleted_at.is_(None)))
        ).scalar_one_or_none()
        if skill is None:
            return False
        skill.deleted_at = datetime.now(tz=UTC)
        await db.commit()
        return True


# --------------------------------------------------------------------------- #
#  Resolution for a run                                                        #
# --------------------------------------------------------------------------- #


def skill_ids_from_config(skill_config: dict[str, Any] | None) -> list[uuid.UUID]:
    """Read the ordered skill ids out of an agent's ``skill_config``.

    Tolerant by design: the column is free-form JSON that a product writes, so
    an unparseable entry is dropped with a log line rather than failing a run
    that would otherwise work.
    """
    if not skill_config:
        return []
    raw = skill_config.get("skill_ids") or []
    if not isinstance(raw, list):
        logger.warning("skill_service.skill_ids_not_a_list", extra={"type": type(raw).__name__})
        return []
    ids: list[uuid.UUID] = []
    for entry in raw:
        try:
            ids.append(entry if isinstance(entry, uuid.UUID) else uuid.UUID(str(entry)))
        except (ValueError, AttributeError, TypeError):
            logger.warning("skill_service.invalid_skill_id", extra={"value": str(entry)})
    return ids


async def resolve_skills(
    skill_config: dict[str, Any] | None,
    *,
    owner_id: uuid.UUID | None = None,
    db: AsyncSession | None = None,
) -> list[dict[str, str]]:
    """Resolve an agent's ``skill_config`` into the skills a run should carry.

    Returns ``[{"name", "description", "body"}, ...]`` in the order the agent
    declared them — the order is the agent's, not the database's, because it is
    the order the metadata block will list them in.

    A referenced skill that has been deleted, or that belongs to another owner,
    is skipped with a log line rather than failing the run: losing one skill
    degrades a run, but refusing to start one strands every agent that ever
    referenced a since-deleted skill.

    *db* is the one place in this module a caller may supply its own session —
    :func:`motoro.runner.execute_run` holds one open for the whole run and
    resolving skills inside it should not take a second connection from the
    pool for a single small read.
    """
    ids = skill_ids_from_config(skill_config)
    if not ids:
        return []

    stmt = select(Skill).where(Skill.id.in_(ids), Skill.deleted_at.is_(None))
    if owner_id is not None:
        stmt = stmt.where(or_(Skill.owner_id == owner_id, Skill.is_system.is_(True)))

    if db is not None:
        rows = (await db.execute(stmt)).scalars().all()
    else:
        async with _session("resolve_skills") as own_db:
            rows = (await own_db.execute(stmt)).scalars().all()

    by_id = {row.id: row for row in rows}
    resolved: list[dict[str, str]] = []
    for skill_id in ids:
        skill = by_id.get(skill_id)
        if skill is None:
            logger.warning("skill_service.skill_unresolved", extra={"skill_id": str(skill_id)})
            continue
        resolved.append({"name": skill.name, "description": skill.description, "body": skill.body or ""})
    return resolved
