"""Agent Skills — parsing, storage, and resolution of ``SKILL.md`` documents.

A skill is a markdown file with YAML frontmatter: a ``name`` and a
``description`` the agent always sees, over a body it reads only once it
decides the skill applies. That two-level split is the whole point of the
format — the description is a cheap, always-loaded pointer, and the body is
the expensive part that stays out of the context window until it is wanted.
The engine honours the split (see :mod:`motoro.engine.skills`); this module is
the half that turns a file into a row and a row back into a file.

A skill may also arrive as the *directory* the published format describes —
``code-simplification/SKILL.md`` plus whatever reference documents sit beside
it. :func:`parse_skill_bundle` takes those files as ``(relative path, text)``
pairs and splits them into the ``SKILL.md`` and its level-3 companions, which
are stored as :class:`motoro.models.skill.SkillFile` rows. What is still
refused is a bundled *script*: see :mod:`motoro.models.skill` for why, and
:func:`validate_bundle_path` for where.

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

from motoro.models.skill import Skill, SkillFile

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from contextlib import AbstractAsyncContextManager

    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024

# The entry point of a skill directory, by the format's definition. Matched
# case-insensitively on upload because a browser hands back whatever the
# filesystem stored, and a `skill.md` is unambiguously the same intent.
SKILL_MD = "SKILL.md"

# Bundle caps. Not the format's — it has none, because it assumes a filesystem
# where an unread file costs nothing. Core's bundle is loaded whole when a run
# resolves the skill (see resolve_skills), so "cheap until read" stops being
# true of memory even though it stays true of context, and something has to
# bound it. Generous against real skills, whose level-3 material is prose.
MAX_BUNDLE_FILES = 50
MAX_BUNDLE_BYTES = 1_000_000
MAX_BUNDLE_PATH_LENGTH = 255

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


@dataclass(frozen=True)
class ParsedSkillBundle:
    """A whole skill directory: its ``SKILL.md`` and its level-3 companions.

    ``files`` is ordered, and that order is preserved into storage — it is what
    the "bundled files" listing an agent sees is sorted by, so it should be
    stable between an upload and a re-upload rather than whatever order the
    caller's filesystem or the database happened to produce.
    """

    skill: ParsedSkill
    files: tuple[tuple[str, str], ...]


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

    The round trip is what keeps the stored form spec-conformant: write this to
    ``<root>/<name>/SKILL.md``, write each :class:`SkillFile` to
    ``<root>/<name>/<path>``, and the result is a valid skill directory — so a
    stored skill can always leave core in the format it arrived in.
    ``yaml.safe_dump`` does the quoting, so a description containing a colon or
    a quote survives the trip.
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
#  Bundles (the directory form)                                                #
# --------------------------------------------------------------------------- #

# Extensions core will store as a bundled level-3 file. An allow-list rather
# than a deny-list: the question is not "is this dangerous" but "can a Motoro
# agent do anything at all with it", and the only answer is "read it into the
# context window". Anything outside this list either needs a shell (a script)
# or cannot enter a context window (an image, a font, a .docx), so storing it
# would be storing something no run can ever reach.
BUNDLE_TEXT_SUFFIXES = frozenset({".md", ".markdown", ".txt", ".json", ".yaml", ".yml", ".csv", ".toml"})

# Suffixes worth naming in the error, because they are the ones a real skill
# from the wild actually ships and whose rejection therefore needs explaining
# rather than merely reporting.
_SCRIPT_SUFFIXES = frozenset({".py", ".sh", ".bash", ".js", ".ts", ".rb", ".pl", ".ps1"})


def _normalise_bundle_path(path: str) -> str:
    """Forward slashes, no leading ``./``, otherwise untouched.

    Deliberately *not* ``lstrip("./")``: that strips a character *set*, so it
    would quietly turn ``/etc/passwd`` into ``etc/passwd`` and ``../x`` into
    ``x`` — normalising away the very things :func:`validate_bundle_path` then
    checks for.
    """
    cleaned = (path or "").strip().replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned


def validate_bundle_path(path: str) -> str:
    """Return *path* normalised for storage, or raise :class:`SkillFormatError`.

    Normalised means: forward slashes, no leading ``./``, relative to the skill
    directory. Rejected means anything that is not a plain relative path to a
    readable text file — absolute paths, ``..`` traversal, hidden segments, and
    the suffixes outside :data:`BUNDLE_TEXT_SUFFIXES`.

    The traversal checks matter even though nothing here touches a filesystem:
    a product rendering a bundle back out to disk (the ``render_skill_md``
    round trip) would otherwise write wherever the stored path pointed, so the
    guarantee has to hold at the boundary where the path is accepted rather
    than at each place it is later used.
    """
    cleaned = _normalise_bundle_path(path)
    if not cleaned:
        raise SkillFormatError("A bundled skill file must have a path.")
    if len(cleaned) > MAX_BUNDLE_PATH_LENGTH:
        raise SkillFormatError(
            f"Bundled file path '{cleaned[:60]}...' is longer than {MAX_BUNDLE_PATH_LENGTH} characters."
        )
    if cleaned.startswith("/"):
        raise SkillFormatError(f"Bundled file path '{cleaned}' must be relative to the skill directory.")
    segments = cleaned.split("/")
    if any(segment in ("", ".", "..") for segment in segments):
        raise SkillFormatError(f"Bundled file path '{cleaned}' may not contain '.' or '..' segments.")
    if any(segment.startswith(".") for segment in segments):
        # .git, .DS_Store and friends: a folder picker hands over the whole
        # subtree, and none of it is part of the skill.
        raise SkillFormatError(f"Bundled file path '{cleaned}' may not contain hidden segments.")
    suffix = ("." + segments[-1].rsplit(".", 1)[-1].lower()) if "." in segments[-1] else ""
    if suffix in _SCRIPT_SUFFIXES:
        raise SkillFormatError(
            f"'{cleaned}' is a script, and a Motoro agent has no shell to run one in — its only "
            "way to act is an MCP tool call. Register the code as an MCP server instead, and keep "
            "the skill to the instructions that say when to call it."
        )
    if suffix not in BUNDLE_TEXT_SUFFIXES:
        allowed = ", ".join(sorted(BUNDLE_TEXT_SUFFIXES))
        raise SkillFormatError(
            f"'{cleaned}' is not a text file a skill can be read from. A bundled file reaches an "
            f"agent by being read into its context window, so it must be one of: {allowed}."
        )
    return cleaned


def parse_skill_bundle(files: Iterable[tuple[str, str]]) -> ParsedSkillBundle:
    """Parse a whole skill directory into its ``SKILL.md`` and level-3 files.

    *files* is ``(relative path, text)`` pairs — relative to the skill
    directory itself, so ``SKILL.md`` and ``references/schema.md``, not
    ``code-simplification/SKILL.md``. Stripping the leading directory segment is
    the caller's job, because only the caller knows whether the user picked the
    skill folder or its parent.

    Raises :class:`SkillFormatError` if there is no ``SKILL.md``, if it does not
    parse, or if any companion file fails :func:`validate_bundle_path`. Refusing
    the whole upload on one bad file is deliberate: a partially-stored skill is
    one whose ``SKILL.md`` links to documents that are silently not there.
    """
    entries = list(files)
    skill_md: str | None = None
    bundled: list[tuple[str, str]] = []
    total_bytes = 0

    for raw_path, text in entries:
        normalised = _normalise_bundle_path(raw_path)
        if normalised.lower() == SKILL_MD.lower():
            if skill_md is not None:
                raise SkillFormatError("That folder contains more than one SKILL.md.")
            skill_md = text
            continue
        bundled.append((validate_bundle_path(raw_path), text))
        total_bytes += len(text.encode("utf-8"))

    if skill_md is None:
        raise SkillFormatError(
            "That folder has no SKILL.md. An Agent Skill is a directory whose entry point is a "
            "SKILL.md holding the name/description frontmatter — pick the skill's own folder, not "
            "the folder containing it."
        )
    if len(bundled) > MAX_BUNDLE_FILES:
        raise SkillFormatError(
            f"That skill bundles {len(bundled)} files, more than the {MAX_BUNDLE_FILES} limit."
        )
    if total_bytes > MAX_BUNDLE_BYTES:
        raise SkillFormatError(
            f"That skill's bundled files total {total_bytes // 1000}KB, more than the "
            f"{MAX_BUNDLE_BYTES // 1000}KB limit."
        )

    seen: set[str] = set()
    for path, _text in bundled:
        lowered = path.lower()
        if lowered in seen:
            raise SkillFormatError(f"That folder contains '{path}' more than once (paths are case-insensitive).")
        seen.add(lowered)

    return ParsedSkillBundle(skill=parse_skill_markdown(skill_md), files=tuple(bundled))


def bundle_paths(skill: Skill) -> list[str]:
    """The bundled file paths of *skill*, in upload order."""
    return [f.path for f in skill.files]


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
    files: Iterable[tuple[str, str]] = (),
) -> Skill:
    """Persist a skill from already-separated fields.

    Both metadata fields are validated here as well as in
    :func:`parse_skill_markdown`, so a caller that builds a skill in a form
    (rather than uploading a file) cannot bypass the format's rules. The same
    goes for *files*, whose paths run through :func:`validate_bundle_path` here
    and not only in :func:`parse_skill_bundle`.
    """
    skill = Skill(
        name=validate_skill_name(name.strip()),
        description=validate_skill_description(description),
        body=body.strip(),
        owner_id=owner_id,
        is_system=is_system,
        source_filename=source_filename,
        files=[
            SkillFile(path=validate_bundle_path(path), content=content, position=index)
            for index, (path, content) in enumerate(files)
        ],
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


async def create_skill_from_bundle(
    files: Iterable[tuple[str, str]],
    *,
    owner_id: uuid.UUID | None = None,
    is_system: bool = False,
    source_filename: str | None = None,
) -> Skill:
    """Parse a whole skill directory and persist it, bundled files and all.

    The directory-shaped counterpart to :func:`create_skill_from_markdown`, and
    the entry point a folder upload should use. *source_filename* is the folder
    the user picked, for the same display-only purpose as the single-file case.
    """
    bundle = parse_skill_bundle(files)
    return await create_skill(
        name=bundle.skill.name,
        description=bundle.skill.description,
        body=bundle.skill.body,
        owner_id=owner_id,
        is_system=is_system,
        source_filename=source_filename,
        files=bundle.files,
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
    files: Iterable[tuple[str, str]] | None = None,
) -> Skill | None:
    """Update a live skill. ``None`` means "leave unchanged" for every field.

    Returns ``None`` if *skill_id* does not name a live skill.

    *files*, when given, is a **full replacement** of the bundle rather than a
    merge, and ``[]`` empties it. A merge would have no way to express "this
    re-upload dropped FORMS.md", and a stale document the SKILL.md no longer
    mentions is exactly the kind of thing an agent still reads.
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
        if files is not None:
            replacements = [
                SkillFile(path=validate_bundle_path(path), content=content, position=index)
                for index, (path, content) in enumerate(files)
            ]
            # Two flushes, deliberately. Clearing the collection is what fires
            # delete-orphan; flushing before the inserts is what stops a
            # re-upload keeping the same path from hitting
            # uq_skill_files_skill_path, since a single flush orders the INSERTs
            # ahead of the DELETEs.
            skill.files = []
            await db.flush()
            skill.files = replacements
        await db.commit()
        await db.refresh(skill)
        return skill


async def update_skill_from_markdown(skill_id: uuid.UUID, text: str) -> Skill | None:
    """Replace a live skill's contents from a re-uploaded ``SKILL.md``.

    Leaves the bundled files alone — this is the single-file path, and a
    ``SKILL.md`` on its own says nothing about whether its companions changed.
    Use :func:`update_skill_from_bundle` to replace the whole directory.
    """
    parsed = parse_skill_markdown(text)
    return await update_skill(skill_id, name=parsed.name, description=parsed.description, body=parsed.body)


async def update_skill_from_bundle(skill_id: uuid.UUID, files: Iterable[tuple[str, str]]) -> Skill | None:
    """Replace a live skill's whole directory from a re-uploaded folder."""
    bundle = parse_skill_bundle(files)
    return await update_skill(
        skill_id,
        name=bundle.skill.name,
        description=bundle.skill.description,
        body=bundle.skill.body,
        files=bundle.files,
    )


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
) -> list[dict[str, Any]]:
    """Resolve an agent's ``skill_config`` into the skills a run should carry.

    Returns ``[{"name", "description", "body", "files"}, ...]`` in the order the
    agent declared them — the order is the agent's, not the database's, because
    it is the order the metadata block will list them in.

    ``files`` is ``{path: contents}`` for the skill's bundled level-3
    documents, loaded whole here rather than fetched when the model asks for
    one. Progressive disclosure is about the *context window*, not about the
    read: :mod:`motoro.engine.skills` renders from a plain dict so it stays a
    set of pure functions with no session to thread through a pattern's turn
    loop, and MAX_BUNDLE_BYTES is what makes eager loading affordable.

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
    resolved: list[dict[str, Any]] = []
    for skill_id in ids:
        skill = by_id.get(skill_id)
        if skill is None:
            logger.warning("skill_service.skill_unresolved", extra={"skill_id": str(skill_id)})
            continue
        resolved.append(
            {
                "name": skill.name,
                "description": skill.description,
                "body": skill.body or "",
                "files": {f.path: f.content for f in skill.files},
            }
        )
    return resolved
