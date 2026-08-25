"""Motoro's bundled OKF (Open Knowledge Format) MCP server.

OKF (https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md)
is a directory tree of markdown files with YAML frontmatter, "continuously
written and maintained by agents" rather than a read-only archive. Every tool
here mediates access rather than exposing the filesystem directly — reads get
pre-parsed, pre-filtered JSON instead of raw markdown+YAML to parse in
context; writes go through operations that enforce the spec's own structural
rules (an attested computation's ``parameters``/``runtime``/``executor``/
``attester`` can only be set by :func:`update_concept` itself, never
overwritten by an agent — see :func:`supply_computation_value`).

Configuration
-------------
The bundle root is read once, from ``AGENTIC_OKF_BUNDLE_DIR`` — never a path
an agent supplies — so a product spawns one server instance per bundle it
wants to expose, and each instance is jailed to exactly that directory. This
mirrors ASAREE's ``ASAREE_DATASET_WORKSPACE_DIR`` convention for
``asaree-workspace``.

Concurrency
-----------
The spec says nothing about concurrent access — this is purely an
implementation choice. Every mutation acquires a per-concept ``filelock``
(scoped to that one concept file) plus a shared lock over ``log.md`` (every
mutation appends to it), always in that fixed order, so two concurrent
mutations of different concepts never serialize behind each other but can
never deadlock either. File locks, not an in-process ``asyncio.Lock``, so the
guarantee holds even if this ever runs as more than one process against the
same bundle.

Actor attribution
------------------
The spec's ``generated``/``verified`` fields are structured (``{by, at}``),
and ``by`` follows a defined actor convention (SPEC.md §7): ``<producer>/
<version>`` for an agent/tool, ``human:<id>`` for a person, ``process:<id>``
for automation. Every write here stamps ``by`` from the calling run's ambient
identity (:func:`_actor_from_ctx`) — an agent run's own name/model when
present (``motoro.mcp.adapters.META_KEY_AGENT_NAME``/``META_KEY_MODEL``,
threaded onto every tool call automatically), else the run's owner id as a
``process:`` actor, else ``process:unknown``. Never a value the model can set
directly through tool arguments.
"""

from __future__ import annotations

import base64
import json
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

import yaml
from filelock import FileLock
from mcp.server import FastMCP
from mcp.server.fastmcp import Context

from motoro.mcp.adapters import META_KEY_AGENT_NAME, META_KEY_MODEL, META_KEY_OWNER_ID

mcp = FastMCP("motoro-okf")

_BUNDLE_ROOT_ENV = "AGENTIC_OKF_BUNDLE_DIR"
# index.md / log.md at the bundle root are the spec's own special files, not
# concepts — never listed, searched, or addressable by create/update/etc.
_RESERVED_ROOT_STEMS = {"index", "log"}
# An attested computation's own definition (SPEC.md §6) — update_concept
# refuses to touch these; supply_computation_value is the only door in.
_ATTESTED_COMPUTATION_LOCKED_FIELDS = ("parameters", "runtime", "executor", "attester")
_MARKDOWN_LINK_RE = re.compile(r"\]\(([^)\s]+)\)")
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?\n)---\s*\n?(.*)\Z", re.S)


# --------------------------------------------------------------------------- #
#  JSON encoding
# --------------------------------------------------------------------------- #


def _json_default(value: Any) -> str:
    """Render what YAML parses natively but JSON has no type for.

    Frontmatter is YAML, and YAML resolves an unquoted ``2026-08-25T16:00:00Z``
    to a real ``datetime`` — so a hand-authored bundle whose ``generated.at``
    (or ``verified[].at``, or a date-valued field of its own) isn't quoted hands
    :func:`_dumps` an object ``json.dumps`` refuses, and the whole tool call
    fails with "Object of type datetime is not JSON serializable" instead of
    returning the concepts. A bundle written *by* this server never trips it —
    it stamps ``at`` as an ISO string, which ``yaml.safe_dump`` quotes — so this
    is specifically about bundles authored elsewhere, which the spec's
    "continuously written and maintained" tree is full of.

    ISO 8601 rather than ``str``, so a date read back out matches the form this
    server writes (``datetime.now(UTC).isoformat()``) instead of YAML's
    space-separated ``str(datetime)``. Anything else unexpected degrades to
    ``str`` — a read of an odd frontmatter value should return something the
    model can use, not fail.
    """
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    return str(value)


def _dumps(payload: Any) -> str:
    """Serialize a tool result. Every tool returns through here, so a
    JSON-hostile frontmatter value can't fail one tool and not its neighbour —
    see :func:`_json_default`."""
    return json.dumps(payload, default=_json_default)


class OKFError(Exception):
    """A malformed request — caught at each tool's boundary and reported as
    ``{"error": ...}`` rather than propagating, matching
    ``asaree.mcp_servers.workspace_server``'s convention."""


# --------------------------------------------------------------------------- #
#  Bundle root / path jailing
# --------------------------------------------------------------------------- #


def _bundle_root() -> Path:
    raw = os.environ.get(_BUNDLE_ROOT_ENV)
    if not raw:
        raise OKFError(f"{_BUNDLE_ROOT_ENV} is not set; this server has no bundle to serve.")
    root = Path(raw).resolve()
    if not root.is_dir():
        raise OKFError(f"{_BUNDLE_ROOT_ENV}={raw!r} is not a directory.")
    return root


def _resolve_path(root: Path, relative: str, *, suffix: str = "", must_exist: bool = False) -> Path:
    """Jail *relative* (plus *suffix*) inside *root*; raise OKFError on any escape attempt."""
    if not relative or relative.startswith("/") or ".." in Path(relative).parts:
        raise OKFError(f"invalid path {relative!r}")
    candidate = (root / f"{relative}{suffix}").resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise OKFError(f"path {relative!r} escapes the bundle root") from None
    if must_exist and not candidate.is_file():
        raise OKFError(f"no such file: {relative}{suffix}")
    return candidate


def _resolve_reference_path(root: Path, ref_path: str, *, must_exist: bool = False) -> Path:
    if not ref_path.startswith("references/"):
        raise OKFError(f"ref_path must be under references/, got {ref_path!r}")
    return _resolve_path(root, ref_path, must_exist=must_exist)


# --------------------------------------------------------------------------- #
#  Concept frontmatter parsing/writing
# --------------------------------------------------------------------------- #


def _parse_concept_file(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    m = _FRONTMATTER_RE.match(text)
    if not m:
        raise OKFError(f"{path.name} has no YAML frontmatter block")
    frontmatter = yaml.safe_load(m.group(1)) or {}
    if not isinstance(frontmatter, dict):
        raise OKFError(f"{path.name}'s frontmatter is not a mapping")
    return frontmatter, m.group(2)


def _write_concept_file(path: Path, frontmatter: dict[str, Any], body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "---\n" + yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True) + "---\n" + body
    path.write_text(text, encoding="utf-8")


def _iter_concept_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*.md")):
        rel = path.relative_to(root)
        if "references" in rel.parts:
            continue
        if path.parent == root and path.stem in _RESERVED_ROOT_STEMS:
            continue
        yield path


def _concept_id(root: Path, path: Path) -> str:
    return path.relative_to(root).with_suffix("").as_posix()


# --------------------------------------------------------------------------- #
#  Ambient identity -> OKF actor string (SPEC.md §7)
# --------------------------------------------------------------------------- #


def _meta_mapping_from_ctx(ctx: Any) -> dict[str, Any] | None:
    """Duck-typed extraction of the request ``_meta`` mapping from a FastMCP
    ``Context`` — mirrors ``asaree_workspace_core.context.meta_mapping_from_ctx``,
    inlined here rather than imported since this package has no reason to
    depend on a product's own workspace-core library."""
    if ctx is None:
        return None
    request_context = getattr(ctx, "request_context", None)
    if request_context is None:
        return None
    meta = getattr(request_context, "meta", None)
    if meta is None:
        return None
    return getattr(meta, "model_extra", None) or {}


def _actor_from_ctx(ctx: Any) -> str:
    """Resolve a SPEC.md §7 actor string from ambient run identity.

    ``<agent_name>/<model>`` when the call came from a real agent run (both
    keys are only ever present together — see ``adapters._build_run_meta``),
    else ``process:<owner_id>`` for a direct/manual call on behalf of a known
    owner, else ``process:unknown``. Never a value the model can set directly
    — there is deliberately no actor/by argument on any tool below.
    """
    meta = _meta_mapping_from_ctx(ctx)
    agent_name = meta.get(META_KEY_AGENT_NAME) if meta else None
    if isinstance(agent_name, str) and agent_name:
        model = meta.get(META_KEY_MODEL) if meta else None
        return f"{agent_name}/{model}" if isinstance(model, str) and model else f"{agent_name}/unknown"
    owner_id = meta.get(META_KEY_OWNER_ID) if meta else None
    if isinstance(owner_id, str) and owner_id:
        return f"process:{owner_id}"
    return "process:unknown"


# --------------------------------------------------------------------------- #
#  Locking + log.md
# --------------------------------------------------------------------------- #


def _lock_path(root: Path, key: str) -> Path:
    path = root / ".okf-locks" / f"{key.replace('/', '__')}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def _mutate_lock(root: Path, key: str) -> Iterator[None]:
    """Acquire *key*'s own lock, then the shared log.md lock, in that fixed
    order — every mutation uses this same order, so two concurrent mutations
    can never deadlock on each other."""
    with FileLock(str(_lock_path(root, key))), FileLock(str(_lock_path(root, "log.md"))):
        yield


def _append_log(root: Path, kind: str, concept_id: str, actor: str, note: str = "") -> None:
    """Append a dated entry to the bundle's log.md, matching the spec's
    **Update**/**Creation**/**Deprecation** convention. Caller must already
    hold the log lock (i.e. call from inside :func:`_mutate_lock`)."""
    log_path = root / "log.md"
    date = datetime.now(UTC).strftime("%Y-%m-%d")
    line = f"- **{kind}** `{concept_id}` by `{actor}`" + (f" — {note}" if note else "") + "\n"
    existing = log_path.read_text(encoding="utf-8") if log_path.exists() else "# Log\n\n"
    header = f"## {date}\n"
    if header in existing:
        idx = existing.index(header) + len(header)
        existing = existing[:idx] + line + existing[idx:]
    else:
        existing = existing.rstrip("\n") + "\n\n" + header + line
    log_path.write_text(existing, encoding="utf-8")


# --------------------------------------------------------------------------- #
#  Tools — discovery
# --------------------------------------------------------------------------- #


@mcp.tool()
def list_concepts(type: str = "", tags: str = "", verified_only: bool = False) -> str:
    """List concepts in the bundle as lightweight summaries — progressive
    disclosure; call get_concept for any one concept's full body.

    Args:
        type: exact match on the concept's `type` field; "" = any type.
        tags: comma-separated; a concept matches if it has ANY of these tags; "" = no tag filter.
        verified_only: only include concepts with at least one `verified` entry.
    """
    try:
        root = _bundle_root()
    except OKFError as e:
        return _dumps({"error": str(e)})
    wanted_tags = {t.strip() for t in tags.split(",") if t.strip()}
    results = []
    for path in _iter_concept_files(root):
        try:
            fm, _ = _parse_concept_file(path)
        except OKFError:
            continue  # graceful degradation (SPEC.md): a malformed file is skipped, not fatal
        if type and fm.get("type") != type:
            continue
        concept_tags = set(fm.get("tags") or [])
        if wanted_tags and not (wanted_tags & concept_tags):
            continue
        if verified_only and not fm.get("verified"):
            continue
        results.append(
            {
                "id": _concept_id(root, path),
                "type": fm.get("type"),
                "title": fm.get("title"),
                "tags": sorted(concept_tags),
                "generated": fm.get("generated"),
                "verified": fm.get("verified"),
                "status": fm.get("status"),
            }
        )
    return _dumps({"concepts": results, "count": len(results)})


@mcp.tool()
def search_concepts(query: str, type: str = "") -> str:
    """Lexical, case-insensitive substring search over concept
    titles/descriptions/bodies. No embeddings/vector index.

    Args:
        query: matched as a case-insensitive substring.
        type: optional exact-match filter on `type`.
    """
    try:
        root = _bundle_root()
    except OKFError as e:
        return _dumps({"error": str(e)})
    q = query.strip().lower()
    if not q:
        return _dumps({"error": "query must not be empty"})
    results = []
    for path in _iter_concept_files(root):
        try:
            fm, body = _parse_concept_file(path)
        except OKFError:
            continue
        if type and fm.get("type") != type:
            continue
        haystack = f"{fm.get('title', '')} {fm.get('description', '')} {body}".lower()
        if q in haystack:
            results.append(
                {
                    "id": _concept_id(root, path),
                    "type": fm.get("type"),
                    "title": fm.get("title"),
                    "tags": sorted(fm.get("tags") or []),
                }
            )
    return _dumps({"concepts": results, "count": len(results)})


@mcp.tool()
def get_concept(id: str) -> str:
    """Return a concept's full parsed frontmatter, body, and outgoing markdown
    links (resolved once here so the model never has to regex the body itself).

    Args:
        id: the concept's id — its bundle-relative path, without the .md suffix.
    """
    try:
        root = _bundle_root()
        path = _resolve_path(root, id, suffix=".md", must_exist=True)
        frontmatter, body = _parse_concept_file(path)
    except OKFError as e:
        return _dumps({"error": str(e)})
    links = sorted(set(_MARKDOWN_LINK_RE.findall(body)))
    return _dumps({**frontmatter, "id": id, "body": body, "links": links})


# --------------------------------------------------------------------------- #
#  Tools — mutation
# --------------------------------------------------------------------------- #


@mcp.tool()
def create_concept(
    type: str,
    title: str,
    path: str,
    description: str = "",
    tags: str = "",
    body: str = "",
    ctx: Context[Any, Any, Any] | None = None,
) -> str:
    """Create a new concept. Sets `generated` (by/at) automatically; `verified`
    is left unset — a new concept has not been confirmed by anyone yet.
    Refuses an existing path (use update_concept) or one escaping the bundle.

    Args:
        type: the concept's `type` (required by the OKF spec).
        title: human-readable title.
        path: bundle-relative location, no .md suffix, e.g. "models/discharge-risk/xgboost-v1".
        description: optional one-line description.
        tags: comma-separated tags.
        body: the concept's markdown body (Schema/Examples/Computation sections, etc).
    """
    try:
        root = _bundle_root()
        concept_path = _resolve_path(root, path, suffix=".md")
    except OKFError as e:
        return _dumps({"error": str(e)})
    actor = _actor_from_ctx(ctx)
    with _mutate_lock(root, path):
        if concept_path.exists():
            return _dumps({"error": f"concept already exists at {path!r}; use update_concept"})
        frontmatter: dict[str, Any] = {
            "type": type,
            "title": title,
            "generated": {"by": actor, "at": datetime.now(UTC).isoformat()},
        }
        if description:
            frontmatter["description"] = description
        if tags:
            frontmatter["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
        _write_concept_file(concept_path, frontmatter, body)
        _append_log(root, "Creation", path, actor)
    return _dumps({"id": path, "created": True, "generated": frontmatter["generated"]})


@mcp.tool()
def update_concept(
    id: str,
    fields: dict[str, Any] | None = None,
    body: str | None = None,
    append_body: str = "",
    ctx: Context[Any, Any, Any] | None = None,
) -> str:
    """Update an existing concept's frontmatter and/or body. Bumps `generated`
    to now automatically.

    Refuses to touch `parameters`/`runtime`/`executor`/`attester` on a concept
    that already declares an `attester` (an attested computation) — per the
    OKF spec, an agent may only supply computed VALUES for such a concept
    (see supply_computation_value), never author or edit the computation.

    Args:
        id: the concept's id.
        fields: frontmatter fields to shallow-merge in.
        body: if given, replaces the body outright.
        append_body: if given (and body is not), appended to the existing body.
    """
    try:
        root = _bundle_root()
        concept_path = _resolve_path(root, id, suffix=".md", must_exist=True)
    except OKFError as e:
        return _dumps({"error": str(e)})
    fields = fields or {}
    actor = _actor_from_ctx(ctx)
    # The read, the attester guard, and the write must all happen inside one
    # locked critical section — reading beforehand would let two concurrent
    # callers both read the same pre-write state and the second clobber the
    # first (a lost update the lock would otherwise prevent).
    with _mutate_lock(root, id):
        try:
            frontmatter, current_body = _parse_concept_file(concept_path)
        except OKFError as e:
            return _dumps({"error": str(e)})
        if frontmatter.get("attester"):
            touched = [f for f in _ATTESTED_COMPUTATION_LOCKED_FIELDS if f in fields]
            if touched:
                return _dumps(
                    {
                        "error": f"{id!r} is an attested computation; {touched} may not be edited "
                        "directly — use supply_computation_value to supply parameter values instead."
                    }
                )
        frontmatter.update(fields)
        frontmatter["generated"] = {"by": actor, "at": datetime.now(UTC).isoformat()}
        new_body = body if body is not None else (current_body + append_body if append_body else current_body)
        _write_concept_file(concept_path, frontmatter, new_body)
        _append_log(root, "Update", id, actor)
    return _dumps({"id": id, "updated": True, "generated": frontmatter["generated"]})


@mcp.tool()
def mark_verified(id: str, note: str = "", ctx: Context[Any, Any, Any] | None = None) -> str:
    """Re-confirm a concept without regenerating it — appends a `verified`
    entry ({by, at}) without touching `generated` or the body, matching the
    spec's distinction ("facts can be re-confirmed without regeneration").

    Args:
        id: the concept's id.
        note: optional free-text note for the log entry.
    """
    try:
        root = _bundle_root()
        concept_path = _resolve_path(root, id, suffix=".md", must_exist=True)
    except OKFError as e:
        return _dumps({"error": str(e)})
    actor = _actor_from_ctx(ctx)
    with _mutate_lock(root, id):
        try:
            frontmatter, body = _parse_concept_file(concept_path)
        except OKFError as e:
            return _dumps({"error": str(e)})
        entry = {"by": actor, "at": datetime.now(UTC).isoformat()}
        existing = frontmatter.get("verified")
        if existing is None:
            frontmatter["verified"] = entry
        elif isinstance(existing, list):
            frontmatter["verified"] = [*existing, entry]
        else:
            frontmatter["verified"] = [existing, entry]
        _write_concept_file(concept_path, frontmatter, body)
        _append_log(root, "Update", id, actor, note or "verified")
    return _dumps({"id": id, "verified": frontmatter["verified"]})


@mcp.tool()
def supply_computation_value(
    id: str, parameter: str, value: Any, ctx: Context[Any, Any, Any] | None = None
) -> str:
    """Supply a value for one of an attested computation's declared
    parameters — the ONLY way to touch such a concept. Validates *parameter*
    against the concept's own `parameters` list and refuses anything not
    declared there (OKF spec: "MAY only supply values for the declared
    parameters; MUST NOT author or edit the computation").

    Args:
        id: the concept's id.
        parameter: must be one of the concept's own declared parameter names.
        value: the value to record for that parameter.
    """
    try:
        root = _bundle_root()
        concept_path = _resolve_path(root, id, suffix=".md", must_exist=True)
    except OKFError as e:
        return _dumps({"error": str(e)})
    actor = _actor_from_ctx(ctx)
    with _mutate_lock(root, id):
        try:
            frontmatter, body = _parse_concept_file(concept_path)
        except OKFError as e:
            return _dumps({"error": str(e)})
        declared = frontmatter.get("parameters") or []
        declared_names = {p.get("name") if isinstance(p, dict) else p for p in declared}
        if parameter not in declared_names:
            return _dumps(
                {
                    "error": f"{parameter!r} is not a declared parameter of {id!r}; "
                    f"expected one of {sorted(n for n in declared_names if n)}"
                }
            )
        values = frontmatter.get("values") or {}
        values[parameter] = value
        frontmatter["values"] = values
        frontmatter["generated"] = {"by": actor, "at": datetime.now(UTC).isoformat()}
        _write_concept_file(concept_path, frontmatter, body)
        _append_log(root, "Update", id, actor, f"supplied value for parameter {parameter!r}")
    return _dumps({"id": id, "parameter": parameter, "values": frontmatter["values"]})


# --------------------------------------------------------------------------- #
#  Tools — references (non-markdown files: scripts, binaries)
# --------------------------------------------------------------------------- #


@mcp.tool()
def get_reference(ref_path: str) -> str:
    """Read a file under references/ verbatim.

    Returns ``{"encoding": "text", "content": "..."}`` for valid UTF-8 text,
    or ``{"encoding": "base64", "content": "..."}`` otherwise (e.g. a binary
    attachment).

    Args:
        ref_path: bundle-relative path under references/, e.g.
            "references/attesters/sql-equality.py".
    """
    try:
        root = _bundle_root()
        path = _resolve_reference_path(root, ref_path, must_exist=True)
    except OKFError as e:
        return _dumps({"error": str(e)})
    raw = path.read_bytes()
    try:
        return _dumps({"encoding": "text", "content": raw.decode("utf-8")})
    except UnicodeDecodeError:
        return _dumps({"encoding": "base64", "content": base64.b64encode(raw).decode("ascii")})


@mcp.tool()
def write_reference(
    ref_path: str, content: str, encoding: str = "text", ctx: Context[Any, Any, Any] | None = None
) -> str:
    """Write (create or overwrite) a file under references/.

    Args:
        ref_path: bundle-relative path under references/.
        content: the file's content — raw text if encoding="text", base64 if encoding="base64".
        encoding: "text" or "base64".
    """
    if encoding not in ("text", "base64"):
        return _dumps({"error": f"encoding must be 'text' or 'base64', got {encoding!r}"})
    try:
        root = _bundle_root()
        path = _resolve_reference_path(root, ref_path)
    except OKFError as e:
        return _dumps({"error": str(e)})
    actor = _actor_from_ctx(ctx)
    with _mutate_lock(root, ref_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if encoding == "text":
            path.write_text(content, encoding="utf-8")
        else:
            path.write_bytes(base64.b64decode(content))
        _append_log(root, "Update", ref_path, actor, "reference file written")
    return _dumps({"ref_path": ref_path, "written": True})


# --------------------------------------------------------------------------- #
#  Housekeeping
# --------------------------------------------------------------------------- #


@mcp.tool()
def reset_session() -> str:
    """No-op compatibility shim — this server holds no in-process session;
    every write lives on disk. Retained so a driver's between-run call keeps
    working (asaree.services.mcp_service.reset_server_session calls this on
    any registered server generically)."""
    return _dumps({"note": "stateless server; no in-process session to reset", "cleared": {}})


@mcp.tool()
def ping() -> str:
    """Health check — returns 'pong' to verify the server is running."""
    return "pong"


if __name__ == "__main__":
    mcp.run()
