"""Rendering Agent Skills into a run — the three disclosure levels.

The Agent Skills format is built around progressive disclosure: a skill's
``name`` and ``description`` are always in context (cheap — a line or two per
skill), its body is loaded only when the agent judges the skill relevant, and
its bundled reference documents are read only when the body sends it to one.
That is what makes a dozen skills affordable where a dozen inlined instruction
documents would not be.

Two renderings, for two situations:

- :func:`render_skill_index` + :func:`build_load_skill_tool` +
  :func:`build_read_skill_file_tool` — the real thing. The index (level 1) goes
  in the stable system prefix; the body (level 2) arrives as a tool result when
  the model asks for it; a bundled file (level 3) arrives the same way, when the
  body points at one. Requires a pattern with a tool loop.
- :func:`inline_skills` — the fallback for a pattern that has no tool loop
  (``single_agent_baseline``, or any pattern that has not opted in). Every body
  is inlined up front. Honest but expensive; the disclosure is lost, not the
  content. Level 3 is the exception it cannot cover: with no tool loop there is
  no way to ask for a file, and inlining an entire bundle every turn is not a
  trade worth making, so the paths are listed and their unavailability is said
  out loud rather than left for the model to discover mid-task.

``load_skill`` and ``read_skill_file`` are engine-injected pseudo-tools, the
same device as ``final_answer``: bound to the request so the provider validates
the call shape, then intercepted by name before dispatch, since there is no MCP
server behind them. The same name-collision resolution applies, for the same
reason — a real MCP tool with either name would otherwise be silently shadowed.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

LOAD_SKILL_TOOL = "load_skill"
READ_SKILL_FILE_TOOL = "read_skill_file"
RUN_SKILL_SCRIPT_TOOL = "run_skill_script"

_INDEX_HEADER = """\
## Available skills

Each entry below is a skill: a set of instructions for a particular kind of \
task, which you can open when it applies. You are given only the summaries — \
call `{tool_name}` with a skill's name to read its full instructions before \
doing work that matches it. Do not guess at a skill's contents from its \
summary, and do not open one that does not apply.
"""

_INLINE_HEADER = """\
## Active skills

The instructions below are skills: procedures to follow for the kinds of task \
they describe. Apply a skill when the work matches what it covers; ignore it \
otherwise.
"""


class SkillSelection(BaseModel):
    """Skill names chosen from tier-1 metadata for a no-tool-loop run."""

    names: list[str] = Field(default_factory=list)


def _skill_names(skills: list[dict[str, Any]]) -> list[str]:
    return [str(s.get("name") or "") for s in skills if s.get("name")]


def _files_of(skill: dict[str, Any]) -> dict[str, dict[str, str]]:
    """One skill's bundled level-3 files as ``{path: contents}``.

    Tolerant of a missing or malformed ``files`` entry: ``RunContext.skills`` is
    a plain snapshot a product may have built itself, and a skill whose bundle
    cannot be read should still contribute its instructions.
    """
    files = skill.get("files")
    if not isinstance(files, dict):
        return {}
    normalized: dict[str, dict[str, str]] = {}
    for path, value in files.items():
        clean_path = str(path).replace("\\", "/").removeprefix("./")
        parts = clean_path.split("/")
        if not clean_path or clean_path.startswith("/") or any(part in {"", ".", ".."} for part in parts):
            continue
        if isinstance(value, dict):
            normalized[clean_path] = {
                "content": str(value.get("content") or ""),
                "encoding": str(value.get("encoding") or "utf-8"),
            }
        else:
            normalized[clean_path] = {"content": str(value), "encoding": "utf-8"}
    return normalized


def skill_script_paths(skills: list[dict[str, Any]]) -> list[str]:
    """Python resources that the runtime can execute on demand."""
    return [path for path in skill_file_paths(skills) if path.lower().endswith(".py")]


def skill_file_paths(skills: list[dict[str, Any]]) -> list[str]:
    """Every ``skill/path`` a :func:`build_read_skill_file_tool` call may name.

    Qualified by skill name, so two skills bundling a ``REFERENCE.md`` each stay
    distinguishable, and so the enum below is a single flat list rather than a
    pair of arguments the model has to get consistent with each other.
    """
    return [
        f"{str(skill.get('name') or '').strip()}/{path}"
        for skill in skills
        if str(skill.get("name") or "").strip()
        for path in _files_of(skill)
    ]


def render_skill_index(skills: list[dict[str, Any]], *, tool_name: str = LOAD_SKILL_TOOL) -> str:
    """Render the always-loaded metadata block (level 1).

    One line per skill, name and description only. This is what the ~100
    tokens-per-skill budget in the format's own accounting buys, and it is why
    a skill's description has to say *when* to use it and not just what it does
    — this block is the entire basis on which the model decides to open one.
    """
    if not skills:
        return ""
    lines = [_INDEX_HEADER.format(tool_name=tool_name)]
    for skill in skills:
        name = str(skill.get("name") or "").strip()
        if not name:
            continue
        description = " ".join(str(skill.get("description") or "").split())
        lines.append(f"- **{name}**: {description}")
    return "\n".join(lines)


def render_skill_body(skills: list[dict[str, Any]], name: str, *, file_tool_name: str = "") -> str:
    """Render one skill's full instructions (level 2), as a tool result.

    A name that matches nothing returns an error string rather than raising:
    the model is the caller here, and the useful response to it having invented
    a name is to say so and list the real ones, not to fail the run.

    When the skill bundles level-3 files and *file_tool_name* is bound, their
    paths are appended. The body will already link to them in prose ("see
    FORMS.md"), but a markdown link is not a callable thing — this is what turns
    the reference into an instruction the model can act on, and it is appended
    here rather than to the index because a path is only useful once the body
    that explains it is in context.
    """
    wanted = (name or "").strip().lower()
    for skill in skills:
        if str(skill.get("name") or "").strip().lower() == wanted:
            body = str(skill.get("body") or "").strip()
            title = str(skill.get("name") or "").strip()
            rendered = (
                f"# Skill: {title}\n\n{body}" if body else f"Skill '{title}' has no instructions beyond its summary."
            )
            compatibility = (skill.get("frontmatter") or {}).get("compatibility")
            if compatibility:
                rendered = f"{rendered}\n\nCompatibility: {compatibility}"
            paths = list(_files_of(skill))
            if paths and file_tool_name:
                listing = "\n".join(f"- {title}/{path}" for path in paths)
                rendered = (
                    f"{rendered}\n\n---\n\nThis skill bundles the following files. Read one with "
                    f"`{file_tool_name}` when the instructions above send you to it; do not guess "
                    f"at its contents.\n{listing}"
                )
            return rendered
    available = ", ".join(_skill_names(skills)) or "(none)"
    return f"No skill named '{name}'. Available skills: {available}."


def render_skill_file(skills: list[dict[str, Any]], path: str) -> str:
    """Render one bundled level-3 file, as a tool result.

    *path* is the qualified ``skill-name/relative/path`` form
    :func:`skill_file_paths` produces. Same failure posture as
    :func:`render_skill_body`: an unknown path answers with what is actually
    available rather than raising, because the caller is the model.
    """
    wanted = (path or "").strip().lower()
    while wanted.startswith("./"):
        wanted = wanted[2:]
    for skill in skills:
        title = str(skill.get("name") or "").strip()
        if not title:
            continue
        for candidate, resource in _files_of(skill).items():
            if f"{title}/{candidate}".lower() == wanted:
                text = resource["content"].strip()
                if not text:
                    return f"'{title}/{candidate}' is empty."
                if resource["encoding"] == "base64":
                    return (
                        f"# {title}/{candidate}\n\nBinary resource encoded as base64 ({len(text)} characters):\n{text}"
                    )
                return f"# {title}/{candidate}\n\n{text}"
    available = ", ".join(skill_file_paths(skills)) or "(none)"
    return f"No bundled file at '{path}'. Available files: {available}."


def inline_skills(system_prompt: str, skills: list[dict[str, Any]]) -> str:
    """Append every skill's full body to *system_prompt* (the no-tool-loop path).

    Used when the active pattern cannot offer ``load_skill`` — without a tool
    loop the model has no way to ask for a body, so the choice is inlining or
    the skill having no effect at all. Prefer the indexed path where it exists.

    Bundled level-3 files are named but not inlined. The body will send the
    model to them regardless ("see FORMS.md"), so silence would leave it either
    stuck or inventing contents; and inlining a whole bundle into a prompt that
    is resent every turn is a cost the fallback's "expensive but correct"
    bargain does not stretch to.
    """
    if not skills:
        return system_prompt
    parts = [_INLINE_HEADER]
    for skill in skills:
        name = str(skill.get("name") or "").strip()
        if not name:
            continue
        description = " ".join(str(skill.get("description") or "").split())
        body = str(skill.get("body") or "").strip()
        section = f"### Skill: {name}\n{description}"
        if body:
            section = f"{section}\n\n{body}"
        paths = list(_files_of(skill))
        if paths:
            listing = ", ".join(paths)
            section = (
                f"{section}\n\nThis skill bundles reference files ({listing}) that are NOT "
                "available in this run — work from the instructions above, and say so rather than "
                "guessing if they are not enough."
            )
        parts.append(section)
    block = "\n\n".join(parts)
    return f"{system_prompt}\n\n{block}" if system_prompt else block


async def select_relevant_skills(
    llm_service: Any,
    model_config: Any,
    user_input: str,
    skills: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], Any | None]:
    """Choose bodies to inline using only tier-1 metadata.

    Patterns without a tool loop cannot expose ``load_skill``. A small
    structured call preserves the important part of progressive disclosure:
    the model sees every name/description initially and only relevant bodies
    enter the main run. Failure is conservative and keeps every skill active.
    """
    if not skills or llm_service is None:
        return skills, None
    catalog = render_skill_index(skills, tool_name="select by exact name")
    try:
        selection, record = await llm_service.complete(
            config=model_config,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Select only the skills whose descriptions apply to the user's task. "
                        "Return exact names and an empty list when none applies. Do not infer "
                        "anything beyond the supplied metadata.\n\n" + catalog
                    ),
                },
                {"role": "user", "content": user_input},
            ],
            response_model=SkillSelection,
        )
    except Exception:
        return skills, None
    wanted = {name.casefold() for name in selection.names}
    selected = [skill for skill in skills if str(skill.get("name") or "").casefold() in wanted]
    return selected, record


def _unclaimed_name(preferred: str, bound_names: set[str]) -> str:
    name = preferred
    suffix = 0
    while name in bound_names:
        suffix += 1
        name = f"{preferred}_{suffix}"
    return name


def resolve_load_skill_name(bound_names: set[str]) -> str:
    """Pick a ``load_skill`` name no bound tool already claims.

    Same collision problem as ``final_answer``: this name is intercepted before
    dispatch, so a real MCP tool sharing it would never execute.
    """
    return _unclaimed_name(LOAD_SKILL_TOOL, bound_names)


def resolve_read_skill_file_name(bound_names: set[str]) -> str:
    """Pick a ``read_skill_file`` name no bound tool already claims.

    Callers must pass the already-resolved ``load_skill`` name in
    *bound_names* — the two pseudo-tools are bound in the same request and are
    just as capable of colliding with each other as with a real one.
    """
    return _unclaimed_name(READ_SKILL_FILE_TOOL, bound_names)


def resolve_run_skill_script_name(bound_names: set[str]) -> str:
    return _unclaimed_name(RUN_SKILL_SCRIPT_TOOL, bound_names)


async def run_skill_script(skills: list[dict[str, Any]], path: str, args: list[str] | None = None) -> str:
    """Execute one bundled Python script in a temporary materialized skill directory.

    This is process isolation, not a security sandbox. Products should expose
    uploaded skills only within their existing trust model.
    """
    wanted = (path or "").strip().lower().removeprefix("./")
    for skill in skills:
        title = str(skill.get("name") or "").strip()
        files = _files_of(skill)
        for candidate, resource in files.items():
            if f"{title}/{candidate}".lower() != wanted:
                continue
            if not candidate.lower().endswith(".py"):
                return f"'{path}' is not an executable Python resource."
            if resource["encoding"] != "utf-8":
                return f"'{path}' is binary and cannot be executed."
            with tempfile.TemporaryDirectory(prefix="motoro-skill-") as raw_root:
                root = Path(raw_root)
                for resource_path, item in files.items():
                    target = root / resource_path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if item["encoding"] == "base64":
                        target.write_bytes(base64.b64decode(item["content"]))
                    else:
                        target.write_text(item["content"], encoding="utf-8")
                try:
                    completed = await asyncio.to_thread(
                        subprocess.run,
                        [sys.executable, str(root / candidate), *(args or [])],
                        cwd=root,
                        env={key: value for key, value in os.environ.items() if key in {"PATH", "LANG", "LC_ALL"}},
                        capture_output=True,
                        text=True,
                        timeout=300,
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    return json.dumps({"exit_code": None, "stdout": "", "stderr": "Script timed out after 300s."})
                return json.dumps(
                    {
                        "exit_code": completed.returncode,
                        "stdout": completed.stdout[-8000:],
                        "stderr": completed.stderr[-4000:],
                    }
                )
    return f"No bundled script at '{path}'. Available scripts: {', '.join(skill_script_paths(skills)) or '(none)'}"


def build_load_skill_tool(skills: list[dict[str, Any]], name: str = LOAD_SKILL_TOOL) -> dict[str, Any]:
    """The pseudo-tool that opens a skill.

    The available names are bound as an ``enum`` rather than described in prose,
    so a hallucinated skill name is rejected by the provider's own schema
    validation instead of costing a round trip to find out.
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": (
                "Read the full instructions for one of the available skills. Call this "
                "before starting work the skill covers, then follow what it says."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": _skill_names(skills),
                        "description": "The name of the skill to open.",
                    }
                },
                "required": ["name"],
            },
        },
    }


def build_read_skill_file_tool(skills: list[dict[str, Any]], name: str = READ_SKILL_FILE_TOOL) -> dict[str, Any]:
    """The pseudo-tool that opens one of a skill's bundled files (level 3).

    Bound only when some skill actually has files — see
    :func:`skill_file_paths`, whose emptiness is the caller's signal to skip
    this entirely. A tool whose ``enum`` would be empty is worse than no tool:
    some providers reject the schema outright, and the ones that don't have
    just been handed an unusable affordance.
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": (
                "Read one of the reference files bundled with a skill, named as "
                "'<skill-name>/<path>'. Call this when a skill's instructions send you to one "
                "of its files. These are documents, not programs -- there is nothing to run."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "enum": skill_file_paths(skills),
                        "description": "The skill-qualified path of the file to read.",
                    }
                },
                "required": ["path"],
            },
        },
    }


def build_run_skill_script_tool(skills: list[dict[str, Any]], name: str = RUN_SKILL_SCRIPT_TOOL) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": (
                "Run a Python script bundled with an activated skill and return its stdout, stderr, and exit code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "enum": skill_script_paths(skills)},
                    "args": {"type": "array", "items": {"type": "string"}, "default": []},
                },
                "required": ["path"],
            },
        },
    }
