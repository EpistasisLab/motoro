"""Rendering Agent Skills into a run — the two disclosure levels.

The Agent Skills format is built around progressive disclosure: a skill's
``name`` and ``description`` are always in context (cheap — a line or two per
skill), and its body is loaded only when the agent judges the skill relevant.
That is what makes a dozen skills affordable where a dozen inlined instruction
documents would not be.

Two renderings, for two situations:

- :func:`render_skill_index` + :func:`build_load_skill_tool` — the real thing.
  The index goes in the stable system prefix; the body arrives as a tool result
  when the model asks for it. Requires a pattern with a tool loop.
- :func:`inline_skills` — the fallback for a pattern that has no tool loop
  (``single_agent_baseline``, or any pattern that has not opted in). Every body
  is inlined up front. Honest but expensive; the disclosure is lost, not the
  content.

``load_skill`` is an engine-injected pseudo-tool, the same device as
``final_answer``: bound to the request so the provider validates the call
shape, then intercepted by name before dispatch, since there is no MCP server
behind it. The same name-collision resolution applies, for the same reason — a
real MCP tool called ``load_skill`` would otherwise be silently shadowed.
"""

from __future__ import annotations

from typing import Any

LOAD_SKILL_TOOL = "load_skill"

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


def _skill_names(skills: list[dict[str, Any]]) -> list[str]:
    return [str(s.get("name") or "") for s in skills if s.get("name")]


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


def render_skill_body(skills: list[dict[str, Any]], name: str) -> str:
    """Render one skill's full instructions (level 2), as a tool result.

    A name that matches nothing returns an error string rather than raising:
    the model is the caller here, and the useful response to it having invented
    a name is to say so and list the real ones, not to fail the run.
    """
    wanted = (name or "").strip().lower()
    for skill in skills:
        if str(skill.get("name") or "").strip().lower() == wanted:
            body = str(skill.get("body") or "").strip()
            title = str(skill.get("name") or "").strip()
            if not body:
                return f"Skill '{title}' has no instructions beyond its summary."
            return f"# Skill: {title}\n\n{body}"
    available = ", ".join(_skill_names(skills)) or "(none)"
    return f"No skill named '{name}'. Available skills: {available}."


def inline_skills(system_prompt: str, skills: list[dict[str, Any]]) -> str:
    """Append every skill's full body to *system_prompt* (the no-tool-loop path).

    Used when the active pattern cannot offer ``load_skill`` — without a tool
    loop the model has no way to ask for a body, so the choice is inlining or
    the skill having no effect at all. Prefer the indexed path where it exists.
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
        parts.append(section)
    block = "\n\n".join(parts)
    return f"{system_prompt}\n\n{block}" if system_prompt else block


def resolve_load_skill_name(bound_names: set[str]) -> str:
    """Pick a ``load_skill`` name no bound tool already claims.

    Same collision problem as ``final_answer``: this name is intercepted before
    dispatch, so a real MCP tool sharing it would never execute.
    """
    name = LOAD_SKILL_TOOL
    suffix = 0
    while name in bound_names:
        suffix += 1
        name = f"{LOAD_SKILL_TOOL}_{suffix}"
    return name


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
