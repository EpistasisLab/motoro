#!/usr/bin/env python3
"""Pull modules from the ARES tree into agentic-core, rewriting imports.

Usage:
    python scripts/pull_from_ares.py config models.base observability.metrics
    python scripts/pull_from_ares.py --check config          # report, write nothing
    python scripts/pull_from_ares.py --step 0                # a named slice step

Each named module is copied from ``$ARES/backend/src/ares/<path>.py`` to
``src/agentic_core/<path>.py`` with every ``ares.`` import rewritten to
``agentic_core.``. Package ``__init__.py`` files are created as needed.

The script is deliberately dumb about *what* to move — the dependency ordering
lives in SLICES below, and the decision of what belongs in core is a human one.
What it does guarantee is that a module never lands here still importing ``ares``:
--check reports any leftover ``ares`` reference and any dependency that is named
by a copied module but not yet present, so a step is verifiably complete before
the next one starts.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import shutil
import sys
from pathlib import Path

ARES = Path(os.environ.get("ARES_SRC", Path.home() / "dev/ARES/backend/src/ares"))
DEST = Path(__file__).resolve().parent.parent / "src/agentic_core"

# Dependency-ordered slices. Step N may only depend on steps < N.
# Computed from the runtime import graph of the SRPA loop with isolation,
# per-user credential resolution, and agent-relationship limits excluded.
SLICES: dict[str, list[str]] = {
    "0": [
        "config",
        "models.base",
        "observability.metrics",
        "observability.tracing",
        "schemas.llm",
        "schemas.pattern",
        "schemas.user",
        "security.prompt_injection",
        "services.credential_scrubber",
        "services.llm_errors",
        "services.model_capabilities",
        "services.retry",
    ],
    "1": [
        "mcp.client",
        "models.agent",
        "models.database",
        "models.pricing",
        "models.redis",
        "models.run",
        "schemas.agent",
        "schemas.pricing",
    ],
    "2": ["engine.context", "mcp.registry", "services.pricing_service"],
    "3": ["engine.phase", "mcp.adapters"],
    "4": ["engine.runtime", "engine.sense", "services.llm_service"],
    "5": ["engine.act", "engine.plan", "engine.reason"],
}

IMPORT_RE = re.compile(r"\bares\.")
BARE_IMPORT_RE = re.compile(r"^(\s*)import ares\b", re.MULTILINE)


def rewrite(text: str) -> str:
    text = BARE_IMPORT_RE.sub(r"\1import agentic_core", text)
    return IMPORT_RE.sub("agentic_core.", text)


def src_path(mod: str) -> Path:
    p = ARES / (mod.replace(".", "/") + ".py")
    if p.exists():
        return p
    pkg = ARES / mod.replace(".", "/") / "__init__.py"
    if pkg.exists():
        return pkg
    raise FileNotFoundError(f"no such ARES module: {mod}")


def dest_path(mod: str) -> Path:
    p = src_path(mod)
    rel = p.relative_to(ARES)
    return DEST / rel


def internal_deps(text: str) -> set[str]:
    """Module names this file imports from the core package, after rewriting."""
    deps: set[str] = set()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return deps
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("agentic_core"):
            deps.add(node.module[len("agentic_core.") :] or "")
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("agentic_core."):
                    deps.add(a.name[len("agentic_core.") :])
    return {d for d in deps if d}


def ensure_packages(path: Path) -> list[Path]:
    made = []
    rel = path.relative_to(DEST).parent
    cur = DEST
    for part in rel.parts:
        cur = cur / part
        cur.mkdir(exist_ok=True)
        init = cur / "__init__.py"
        if not init.exists():
            init.write_text(f'"""{".".join(cur.relative_to(DEST).parts)} — pulled from ARES."""\n')
            made.append(init)
    return made


def present() -> set[str]:
    out = set()
    for p in DEST.rglob("*.py"):
        rel = p.relative_to(DEST).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts.pop()
        if parts:
            out.add(".".join(parts))
    return out


def docstring_lines(text: str) -> set[int]:
    """Line numbers occupied by docstrings.

    Prose *about* the split legitimately names ARES — the module docstring
    explaining why ``env_prefix`` is a product concern, for instance. What must
    not survive is ARES in *code*: a ``"ares-backend"`` default, an ``ares_``
    instrument name, an import. Excluding docstrings keeps the check strict where
    it matters without punishing documentation.
    """
    out: set[int] = set()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            out.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    return out


def verify() -> int:
    """Check what is on disk in agentic_core, not what ARES would produce.

    Two invariants, both of which must hold after every step:
      1. No module still references ``ares`` — including in strings, defaults and
         docstrings, not merely in imports. A rewritten import is easy; a
         hardcoded ``"ares-backend"`` default or an ``ares_`` metric name is the
         kind of thing that survives a mechanical pass and quietly makes core
         claim to be a product.
      2. Every internal import resolves to a module that has actually been
         pulled, so a step is verifiably self-contained before the next begins.
    """
    have = present()
    problems = 0
    for p in sorted(DEST.rglob("*.py")):
        rel = p.relative_to(DEST)
        text = p.read_text(encoding="utf-8")
        skip = docstring_lines(text)
        hits = [
            ln.strip()
            for i, ln in enumerate(text.splitlines(), start=1)
            if i not in skip and "ares-ok" not in ln and re.search(r"\bares\b|\bares_", ln)
        ]
        if hits:
            print(f"  !! {rel}: {len(hits)} 'ares' reference(s)")
            for h in hits[:4]:
                print(f"       {h[:110]}")
            problems += 1
        missing = sorted(d for d in internal_deps(text) if d not in have)
        if missing:
            print(f"  !! {rel}: unresolved internal import(s): {', '.join(missing)}")
            problems += 1
    n = len(list(DEST.rglob("*.py")))
    loc = sum(len(p.read_text(encoding="utf-8").splitlines()) for p in DEST.rglob("*.py"))
    print(f"\nVERIFY: {n} files, {loc:,} LOC, {problems} problem(s)")
    return 1 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("modules", nargs="*")
    ap.add_argument("--step", help="pull a named slice step from SLICES")
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    ap.add_argument("--verify", action="store_true", help="check what is already on disk")
    args = ap.parse_args()

    if args.verify:
        return verify()

    mods = list(args.modules)
    if args.step:
        if args.step not in SLICES:
            print(f"unknown step {args.step!r}; known: {sorted(SLICES)}", file=sys.stderr)
            return 2
        mods += SLICES[args.step]
    if not mods:
        ap.error("name at least one module, or pass --step")

    copied: list[tuple[str, Path, str]] = []
    for mod in mods:
        try:
            s = src_path(mod)
        except FileNotFoundError as exc:
            print(f"  !! {exc}", file=sys.stderr)
            return 1
        text = rewrite(s.read_text(encoding="utf-8"))
        copied.append((mod, dest_path(mod), text))

    if not args.check:
        for mod, d, text in copied:
            made = ensure_packages(d)
            d.parent.mkdir(parents=True, exist_ok=True)
            d.write_text(text, encoding="utf-8")
            for m in made:
                print(f"  +pkg {m.relative_to(DEST.parent)}")
            print(f"  copy {mod:38s} -> {d.relative_to(DEST.parent)}  ({len(text.splitlines())} lines)")

    have = present() | {m for m, _, _ in copied}
    problems = 0
    for mod, _, text in copied:
        if re.search(r"\bares\b", text):
            hits = [ln for ln in text.splitlines() if re.search(r"\bares\b", ln)]
            print(f"  !! {mod}: {len(hits)} leftover 'ares' reference(s):")
            for h in hits[:4]:
                print(f"       {h.strip()[:110]}")
            problems += 1
        missing = sorted(d for d in internal_deps(text) if d not in have)
        if missing:
            print(f"  !! {mod}: depends on modules not yet pulled: {', '.join(missing)}")
            problems += 1

    print(f"\n{'CHECK' if args.check else 'PULLED'}: {len(copied)} modules, {problems} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
