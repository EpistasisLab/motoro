"""Boundary invariants for the migrated core.

These are cheap and they guard the two things a mechanical migration gets wrong:
importing a module that was never pulled, and leaving a product's identity baked
into core. Both failed at least once while step 0 was being assembled, which is
why they are tests rather than a checklist.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
import re
from pathlib import Path

import pytest

import agentic_core

SRC = Path(agentic_core.__file__).parent

STEP_0 = [
    "config",
    "models.base",
    "observability.metrics",
    "observability.tracing",
    "schemas.llm",
    "schemas.pattern",
    "security.prompt_injection",
    "services.credential_scrubber",
    "services.llm_errors",
    "services.model_capabilities",
    "services.retry",
]


@pytest.mark.parametrize("mod", STEP_0)
def test_module_imports(mod: str) -> None:
    """Every migrated module imports standalone.

    An AST scan of third-party imports is not enough on its own: the ARES
    ``schemas/user.py`` passed such a scan but failed here, because ``EmailStr``
    pulls in ``email-validator`` at class-construction time rather than via an
    import statement. Only actually importing the module finds that.
    """
    importlib.import_module(f"agentic_core.{mod}")


def _iter_source() -> list[tuple[Path, str]]:
    return [(p, p.read_text(encoding="utf-8")) for p in sorted(SRC.rglob("*.py"))]


def _docstring_lines(text: str) -> set[int]:
    out: set[int] = set()
    tree = ast.parse(text)
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


def test_no_product_references_in_code() -> None:
    """No module names a product in code.

    Docstrings are exempt — prose explaining the split legitimately mentions
    ARES. String defaults, identifiers and instrument names are not: an
    ``"ares-backend"`` default or an ``ares_`` metric name survives an import
    rewrite untouched and quietly makes core impersonate a product.
    """
    offenders: list[str] = []
    for path, text in _iter_source():
        skip = _docstring_lines(text)
        for i, line in enumerate(text.splitlines(), start=1):
            if i in skip or "ares-ok" in line:
                continue
            if re.search(r"\bares\b|\bares_|\becoxai\b", line):
                offenders.append(f"{path.relative_to(SRC)}:{i}: {line.strip()[:100]}")
    assert not offenders, "product references in core:\n" + "\n".join(offenders)


def test_no_product_imports() -> None:
    """Core never imports a product package. The contract, as a test.

    ``lint-imports`` enforces this too; duplicating it here means a plain
    ``pytest`` run catches it without the extra tool.
    """
    offenders: list[str] = []
    for path, text in _iter_source():
        for node in ast.walk(ast.parse(text)):
            names: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            for name in names:
                if name.split(".")[0] in {"ares", "ecoxai"}:
                    offenders.append(f"{path.relative_to(SRC)}:{node.lineno}: imports {name}")
    assert not offenders, "core imports a product:\n" + "\n".join(offenders)


def test_every_internal_import_resolves() -> None:
    """No module imports a sibling that has not been pulled yet.

    This is what keeps a slice self-contained: a step is done when nothing in it
    reaches for a module that is still only in ARES.
    """
    present = {
        name
        for _, name, _ in pkgutil.walk_packages([str(SRC)], prefix="agentic_core.")
    } | {"agentic_core"}
    offenders: list[str] = []
    for path, text in _iter_source():
        for node in ast.walk(ast.parse(text)):
            targets: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("agentic_core"):
                targets = [node.module]
            elif isinstance(node, ast.Import):
                targets = [a.name for a in node.names if a.name.startswith("agentic_core")]
            for t in targets:
                if t not in present:
                    offenders.append(f"{path.relative_to(SRC)}:{node.lineno}: {t}")
    assert not offenders, "unresolved internal imports:\n" + "\n".join(offenders)


def test_defaults_do_not_claim_a_product_identity() -> None:
    """An unconfigured core identifies as itself, not as a product."""
    from agentic_core import CoreSettings

    s = CoreSettings()
    assert s.otel_service_name == "agentic-core"
    assert s.metrics_prefix == "agentic_core"
    assert "ares" not in s.database_url


def test_configure_rejects_late_reconfiguration() -> None:
    """Installing settings after a read has happened is an error, not a no-op.

    Half a process on defaults and half on the product's values is far harder to
    diagnose than a startup failure.
    """
    from agentic_core.config import CoreSettings, configure, get_settings, reset_for_testing

    reset_for_testing()
    try:
        get_settings()  # materializes defaults, marks settings as read
        with pytest.raises(RuntimeError, match="already read"):
            configure(CoreSettings(otel_service_name="late"))
    finally:
        reset_for_testing()
