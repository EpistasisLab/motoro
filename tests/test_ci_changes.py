"""Exercise CI classification against real Git diffs and its required-status gate."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ci_changes.py"


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        [
            "git",
            "-c",
            "user.name=CI test",
            "-c",
            "user.email=ci@example.test",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "commit.gpgSign=false",
            *args,
        ],
        cwd=repo,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    (repo / "README.md").write_text("Initial docs\n")
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("VALUE = 1\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "Initial tree")
    return repo


def commit(repo: Path, path: str) -> str:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("Changed content\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "Change a file")
    return git(repo, "rev-parse", "HEAD")


def classify(repo: Path, event_name: str, base: str | None, head: str) -> bool:
    event = {"before": base, "pull_request": {"base": {"sha": base}}}
    event_file = repo.parent / "event.json"
    output = repo.parent / "output.txt"
    event_file.write_text(json.dumps(event))
    output.write_text("")
    subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=repo,
        check=True,
        capture_output=True,
        env={
            **os.environ,
            "GITHUB_EVENT_NAME": event_name,
            "GITHUB_EVENT_PATH": str(event_file),
            "GITHUB_SHA": head,
            "GITHUB_OUTPUT": str(output),
        },
    )
    assert output.read_text() in {"run_validation=true\n", "run_validation=false\n"}
    return output.read_text() == "run_validation=true\n"


@pytest.mark.parametrize("event", ["push", "pull_request"])
@pytest.mark.parametrize(
    "path,full",
    [
        ("README.md", False),
        ("CONTRIBUTING.md", False),
        ("CLAUDE.md", False),
        ("docs/guide.md", False),
        ("docs/images/example.svg", False),
        ("src/app.py", True),
        ("tests/fixtures/prompt.md", True),
        ("pyproject.toml", True),
        ("uv.lock", True),
        (".github/workflows/ci.yml", True),
        ("scripts/build.sh", True),
        ("Dockerfile", True),
        ("docs/build.py", True),
        ("new.unknown", True),
    ],
)
def test_changed_files_select_full_validation(repo: Path, event: str, path: str, full: bool) -> None:
    base = git(repo, "rev-parse", "HEAD")
    assert classify(repo, event, base, commit(repo, path)) is full


@pytest.mark.parametrize("event", ["push", "pull_request"])
def test_docs_at_head_cannot_hide_code_in_earlier_commit(repo: Path, event: str) -> None:
    base = git(repo, "rev-parse", "HEAD")
    commit(repo, "src/app.py")
    head = commit(repo, "README.md")
    assert classify(repo, event, base, head)


@pytest.mark.parametrize("event", ["push", "pull_request"])
def test_all_documentation_commits_skip_full_validation(repo: Path, event: str) -> None:
    base = git(repo, "rev-parse", "HEAD")
    commit(repo, "README.md")
    head = commit(repo, "docs/guide.md")
    assert not classify(repo, event, base, head)


@pytest.mark.parametrize("path,full", [("src/app.py", True), ("README.md", False)])
def test_deletions_are_classified(repo: Path, path: str, full: bool) -> None:
    base = git(repo, "rev-parse", "HEAD")
    git(repo, "rm", path)
    git(repo, "commit", "-m", "Delete a file")
    assert classify(repo, "push", base, git(repo, "rev-parse", "HEAD")) is full


def test_renaming_code_into_docs_still_requires_validation(repo: Path) -> None:
    base = git(repo, "rev-parse", "HEAD")
    git(repo, "mv", "src/app.py", "example.md")
    git(repo, "commit", "-m", "Rename code into documentation")
    assert classify(repo, "push", base, git(repo, "rev-parse", "HEAD"))


def test_pr_uses_tested_merge_tree_instead_of_source_head(repo: Path) -> None:
    git(repo, "checkout", "-b", "docs")
    source = commit(repo, "docs/guide.md")
    git(repo, "checkout", "main")
    target = commit(repo, "src/app.py")
    git(repo, "merge", "--no-ff", "docs", "-m", "Test PR merge")
    merged = git(repo, "rev-parse", "HEAD")
    # Comparing target to the source head would mistake target-only code for
    # a deletion. The tested merge correctly contains only documentation changes.
    assert classify(repo, "pull_request", target, source)
    assert not classify(repo, "pull_request", target, merged)
    assert not classify(repo, "push", target, merged)


@pytest.mark.parametrize(
    "event,base",
    [
        ("push", None),
        ("push", "0" * 40),
        ("push", "unavailable-commit"),
        ("workflow_dispatch", None),
        ("unfamiliar_event", None),
    ],
)
def test_manual_or_unavailable_comparison_runs_validation(repo: Path, event: str, base: str | None) -> None:
    assert classify(repo, event, base, git(repo, "rev-parse", "HEAD"))


def test_empty_diff_skips_full_validation(repo: Path) -> None:
    head = git(repo, "rev-parse", "HEAD")
    assert not classify(repo, "push", head, head)


@pytest.mark.parametrize(
    "detection,required,validation,passes",
    [
        ("success", "false", "skipped", True),
        ("success", "true", "success", True),
        ("failure", "false", "skipped", False),
        ("cancelled", "false", "skipped", False),
        ("success", "true", "failure", False),
        ("success", "true", "skipped", False),
        ("success", "true", "cancelled", False),
        ("success", "", "skipped", False),
    ],
)
def test_required_check_cannot_pass_without_required_validation(
    detection: str,
    required: str,
    validation: str,
    passes: bool,
) -> None:
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())
    check = workflow["jobs"]["test"]
    assert check["name"] == "Lint, Type Check & Test"
    assert check["if"] == "always()"
    assert set(check["needs"]) == {"changes", "validate"}
    result = subprocess.run(
        ["bash", "-c", check["steps"][0]["run"]],
        capture_output=True,
        env={
            **os.environ,
            "DETECTION_RESULT": detection,
            "RUN_VALIDATION": required,
            "VALIDATION_RESULT": validation,
        },
    )
    assert (result.returncode == 0) is passes
