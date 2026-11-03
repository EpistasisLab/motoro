"""Select full CI from the complete tested diff, using only the standard library."""

import json
import os
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any


def is_documentation(path: str) -> bool:
    file = PurePosixPath(path)
    if len(file.parts) == 1:
        return file.suffix.lower() == ".md"
    return file.parts[0] == "docs" and file.suffix.lower() in {
        ".md",
        ".rst",
        ".txt",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".webp",
        ".pdf",
        ".ico",
    }


def requires_validation(event_name: str, event: dict[str, Any], head: str) -> bool:
    if event_name == "pull_request":
        base = event.get("pull_request", {}).get("base", {}).get("sha")
    elif event_name == "push":
        base = event.get("before")
    else:
        return True  # Manual runs and unfamiliar event types always validate.
    if not base or not head or set(base) == {"0"}:
        return True
    try:
        # Include every commit in the event and both sides of renamed files.
        # GITHUB_SHA is the tested PR merge commit, not just its source head.
        changed = subprocess.check_output(
            [
                "git",
                "diff",
                "--name-only",
                "--no-renames",
                "-z",
                base,
                head,
                "--",
            ]
        )
    except subprocess.CalledProcessError:
        return True  # An unavailable comparison must never skip validation.
    paths = changed.decode("utf-8", errors="surrogateescape").split("\0")
    return any(not is_documentation(path) for path in paths if path)


def main() -> None:
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    full = requires_validation(os.environ["GITHUB_EVENT_NAME"], event, os.environ.get("GITHUB_SHA", ""))
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        output.write(f"run_validation={str(full).lower()}\n")
    print("Full validation required." if full else "Documentation-only or empty diff: full validation skipped.")


if __name__ == "__main__":
    main()
