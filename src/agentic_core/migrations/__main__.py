"""Command-line entry point for core's schema — a deploy step, not app startup.

    python -m agentic_core.migrations upgrade  --url postgresql+asyncpg://...
    python -m agentic_core.migrations current  --url ...
    python -m agentic_core.migrations stamp    --url ...      # adopt an existing schema
    python -m agentic_core.migrations downgrade --url ... --revision base

Applying migrations belongs in a deploy step — an init container, a CI job, a
release task — that runs **once**, before the application starts. It does not
belong in application startup:

* Every replica would race the same migration on a rolling deploy, and for a
  window old and new code run against a half-migrated schema.
* An API process and a worker process would both try to migrate.
* A failure at migrate time becomes a crash-looping app instead of a failed
  deploy you can read and roll back.

So core exposes this as a CLI rather than doing it for you. An application should
*verify* the schema is current and refuse to start if it is not — never silently
migrate. ``current`` is the check.

``--url`` is explicit because core cannot know a product's settings prefix: a
product reading ``ARES_DATABASE_URL`` or ``AGENTIC_DATABASE_URL`` passes it
through (``--url "$AGENTIC_DATABASE_URL"``). With no flag, ``DATABASE_URL`` from
the environment or ``.env`` is used.
"""

from __future__ import annotations

import argparse
import asyncio
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m agentic_core.migrations")
    ap.add_argument("command", choices=["upgrade", "downgrade", "current", "stamp"])
    ap.add_argument(
        "--url",
        default=None,
        help="Database URL. Defaults to DATABASE_URL from the environment or .env.",
    )
    ap.add_argument(
        "--revision",
        default=None,
        help="Target revision. Defaults: head for upgrade/stamp, -1 for downgrade.",
    )
    args = ap.parse_args(argv)

    from agentic_core.config import CoreSettings, configure
    from agentic_core.migrations import current_revision, downgrade, stamp, upgrade

    # A bare CoreSettings reads DATABASE_URL (no product prefix) plus .env.
    configure(CoreSettings())
    url = args.url  # None => make_config falls back to settings.database_url

    if args.command == "current":
        rev = asyncio.run(current_revision(url))
        print(rev or "not migrated")
        return 0

    if args.command == "upgrade":
        upgrade(url, args.revision or "head")
    elif args.command == "downgrade":
        # No default to "base": a downgrade that drops every table should be
        # asked for explicitly, not reachable by omitting a flag.
        upgrade_target = args.revision or "-1"
        downgrade(url, upgrade_target)
    elif args.command == "stamp":
        stamp(url, args.revision or "head")

    rev = asyncio.run(current_revision(url))
    print(f"{args.command} -> {rev or 'not migrated'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
