"""Command-line entry point for core's schema — a deploy step, not app startup.

    python -m motoro.migrations upgrade  --url postgresql+asyncpg://...
    python -m motoro.migrations current  --url ...
    python -m motoro.migrations stamp    --url ...      # adopt an existing schema
    python -m motoro.migrations downgrade --url ... --revision base

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
product reading ``ARES_DATABASE_URL`` or ``MOTORO_DATABASE_URL`` passes it
through (``--url "$MOTORO_DATABASE_URL"``). With no flag, ``DATABASE_URL`` from
the environment or ``.env`` is used.
"""

from __future__ import annotations

import argparse
import asyncio
import sys


def _sync_catalog() -> str:
    """Project the pattern registry into ``architectural_patterns``.

    Unlike the Alembic commands, this cannot take a URL as an argument: it goes
    through the service, which manages its own sessions off the configured
    settings. ``main`` therefore installs ``--url`` into the settings up front
    rather than mutating them here.
    """
    from motoro.services.pattern_catalog import sync_pattern_catalog

    counts = asyncio.run(sync_pattern_catalog())
    return (
        f"sync-catalog -> {counts['inserted']} inserted, "
        f"{counts['updated']} updated, {counts['stale']} unregistered rows left alone"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m motoro.migrations")
    ap.add_argument(
        "command",
        choices=["deploy", "upgrade", "downgrade", "current", "stamp", "sync-catalog"],
        help=(
            "deploy = upgrade then sync-catalog, the whole 'prepare core's database' step. "
            "sync-catalog projects the pattern registry into architectural_patterns; it is data "
            "rather than schema, but it belongs to the same deploy step, must run after the "
            "schema exists, and is idempotent."
        ),
    )
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

    from motoro.config import CoreSettings, configure
    from motoro.migrations import current_revision, downgrade, stamp, upgrade

    # A bare CoreSettings reads DATABASE_URL (no product prefix) plus .env.
    # `--url` is installed here, before anything reads settings, so the two paths
    # agree: Alembic takes the URL as an argument, while sync-catalog goes through
    # a service that opens its own sessions off the settings.
    url = args.url  # None => make_config falls back to settings.database_url
    configure(CoreSettings(database_url=url) if url else CoreSettings())

    if args.command == "current":
        rev = asyncio.run(current_revision(url))
        print(rev or "not migrated")
        return 0

    if args.command == "sync-catalog":
        print(_sync_catalog())
        return 0

    if args.command == "deploy":
        upgrade(url, "head")
        rev = asyncio.run(current_revision(url))
        print(f"upgrade -> {rev or 'not migrated'}")
        print(_sync_catalog())
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
