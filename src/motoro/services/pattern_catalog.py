"""Project the pattern registry into the ``architectural_patterns`` table.

Core does not read this table. Validation and parameter defaults come from the
registry in-process (:mod:`motoro.engine.patterns.catalog`), which is what
keeps core working against an empty one. The table exists for *products*: a UI
listing the available patterns, an advisor ingesting their descriptions into a
knowledge base, an experiment designer picking factors from the implemented set.
With core owning its own database, a product reaches it through core's API rather
than a join, so this is the projection those calls read.

The direction is one-way and deliberate. The plugin class is the source of truth;
this writes what the plugins say. Nothing reads the table back into the runtime,
so a hand-edited row cannot change how a pattern behaves — it will simply be
overwritten by the next sync.

Run it as a deploy step, next to the migration::

    python -m motoro.migrations sync-catalog --url "$AGENTIC_DATABASE_URL"

``is_implemented`` is not stored data here but a derived fact: a pattern is
implemented when its plugin class is discoverable. ARES maintains that column by
hand, with nothing reconciling it against which plugins exist, so its catalog can
claim a pattern is implemented when no code implements it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from motoro.engine.patterns.catalog import display_name_for
from motoro.engine.patterns.registry import PluginRegistry

if TYPE_CHECKING:
    from collections.abc import Sequence

    from motoro.models.pattern import ArchitecturalPattern

log = structlog.get_logger()


def catalog_rows() -> list[dict[str, Any]]:
    """Build one catalog row per registered plugin, from the plugin classes."""
    PluginRegistry.discover()
    rows: list[dict[str, Any]] = []
    for slug, plugin_cls in sorted(PluginRegistry.all().items()):
        rows.append(
            {
                "slug": slug,
                "name": display_name_for(plugin_cls),
                "category": plugin_cls.category,
                "description": plugin_cls.description,
                "phase": plugin_cls.complexity_phase,
                "configuration_schema": plugin_cls.configuration_schema,
                "requires_multi_agent": plugin_cls.requires_multi_agent,
                "dependencies": list(plugin_cls.dependencies),
                "version": plugin_cls.version,
                # Derived, not declared: the class exists, so the pattern exists.
                "is_implemented": True,
            }
        )
    return rows


async def sync_pattern_catalog() -> dict[str, int]:
    """Upsert a catalog row for every registered pattern.

    Returns counts of ``{"inserted", "updated", "stale"}``. Idempotent: running it
    twice changes nothing the second time.

    Rows for patterns that are *not* registered are left alone rather than
    deleted, and reported as ``stale``. Deleting them would be wrong here — core
    ships 2 of 37 patterns today, so "not registered in this process" does not
    mean "does not exist"; a product could equally be running a core build with a
    subset imported. Products that want the strict view filter on
    ``is_implemented``.
    """
    from sqlalchemy import select
    from sqlalchemy.dialects.postgresql import insert

    from motoro.models.database import system_session
    from motoro.models.pattern import ArchitecturalPattern

    rows = catalog_rows()
    if not rows:
        log.warning("pattern_catalog.sync.no_patterns_registered")
        return {"inserted": 0, "updated": 0, "stale": 0}

    slugs = [r["slug"] for r in rows]
    async with system_session(reason="sync_pattern_catalog") as db:
        existing = set(
            (await db.execute(select(ArchitecturalPattern.slug).where(ArchitecturalPattern.slug.in_(slugs))))
            .scalars()
            .all()
        )

        stmt = insert(ArchitecturalPattern).values(rows)
        # ON CONFLICT DO UPDATE, not DO NOTHING: this is a projection, so an
        # existing row whose description or schema has changed in code must be
        # corrected. DO NOTHING would freeze the catalog at whatever the first
        # sync wrote — which is how ARES ended up needing migrations 0009, 0010
        # and 0025 to amend seeded rows after the fact.
        stmt = stmt.on_conflict_do_update(
            index_elements=[ArchitecturalPattern.slug],
            set_={
                "name": stmt.excluded.name,
                "category": stmt.excluded.category,
                "description": stmt.excluded.description,
                "phase": stmt.excluded.phase,
                "configuration_schema": stmt.excluded.configuration_schema,
                "requires_multi_agent": stmt.excluded.requires_multi_agent,
                "dependencies": stmt.excluded.dependencies,
                "version": stmt.excluded.version,
                "is_implemented": stmt.excluded.is_implemented,
            },
        )
        await db.execute(stmt)

        all_slugs = set((await db.execute(select(ArchitecturalPattern.slug))).scalars().all())
        await db.commit()

    result = {
        "inserted": len(set(slugs) - existing),
        "updated": len(existing),
        "stale": len(all_slugs - set(slugs)),
    }
    log.info("pattern_catalog.synced", **result)
    return result


async def list_catalog(*, implemented_only: bool = False) -> Sequence[ArchitecturalPattern]:
    """Read the catalog back. The one core API a product needs for this table.

    Exists so a product never opens a session against core's database to render a
    patterns page. Returns ORM objects, usable after core closes its session
    because the sessionmaker sets ``expire_on_commit=False``.
    """
    from sqlalchemy import select

    from motoro.models.database import system_session
    from motoro.models.pattern import ArchitecturalPattern

    stmt = select(ArchitecturalPattern).order_by(ArchitecturalPattern.slug)
    if implemented_only:
        stmt = stmt.where(ArchitecturalPattern.is_implemented.is_(True))
    async with system_session(reason="list_catalog") as db:
        return (await db.execute(stmt)).scalars().all()
