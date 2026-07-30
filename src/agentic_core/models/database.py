"""Async database engine and session management.

Two named entry points, so a caller has to say up front what its session is for:

* ``get_db`` — an async-generator dependency, suitable for
  ``Depends(get_db)`` in a product's FastAPI app. Core defines it but serves
  nothing itself.
* ``system_session(reason=...)`` — for the engine and worker, where there is no
  request to tie a session to. The mandatory *reason* is what makes the session
  auditable rather than anonymous.

``_get_session_factory`` is private on purpose: it forces every caller to pick
one of the named entry points rather than opening an unrestricted session.

The ARES original has a third entry point, ``scoped_session(viewer)``, which
attaches a ``Viewer`` to ``session.info`` for its per-user read guard. That is
left behind with the rest of the isolation subsystem — whether core should own
authorization at all is undecided. A product adding it back layers a
``scoped_session`` of its own over ``_get_session_factory``.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from agentic_core.config import settings

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Get or create the async database engine (lazy singleton)."""
    global _engine  # noqa: PLW0603
    if _engine is None:
        _engine = create_async_engine(
            settings.database_url,
            echo=settings.debug,
            pool_pre_ping=True,
            pool_recycle=settings.db_pool_recycle_seconds,
        )
    return _engine


async def dispose_engine() -> None:
    """Close the engine's connection pool and drop the cached singleton.

    A product calls this on shutdown so pooled connections are closed rather than
    reclaimed by the OS. Tests need it for a second reason: the engine is cached
    across calls but an asyncpg connection belongs to the event loop that opened
    it, so a suite giving each test a fresh loop must dispose between tests or the
    next one inherits connections attached to a dead loop.
    """
    global _engine, _session_factory  # noqa: PLW0603
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None


def _get_session_factory() -> async_sessionmaker[AsyncSession]:
    """The raw session factory. Private — use ``get_db``, ``system_session``,
    instead of calling this directly."""
    global _session_factory  # noqa: PLW0603
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async database session for the current request."""
    factory = _get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise



# Read by ``agentic_core.security.isolation.guard`` to exempt a system session from the
# do_orm_execute guard (M113 #1469). Duplicated as a literal rather than
# imported from ``guard.py`` so this module's import graph stays as light as
# it was before the guard existed — the two modules only need to agree on the
# string, not share code.
_SYSTEM_BYPASS_KEY = "agentic_core_system_bypass"


@asynccontextmanager
async def system_session(*, reason: str) -> AsyncGenerator[AsyncSession, None]:
    """An unscoped session for the engine, worker, and startup/background code.

    *reason* is required: a short, grep-able description of why this call
    site needs unrestricted access (e.g. ``"heartbeat: worker liveness"``).
    The ``do_orm_execute`` guard (#1469) exempts sessions opened here
    entirely, in every mode — this is the audited, deliberate bypass, not
    something for the guard to warn about.
    """
    if not reason:
        raise ValueError("system_session() requires a reason")
    factory = _get_session_factory()
    async with factory() as session:
        session.info[_SYSTEM_BYPASS_KEY] = reason
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


