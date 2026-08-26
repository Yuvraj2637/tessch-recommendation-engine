"""Database engine and session management.

The same code path serves SQLite and PostgreSQL. SQLite gets two extra
pragmas applied on every connection:

* ``foreign_keys=ON`` — SQLite ignores foreign keys by default, which would
  silently let the simulator write events for products that don't exist.
* ``journal_mode=WAL`` — lets the API read while a feature job writes.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.core.config import settings
from backend.database.models import Base


def _configure_sqlite(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_connection, _connection_record):  # pragma: no cover
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()


def create_db_engine(url: str | None = None, echo: bool | None = None) -> Engine:
    """Build an Engine for ``url`` (defaults to the configured DATABASE_URL)."""
    url = url or settings.database_url
    echo = settings.sql_echo if echo is None else echo

    kwargs: dict[str, Any] = {"echo": echo, "future": True}

    if url.startswith("sqlite"):
        # check_same_thread=False so FastAPI's threadpool can share the engine.
        kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in url:
            # An in-memory DB dies with its connection; pin it for tests.
            kwargs["poolclass"] = StaticPool
    else:
        kwargs["pool_pre_ping"] = True

    engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):
        _configure_sqlite(engine)
    return engine


engine: Engine = create_db_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@contextmanager
def session_scope(bind: Engine | None = None) -> Iterator[OrmSession]:
    """Transactional scope: commit on success, roll back on error, always close."""
    factory = SessionLocal if bind is None else sessionmaker(bind=bind, autoflush=False, expire_on_commit=False)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Iterator[OrmSession]:
    """FastAPI dependency yielding a read-oriented session."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def create_all(bind: Engine | None = None) -> None:
    Base.metadata.create_all(bind or engine)


def drop_all(bind: Engine | None = None) -> None:
    Base.metadata.drop_all(bind or engine)


def bulk_insert(table_model: Any, rows: Sequence[dict], bind: Engine | None = None,
                chunk_size: int = 5_000) -> int:
    """Fast multi-row INSERT via executemany, chunked to bound memory.

    Used by the loader for hundreds of thousands of events, where the ORM's
    per-object overhead would be the bottleneck.
    """
    if not rows:
        return 0

    target = table_model.__table__
    active = bind or engine
    written = 0
    with active.begin() as conn:
        for start in range(0, len(rows), chunk_size):
            chunk = rows[start : start + chunk_size]
            conn.execute(target.insert(), chunk)
            written += len(chunk)
    return written


def table_counts(bind: Engine | None = None) -> dict[str, int]:
    """Row count per table — used by scripts to report what they loaded."""
    active = bind or engine
    counts: dict[str, int] = {}
    with active.connect() as conn:
        for name in Base.metadata.tables:
            counts[name] = conn.execute(text(f"SELECT COUNT(*) FROM {name}")).scalar_one()
    return counts


__all__ = [
    "engine",
    "SessionLocal",
    "create_db_engine",
    "session_scope",
    "get_session",
    "create_all",
    "drop_all",
    "bulk_insert",
    "table_counts",
]
