"""Shared FastAPI dependencies.

The only interesting one is the database connection, and the interesting part of
that is threading. ``sqlite3`` connections cannot be shared across threads, and
FastAPI runs synchronous route handlers in a thread pool — so a single
module-level connection would work in development and then fail intermittently
under concurrent requests, which is the worst possible failure mode because it
looks like a random bug rather than a design error.

A connection per request avoids the problem entirely. That is affordable here
because opening a SQLite file is a filesystem operation measured in microseconds,
not a network handshake to a remote server.
"""

from __future__ import annotations

import sqlite3
from typing import Iterator

from fastapi import Depends, HTTPException, status

from backend.core.config import Settings, get_settings
from backend.database import queries


def get_db() -> Iterator[sqlite3.Connection]:
    """Yield a per-request SQLite connection, always closed afterwards.

    Returns 503 rather than 500 when the database file is missing, because that is
    a setup problem with a specific fix ("run python -m scripts.init_db") rather
    than an internal error — and saying so is more useful than a stack trace.
    """
    if not queries.database_exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Database not found or empty. Create it with: "
                "python -m scripts.init_db && python -m scripts.generate_events"
            ),
        )
    conn = queries.connect()
    try:
        yield conn
    finally:
        conn.close()


def get_config() -> Settings:
    return get_settings()


def resolve_top_k(
    top_k: int | None = None, settings: Settings = Depends(get_config)
) -> int:
    """Clamp a caller-supplied list length into a sane range.

    The upper bound is not arbitrary caution: ``top_k`` reaches
    ``np.argpartition`` and a DataFrame construction, so an unbounded value read
    from a query string is a way for a caller to make the server do unbounded
    work.
    """
    requested = settings.default_top_k if top_k is None else int(top_k)
    return max(1, min(requested, 50))


__all__ = ["get_db", "get_config", "resolve_top_k"]
