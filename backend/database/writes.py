"""Writes on the live path — tracked views and served recommendations.

Kept separate from ``queries.py`` (which is read-only) so that the distinction is
enforced by imports rather than by discipline. Anything that mutates the database
during normal operation is in this file, and it is a short file on purpose.

Two design notes worth stating:

**Anonymous users are created on demand.** ``user_events.user_id`` is a foreign
key to ``users``, and there is no login system, so the first tracked event from a
new browser has nothing to point at. Rather than dropping the foreign key — which
would let genuinely corrupt rows in — the ingest path inserts a placeholder user
row first. The visitor becomes a real row in ``users`` with NULL demographics and
an ``anon_`` prefixed id, which keeps the constraint meaningful and keeps live
visitors distinguishable from simulated ones at a glance.

**Every write is explicitly committed.** SQLite's default isolation means an
uncommitted insert is invisible to the next connection, and the tracker's beacon
is fire-and-forget: nobody is watching for a failure. A view that is recorded but
not committed is worse than one that was never recorded, because it looks like it
worked.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Iterable, Sequence

from backend.core.constants import EVENT_PRODUCT_VIEW, SOURCE_LIVE


def _iso_now(dt: datetime | None = None) -> str:
    """Format datetime as ISO string for clean SQLite storage without deprecation warnings."""
    target = dt or datetime.now(timezone.utc)
    if target.tzinfo is not None:
        target = target.astimezone(timezone.utc).replace(tzinfo=None)
    return target.strftime("%Y-%m-%d %H:%M:%S")


def ensure_user(
    conn: sqlite3.Connection,
    user_id: str,
    *,
    now: datetime | None = None,
    commit: bool = True,
) -> bool:
    """Insert a placeholder ``users`` row if this id is not known yet.

    Returns True if a row was created. Safe to call on every event: the
    ``INSERT OR IGNORE`` makes it a no-op for a user who already exists, which is
    the common case, and costs one indexed primary-key probe.
    """
    timestamp = _iso_now(now)
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO users
            (user_id, username, email, age_group, gender, style_preference,
             device_preference, is_new_user, signup_at, created_at, last_active_at)
        VALUES (?, ?, NULL, NULL, NULL, NULL, NULL, 1, NULL, ?, ?)
        """,
        (user_id, user_id, timestamp, timestamp),
    )
    created = cur.rowcount > 0
    if not created:
        # Touch last_active_at so "when did we last hear from this visitor?" is
        # answerable without scanning the event log.
        conn.execute(
            "UPDATE users SET last_active_at = ? WHERE user_id = ?",
            (timestamp, user_id),
        )
    if commit:
        conn.commit()
    return created


def insert_view_event(
    conn: sqlite3.Connection,
    user_id: str,
    product_id: str,
    duration_seconds: float,
    *,
    timestamp: datetime | None = None,
    surface: str | None = None,
    position: int | None = None,
    session_id: str | None = None,
    source: str = SOURCE_LIVE,
    commit: bool = True,
) -> str:
    """Record one dwell-time observation. Returns the generated ``event_id``.

    ``session_id`` is passed through as NULL by default. The mock storefront has
    no server-side session concept, and inventing one would mean writing a
    ``sessions`` row per page load with fields (intent, device, event counts) that
    would all be guesses. Neither recommender uses ``session_id``, so a NULL is
    the honest value.

    Validation happens in ``backend/services/ingest.py`` before this is called.
    This function's job is to write, not to judge.
    """
    event_id = uuid.uuid4().hex
    conn.execute(
        """
        INSERT INTO user_events
            (event_id, user_id, product_id, session_id, event_type,
             duration_seconds, surface, position, search_query, event_value,
             source, timestamp, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, NULL)
        """,
        (
            event_id,
            user_id,
            product_id,
            session_id,
            EVENT_PRODUCT_VIEW,
            float(duration_seconds),
            surface,
            position,
            source,
            _iso_now(timestamp),
        ),
    )
    if commit:
        conn.commit()
    return event_id


def log_recommendations(
    conn: sqlite3.Connection,
    user_id: str,
    items: Sequence[tuple[str, int, float]],
    model_name: str,
    model_version: str,
    *,
    session_id: str | None = None,
    served_at: datetime | None = None,
    commit: bool = True,
) -> int:
    """Record what was shown, in what order, by which model.

    ``items`` is ``(product_id, rank, score)`` tuples.
    """
    if not items:
        return 0

    timestamp = _iso_now(served_at)
    rows = [
        (
            uuid.uuid4().hex,
            user_id,
            product_id,
            session_id,
            int(rank),
            float(score),
            model_name,
            model_version,
            None,
            0,
            0,
            0,
            timestamp,
            None,
        )
        for product_id, rank, score in items
    ]
    conn.executemany(
        """
        INSERT INTO recommendation_logs
            (log_id, user_id, product_id, session_id, recommendation_rank, score,
             model_name, model_version, experiment_group, was_clicked,
             was_carted, was_purchased, served_at, interacted_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    if commit:
        conn.commit()
    return len(rows)


def mark_clicked(
    conn: sqlite3.Connection,
    user_id: str,
    product_id: str,
    *,
    now: datetime | None = None,
    commit: bool = True,
) -> int:
    """Attribute a click back to the most recent impression of that product."""
    timestamp = _iso_now(now)
    cur = conn.execute(
        """
        UPDATE recommendation_logs
           SET was_clicked = 1, interacted_at = ?
         WHERE log_id = (
               SELECT log_id FROM recommendation_logs
                WHERE user_id = ? AND product_id = ? AND was_clicked = 0
                ORDER BY served_at DESC
                LIMIT 1
         )
        """,
        (timestamp, user_id, product_id),
    )
    if commit:
        conn.commit()
    return cur.rowcount


def delete_live_events(conn: sqlite3.Connection, *, commit: bool = True) -> int:
    """Remove every ``source='live'`` event. Used to reset a demo."""
    cur = conn.execute("DELETE FROM user_events WHERE source = ?", (SOURCE_LIVE,))
    removed = cur.rowcount
    if commit:
        conn.commit()
    return removed


__all__ = [
    "ensure_user",
    "insert_view_event",
    "log_recommendations",
    "mark_clicked",
    "delete_live_events",
]
