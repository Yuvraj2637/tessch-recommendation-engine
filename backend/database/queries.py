"""Read path: SQLite -> pandas DataFrames.

Why stdlib ``sqlite3`` here when the project already depends on SQLAlchemy?

SQLAlchemy owns the *schema* (``backend/database/models.py``) and the bulk
simulation load, where the ORM's declarative models and ``executemany`` batching
genuinely earn their keep. The serving path has the opposite shape: it runs a
handful of fixed aggregate queries and immediately hands the result to pandas.
An ORM adds object materialisation cost and no expressiveness for that, so the
read layer issues parameterised SQL through the standard library instead.

The practical payoff is that this module, and therefore every recommendation
algorithm downstream of it, depends only on ``pandas`` and the standard library.
That keeps the algorithms testable without a web framework or a database server
in the picture.

Every function here takes an explicit connection. Nothing opens a global
connection at import time, because SQLite connections are not safe to share
across threads and a web server is threaded.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from backend.core.config import settings
from backend.core.constants import (
    EVENT_PRODUCT_VIEW,
    SOURCE_LIVE,
    SOURCE_SIMULATED,
)


# ── Connections ──────────────────────────────────────────────────────────────
def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open a connection with the pragmas this project relies on.

    ``foreign_keys=ON`` is not optional: SQLite silently ignores foreign keys by
    default, which would let the ingest endpoint accept events for product IDs
    that do not exist and quietly poison the similarity matrix weeks later.

    Write-ahead logging is **off by default**, which is worth explaining because
    WAL is normally the obvious choice. WAL needs a shared-memory file next to
    the database, and cloud-synced folders (OneDrive, Dropbox, Google Drive) and
    network shares cannot provide one. On those filesystems requesting WAL
    raises "disk I/O error" — and because the journal mode is recorded in the
    database header rather than the connection, a single successful WAL switch
    leaves the file permanently unopenable on that filesystem. That failure mode
    is far more expensive than the write concurrency WAL would have bought a
    project of this size, so it is opt-in via ``SQLITE_WAL=true`` for anyone
    running from a local disk.

    When WAL *is* requested, it is verified with a canary read before being
    trusted, and reverted if that read fails.
    """
    path = Path(db_path) if db_path is not None else settings.sqlite_path
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        str(path),
        # FastAPI serves requests from a threadpool; each request opens its own
        # connection (see get_connection), so no connection is ever shared.
        check_same_thread=False,
        timeout=15.0,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")

    if settings.sqlite_wal:
        if not _try_journal_mode(conn, "WAL"):
            _try_journal_mode(conn, "DELETE")
    _try_pragma(conn, "synchronous=NORMAL")
    return conn


def _try_pragma(conn: sqlite3.Connection, pragma: str) -> bool:
    """Apply an optimisation pragma, tolerating filesystems that refuse it."""
    try:
        conn.execute(f"PRAGMA {pragma}")
        return True
    except sqlite3.OperationalError:
        return False


def _try_journal_mode(conn: sqlite3.Connection, mode: str) -> bool:
    """Switch journal mode and prove the database is still readable afterwards.

    The canary read is the point of this function. Setting ``journal_mode=WAL``
    can succeed while leaving the database unusable on filesystems that cannot
    back the shared-memory file, so the switch is only reported as successful if
    a subsequent query actually returns.
    """
    try:
        conn.execute(f"PRAGMA journal_mode={mode}")
        conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
        return True
    except sqlite3.OperationalError:
        return False


def journal_mode(conn: sqlite3.Connection) -> str:
    """The journal mode actually in force — useful in a startup banner."""
    return str(conn.execute("PRAGMA journal_mode").fetchone()[0])


def repair_journal_mode(db_path: str | Path | None = None) -> str | None:
    """Force a database out of WAL and back to the portable rollback journal.

    Recovery path for a database that was opened with WAL on a local disk and
    then moved into a synced folder, which is an easy accident to have and
    presents as a flat "disk I/O error" on every query. Returns the resulting
    journal mode, or None if the file could not be opened at all.
    """
    path = Path(db_path) if db_path is not None else settings.sqlite_path
    if not path.exists():
        return None
    conn = None
    try:
        conn = sqlite3.connect(str(path), timeout=15.0)
        conn.execute("PRAGMA journal_mode=DELETE")
        return str(conn.execute("PRAGMA journal_mode").fetchone()[0])
    except sqlite3.Error:
        return None
    finally:
        if conn is not None:
            conn.close()


def database_exists(db_path: str | Path | None = None) -> bool:
    """True if the database file exists and has the ``user_events`` table.

    Deliberately opens read-only and closes explicitly. Using ``with
    sqlite3.connect(...)`` here would commit on exit, which is both meaningless
    for a read and an error on some filesystems.
    """
    path = Path(db_path) if db_path is not None else settings.sqlite_path
    if not path.exists():
        return False
    conn = None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='user_events'"
        ).fetchone()
        return row is not None
    except sqlite3.Error:
        return False
    finally:
        if conn is not None:
            conn.close()


# ── Catalogue ────────────────────────────────────────────────────────────────
PRODUCT_COLUMNS = (
    "product_id",
    "product_name",
    "category",
    "upper_type",
    "color",
    "color_family",
    "style",
    "price",
    "material",
    "collection",
    "description",
    "availability",
    "image_url",
    "is_core",
)


def load_products(
    conn: sqlite3.Connection,
    available_only: bool = False,
    is_core_only: bool = False,
) -> pd.DataFrame:
    """The product catalogue, indexed positionally (not by product_id).

    ``available_only`` filters to in-stock items. It defaults to False because
    the similarity model should be *built* over the whole catalogue — dropping
    out-of-stock items from the matrix would distort the co-view statistics of
    the items that remain. Filtering belongs at serving time, not training time.

    ``is_core_only`` filters to the 25 real curated TESSCH products. The
    storefront uses this so visitors never see the generated filler products.
    The ML model training always passes ``is_core_only=False`` to keep the
    user-item matrix sparse and realistic.
    """
    sql = f"SELECT {', '.join(PRODUCT_COLUMNS)} FROM products"  # noqa: S608 - fixed column allowlist
    conditions = []
    if available_only:
        conditions.append("availability = 1")
    if is_core_only:
        conditions.append("is_core = 1")
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY product_id"
    df = pd.read_sql(sql, conn)
    if not df.empty:
        df["availability"] = df["availability"].astype(bool)
        df["is_core"] = df["is_core"].astype(bool)
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df


def catalog_ids(conn: sqlite3.Connection) -> set[str]:
    """Every known product_id — used by ingest validation."""
    return {row[0] for row in conn.execute("SELECT product_id FROM products")}


def product_exists(conn: sqlite3.Connection, product_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM products WHERE product_id = ? LIMIT 1", (product_id,)
    ).fetchone()
    return row is not None


def user_exists(conn: sqlite3.Connection, user_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM users WHERE user_id = ? LIMIT 1", (user_id,)
    ).fetchone()
    return row is not None


# ── Behaviour ────────────────────────────────────────────────────────────────
def load_dwell_events(
    conn: sqlite3.Connection,
    source: str | Iterable[str] | None = None,
    min_seconds: float = 0.0,
) -> pd.DataFrame:
    """Every ``product_view`` with a usable dwell time.

    This is the one query the whole project rests on. Columns returned:
    ``user_id``, ``product_id``, ``seconds``, ``timestamp``, ``source``.

    ``source`` filters to ``'live'``, ``'simulated'``, or a collection of both.
    Passing None (the default) returns the combined dataset, which is what the
    recommenders use — but every caller that reports a metric should also run
    with ``source='live'`` so the write-up can separate the two.

    Rows with a NULL dwell time are excluded rather than treated as zero. A NULL
    here means the tracker never got a chance to report (the browser was killed),
    which is missing data, not evidence of a zero-second view.
    """
    sql = (
        "SELECT user_id, product_id, duration_seconds AS seconds, timestamp, source "
        "FROM user_events "
        "WHERE event_type = ? AND product_id IS NOT NULL AND duration_seconds IS NOT NULL "
        "AND duration_seconds > ?"
    )
    params: list[Any] = [EVENT_PRODUCT_VIEW, float(min_seconds)]

    sources = _normalise_sources(source)
    if sources is not None:
        placeholders = ", ".join("?" for _ in sources)
        sql += f" AND source IN ({placeholders})"
        params.extend(sources)

    df = pd.read_sql(sql, conn, params=params, parse_dates=["timestamp"])
    if not df.empty:
        df["seconds"] = pd.to_numeric(df["seconds"], errors="coerce")
        df = df.dropna(subset=["seconds"])
    return df


def _normalise_sources(source: str | Iterable[str] | None) -> list[str] | None:
    if source is None:
        return None
    if isinstance(source, str):
        return [source]
    values = list(source)
    return values or None


def load_user_history(
    conn: sqlite3.Connection, user_id: str, limit: int = 200
) -> pd.DataFrame:
    """A single user's recent product views, newest first.

    Used by the "Picked for you" endpoint, which only ever needs one user's
    rows. Scanning the whole event table to serve one visitor would be the
    obvious way to make this endpoint slow as the log grows.
    """
    sql = (
        "SELECT product_id, duration_seconds AS seconds, timestamp, source "
        "FROM user_events "
        "WHERE user_id = ? AND event_type = ? "
        "  AND product_id IS NOT NULL AND duration_seconds IS NOT NULL "
        "ORDER BY timestamp DESC LIMIT ?"
    )
    df = pd.read_sql(
        sql,
        conn,
        params=[user_id, EVENT_PRODUCT_VIEW, int(limit)],
        parse_dates=["timestamp"],
    )
    if not df.empty:
        df["seconds"] = pd.to_numeric(df["seconds"], errors="coerce")
        df = df.dropna(subset=["seconds"])
    return df


def popular_products(
    conn: sqlite3.Connection,
    limit: int = 8,
    source: str | Iterable[str] | None = None,
    is_core_only: bool = True,
) -> pd.DataFrame:
    """Total dwell time per product across all users, descending.

    This is the cold-start fallback: it is what a brand-new visitor with no
    history sees, and it is also the non-personalised baseline the personalised
    endpoints should be compared against.
    """
    sql = (
        "SELECT ue.product_id, SUM(ue.duration_seconds) AS total_seconds, "
        "       COUNT(*) AS view_count, COUNT(DISTINCT ue.user_id) AS viewer_count "
        "FROM user_events ue "
    )
    if is_core_only:
        sql += "JOIN products p ON ue.product_id = p.product_id AND p.is_core = 1 "
    sql += (
        "WHERE ue.event_type = ? AND ue.product_id IS NOT NULL AND ue.duration_seconds IS NOT NULL"
    )
    params: list[Any] = [EVENT_PRODUCT_VIEW]

    sources = _normalise_sources(source)
    if sources is not None:
        placeholders = ", ".join("?" for _ in sources)
        sql += f" AND ue.source IN ({placeholders})"
        params.extend(sources)

    sql += " GROUP BY ue.product_id ORDER BY total_seconds DESC LIMIT ?"
    params.append(int(limit))
    return pd.read_sql(sql, conn, params=params)


# ── Provenance ───────────────────────────────────────────────────────────────
def data_provenance(conn: sqlite3.Connection) -> dict[str, Any]:
    """How much of the dataset is real traffic versus generated.

    Every claim in the results write-up is derived from this, so that "which
    parts were simulated?" has a queryable answer rather than a remembered one.
    """
    rows = conn.execute(
        "SELECT source, COUNT(*) AS events, "
        "       COUNT(DISTINCT user_id) AS users, "
        "       COUNT(DISTINCT product_id) AS products "
        "FROM user_events WHERE event_type = ? GROUP BY source",
        (EVENT_PRODUCT_VIEW,),
    ).fetchall()

    by_source = {
        r["source"]: {
            "views": r["events"],
            "users": r["users"],
            "products": r["products"],
        }
        for r in rows
    }
    live = by_source.get(SOURCE_LIVE, {"views": 0, "users": 0, "products": 0})
    simulated = by_source.get(SOURCE_SIMULATED, {"views": 0, "users": 0, "products": 0})
    total_views = live["views"] + simulated["views"]

    return {
        "live": live,
        "simulated": simulated,
        "total_views": total_views,
        "live_share": (live["views"] / total_views) if total_views else 0.0,
    }


def table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Row count per table, for scripts that report what they loaded."""
    names = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
    ]
    return {
        name: conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]  # noqa: S608
        for name in names
    }


__all__ = [
    "connect",
    "journal_mode",
    "repair_journal_mode",
    "database_exists",
    "load_products",
    "catalog_ids",
    "product_exists",
    "user_exists",
    "load_dwell_events",
    "load_user_history",
    "popular_products",
    "data_provenance",
    "table_counts",
    "PRODUCT_COLUMNS",
]
