"""Create the database tables.

    python -m scripts.init_db            # create if missing
    python -m scripts.init_db --reset    # drop and recreate (destroys data)

``backend/database/models.py`` is the source of truth, so the SQLAlchemy path is
preferred. But there is a deliberate fallback to executing
``backend/database/schema.sql`` with stdlib ``sqlite3`` when SQLAlchemy is not
installed, because "I cannot even create the database until I fix my virtualenv"
is a bad first experience for anyone cloning this repo.

The fallback is safe to rely on because ``schema.sql`` is generated from the models
by ``scripts/dump_schema.py`` and the two are asserted equivalent in
``tests/test_schema.py`` — if they ever drift, that test fails rather than the
fallback quietly building a different database.

This only creates empty tables. To populate them, run
``python -m scripts.generate_events``.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from backend.core.config import get_settings
from backend.database import queries

SCHEMA_SQL = Path(__file__).resolve().parents[1] / "backend" / "database" / "schema.sql"


def create_with_sqlalchemy() -> bool:
    """Create tables from the ORM models. Returns False if SQLAlchemy is absent."""
    try:
        from backend.database.session import create_all
    except ImportError:
        return False
    create_all()
    return True


def create_with_schema_sql(db_path: Path) -> None:
    """Create tables by executing the checked-in DDL."""
    if not SCHEMA_SQL.is_file():
        raise FileNotFoundError(f"missing {SCHEMA_SQL}")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL.read_text(encoding="utf-8"))
        conn.commit()
    finally:
        conn.close()


def reset(db_path: Path) -> None:
    """Delete the database file.

    Dropping tables individually would leave a file whose journal mode is whatever
    it was before — and a WAL-poisoned file on a cloud-synced folder stays
    unopenable through a table drop. Removing the file is the only reliable reset.
    """
    for suffix in ("", "-wal", "-shm", "-journal"):
        candidate = Path(str(db_path) + suffix)
        if candidate.exists():
            candidate.unlink()
            print(f"  removed {candidate.name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="delete the existing database file first (destroys all data)",
    )
    parser.add_argument(
        "--force-sql",
        action="store_true",
        help="use schema.sql even if SQLAlchemy is available",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    if not settings.is_sqlite:
        print(f"DATABASE_URL is not SQLite ({settings.database_url}).", file=sys.stderr)
        return 1

    db_path = settings.sqlite_path
    print(f"database: {db_path}")

    if args.reset:
        reset(db_path)

    used = "schema.sql"
    if args.force_sql or not create_with_sqlalchemy():
        create_with_schema_sql(db_path)
        if args.force_sql:
            print("  created from schema.sql (forced)")
        else:
            print("  created from schema.sql (SQLAlchemy not installed)")
    else:
        used = "models.py"
        print("  created from models.py via SQLAlchemy")

    conn = queries.connect()
    try:
        counts = queries.table_counts(conn)
        print(f"  journal_mode: {queries.journal_mode(conn)}")
        print(f"  tables: {len(counts)} ({', '.join(sorted(counts))})")
        if counts.get("products", 0) == 0:
            print()
            print("Tables are empty. Populate them with:")
            print("  python -m scripts.generate_events")
    finally:
        conn.close()

    print(f"\nDone (source: {used}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
