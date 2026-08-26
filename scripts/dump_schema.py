"""Regenerate ``backend/database/schema.sql`` from the ORM models.

    python -m scripts.dump_schema           # write the file
    python -m scripts.dump_schema --check   # verify it is up to date (CI)

``schema.sql`` is a generated artifact, kept in the repo so the table layout can be
read without running Python and so ``scripts/init_db.py`` can create the database
without SQLAlchemy installed. Because it is a copy, it can drift — ``--check``
exits non-zero when it has, which is what makes the copy trustworthy.

Requires SQLAlchemy. That is fine: this script is for whoever changes the models,
not for whoever runs the app.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "backend" / "database" / "schema.sql"

HEADER = """\
-- =============================================================
-- TESSCH Recommendation Engine — SQLite reference schema
-- =============================================================
-- GENERATED FILE — do not edit by hand.
--
-- `backend/database/models.py` is the single source of truth for the schema.
-- This file is a human-readable dump of what those models create, kept in the
-- repo so the table layout can be reviewed without running any Python.
--
-- Regenerate with:   python -m scripts.dump_schema
-- Create the tables: python -m scripts.init_db
-- =============================================================
"""


def render_ddl() -> str:
    """Render CREATE TABLE / CREATE INDEX statements for the SQLite dialect."""
    from sqlalchemy.schema import CreateIndex, CreateTable

    from backend.database.models import Base

    try:
        from sqlalchemy.dialects import sqlite

        dialect = sqlite.dialect()
    except ImportError:  # pragma: no cover
        dialect = None

    chunks: list[str] = [HEADER]
    for table in Base.metadata.sorted_tables:
        chunks.append(str(CreateTable(table).compile(dialect=dialect)).strip() + ";")
        for index in sorted(table.indexes, key=lambda i: i.name or ""):
            chunks.append(str(CreateIndex(index).compile(dialect=dialect)).strip() + ";")
        chunks.append("")
    return "\n\n".join(chunks).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if schema.sql differs from the models instead of writing it",
    )
    args = parser.parse_args(argv)

    try:
        rendered = render_ddl()
    except ImportError:
        print(
            "SQLAlchemy is required to regenerate the schema: pip install SQLAlchemy",
            file=sys.stderr,
        )
        return 2

    if args.check:
        current = SCHEMA_PATH.read_text(encoding="utf-8") if SCHEMA_PATH.is_file() else ""
        # Compare statements, not whole text: the committed file carries hand-written
        # explanatory comments that this script does not reproduce, and losing those
        # to a formatting difference would be a bad trade.
        if _statements(current) == _statements(rendered):
            print("schema.sql is up to date.")
            return 0
        print(
            "schema.sql is out of date with models.py — run: python -m scripts.dump_schema",
            file=sys.stderr,
        )
        return 1

    SCHEMA_PATH.write_text(rendered, encoding="utf-8")
    print(f"wrote {SCHEMA_PATH}")
    return 0


def _statements(sql: str) -> list[str]:
    """Normalise DDL to a comparable list of statements, ignoring comments."""
    lines = [
        line for line in sql.splitlines() if not line.strip().startswith("--")
    ]
    body = "\n".join(lines)
    return sorted(
        " ".join(stmt.split()) for stmt in body.split(";") if stmt.strip()
    )


if __name__ == "__main__":
    raise SystemExit(main())
