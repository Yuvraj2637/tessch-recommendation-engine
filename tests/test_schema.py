"""Tests for database schema creation and constraints."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from backend.database import queries, session
from backend.database.models import Base


def test_schema_creation_and_tables(tmp_path: Path):
    """Test that all tables are created properly in SQLite."""
    db_path = tmp_path / "test.db"
    engine = session.create_db_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(bind=engine)

    conn = sqlite3.connect(db_path)
    try:
        counts = queries.table_counts(conn)
        expected_tables = {
            "products",
            "users",
            "sessions",
            "user_events",
            "user_item_features",
            "recommendation_logs",
            "orders",
        }
        assert expected_tables.issubset(set(counts.keys()))
        for table in expected_tables:
            assert counts[table] == 0
    finally:
        conn.close()


def test_foreign_key_enforcement(tmp_path: Path):
    """Test that SQLite foreign key pragma prevents orphan events."""
    db_path = tmp_path / "fk_test.db"
    engine = session.create_db_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(bind=engine)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        failed = False
        try:
            conn.execute(
                """
                INSERT INTO user_events (
                    event_id, user_id, product_id, event_type, duration_seconds, source, timestamp
                ) VALUES (
                    'evt_001', 'non_existent_user', 'T999', 'product_view', 25.0, 'live', '2026-08-26 00:00:00'
                )
                """
            )
            conn.commit()
        except sqlite3.IntegrityError:
            failed = True
        assert failed, "Inserting orphan event should trigger SQLite IntegrityError"
    finally:
        conn.close()
