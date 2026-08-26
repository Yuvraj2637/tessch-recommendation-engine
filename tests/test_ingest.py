"""Tests for Phase 1 ingest and validation."""

from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from backend.core.config import IngestSettings
from backend.database import queries, writes
from backend.database.models import Base, Product
from backend.database.session import create_db_engine
from backend.services import ingest


def get_test_conn(tmp_path: Path):
    db_path = tmp_path / "ingest_test.db"
    engine = create_db_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(bind=engine)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    # Seed a sample product
    conn.execute(
        """
        INSERT INTO products (
            product_id, product_name, category, price, availability, is_core, created_at, updated_at
        ) VALUES (
            'T001', 'Matrix Runner Low', 'Athletic', 14000.0, 1, 1, '2026-08-26 00:00:00', '2026-08-26 00:00:00'
        )
        """
    )
    conn.commit()
    return conn


def test_validate_user_id():
    """Anonymous user ID must match anon_<8-16 hex chars> format."""
    assert ingest.validate_user_id("anon_12345678") == "anon_12345678"
    assert ingest.validate_user_id("anon_abcdef012345") == "anon_abcdef012345"

    raised = False
    try:
        ingest.validate_user_id("user_12345")
    except ingest.ValidationError:
        raised = True
    assert raised, "Expected ValidationError for invalid user ID format"

    raised = False
    try:
        ingest.validate_user_id("")
    except ingest.ValidationError:
        raised = True
    assert raised, "Expected ValidationError for empty user ID"


def test_validate_duration():
    """Duration must be finite, >= min_dwell_seconds, and clamped at max_dwell_seconds."""
    cfg = IngestSettings(min_dwell_seconds=1.0, max_dwell_seconds=1800.0)

    # Sub-second view is discarded (mis-click)
    raised = False
    try:
        ingest.validate_duration(0.5, cfg)
    except ingest.ValidationError:
        raised = True
    assert raised, "Sub-second view should raise ValidationError"

    # Negative duration is rejected
    raised = False
    try:
        ingest.validate_duration(-10.0, cfg)
    except ingest.ValidationError:
        raised = True
    assert raised, "Negative duration should raise ValidationError"

    # Non-finite values rejected
    raised = False
    try:
        ingest.validate_duration(float("nan"), cfg)
    except ingest.ValidationError:
        raised = True
    assert raised, "NaN duration should raise ValidationError"

    # Normal duration
    val, clamped = ingest.validate_duration(45.5, cfg)
    assert val == 45.5
    assert not clamped

    # Clamping extreme duration (e.g. 5000s)
    val, clamped = ingest.validate_duration(5000.0, cfg)
    assert val == 1800.0
    assert clamped


def test_record_view_success(tmp_path: Path):
    """Recording a valid view creates the user and logs the event."""
    conn = get_test_conn(tmp_path)
    try:
        result = ingest.record_view(
            conn,
            user_id="anon_a1b2c3d4e5f6",
            product_id="T001",
            duration_seconds=22.4,
        )
        assert result.user_id == "anon_a1b2c3d4e5f6"
        assert result.product_id == "T001"
        assert result.duration_seconds == 22.4
        assert not result.clamped
        assert result.user_created

        # Verify directly in SQLite
        events = conn.execute(
            "SELECT user_id, product_id, duration_seconds FROM user_events WHERE event_id = ?",
            (result.event_id,),
        ).fetchone()
        assert events == ("anon_a1b2c3d4e5f6", "T001", 22.4)
    finally:
        conn.close()


def test_record_view_unknown_product(tmp_path: Path):
    """Viewing a non-existent product should raise ValidationError."""
    conn = get_test_conn(tmp_path)
    try:
        raised = False
        try:
            ingest.record_view(
                conn,
                user_id="anon_a1b2c3d4e5f6",
                product_id="T999",
                duration_seconds=30.0,
            )
        except ingest.ValidationError:
            raised = True
        assert raised, "Non-existent product should raise ValidationError"
    finally:
        conn.close()
