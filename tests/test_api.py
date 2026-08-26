"""Tests for FastAPI backend routes and router logic directly."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from backend.api import schemas
from backend.api.routers import events, ops, products, recommendations
from backend.database import queries, session
from backend.database.models import Base


def get_test_db(tmp_path: Path) -> sqlite3.Connection:
    db_path = tmp_path / "api_test.db"
    engine = session.create_db_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(bind=engine)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    # Seed products
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


def test_api_health(tmp_path: Path):
    """Test ops.health endpoint logic."""
    conn = get_test_db(tmp_path)
    try:
        health = ops.health(conn)
        assert hasattr(health, "status")
        assert hasattr(health, "products")
        assert hasattr(health, "view_events")
        assert health.products == 1
    finally:
        conn.close()


def test_api_provenance(tmp_path: Path):
    """Test ops.provenance reporting simulated vs live split."""
    conn = get_test_db(tmp_path)
    try:
        prov = ops.provenance(conn)
        assert hasattr(prov, "simulated")
        assert hasattr(prov, "live")
        assert hasattr(prov, "live_share")
        assert prov.live_share == 0.0
    finally:
        conn.close()


def test_api_products_list_and_get(tmp_path: Path):
    """Test products list and single item retrieval."""
    conn = get_test_db(tmp_path)
    try:
        page = products.list_products(conn, limit=10, offset=0, category=None, available_only=False)
        assert len(page.items) == 1
        assert page.items[0].product_id == "T001"

        single = products.get_product("T001", conn)
        assert single.product_id == "T001"
        assert single.product_name == "Matrix Runner Low"
    finally:
        conn.close()


def test_api_tracking_event(tmp_path: Path):
    """Test recording view events and validation."""
    conn = get_test_db(tmp_path)
    try:
        payload = schemas.ViewEventIn(
            user_id="anon_abcdef012345",
            product_id="T001",
            duration_seconds=15.5,
            surface="product_page",
        )
        result = events.record_view(payload, conn)
        assert result.user_id == "anon_abcdef012345"
        assert result.product_id == "T001"
        assert result.duration_seconds == 15.5
        assert not result.clamped
        assert result.user_created
    finally:
        conn.close()


def test_api_recommendations_flow(tmp_path: Path):
    """Test personalized and similar product recommendation endpoints."""
    conn = get_test_db(tmp_path)
    try:
        # Phase 2: User recommendation with no history -> fallback
        rec_user = recommendations.for_user("anon_999988887777", conn=conn, top_k=5, log=False)
        assert rec_user.strategy in {"popular", "empty"}

        # Phase 3: Item similarity
        rec_sim = recommendations.similar_to("T001", conn=conn, top_k=5, exclude_seen_by=None)
        assert rec_sim.strategy in {"item_item_cosine", "popular", "empty"}
    finally:
        conn.close()
