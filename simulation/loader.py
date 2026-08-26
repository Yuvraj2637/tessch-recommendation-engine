"""Persist simulated world data to the database, or to parquet.

Timezone discipline: every timestamp is an aware UTC ``datetime`` everywhere in
the Python layer. SQLite has no timezone type, so this module converts to naive
UTC on the way in and back to aware UTC on the way out. PostgreSQL keeps the
offset natively and the same functions are no-ops there. Doing the conversion in
exactly one place is what stops "off by 5.5 hours" bugs from appearing in the
recency features later.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Sequence

from sqlalchemy import Engine, delete

from backend.core.config import PROCESSED_DIR, RAW_EVENTS_DIR, ensure_directories
from backend.core.constants import SOURCE_SIMULATED
from backend.database.models import (
    Order,
    Product,
    RecommendationLog,
    Session,
    User,
    UserEvent,
    UserItemFeature,
)
from backend.database.session import bulk_insert, create_all, engine as default_engine
from simulation.catalog import Catalogue
from simulation.generator import SimulationOutput


def _to_db_dt(value: datetime | None) -> datetime | None:
    """Normalise to naive UTC for storage. SQLite has no tz; Postgres gets it back."""
    if value is None or value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _as_uuid(value: Any) -> uuid.UUID | None:
    """Coerce to a ``UUID`` object, which is what the ``Uuid`` column expects.

    The simulator already produces UUID objects, but accepting strings here means
    hand-built rows and fixtures work too, instead of failing deep inside the
    SQLAlchemy type processor with an opaque ``'str' has no attribute 'hex'``.
    """
    if value is None or isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))


def _from_db_dt(value: datetime | None) -> datetime | None:
    """Restore awareness for values read back from SQLite."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _product_rows(catalogue: Catalogue) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for product in catalogue.products:
        rows.append(
            {
                "product_id": product["product_id"],
                "product_name": product["product_name"],
                "category": product["category"],
                "upper_type": product.get("upper_type"),
                "color": product.get("color"),
                "color_family": product.get("color_family"),
                "style": product.get("style"),
                "price": float(product["price"]),
                "material": product.get("material"),
                "collection": product.get("collection"),
                "description": product.get("description"),
                "availability": bool(product.get("availability", True)),
                "image_url": product.get("image_url"),
                "is_core": bool(product.get("is_core", False)),
            }
        )
    return rows


def _user_rows(users: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for user in users:
        rows.append(
            {
                "user_id": user["user_id"],
                "username": user.get("username"),
                "email": user.get("email"),
                "age_group": user.get("age_group"),
                "gender": user.get("gender"),
                "style_preference": user.get("style_preference"),
                "device_preference": user.get("device_preference"),
                "is_new_user": bool(user.get("is_new_user", True)),
                "signup_at": _to_db_dt(user.get("signup_at")),
                "created_at": _to_db_dt(user.get("created_at")),
                "last_active_at": _to_db_dt(user.get("last_active_at")),
            }
        )
    return rows


def _session_rows(sessions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for s in sessions:
        rows.append(
            {
                "session_id": s["session_id"],
                "user_id": s["user_id"],
                "started_at": _to_db_dt(s["started_at"]),
                "ended_at": _to_db_dt(s.get("ended_at")),
                "duration_seconds": s.get("duration_seconds"),
                "events_count": int(s.get("events_count", 0)),
                "products_viewed": int(s.get("products_viewed", 0)),
                "unique_products": int(s.get("unique_products", 0)),
                "repeat_views": int(s.get("repeat_views", 0)),
                "wishlist_count": int(s.get("wishlist_count", 0)),
                "cart_count": int(s.get("cart_count", 0)),
                "purchase_count": int(s.get("purchase_count", 0)),
                "search_count": int(s.get("search_count", 0)),
                "device_type": s.get("device_type"),
                "intent": s.get("intent"),
            }
        )
    return rows


def _event_rows(events: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for e in events:
        rows.append(
            {
                "event_id": _as_uuid(e["event_id"]),
                "user_id": e["user_id"],
                "product_id": e.get("product_id"),
                "session_id": e.get("session_id"),
                "event_type": e["event_type"],
                "duration_seconds": e.get("duration_seconds"),
                "surface": e.get("surface"),
                "position": e.get("position"),
                "search_query": e.get("search_query"),
                "event_value": e.get("event_value"),
                # Everything this module writes is, by definition, generated.
                # Stamping it here means no later report has to guess.
                "source": SOURCE_SIMULATED,
                "timestamp": _to_db_dt(e["timestamp"]),
                "metadata": e.get("metadata", {}),
            }
        )
    return rows


def _order_rows(orders: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for o in orders:
        rows.append(
            {
                "order_id": _as_uuid(o["order_id"]),
                "user_id": o["user_id"],
                "session_id": o.get("session_id"),
                "product_id": o["product_id"],
                "quantity": int(o.get("quantity", 1)),
                "unit_price": float(o["unit_price"]),
                "total_amount": float(o["total_amount"]),
                "status": o.get("status", "completed"),
                "ordered_at": _to_db_dt(o["ordered_at"]),
            }
        )
    return rows


def _rec_log_rows(logs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for log in logs:
        rows.append(
            {
                "user_id": log["user_id"],
                "product_id": log["product_id"],
                "session_id": log.get("session_id"),
                "recommendation_rank": int(log["recommendation_rank"]),
                "score": log.get("score"),
                "model_name": log.get("model_name", "legacy_popularity"),
                "model_version": log.get("model_version", "0.1"),
                "experiment_group": log.get("experiment_group"),
                "served_at": _to_db_dt(log["served_at"]),
                "was_clicked": bool(log.get("was_clicked", False)),
                "was_carted": bool(log.get("was_carted", False)),
                "was_purchased": bool(log.get("was_purchased", False)),
                "interacted_at": _to_db_dt(log.get("interacted_at")),
            }
        )
    return rows


def load_simulation(
    output: SimulationOutput,
    bind: Engine | None = None,
    reset: bool = True,
) -> dict[str, int]:
    """Write a simulation output into the database. Returns per-table row counts.

    ``reset=True`` (the default) clears the affected tables first, so re-running
    Stage 1 is idempotent rather than accumulating duplicate users.
    """
    active = bind or default_engine
    ensure_directories()
    create_all(active)

    if reset:
        # Children before parents so foreign keys stay satisfied throughout.
        with active.begin() as conn:
            for table in (
                UserEvent,
                RecommendationLog,
                Order,
                UserItemFeature,
                Session,
                User,
                Product,
            ):
                conn.execute(delete(table.__table__))

    counts: dict[str, int] = {}
    counts["products"] = bulk_insert(Product, _product_rows(Catalogue(output.products)), active)
    counts["users"] = bulk_insert(User, _user_rows(output.users), active)
    counts["sessions"] = bulk_insert(Session, _session_rows(output.sessions), active)
    counts["user_events"] = bulk_insert(UserEvent, _event_rows(output.events), active)
    counts["orders"] = bulk_insert(Order, _order_rows(output.orders), active)
    counts["recommendation_logs"] = bulk_insert(
        RecommendationLog, _rec_log_rows(output.recommendation_logs), active
    )
    return counts


def write_parquet(output: SimulationOutput, prefix: str = "events") -> dict[str, str]:
    """Alternative to the database: write the world to parquet files.

    Useful for getting straight to modelling, and for keeping the ground-truth
    affinity matrix *outside* the application database where no model can
    accidentally join against it.
    """
    ensure_directories()
    import pandas as pd

    paths: dict[str, str] = {}
    frames = {
        "products": output.products,
        "users": output.users,
        "sessions": output.sessions,
        "user_events": output.events,
        "orders": output.orders,
        "recommendation_logs": output.recommendation_logs,
    }

    for name, rows in frames.items():
        df = pd.DataFrame(rows)
        if name == "user_events" and "metadata" in df.columns:
            # Nested dicts with heterogeneous keys have no stable parquet
            # schema; serialise them so the column round-trips cleanly.
            df["metadata"] = df["metadata"].map(json.dumps)
        # UUID objects likewise have no parquet type — store them as text.
        for column in ("event_id", "order_id", "log_id"):
            if column in df.columns:
                df[column] = df[column].astype(str)
        path = PROCESSED_DIR / f"{prefix}_{name}.parquet"
        df.to_parquet(path, index=False)
        paths[name] = str(path)

    ground_truth_dir = RAW_EVENTS_DIR / "ground_truth"
    ground_truth_dir.mkdir(parents=True, exist_ok=True)

    hidden_path = ground_truth_dir / "hidden_users.parquet"
    pd.DataFrame(output.hidden_users).to_parquet(hidden_path, index=False)
    paths["hidden_users"] = str(hidden_path)

    affinity_path = ground_truth_dir / "affinity.parquet"
    pd.DataFrame(output.affinity_long).to_parquet(affinity_path, index=False)
    paths["affinity"] = str(affinity_path)

    meta_path = PROCESSED_DIR / f"{prefix}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(output.meta, handle, indent=2, default=str)
    paths["meta"] = str(meta_path)
    return paths


__all__ = ["load_simulation", "write_parquet"]
