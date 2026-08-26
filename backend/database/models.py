"""SQLAlchemy ORM models for the TESSCH recommendation engine.

These mirror `backend/database/schema.sql` (the PostgreSQL reference DDL) but
are written dialect-neutrally so the identical code runs on SQLite during
development and PostgreSQL in production. The only difference is the
``DATABASE_URL``.

Two dialect variants do the work:
  * ``Uuid``  renders ``UUID`` on PostgreSQL and ``CHAR(32)`` on SQLite.
  * ``Money`` renders ``NUMERIC(10,2)`` on PostgreSQL and ``FLOAT`` on SQLite,
    because SQLite has no native fixed-point type and round-tripping ``Decimal``
    through it costs more than it buys us for a recommender.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    Numeric,
    String,
    Text,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# ── Dialect-aware column types ───────────────────────────────────────────────
Money = Numeric(10, 2).with_variant(Float(), "sqlite")
Score = Numeric(12, 6).with_variant(Float(), "sqlite")
JsonBlob = JSON().with_variant(JSONB(), "postgresql")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> uuid.UUID:
    return uuid.uuid4()


class Base(DeclarativeBase):
    """Declarative base for every TESSCH table."""


# ── Catalogue ────────────────────────────────────────────────────────────────
class Product(Base):
    __tablename__ = "products"

    product_id: Mapped[str] = mapped_column(String(10), primary_key=True)
    product_name: Mapped[str] = mapped_column(String(200), nullable=False)
    category: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    upper_type: Mapped[str | None] = mapped_column(String(50))
    color: Mapped[str | None] = mapped_column(String(50))
    color_family: Mapped[str | None] = mapped_column(String(20), index=True)
    style: Mapped[str | None] = mapped_column(String(50), index=True)
    price: Mapped[float] = mapped_column(Money, nullable=False)
    material: Mapped[str | None] = mapped_column(String(100))
    collection: Mapped[str | None] = mapped_column(String(50), index=True)
    description: Mapped[str | None] = mapped_column(Text)
    availability: Mapped[bool] = mapped_column(Boolean, default=True)
    image_url: Mapped[str | None] = mapped_column(String(300))
    #: True for the 25 curated real TESSCH products; False for generated filler
    #: used to keep the user-item matrix sparse. The storefront only shows
    #: is_core=True products so visitors never see fake names.
    is_core: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Product {self.product_id} {self.product_name!r}>"


# ── People ───────────────────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    email: Mapped[str | None] = mapped_column(String(200), unique=True)
    age_group: Mapped[str | None] = mapped_column(String(20))
    gender: Mapped[str | None] = mapped_column(String(20))

    #: What the user told us at signup. Deliberately noisy — real self-reported
    #: preferences disagree with real behaviour ~30% of the time. This is a
    #: legitimate cold-start feature; the latent archetype that actually drives
    #: behaviour is *not* stored here (see data/raw_events/ground_truth/).
    style_preference: Mapped[str | None] = mapped_column(String(50))

    device_preference: Mapped[str | None] = mapped_column(String(20))
    is_new_user: Mapped[bool] = mapped_column(Boolean, default=True)
    signup_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    last_active_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    sessions: Mapped[list["Session"]] = relationship(back_populates="user")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<User {self.user_id}>"


# ── Behaviour ────────────────────────────────────────────────────────────────
class Session(Base):
    """One continuous browsing visit.

    The rollup counters are denormalised on purpose: session-level features are
    read on every recommendation request, and recomputing them from the event
    log per request would dominate API latency.
    """

    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String(30), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(20), ForeignKey("users.user_id"), nullable=False, index=True
    )
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_seconds: Mapped[int | None] = mapped_column(Integer)

    events_count: Mapped[int] = mapped_column(Integer, default=0)
    products_viewed: Mapped[int] = mapped_column(Integer, default=0)
    unique_products: Mapped[int] = mapped_column(Integer, default=0)
    repeat_views: Mapped[int] = mapped_column(Integer, default=0)
    wishlist_count: Mapped[int] = mapped_column(Integer, default=0)
    cart_count: Mapped[int] = mapped_column(Integer, default=0)
    purchase_count: Mapped[int] = mapped_column(Integer, default=0)
    search_count: Mapped[int] = mapped_column(Integer, default=0)

    device_type: Mapped[str | None] = mapped_column(String(20))
    #: Simulator label ('casual_browse' | 'focused_research' | ...). Not
    #: available at serving time on a real site — analysis only.
    intent: Mapped[str | None] = mapped_column(String(30))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    user: Mapped["User"] = relationship(back_populates="sessions")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Session {self.session_id} user={self.user_id}>"


class UserEvent(Base):
    """The raw event log — the single source of truth for the whole system.

    Everything downstream (features, matrices, models, metrics) is derived from
    this table. Nothing is ever written back into it.
    """

    __tablename__ = "user_events"

    event_id: Mapped[uuid.UUID] = mapped_column(Uuid(), primary_key=True, default=_new_uuid)
    user_id: Mapped[str] = mapped_column(
        String(20), ForeignKey("users.user_id"), nullable=False
    )
    product_id: Mapped[str | None] = mapped_column(
        String(10), ForeignKey("products.product_id")
    )
    session_id: Mapped[str | None] = mapped_column(
        String(30), ForeignKey("sessions.session_id")
    )
    event_type: Mapped[str] = mapped_column(String(30), nullable=False)

    #: Dwell time, only meaningful for ``product_view``.
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    #: Where on the site this happened — see ``constants.SURFACES``.
    surface: Mapped[str | None] = mapped_column(String(30))
    #: 1-based slot for impressions/clicks in a grid; drives position-bias work.
    position: Mapped[int | None] = mapped_column(Integer)
    search_query: Mapped[str | None] = mapped_column(String(300))
    #: Purchase amount, or any other numeric payload.
    event_value: Mapped[float | None] = mapped_column(Money)

    #: ``'live'`` for events tracked from a real browser, ``'simulated'`` for
    #: events produced by the generator. Every report that quotes a number has
    #: to be able to say which of the two it came from, so this is a stored
    #: column rather than something inferred from user_id patterns later.
    source: Mapped[str] = mapped_column(String(12), default="simulated", index=True)

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    event_metadata: Mapped[dict | None] = mapped_column("metadata", JsonBlob, default=dict)

    __table_args__ = (
        Index("idx_user_events_user_id", "user_id"),
        Index("idx_user_events_product_id", "product_id"),
        Index("idx_user_events_session_id", "session_id"),
        Index("idx_user_events_event_type", "event_type"),
        Index("idx_user_events_timestamp", "timestamp"),
        Index("idx_user_events_user_product", "user_id", "product_id"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<UserEvent {self.event_type} {self.user_id}->{self.product_id}>"


# ── Derived features ─────────────────────────────────────────────────────────
class UserItemFeature(Base):
    """Pre-computed behavioural features for one (user, product) pair.

    Written by the Stage 2 feature pipeline, read by the models and the API.
    """

    __tablename__ = "user_item_features"

    user_id: Mapped[str] = mapped_column(
        String(20), ForeignKey("users.user_id"), primary_key=True
    )
    product_id: Mapped[str] = mapped_column(
        String(10), ForeignKey("products.product_id"), primary_key=True
    )

    impression_count: Mapped[int] = mapped_column(Integer, default=0)
    total_views: Mapped[int] = mapped_column(Integer, default=0)
    total_time_spent: Mapped[float] = mapped_column(Float, default=0.0)
    avg_dwell_time: Mapped[float] = mapped_column(Float, default=0.0)
    max_dwell_time: Mapped[float] = mapped_column(Float, default=0.0)
    repeat_view_count: Mapped[int] = mapped_column(Integer, default=0)
    distinct_sessions: Mapped[int] = mapped_column(Integer, default=0)
    hover_count: Mapped[int] = mapped_column(Integer, default=0)
    click_count: Mapped[int] = mapped_column(Integer, default=0)
    wishlist_count: Mapped[int] = mapped_column(Integer, default=0)
    cart_count: Mapped[int] = mapped_column(Integer, default=0)
    purchase_count: Mapped[int] = mapped_column(Integer, default=0)
    share_count: Mapped[int] = mapped_column(Integer, default=0)

    days_since_last_view: Mapped[float | None] = mapped_column(Float)
    days_since_last_interaction: Mapped[float | None] = mapped_column(Float)
    days_since_last_purchase: Mapped[float | None] = mapped_column(Float)
    first_interaction_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_interaction_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    raw_interest_score: Mapped[float] = mapped_column(Score, default=0.0)
    time_decayed_score: Mapped[float] = mapped_column(Score, default=0.0)

    #: Supervised label for the ranking model (Stage 4). Set by the labelling
    #: step, which uses a strictly later time window than the features.
    target: Mapped[int] = mapped_column(Integer, default=0)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    __table_args__ = (
        Index("idx_uif_user_id", "user_id"),
        Index("idx_uif_product_id", "product_id"),
        Index("idx_uif_score", "time_decayed_score"),
    )


# ── Serving + outcomes ───────────────────────────────────────────────────────
class RecommendationLog(Base):
    """Every recommendation the system has ever served.

    This is what makes online evaluation possible at all: without a record of
    what was shown, in what order, by which model, a later click tells you
    nothing.
    """

    __tablename__ = "recommendation_logs"

    log_id: Mapped[uuid.UUID] = mapped_column(Uuid(), primary_key=True, default=_new_uuid)
    user_id: Mapped[str] = mapped_column(
        String(20), ForeignKey("users.user_id"), nullable=False
    )
    product_id: Mapped[str] = mapped_column(
        String(10), ForeignKey("products.product_id"), nullable=False
    )
    session_id: Mapped[str | None] = mapped_column(
        String(30), ForeignKey("sessions.session_id")
    )
    recommendation_rank: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[float | None] = mapped_column(Score)
    model_name: Mapped[str] = mapped_column(String(50), nullable=False)
    model_version: Mapped[str] = mapped_column(String(20), default="1.0")
    #: A/B bucket, so a later analysis can compare arms without guessing.
    experiment_group: Mapped[str | None] = mapped_column(String(20))

    was_clicked: Mapped[bool] = mapped_column(Boolean, default=False)
    was_carted: Mapped[bool] = mapped_column(Boolean, default=False)
    was_purchased: Mapped[bool] = mapped_column(Boolean, default=False)
    served_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    interacted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("idx_rec_logs_user_id", "user_id"),
        Index("idx_rec_logs_product_id", "product_id"),
        Index("idx_rec_logs_served_at", "served_at"),
    )


class Order(Base):
    __tablename__ = "orders"

    order_id: Mapped[uuid.UUID] = mapped_column(Uuid(), primary_key=True, default=_new_uuid)
    user_id: Mapped[str] = mapped_column(
        String(20), ForeignKey("users.user_id"), nullable=False, index=True
    )
    session_id: Mapped[str | None] = mapped_column(
        String(30), ForeignKey("sessions.session_id")
    )
    product_id: Mapped[str] = mapped_column(
        String(10), ForeignKey("products.product_id"), nullable=False, index=True
    )
    quantity: Mapped[int] = mapped_column(Integer, default=1)
    unit_price: Mapped[float] = mapped_column(Money, nullable=False)
    total_amount: Mapped[float] = mapped_column(Money, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="completed")
    ordered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


ALL_TABLES = (
    Product,
    User,
    Session,
    UserEvent,
    UserItemFeature,
    RecommendationLog,
    Order,
)

__all__ = [
    "Base",
    "Product",
    "User",
    "Session",
    "UserEvent",
    "UserItemFeature",
    "RecommendationLog",
    "Order",
    "ALL_TABLES",
]
