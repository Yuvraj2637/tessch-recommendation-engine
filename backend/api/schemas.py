"""Request and response shapes for the HTTP API.

Pydantic does the validation that is about *form* — is this field a number, is it
present, is it in range — and ``backend/services/ingest.py`` does the validation
that is about *meaning* — is this product real, is this dwell time plausible.
Keeping the two separate means the meaning-level rules are testable without
constructing an HTTP request, and they stay enforced no matter which entry point
writes the event.

The examples on each model are not decoration. FastAPI renders them into the
interactive ``/docs`` page, which is the demo surface for this project: someone
can open it, hit "Try it out", and exercise the whole system without writing a
line of client code.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ── Tracking ────────────────────────────────────────────────────────────────
class ViewEventIn(BaseModel):
    """One dwell-time observation, as posted by the browser tracker.

    ``duration_seconds`` has a permissive upper bound here (24 hours) on purpose.
    The strict ceiling lives in ``IngestSettings`` where it is configurable and
    where exceeding it *clamps* rather than rejects. Rejecting at the schema layer
    would throw away a real interaction because of a bad clock, and would make the
    limit un-tunable without a code change.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "anon_3f8a91c4d2b7",
                "product_id": "T064",
                "duration_seconds": 47.3,
                "timestamp": "2026-08-26T10:15:00Z",
                "surface": "product_page",
            }
        }
    )

    user_id: str = Field(
        ...,
        max_length=20,
        description="Anonymous browser id, of the form anon_<8-16 hex chars>.",
    )
    product_id: str = Field(..., max_length=10, description="Catalogue id, e.g. T064.")
    duration_seconds: float = Field(
        ...,
        ge=0,
        le=86_400,
        description="Seconds the product page was in view.",
    )
    timestamp: datetime | None = Field(
        default=None,
        description="Client time of the view. Server time is used if absent or implausible.",
    )
    surface: str | None = Field(
        default=None,
        max_length=30,
        description="Where the visit came from: product_page, recommendation, search.",
    )
    position: int | None = Field(
        default=None,
        ge=0,
        description="Slot index if the product was reached from a list. Used to reason about exposure bias.",
    )


class ViewEventOut(BaseModel):
    """Acknowledgement for a stored view."""

    status: Literal["recorded"] = "recorded"
    event_id: str
    user_id: str
    product_id: str
    duration_seconds: float
    clamped: bool = Field(
        default=False,
        description="True if the reading exceeded the configured ceiling and was capped.",
    )
    user_created: bool = Field(
        default=False, description="True if this was the visitor's first tracked event."
    )


class ViewEventBatchIn(BaseModel):
    """Several views at once.

    The tracker sends one event per page, but a batch endpoint matters for the
    case where a visitor closes several tabs at once or comes back online after
    losing connectivity — without it those queued events would each need their own
    request and some would be lost on unload.
    """

    events: list[ViewEventIn] = Field(..., min_length=1, max_length=100)


class ViewEventBatchOut(BaseModel):
    accepted: int
    rejected: int
    clamped: int
    errors: list[str] = Field(
        default_factory=list,
        description="One client-safe reason per rejected event. Partial success is normal.",
    )


# ── Recommendations ─────────────────────────────────────────────────────────
class ProductCard(BaseModel):
    """A product as the storefront renders it.

    ``score`` is nullable because the popularity fallback has no model score to
    report, and reporting a fabricated one would make the two strategies
    indistinguishable to anyone reading the response.
    """

    model_config = ConfigDict(extra="allow")

    product_id: str
    product_name: str
    category: str | None = None
    style: str | None = None
    color: str | None = None
    color_family: str | None = None
    price: float | None = None
    material: str | None = None
    collection: str | None = None
    availability: bool | None = None
    image_url: str | None = None
    rank: int | None = None
    score: float | None = None
    total_seconds: float | None = None
    view_count: int | None = None


class RecommendationsOut(BaseModel):
    """A recommendation list, with the provenance needed to explain it.

    ``strategy`` is the field that keeps this honest. A caller can always tell
    whether it received genuine personalisation, behavioural similarity, or a
    popularity fallback — so the UI can label the section truthfully rather than
    claiming personalisation the system did not perform.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": [
                    {
                        "product_id": "T057",
                        "product_name": "Quarry Grey",
                        "category": "Sneaker",
                        "price": 189.0,
                        "rank": 1,
                        "score": 0.619,
                    }
                ],
                "strategy": "item_item_cosine",
                "model_name": "item_item_cosine",
                "model_version": "1.0",
                "anchor_product_id": "T064",
                "reason": "Visitors who viewed Ledger Rust also viewed these.",
                "count": 1,
                "meta": {"backend": "sklearn", "n_users_in_model": 33},
            }
        }
    )

    items: list[ProductCard]
    strategy: Literal["personal_dwell", "item_item_cosine", "popular", "empty"]
    model_name: str
    model_version: str
    anchor_product_id: str | None = None
    reason: str | None = Field(
        default=None, description="Human-readable justification for display."
    )
    count: int
    meta: dict[str, Any] = Field(default_factory=dict)


# ── Catalogue ───────────────────────────────────────────────────────────────
class ProductListOut(BaseModel):
    items: list[ProductCard]
    count: int
    total: int


# ── Ops ─────────────────────────────────────────────────────────────────────
class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    database: bool
    products: int
    view_events: int
    similarity_model: bool
    sklearn: bool = Field(
        description="Whether scikit-learn is installed. False means the NumPy cosine fallback is in use."
    )


class ProvenanceOut(BaseModel):
    """Real-vs-simulated split. Backs the honesty claim in the results write-up.

    Every number this project reports about its own performance has to be
    attributable to one of these two sources, so it is exposed as an endpoint
    rather than computed by hand for the README.
    """

    live: dict[str, int]
    simulated: dict[str, int]
    total_views: int
    live_share: float


class ModelInfoOut(BaseModel):
    trained: bool
    backend: str | None = None
    normalisation: str | None = None
    n_users: int = 0
    n_products: int = 0
    observed_cells: int = 0
    products_with_neighbours: int = 0
    cold_products: int = 0
    coverage: float = 0.0


__all__ = [
    "ViewEventIn",
    "ViewEventOut",
    "ViewEventBatchIn",
    "ViewEventBatchOut",
    "ProductCard",
    "RecommendationsOut",
    "ProductListOut",
    "HealthOut",
    "ProvenanceOut",
    "ModelInfoOut",
]
