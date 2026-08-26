"""The serving layer — turning stored events into a list of products to show.

This module holds the decisions that only exist because something is being served
to a real visitor rather than computed in a notebook: what to do when there is no
data yet, how to attach product details to a bare list of ids, and when to rebuild
the similarity model.

**Cold start is handled here, not in the algorithms.** ``recommendation/`` returns
an empty frame when it has nothing to say, which is correct and testable. But an
empty homepage section is not an acceptable answer to a visitor, so every entry
point in this file falls back to popularity and — importantly — reports which
strategy it used. A recommendation that cannot say where it came from is
impossible to debug and impossible to honestly describe.

**The similarity model is cached in memory.** Fitting takes a few milliseconds at
this catalogue size, but it reads the entire event log to do it, so refitting per
request would make every product page a full table scan. The cache is keyed on the
event count so that new tracked views eventually get reflected without a manual
rebuild. That is a deliberate simplification and its limits are worth naming: it
is per-process, so multiple workers each hold their own copy, and it will not
notice edits that leave the row count unchanged. At this scale that is the right
trade; a real deployment would rebuild the artifact on a schedule and load it from
``artifacts/`` instead, which ``ItemSimilarityModel.save``/``load`` already
supports and ``scripts/build_similarity.py`` already produces.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from backend.core.config import get_settings
from backend.database import queries
from recommendation import personal, similarity as sim_mod
from recommendation.matrix import build_user_item_matrix

Strategy = Literal["personal_dwell", "item_item_cosine", "popular", "empty"]

#: Users must have touched at least this many distinct products before their row
#: contributes to the item-item matrix. A single-product user's row contributes
#: exactly zero to every product pair's dot product, so including them only
#: inflates the apparent user count.
MIN_USER_PRODUCTS_FOR_MATRIX = 2

#: Dwell readings below this are excluded from model training. Matches the ingest
#: floor: rows written before that floor existed, or by the simulator, should be
#: filtered on read rather than trusted.
MIN_TRAINING_SECONDS = 1.0


@dataclass
class RecommendationResponse:
    """A served list, plus the provenance needed to explain it."""

    items: list[dict[str, Any]]
    strategy: Strategy
    model_name: str
    model_version: str
    anchor_product_id: str | None = None
    reason: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "items": self.items,
            "strategy": self.strategy,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "anchor_product_id": self.anchor_product_id,
            "reason": self.reason,
            "count": len(self.items),
            "meta": self.meta,
        }


# ── Product enrichment ──────────────────────────────────────────────────────
#: Fields the storefront needs to render a card. Explicit rather than "everything"
#: so that adding an internal column to `products` cannot accidentally start
#: publishing it through the API.
CARD_FIELDS = (
    "product_id",
    "product_name",
    "category",
    "style",
    "color",
    "color_family",
    "price",
    "material",
    "collection",
    "availability",
    "image_url",
)


def product_cards(
    products: pd.DataFrame, product_ids: list[str]
) -> list[dict[str, Any]]:
    """Look up display fields for a ranked list, preserving the ranking order.

    Order matters and pandas will not preserve it for you: a naive
    ``products[products.product_id.isin(ids)]`` returns catalogue order, silently
    discarding the ranking this whole system exists to compute.
    """
    if not product_ids:
        return []
    available = [c for c in CARD_FIELDS if c in products.columns]
    indexed = products.set_index("product_id")[
        [c for c in available if c != "product_id"]
    ]
    cards: list[dict[str, Any]] = []
    for pid in product_ids:
        if pid not in indexed.index:
            continue
        row = indexed.loc[pid]
        card: dict[str, Any] = {"product_id": pid}
        for column in row.index:
            card[column] = _jsonable(row[column])
        cards.append(card)
    return cards


def _jsonable(value: Any) -> Any:
    """Convert numpy/pandas scalars to plain Python for JSON serialisation."""
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, (bool,)):
        return value
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            return str(value)
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


# ── Phase 2: personalised ───────────────────────────────────────────────────
def recommend_for_user(
    conn: sqlite3.Connection,
    user_id: str,
    top_k: int | None = None,
    *,
    include_reason: bool = True,
) -> RecommendationResponse:
    """"Picked for you" — the products this visitor gave the most attention to.

    Falls back to popularity for a visitor with no history, which is every
    first-time visitor. The response says which happened, so the frontend can
    label the section honestly instead of implying personalisation that did not
    occur.
    """
    settings = get_settings()
    k = int(top_k or settings.default_top_k)

    history = queries.load_user_history(conn, user_id)
    ranking = personal.rank_for_user(history, top_k=k)

    core_df = queries.load_products(conn, is_core_only=True)
    core_ids = set(core_df["product_id"])
    if not ranking.empty:
        ranking = ranking[ranking["product_id"].isin(core_ids)]

    if ranking.empty:
        return _popular_fallback(
            conn,
            k,
            reason="No browsing history yet — showing what other visitors view most.",
        )

    products = queries.load_products(conn)
    ordered = [str(p) for p in ranking["product_id"]]
    cards = product_cards(products, ordered)

    scores = dict(zip(ranking["product_id"], ranking["score"]))
    seconds = dict(zip(ranking["product_id"], ranking["total_seconds"]))
    views = dict(zip(ranking["product_id"], ranking["view_count"]))
    for rank, card in enumerate(cards, start=1):
        pid = card["product_id"]
        card["rank"] = rank
        card["score"] = round(float(scores.get(pid, 0.0)), 4)
        card["total_seconds"] = round(float(seconds.get(pid, 0.0)), 1)
        card["view_count"] = int(views.get(pid, 0))

    anchor = personal.anchor_product(ranking)
    reason = None
    if include_reason and anchor:
        name = _product_name(products, anchor)
        total = seconds.get(anchor, 0.0)
        reason = f"You spent {_humanise_seconds(total)} looking at {name}."

    return RecommendationResponse(
        items=cards,
        strategy="personal_dwell",
        model_name=personal.MODEL_NAME,
        model_version=personal.MODEL_VERSION,
        anchor_product_id=anchor,
        reason=reason,
        meta={"history_events": int(len(history))},
    )


def _product_name(products: pd.DataFrame, product_id: str) -> str:
    match = products.loc[products["product_id"] == product_id, "product_name"]
    return str(match.iloc[0]) if len(match) else product_id


def _humanise_seconds(seconds: float) -> str:
    """"3 minutes" reads better than "187.4 seconds" in user-facing copy."""
    total = int(round(seconds))
    if total < 90:
        return f"{total} seconds"
    minutes = total / 60.0
    if minutes < 10:
        return f"{minutes:.1f} minutes".replace(".0 ", " ")
    return f"{int(round(minutes))} minutes"


# ── Phase 3: similar products ───────────────────────────────────────────────
def recommend_similar(
    conn: sqlite3.Connection,
    product_id: str,
    top_k: int | None = None,
    *,
    exclude_seen_by: str | None = None,
) -> RecommendationResponse:
    """"You might also like" — products that attract the same attention.

    ``exclude_seen_by`` optionally drops products that user has already viewed.
    Off by default: on a sparse catalogue it can empty the list entirely, and on
    a product page a familiar item alongside a new one is reassuring rather than
    redundant.
    """
    settings = get_settings()
    k = int(top_k or settings.default_top_k)

    model = get_similarity_model(conn)
    core_df = queries.load_products(conn, is_core_only=True)
    core_ids = set(core_df["product_id"])
    non_core = (set(model.product_ids) - core_ids) if model is not None else set()

    exclude: set[str] = set()
    if exclude_seen_by:
        history = queries.load_user_history(conn, exclude_seen_by)
        if not history.empty:
            exclude = {str(p) for p in history["product_id"].unique()}

    neighbours = (
        model.similar_to(product_id, top_k=k, exclude=exclude | non_core)
        if model is not None
        else pd.DataFrame()
    )

    if neighbours.empty:
        return _popular_fallback(
            conn,
            k,
            reason=(
                "Not enough viewing data for this product yet — "
                "showing popular products instead."
            ),
            exclude={product_id} | exclude,
        )

    products = queries.load_products(conn)
    cards = product_cards(products, [str(p) for p in neighbours["product_id"]])
    scores = dict(zip(neighbours["product_id"], neighbours["score"]))
    for rank, card in enumerate(cards, start=1):
        card["rank"] = rank
        card["score"] = round(float(scores.get(card["product_id"], 0.0)), 4)

    return RecommendationResponse(
        items=cards,
        strategy="item_item_cosine",
        model_name=sim_mod.MODEL_NAME,
        model_version=sim_mod.MODEL_VERSION,
        anchor_product_id=product_id,
        reason=f"Visitors who viewed {_product_name(products, product_id)} also viewed these.",
        meta={
            "backend": model.backend,
            "n_users_in_model": model.n_users,
            "normalisation": model.normalisation,
        },
    )


# ── Fallback ────────────────────────────────────────────────────────────────
def _popular_fallback(
    conn: sqlite3.Connection,
    top_k: int,
    reason: str,
    exclude: set[str] | None = None,
) -> RecommendationResponse:
    """Most-viewed products. The answer when personalisation has nothing to say.

    Not a real recommender, and labelled as such via ``strategy``. It exists
    because a visitor with no history still needs a populated page, and because
    it doubles as the non-personalised baseline any comparison needs.
    """
    # Over-fetch so that removing excluded ids still leaves a full list.
    popular = queries.popular_products(conn, limit=top_k + len(exclude or ()) + 4)
    ids = [str(p) for p in popular["product_id"]] if not popular.empty else []
    if exclude:
        ids = [p for p in ids if p not in exclude]
    ids = ids[:top_k]

    if not ids:
        return RecommendationResponse(
            items=[],
            strategy="empty",
            model_name="none",
            model_version="0",
            reason="No viewing data recorded yet.",
        )

    products = queries.load_products(conn)
    cards = product_cards(products, ids)
    view_counts = (
        dict(zip(popular["product_id"], popular["viewer_count"]))
        if "viewer_count" in popular.columns
        else {}
    )
    for rank, card in enumerate(cards, start=1):
        card["rank"] = rank
        card["score"] = None
        card["view_count"] = int(view_counts.get(card["product_id"], 0) or 0)

    return RecommendationResponse(
        items=cards,
        strategy="popular",
        model_name="popularity",
        model_version="1.0",
        reason=reason,
    )


# ── Model cache ─────────────────────────────────────────────────────────────
_model_lock = threading.Lock()
_model_cache: dict[str, Any] = {"key": None, "model": None}


def _cache_key(conn: sqlite3.Connection) -> int:
    """Number of usable dwell events — cheap to compute, changes when data does."""
    row = conn.execute(
        """
        SELECT COUNT(*) FROM user_events
         WHERE event_type = 'product_view'
           AND duration_seconds IS NOT NULL
           AND duration_seconds >= ?
        """,
        (MIN_TRAINING_SECONDS,),
    ).fetchone()
    return int(row[0]) if row else 0


def get_similarity_model(
    conn: sqlite3.Connection, force_rebuild: bool = False
) -> sim_mod.ItemSimilarityModel | None:
    """Return the fitted item-item model, refitting only when the data changed.

    Returns None when there is not enough data to fit anything, which callers
    must treat as "fall back", not as an error.
    """
    key = _cache_key(conn)
    with _model_lock:
        if not force_rebuild and _model_cache["key"] == key and _model_cache["model"]:
            return _model_cache["model"]

        model = build_similarity_model(conn)
        _model_cache["key"] = key
        _model_cache["model"] = model
        return model


def build_similarity_model(
    conn: sqlite3.Connection,
) -> sim_mod.ItemSimilarityModel | None:
    """Fit the item-item model from everything currently in the event log."""
    events = queries.load_dwell_events(conn, min_seconds=MIN_TRAINING_SECONDS)
    if events.empty:
        return None

    catalogue = queries.catalog_ids(conn)
    matrix = build_user_item_matrix(
        events,
        product_ids=catalogue,
        normalise="l2",
        min_user_products=MIN_USER_PRODUCTS_FOR_MATRIX,
    )
    if matrix.n_users == 0:
        return None
    return sim_mod.fit_item_similarity(matrix)


def invalidate_model_cache() -> None:
    """Drop the cached model. Used by tests and by the rebuild endpoint."""
    with _model_lock:
        _model_cache["key"] = None
        _model_cache["model"] = None


__all__ = [
    "Strategy",
    "RecommendationResponse",
    "CARD_FIELDS",
    "product_cards",
    "recommend_for_user",
    "recommend_similar",
    "get_similarity_model",
    "build_similarity_model",
    "invalidate_model_cache",
]
