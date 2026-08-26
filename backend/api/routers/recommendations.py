"""Phases 2 and 3 — the two recommendation endpoints.

Both return the same envelope, differing only in ``strategy``. That uniformity is
what lets one piece of frontend code render both sections, and it means adding a
third recommender later requires no new response shape.

Neither endpoint returns 404 for an unknown user or a cold product. Both are
normal states — a first-time visitor and a newly-added product — and both are
answered with the popularity fallback plus a ``strategy`` field saying so. Making
the caller handle an error for the most common case in the system would be a
design mistake.
"""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.api import schemas
from backend.api.deps import get_db, resolve_top_k
from backend.services import recommend

router = APIRouter(prefix="/api/recommendations", tags=["recommendations"])


@router.get(
    "/user/{user_id}",
    response_model=schemas.RecommendationsOut,
    summary='Phase 2 — "Picked for you" from this visitor\'s own dwell time',
)
def for_user(
    user_id: str,
    conn: sqlite3.Connection = Depends(get_db),
    top_k: int = Depends(resolve_top_k),
    log: bool = Query(
        default=False,
        description="Also write these impressions to recommendation_logs.",
    ),
) -> schemas.RecommendationsOut:
    """Rank what this visitor spent the most time on, most recent first on ties.

    Returns the popularity fallback with ``strategy="popular"`` when the visitor
    has no history — which is every first visit, and is why this is a fallback
    rather than an error.
    """
    response = recommend.recommend_for_user(conn, user_id, top_k=top_k)

    if log and response.items:
        from backend.database import writes

        writes.log_recommendations(
            conn,
            user_id,
            [
                (item["product_id"], item.get("rank") or i, item.get("score") or 0.0)
                for i, item in enumerate(response.items, start=1)
            ],
            response.model_name,
            response.model_version,
        )

    return schemas.RecommendationsOut(**response.as_dict())


@router.get(
    "/similar/{product_id}",
    response_model=schemas.RecommendationsOut,
    summary='Phase 3 — "You might also like", from pooled cross-user viewing time',
)
def similar_to(
    product_id: str,
    conn: sqlite3.Connection = Depends(get_db),
    top_k: int = Depends(resolve_top_k),
    exclude_seen_by: str | None = Query(
        default=None,
        description="Optional visitor id; drops products that visitor already viewed.",
    ),
) -> schemas.RecommendationsOut:
    """Products whose viewing pattern across all visitors resembles this one's.

    The scores are cosine similarities in [0, 1] between columns of the
    row-normalised user-item matrix. A product nobody has viewed yet has no
    direction to compare, so it falls back to popularity.
    """
    response = recommend.recommend_similar(
        conn, product_id.upper(), top_k=top_k, exclude_seen_by=exclude_seen_by
    )
    return schemas.RecommendationsOut(**response.as_dict())


@router.get(
    "/popular",
    response_model=schemas.RecommendationsOut,
    summary="Non-personalised baseline — most-viewed products",
)
def popular(
    conn: sqlite3.Connection = Depends(get_db),
    top_k: int = Depends(resolve_top_k),
) -> schemas.RecommendationsOut:
    """Most-viewed products overall.

    Exposed as its own endpoint, not just as an internal fallback, because a
    personalised list is only interesting relative to the unpersonalised one. Being
    able to fetch both side by side is what makes "this is actually personalised"
    something a reviewer can check rather than take on trust.
    """
    response = recommend._popular_fallback(
        conn, top_k, reason="Most-viewed products across all visitors."
    )
    return schemas.RecommendationsOut(**response.as_dict())


@router.post(
    "/rebuild",
    response_model=schemas.ModelInfoOut,
    summary="Refit the item-item similarity model now",
)
def rebuild(conn: sqlite3.Connection = Depends(get_db)) -> schemas.ModelInfoOut:
    """Force a refit instead of waiting for the event-count cache key to change.

    Exists for the demo: after driving some browsing through the storefront you
    want the new co-views reflected immediately, and this makes that a single call
    rather than a server restart.
    """
    recommend.invalidate_model_cache()
    model = recommend.get_similarity_model(conn, force_rebuild=True)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Not enough viewing data to fit a similarity model. "
                "At least two users must each have viewed two or more products."
            ),
        )
    return _model_info(model)


@router.get(
    "/model",
    response_model=schemas.ModelInfoOut,
    summary="What the current similarity model knows",
)
def model_info(conn: sqlite3.Connection = Depends(get_db)) -> schemas.ModelInfoOut:
    """Coverage and shape of the trained model.

    ``cold_products`` is the number this system cannot say anything about yet, and
    it is surfaced rather than hidden because it is the honest measure of the
    cold-start limitation the write-up discusses.
    """
    model = recommend.get_similarity_model(conn)
    if model is None:
        return schemas.ModelInfoOut(trained=False)
    return _model_info(model)


def _model_info(model) -> schemas.ModelInfoOut:
    coverage = model.coverage()
    return schemas.ModelInfoOut(
        trained=True,
        backend=model.backend,
        normalisation=model.normalisation,
        n_users=model.n_users,
        n_products=int(coverage["n_products"]),
        observed_cells=model.observed_cells,
        products_with_neighbours=int(coverage["products_with_neighbours"]),
        cold_products=int(coverage["cold_products"]),
        coverage=float(coverage["coverage"]),
    )
