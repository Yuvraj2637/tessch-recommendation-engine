"""Catalogue endpoints — what the storefront needs to render pages.

Not part of the recommendation system proper, but the mock storefront has to get
its products from somewhere, and reading them from the same database the
recommender trains on means the demo cannot drift out of sync with the model.
"""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.api import schemas
from backend.api.deps import get_db
from backend.database import queries
from backend.services.recommend import product_cards

router = APIRouter(prefix="/api/products", tags=["catalogue"])


@router.get("", response_model=schemas.ProductListOut, summary="List the catalogue")
def list_products(
    conn: sqlite3.Connection = Depends(get_db),
    category: str | None = Query(default=None, description="Filter by category."),
    available_only: bool = Query(
        default=False, description="Hide out-of-stock products."
    ),
    core_only: bool = Query(
        default=True, description="Show only curated core products."
    ),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> schemas.ProductListOut:
    """Paginated catalogue listing.

    ``core_only`` defaults to True so the storefront displays only the genuine
    curated TESSCH shoes from the real website, keeping generated background
    variants strictly for ML training density.
    """
    products = queries.load_products(
        conn, available_only=available_only, is_core_only=core_only
    )
    if category:
        products = products[products["category"].str.lower() == category.lower()]

    total = int(len(products))
    page = products.iloc[offset : offset + limit]
    cards = product_cards(page, [str(p) for p in page["product_id"]])
    return schemas.ProductListOut(items=cards, count=len(cards), total=total)


@router.get(
    "/{product_id}", response_model=schemas.ProductCard, summary="One product"
)
def get_product(
    product_id: str, conn: sqlite3.Connection = Depends(get_db)
) -> schemas.ProductCard:
    """Fetch a single product.

    This one *does* 404 — unlike the recommendation endpoints. A missing product is
    a genuine error here, because the caller asked for a specific thing that does
    not exist, whereas asking for recommendations about it has a sensible fallback.
    """
    pid = product_id.upper()
    products = queries.load_products(conn)
    cards = product_cards(products, [pid])
    if not cards:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No product with id {pid!r}",
        )
    return schemas.ProductCard(**cards[0])
