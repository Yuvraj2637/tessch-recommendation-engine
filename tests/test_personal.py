"""Tests for Phase 2 personalized dwell-time recommendations."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from recommendation import personal


def test_rank_for_user_empty():
    """Empty browsing history produces an empty ranking DataFrame."""
    empty_df = pd.DataFrame(
        columns=["product_id", "seconds", "timestamp"]
    )
    result = personal.rank_for_user(empty_df, top_k=5)
    assert result.empty
    assert list(result.columns) == ["product_id", "total_seconds", "view_count", "score", "last_seen", "rank"]


def test_rank_for_user_single_product():
    """Single product view produces ranking with rank 1."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    df = pd.DataFrame(
        [
            {
                "product_id": "T001",
                "seconds": 45.0,
                "timestamp": now,
            }
        ]
    )
    result = personal.rank_for_user(df, top_k=5)
    assert len(result) == 1
    assert result.iloc[0]["product_id"] == "T001"
    assert result.iloc[0]["rank"] == 1
    assert result.iloc[0]["total_seconds"] == 45.0
    assert result.iloc[0]["view_count"] == 1


def test_rank_for_user_multi_product_ordering():
    """Products are ranked descending by accumulated dwell score."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    df = pd.DataFrame(
        [
            {
                "product_id": "T001",
                "seconds": 20.0,
                "timestamp": now,
            },
            {
                "product_id": "T002",
                "seconds": 80.0,
                "timestamp": now,
            },
            {
                "product_id": "T003",
                "seconds": 50.0,
                "timestamp": now,
            },
        ]
    )
    result = personal.rank_for_user(df, top_k=2)
    assert len(result) == 2
    assert list(result["product_id"]) == ["T002", "T003"]
    assert list(result["rank"]) == [1, 2]
