"""Tests for User-Item Matrix construction and normalisation schemes."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from recommendation.matrix import build_user_item_matrix


def sample_events() -> pd.DataFrame:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return pd.DataFrame(
        [
            {"user_id": "u1", "product_id": "T001", "seconds": 100.0, "timestamp": now},
            {"user_id": "u1", "product_id": "T002", "seconds": 50.0, "timestamp": now},
            {"user_id": "u2", "product_id": "T002", "seconds": 20.0, "timestamp": now},
            {"user_id": "u2", "product_id": "T003", "seconds": 80.0, "timestamp": now},
            {"user_id": "u3", "product_id": "T001", "seconds": 30.0, "timestamp": now},  # single product user
        ]
    )


def test_build_user_item_matrix_l2_normalisation(sample_events: pd.DataFrame):
    """Test L2 row normalisation where each user's row vector has Euclidean norm 1."""
    matrix = build_user_item_matrix(
        sample_events,
        product_ids=["T001", "T002", "T003"],
        normalise="l2",
        min_user_products=2,
        compress=False,
    )
    # u3 is filtered out because min_user_products=2
    assert matrix.n_users == 2
    assert matrix.n_products == 3
    assert matrix.user_ids == ("u1", "u2")

    # Each non-zero row should have L2 norm approximately 1.0
    norms = np.linalg.norm(matrix.values, axis=1)
    np.testing.assert_allclose(norms, np.ones(2), rtol=1e-5)


def test_build_user_item_matrix_none_normalisation(sample_events: pd.DataFrame):
    """Test unnormalised matrix retains raw aggregated seconds."""
    matrix = build_user_item_matrix(
        sample_events,
        product_ids=["T001", "T002", "T003"],
        normalise="none",
        min_user_products=1,
        compress=False,
    )
    assert matrix.n_users == 3
    # u1: T001=100, T002=50, T003=0
    np.testing.assert_allclose(matrix.values[0], [100.0, 50.0, 0.0])


def test_build_user_item_matrix_empty():
    """Empty events table returns empty matrix structure without errors."""
    empty_df = pd.DataFrame(columns=["user_id", "product_id", "seconds", "timestamp"])
    matrix = build_user_item_matrix(empty_df, product_ids=["T001", "T002"])
    assert matrix.n_users == 0
    assert matrix.n_products == 2
    assert matrix.values.shape == (0, 2)
