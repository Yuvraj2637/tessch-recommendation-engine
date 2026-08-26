"""Tests for Phase 3 item-item cosine similarity model."""

from __future__ import annotations

import numpy as np

from recommendation.matrix import UserItemMatrix
from recommendation.similarity import (
    SKLEARN_AVAILABLE,
    ItemSimilarityModel,
    cosine_similarity_matrix,
    fit_item_similarity,
)


def dummy_matrix() -> UserItemMatrix:
    # 3 users, 4 products
    # u0 browses p0, p1
    # u1 browses p0, p1, p2
    # u2 browses p2, p3
    # p0 and p1 should have high similarity; p0 and p3 should have zero similarity
    values = np.array(
        [
            [0.7071, 0.7071, 0.0, 0.0],
            [0.5773, 0.5773, 0.5773, 0.0],
            [0.0, 0.0, 0.7071, 0.7071],
        ],
        dtype="float64",
    )
    return UserItemMatrix(
        values=values,
        user_ids=("u0", "u1", "u2"),
        product_ids=("T001", "T002", "T003", "T004"),
        normalisation="l2",
        observed_cells=int((values > 0).sum()),
    )


def test_cosine_similarity_properties(dummy_matrix: UserItemMatrix):
    """Test mathematical properties: diagonal is 1.0, symmetric, and bounded [0, 1]."""
    sim = cosine_similarity_matrix(dummy_matrix.values)
    n = dummy_matrix.n_products

    # Shape
    assert sim.shape == (n, n)

    # Diagonal is 1.0
    np.testing.assert_allclose(np.diag(sim), np.ones(n))

    # Symmetry
    np.testing.assert_allclose(sim, sim.T)

    # Range [0, 1]
    assert np.all(sim >= 0.0)
    assert np.all(sim <= 1.0)

    # T001 and T002 have non-zero similarity
    assert sim[0, 1] > 0.95
    # T001 and T004 have zero co-viewing overlap
    assert sim[0, 3] == 0.0


def test_sklearn_vs_numpy_agreement(dummy_matrix: UserItemMatrix):
    """Verify Scikit-learn and NumPy cosine implementations yield identical values."""
    sim_numpy = cosine_similarity_matrix(dummy_matrix.values, use_sklearn=False)
    if SKLEARN_AVAILABLE:
        sim_sklearn = cosine_similarity_matrix(dummy_matrix.values, use_sklearn=True)
        np.testing.assert_allclose(sim_numpy, sim_sklearn, rtol=1e-5, atol=1e-5)


def test_fit_and_query_model(dummy_matrix: UserItemMatrix):
    """Test model fitting, similar_to queries, and exclusion logic."""
    model = fit_item_similarity(dummy_matrix)
    assert model.knows("T001")
    assert not model.knows("T999")
    assert model.has_neighbours("T001")

    # Query neighbours for T001
    similar = model.similar_to("T001", top_k=2)
    assert not similar.empty
    assert similar.iloc[0]["product_id"] == "T002"

    # Exclude T002
    similar_ex = model.similar_to("T001", top_k=2, exclude={"T002"})
    assert "T002" not in list(similar_ex["product_id"])


def test_cold_product():
    """Unviewed product has zero similarities and returns empty similar_to."""
    values = np.array([[1.0, 0.0], [1.0, 0.0]], dtype="float64")
    matrix = UserItemMatrix(
        values=values,
        user_ids=("u0", "u1"),
        product_ids=("T001", "T002_cold"),
        normalisation="l2",
        observed_cells=int((values > 0).sum()),
    )
    model = fit_item_similarity(matrix)
    assert not model.has_neighbours("T002_cold")
    assert model.similar_to("T002_cold").empty
