"""Tests for evaluation metrics: coherence, Gini inequality, and stability."""

from __future__ import annotations

import numpy as np
import pandas as pd

from recommendation import evaluate as ev
from recommendation.matrix import UserItemMatrix
from recommendation.similarity import fit_item_similarity


def eval_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"product_id": "T001", "product_name": "Shoe 1", "category": "Athletic", "style": "Running", "color": "Black"},
            {"product_id": "T002", "product_name": "Shoe 2", "category": "Athletic", "style": "Running", "color": "White"},
            {"product_id": "T003", "product_name": "Shoe 3", "category": "Casual", "style": "Slip-on", "color": "Black"},
            {"product_id": "T004", "product_name": "Shoe 4", "category": "Boots", "style": "Tactical", "color": "Brown"},
        ]
    )


def eval_model() -> fit_item_similarity:
    # T001 is similar to T002 (both Athletic)
    values = np.array(
        [
            [0.7071, 0.7071, 0.0, 0.0],
            [0.6, 0.8, 0.0, 0.0],
            [0.0, 0.0, 0.7071, 0.7071],
        ],
        dtype="float64",
    )
    matrix = UserItemMatrix(
        values=values,
        user_ids=("u0", "u1", "u2"),
        product_ids=("T001", "T002", "T003", "T004"),
        normalisation="l2",
        observed_cells=int((values > 0).sum()),
    )
    return fit_item_similarity(matrix)


def test_coherence_report(eval_model, eval_catalog: pd.DataFrame):
    """Test category and attribute coherence calculations."""
    report = ev.coherence_report(eval_model, eval_catalog, top_k=2)
    assert not report.empty
    assert "category" in list(report["attribute"])
    cat_row = report[report["attribute"] == "category"].iloc[0]
    assert cat_row["top1_match_rate"] > 0.0
    assert cat_row["lift"] >= 1.0


def test_user_activity_concentration():
    """Test Gini calculation on user dwell time distribution."""
    events = pd.DataFrame(
        [
            {"user_id": "u1", "seconds": 10.0},
            {"user_id": "u2", "seconds": 10.0},
            {"user_id": "u3", "seconds": 10.0},
            {"user_id": "u4", "seconds": 10.0},
        ]
    )
    res = ev.user_activity_concentration(events)
    # Equal distribution has Gini near 0
    assert res["gini"] == 0.0
    assert res["top_user_share"] == 0.25
