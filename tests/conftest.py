"""Pytest fixtures for TESSCH recommendation engine test suite."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import pytest

from backend.database import session
from backend.database.models import Base
from recommendation.matrix import UserItemMatrix
from recommendation.similarity import fit_item_similarity, ItemSimilarityModel


@pytest.fixture
def dummy_matrix() -> UserItemMatrix:
    """3 users, 4 products fixture for similarity testing."""
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


@pytest.fixture
def sample_events() -> pd.DataFrame:
    """Sample user events for matrix testing."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return pd.DataFrame(
        [
            {"user_id": "u1", "product_id": "T001", "seconds": 100.0, "timestamp": now},
            {"user_id": "u1", "product_id": "T002", "seconds": 50.0, "timestamp": now},
            {"user_id": "u2", "product_id": "T002", "seconds": 20.0, "timestamp": now},
            {"user_id": "u2", "product_id": "T003", "seconds": 80.0, "timestamp": now},
            {"user_id": "u3", "product_id": "T001", "seconds": 30.0, "timestamp": now},
        ]
    )


@pytest.fixture
def eval_catalog() -> pd.DataFrame:
    """Sample catalog for coherence evaluation testing."""
    return pd.DataFrame(
        [
            {"product_id": "T001", "product_name": "Shoe 1", "category": "Athletic", "style": "Running", "color": "Black"},
            {"product_id": "T002", "product_name": "Shoe 2", "category": "Athletic", "style": "Running", "color": "White"},
            {"product_id": "T003", "product_name": "Shoe 3", "category": "Casual", "style": "Slip-on", "color": "Black"},
            {"product_id": "T004", "product_name": "Shoe 4", "category": "Boots", "style": "Tactical", "color": "Brown"},
        ]
    )


@pytest.fixture
def eval_model() -> ItemSimilarityModel:
    """Sample fitted item similarity model for evaluation testing."""
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
