"""Phase 3 — item-item cosine similarity.

Given the user-item matrix, this produces the answer to "which other products
attract the same attention as this one?" — the engine behind "You might also
like".

**Why cosine, and not something else?**

The comparison is between two product columns, each holding one number per user.
Two properties make cosine the right measure for that:

*It ignores magnitude and keeps direction.* A flagship shoe viewed by 200 people
and a niche shoe viewed by 20 can still be recognised as similar if the *pattern*
of who views them agrees. Euclidean distance would call them far apart purely
because one is more popular, which would collapse into recommending bestsellers
to everyone — the exact failure mode a personalisation system exists to avoid.

*It stays bounded and interpretable.* With non-negative dwell scores the result
lands in [0, 1], where 0 means "no user gave attention to both" and 1 means
"identical attention pattern". That makes a similarity score something you can
threshold and reason about, rather than an arbitrary distance.

The main alternative worth naming is Pearson correlation, which additionally
centres each column by its mean. Centring is standard for *explicit* ratings,
where it removes the difference between a harsh and a generous rater. Here the
data is implicit and non-negative: an unviewed product is an absence of evidence,
not a low rating. Subtracting a mean would turn those structural zeros into
negative values and invent dislike where there is only silence.

scikit-learn provides the implementation. A NumPy fallback is included because
this module is otherwise dependency-light and the arithmetic is four lines; both
paths are asserted to agree in ``tests/test_similarity.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from recommendation.matrix import UserItemMatrix

MODEL_NAME = "item_item_cosine"
MODEL_VERSION = "1.0"

try:  # pragma: no cover - exercised by whichever path is installed
    from sklearn.metrics.pairwise import cosine_similarity as _sklearn_cosine

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _sklearn_cosine = None
    SKLEARN_AVAILABLE = False


def cosine_similarity_matrix(values: np.ndarray, use_sklearn: bool = True) -> np.ndarray:
    """Pairwise cosine similarity between the *columns* of ``values``.

    ``values`` is the (n_users, n_products) matrix, so this transposes first:
    we are comparing products to each other, not users.

    Zero columns — products nobody has viewed — have no direction to compare, so
    their similarity is defined as 0 rather than NaN. Returning NaN would
    propagate through any later ``argsort`` and produce silently wrong
    neighbours; returning 0 states the truth, which is that there is no evidence
    of a relationship yet.
    """
    if values.size == 0:
        n = values.shape[1] if values.ndim == 2 else 0
        return np.zeros((n, n), dtype="float64")

    item_vectors = np.ascontiguousarray(values.T, dtype="float64")

    if use_sklearn and SKLEARN_AVAILABLE:
        sim = _sklearn_cosine(item_vectors)
    else:
        sim = _numpy_cosine(item_vectors)

    # Cosine of a vector with itself is 1 by definition, but floating-point
    # accumulation lands on 0.9999999999999998 often enough to matter when the
    # diagonal is used as a correctness check. Pin it, then pin cold items to 0.
    sim = np.clip(sim, -1.0, 1.0)
    np.fill_diagonal(sim, 1.0)

    dead = ~item_vectors.any(axis=1)
    if dead.any():
        sim[dead, :] = 0.0
        sim[:, dead] = 0.0
    return sim


def _numpy_cosine(rows: np.ndarray) -> np.ndarray:
    """Cosine similarity between rows, without scikit-learn."""
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    unit = rows / norms
    return unit @ unit.T


@dataclass(frozen=True)
class ItemSimilarityModel:
    """A trained item-item similarity lookup.

    Small enough to hold in memory and rebuild in milliseconds at this catalogue
    size, so there is no incremental-update machinery here. The honest scaling
    note is that cosine similarity is O(n_products squared) in both time and
    memory: fine for 100 products, a 10,000 x 10,000 float64 matrix (~800 MB) is
    where this design would need approximate nearest neighbours instead.
    """

    similarity: np.ndarray            # (n_products, n_products)
    product_ids: tuple[str, ...]
    normalisation: str = "l2"
    n_users: int = 0
    observed_cells: int = 0
    backend: str = "sklearn"

    def __post_init__(self) -> None:
        n = len(self.product_ids)
        if self.similarity.shape != (n, n):
            raise ValueError(
                f"similarity matrix is {self.similarity.shape}, expected ({n}, {n}) "
                f"to match {n} product ids"
            )

    # ── Lookup ──────────────────────────────────────────────────────────────
    def _index(self) -> dict[str, int]:
        return {pid: i for i, pid in enumerate(self.product_ids)}

    def knows(self, product_id: str) -> bool:
        return product_id in set(self.product_ids)

    def has_neighbours(self, product_id: str) -> bool:
        """True if this product has any non-zero learned similarity.

        False for a cold product. Callers should treat that as "fall back to a
        content or popularity rule", not as an error.
        """
        idx = self._index().get(product_id)
        if idx is None:
            return False
        row = self.similarity[idx].copy()
        row[idx] = 0.0
        return bool((row > 0).any())

    def similar_to(
        self,
        product_id: str,
        top_k: int = 6,
        min_score: float = 1e-9,
        exclude: set[str] | None = None,
    ) -> pd.DataFrame:
        """The ``top_k`` most similar products, best first.

        Returns ``product_id``, ``score``, ``rank``. Empty if the product is
        unknown or cold — deliberately not an exception, because "this item has
        no neighbours yet" is a normal state on a young catalogue and every
        caller has to handle it anyway.

        ``min_score`` drops zero-similarity items rather than padding the list to
        ``top_k`` with products that merely happen to sort first among ties. A
        short honest list beats a full arbitrary one.
        """
        index = self._index()
        idx = index.get(product_id)
        if idx is None:
            return _empty_similar()

        scores = self.similarity[idx].astype("float64").copy()
        scores[idx] = -np.inf  # never recommend the product being viewed

        if exclude:
            for pid in exclude:
                j = index.get(pid)
                if j is not None:
                    scores[j] = -np.inf

        k = int(min(max(top_k, 0), len(scores)))
        if k == 0:
            return _empty_similar()

        # argpartition finds the top k without sorting all n — the same trick
        # scales this lookup to a much larger catalogue than a full sort would.
        candidates = np.argpartition(-scores, k - 1)[:k]
        candidates = candidates[np.argsort(-scores[candidates], kind="stable")]

        rows = [
            {"product_id": self.product_ids[j], "score": float(scores[j])}
            for j in candidates
            if np.isfinite(scores[j]) and scores[j] > min_score
        ]
        out = pd.DataFrame(rows, columns=["product_id", "score"])
        out["rank"] = range(1, len(out) + 1)
        return out

    def neighbour_table(self, top_k: int = 6, min_score: float = 1e-9) -> pd.DataFrame:
        """Top neighbours for every product — used by the evaluation report."""
        frames = []
        for pid in self.product_ids:
            got = self.similar_to(pid, top_k=top_k, min_score=min_score)
            if got.empty:
                continue
            got = got.copy()
            got.insert(0, "source_product_id", pid)
            frames.append(got)
        if not frames:
            return pd.DataFrame(
                columns=["source_product_id", "product_id", "score", "rank"]
            )
        return pd.concat(frames, ignore_index=True)

    def coverage(self) -> dict[str, float | int]:
        """How much of the catalogue the model can actually speak about."""
        total = len(self.product_ids)
        warm = sum(1 for pid in self.product_ids if self.has_neighbours(pid))
        return {
            "n_products": total,
            "products_with_neighbours": warm,
            "cold_products": total - warm,
            "coverage": (warm / total) if total else 0.0,
        }

    # ── Persistence ─────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> Path:
        """Persist to a single ``.npz`` holding the matrix plus its metadata.

        npz rather than pickle: the artifact is a float array and a list of
        strings, both of which npz stores natively, and it stays loadable without
        executing arbitrary code or matching the class definition that wrote it.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            similarity=self.similarity.astype("float32"),
            product_ids=np.array(self.product_ids, dtype=object),
            meta=json.dumps(
                {
                    "model_name": MODEL_NAME,
                    "model_version": MODEL_VERSION,
                    "normalisation": self.normalisation,
                    "n_users": self.n_users,
                    "observed_cells": self.observed_cells,
                    "backend": self.backend,
                }
            ),
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "ItemSimilarityModel":
        with np.load(Path(path), allow_pickle=True) as blob:
            meta = json.loads(str(blob["meta"]))
            return cls(
                # float32 on disk halves the artifact; recommendations are a
                # ranking, and float32 precision is far finer than any ordering
                # difference a user could perceive.
                similarity=blob["similarity"].astype("float64"),
                product_ids=tuple(str(p) for p in blob["product_ids"]),
                normalisation=meta.get("normalisation", "l2"),
                n_users=int(meta.get("n_users", 0)),
                observed_cells=int(meta.get("observed_cells", 0)),
                backend=meta.get("backend", "unknown"),
            )


def _empty_similar() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product_id": pd.Series(dtype="object"),
            "score": pd.Series(dtype="float64"),
            "rank": pd.Series(dtype="int64"),
        }
    )


def fit_item_similarity(
    matrix: UserItemMatrix, use_sklearn: bool = True
) -> ItemSimilarityModel:
    """Train the item-item model from a user-item matrix."""
    sim = cosine_similarity_matrix(matrix.values, use_sklearn=use_sklearn)
    backend = "sklearn" if (use_sklearn and SKLEARN_AVAILABLE) else "numpy"
    return ItemSimilarityModel(
        similarity=sim,
        product_ids=matrix.product_ids,
        normalisation=matrix.normalisation,
        n_users=matrix.n_users,
        observed_cells=matrix.observed_cells,
        backend=backend,
    )


__all__ = [
    "MODEL_NAME",
    "MODEL_VERSION",
    "SKLEARN_AVAILABLE",
    "cosine_similarity_matrix",
    "ItemSimilarityModel",
    "fit_item_similarity",
]
