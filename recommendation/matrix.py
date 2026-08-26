"""Phase 3 — the user-item matrix.

Rows are users, columns are products, each cell is how much attention that user
gave that product. This is the object item-item collaborative filtering is
computed from, and the normalisation applied here decides what "similar" ends up
meaning, so it is worth being precise about.

**Why normalise per user at all?**

Item-item cosine similarity between products *i* and *j* is built from the dot
product of their two columns: ``sum over users of (x[u,i] * x[u,j])``. Every user
contributes a term. If the matrix holds raw summed seconds, then a single heavy
user — and browsing data is always heavy-tailed, the simulated set here has a
user-activity Gini of ~0.55 — contributes terms one or two orders of magnitude
larger than everyone else. Their personal taste stops being one vote among many
and starts being the similarity function. Every pair of products they happened to
view looks related, and pairs they ignored look unrelated.

Row-normalising fixes this by construction. After L2 normalisation each user's
row is a unit vector, so no user can contribute more than 1 to any pair's dot
product regardless of how much they browsed. The matrix then encodes the *shape*
of each person's attention rather than its volume, which is what we actually want
to pool across people.

**Why not normalise per item instead?** Cosine similarity already divides by each
item's norm, so item-level scaling cancels out. Column normalisation would be
redundant; row normalisation would not.

**What normalisation does not fix:** exposure bias. A product in grid slot 8 gets
a ninth of slot 1's clicks in this dataset, and no amount of rescaling recovers
attention that was never given. That limitation is recorded in the results
write-up rather than papered over here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd

from recommendation.scoring import aggregate_dwell

Normalisation = Literal["l2", "user_share", "none"]


@dataclass(frozen=True)
class UserItemMatrix:
    """A dense user-item matrix plus the labels needed to interpret it.

    Dense is the right call at this scale: 100 products against even a few
    thousand users is a matrix of well under a million float64 cells, and
    ``cosine_similarity`` on a dense array avoids sparse-format overhead
    entirely. At catalogue sizes where that stops being true — tens of thousands
    of products — this is the type that would grow a sparse variant, and the
    rest of the pipeline would not need to change because everything downstream
    goes through ``values``.
    """

    values: np.ndarray          # shape (n_users, n_products), float64
    user_ids: tuple[str, ...]
    product_ids: tuple[str, ...]
    normalisation: str
    #: Non-zero cells before normalisation — the honest interaction count.
    observed_cells: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape

    @property
    def n_users(self) -> int:
        return self.values.shape[0]

    @property
    def n_products(self) -> int:
        return self.values.shape[1]

    @property
    def density(self) -> float:
        """Fraction of cells that hold an actual interaction."""
        total = self.n_users * self.n_products
        return (self.observed_cells / total) if total else 0.0

    @property
    def sparsity(self) -> float:
        return 1.0 - self.density

    def product_position(self, product_id: str) -> int | None:
        """Column index for a product, or None if it is not in the matrix."""
        try:
            return self.product_ids.index(product_id)
        except ValueError:
            return None

    def column(self, product_id: str) -> np.ndarray | None:
        idx = self.product_position(product_id)
        return None if idx is None else self.values[:, idx]

    def views_per_product(self) -> np.ndarray:
        """Count of users with a non-zero cell, per product column."""
        return (self.values > 0).sum(axis=0)

    def cold_products(self) -> tuple[str, ...]:
        """Products no user interacted with — they can have no learned neighbours."""
        counts = self.views_per_product()
        return tuple(pid for pid, c in zip(self.product_ids, counts) if c == 0)

    def to_frame(self) -> pd.DataFrame:
        """Labelled DataFrame view — for notebooks and eyeballing, not hot paths."""
        return pd.DataFrame(
            self.values, index=list(self.user_ids), columns=list(self.product_ids)
        )

    def summary(self) -> dict[str, float | int | str]:
        return {
            "n_users": self.n_users,
            "n_products": self.n_products,
            "observed_cells": self.observed_cells,
            "density": round(self.density, 6),
            "sparsity": round(self.sparsity, 6),
            "normalisation": self.normalisation,
            "cold_products": len(self.cold_products()),
        }


def build_user_item_matrix(
    events: pd.DataFrame,
    product_ids: Sequence[str] | None = None,
    normalise: Normalisation = "l2",
    now: datetime | None = None,
    half_life_days: float = 0.0,
    compress: bool = True,
    min_user_products: int = 1,
) -> UserItemMatrix:
    """Build the matrix from a dwell-event log.

    ``product_ids`` pins the column space to the full catalogue. Pass it. Without
    it the columns are only the products someone happened to view, which means
    the matrix silently changes shape as data arrives and a cold product cannot
    even be *asked* about — it would raise a lookup error rather than honestly
    reporting that it has no neighbours yet.

    ``half_life_days`` defaults to 0 (no decay) here, unlike the Phase 2 personal
    ranking. Phase 2 answers "what interests you *now*", where recency is part of
    the question. Phase 3 is learning a stable statement about the catalogue —
    that these two boots attract the same kind of attention — and that
    relationship does not expire just because the co-views happened three weeks
    ago. Decaying it would throw away most of the signal in a 30-day window for
    no benefit.

    ``min_user_products`` drops users who touched fewer than N distinct products.
    Users with a single view carry no co-view information at all (their row
    contributes zero to every pair's dot product), so excluding them costs
    nothing and makes the reported density honest rather than flattered by rows
    that cannot contribute.
    """
    scored = aggregate_dwell(
        events, now=now, half_life_days=half_life_days, compress=compress
    )

    columns = _resolve_columns(scored, product_ids)

    if scored.empty:
        return UserItemMatrix(
            values=np.zeros((0, len(columns)), dtype="float64"),
            user_ids=(),
            product_ids=columns,
            normalisation=normalise,
            observed_cells=0,
        )

    # Restrict to known columns before pivoting, so an event referencing a
    # retired product cannot widen the matrix behind the caller's back.
    scored = scored[scored["product_id"].isin(set(columns))]

    if min_user_products > 1:
        per_user = scored.groupby("user_id")["product_id"].nunique()
        keep = per_user[per_user >= int(min_user_products)].index
        scored = scored[scored["user_id"].isin(set(keep))]

    if scored.empty:
        return UserItemMatrix(
            values=np.zeros((0, len(columns)), dtype="float64"),
            user_ids=(),
            product_ids=columns,
            normalisation=normalise,
            observed_cells=0,
        )

    pivot = (
        scored.pivot_table(
            index="user_id", columns="product_id", values="score", aggfunc="sum"
        )
        .reindex(columns=list(columns))
        .fillna(0.0)
        .sort_index()
    )

    values = pivot.to_numpy(dtype="float64", copy=True)
    observed = int((values > 0).sum())
    values = _normalise_rows(values, normalise)

    return UserItemMatrix(
        values=values,
        user_ids=tuple(str(u) for u in pivot.index),
        product_ids=columns,
        normalisation=normalise,
        observed_cells=observed,
    )


def _resolve_columns(
    scored: pd.DataFrame, product_ids: Iterable[str] | None
) -> tuple[str, ...]:
    if product_ids is not None:
        # dict.fromkeys preserves catalogue order while removing duplicates.
        return tuple(dict.fromkeys(str(p) for p in product_ids))
    if scored.empty:
        return ()
    return tuple(sorted(scored["product_id"].astype(str).unique()))


def _normalise_rows(values: np.ndarray, how: Normalisation) -> np.ndarray:
    """Scale each user's row so no single user can dominate the dot products."""
    if how == "none" or values.size == 0:
        return values

    if how == "l2":
        norms = np.linalg.norm(values, axis=1, keepdims=True)
    elif how == "user_share":
        norms = values.sum(axis=1, keepdims=True)
    else:
        raise ValueError(
            f"unknown normalisation {how!r}; expected 'l2', 'user_share' or 'none'"
        )

    # A user whose whole row is zero has norm 0. Dividing would produce NaN and
    # poison every similarity that touches that column, so leave those rows as
    # exact zeros — they contribute nothing, which is correct.
    norms[norms == 0.0] = 1.0
    return values / norms


__all__ = ["UserItemMatrix", "build_user_item_matrix", "Normalisation"]
