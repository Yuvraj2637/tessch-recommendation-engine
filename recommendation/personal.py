"""Phase 2 — "Picked for you": what did *this* visitor spend time on?

The simplest useful recommender in the system, and the one users notice first.
It answers a single question: of everything this person looked at, what held
their attention longest?

Note that this surface intentionally recommends products the user has **already
seen**. That is not a bug in the ranking — it is the feature. A returning
visitor who spent four minutes comparing two boots last night wants those boots
one click away, not hidden because they are "already known". Discovery of unseen
products is what Phase 3 is for, and the two surfaces are kept separate so that
neither has to compromise.

Everything here is pure pandas so the ranking logic can be tested without a
database or an HTTP layer.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from recommendation.scoring import aggregate_dwell

#: Model identifier written to recommendation_logs, so served rows can later be
#: attributed to the model that produced them.
MODEL_NAME = "dwell_personal"
MODEL_VERSION = "1.0"


def rank_for_user(
    history: pd.DataFrame,
    top_k: int = 8,
    now: datetime | None = None,
    half_life_days: float = 30.0,
    compress: bool = True,
) -> pd.DataFrame:
    """Rank one user's viewed products by recency-weighted dwell time.

    ``history`` is that user's ``product_view`` rows: ``product_id``, ``seconds``,
    ``timestamp``. A ``user_id`` column is optional and ignored — this function
    deliberately handles exactly one user, so passing a multi-user frame is a
    programming error rather than something to silently aggregate over.

    Returns ``product_id``, ``score``, ``total_seconds``, ``view_count``,
    ``last_seen``, ``rank`` — ordered best first, at most ``top_k`` rows.
    Ties on score break toward the more recently seen product.
    """
    if history.empty:
        return _empty_ranking()

    df = history.copy()
    if "user_id" not in df.columns:
        # aggregate_dwell groups by (user_id, product_id); a constant key makes
        # this a single-user aggregation without duplicating that logic.
        df["user_id"] = "_"
    elif df["user_id"].nunique() > 1:
        raise ValueError(
            "rank_for_user expects one user's history; "
            f"got {df['user_id'].nunique()} distinct user_ids"
        )

    scored = aggregate_dwell(
        df, now=now, half_life_days=half_life_days, compress=compress
    )
    if scored.empty:
        return _empty_ranking()

    ranked = (
        scored.sort_values(["score", "last_seen"], ascending=[False, False])
        .head(int(top_k))
        .drop(columns=["user_id"])
        .reset_index(drop=True)
    )
    ranked["rank"] = range(1, len(ranked) + 1)
    return ranked


def _empty_ranking() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product_id": pd.Series(dtype="object"),
            "total_seconds": pd.Series(dtype="float64"),
            "view_count": pd.Series(dtype="int64"),
            "score": pd.Series(dtype="float64"),
            "last_seen": pd.Series(dtype="datetime64[ns]"),
            "rank": pd.Series(dtype="int64"),
        }
    )


def rank_all_users(
    events: pd.DataFrame,
    top_k: int = 8,
    now: datetime | None = None,
    half_life_days: float = 30.0,
) -> pd.DataFrame:
    """Same ranking, computed for every user at once.

    Used by the offline report and by tests that need to assert different users
    receive different lists. The API does not call this — serving one visitor
    should never require aggregating the whole event log.
    """
    scored = aggregate_dwell(events, now=now, half_life_days=half_life_days)
    if scored.empty:
        return _empty_ranking().assign(user_id=pd.Series(dtype="object"))

    scored = scored.sort_values(
        ["user_id", "score", "last_seen"], ascending=[True, False, False]
    )
    scored["rank"] = scored.groupby("user_id", sort=False).cumcount() + 1
    return scored[scored["rank"] <= int(top_k)].reset_index(drop=True)


def anchor_product(ranking: pd.DataFrame) -> str | None:
    """The product to name in a "Because you viewed …" caption.

    The top-ranked product, which is the one the copy can justify honestly.
    """
    if ranking.empty:
        return None
    return str(ranking.iloc[0]["product_id"])


__all__ = [
    "MODEL_NAME",
    "MODEL_VERSION",
    "rank_for_user",
    "rank_all_users",
    "anchor_product",
]
