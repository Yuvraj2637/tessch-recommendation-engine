"""Turning raw dwell time into an interest score.

This module is the join between "what the tracker recorded" and "what the
recommenders rank by". It is deliberately pure: DataFrames in, DataFrames out,
no database and no configuration lookups, so every decision here is visible to a
test.

Three decisions matter, and all three are about not letting one number dominate:

1. **Sum, don't average.** A user who returns to a shoe four times for 30s each
   is more interested than one who sat on it once for 100s. Summing rewards the
   return visit; averaging actively punishes it.

2. **Compress with log1p.** Raw seconds have a long right tail — a p99 dwell of
   ~4 minutes against a median of ~45s. Left raw, a handful of long sessions
   would set the geometry of the entire similarity space. ``log1p`` keeps the
   ordering while pulling the tail in, and maps 0 to 0 so untouched products stay
   at zero rather than becoming slightly negative.

3. **Decay with age.** Interest is not timeless. A shoe studied last night says
   more about what to show now than one glanced at three weeks ago, so scores are
   multiplied by ``0.5 ** (days_ago / half_life)``.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

#: Columns ``aggregate_dwell`` expects on its input frame.
REQUIRED_COLUMNS = ("user_id", "product_id", "seconds", "timestamp")


def utc_now_naive() -> datetime:
    """Current UTC time as a naive datetime.

    SQLite stores naive UTC (see simulation/loader.py), so every comparison in
    this module happens in naive UTC. Mixing naive and aware datetimes raises in
    pandas, and silently shifting by a timezone offset would quietly corrupt
    every recency feature — which is exactly the kind of bug that only shows up
    as "the recommendations feel stale".
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _as_naive_utc(series: pd.Series) -> pd.Series:
    """Coerce a timestamp column to naive UTC."""
    out = pd.to_datetime(series, errors="coerce")
    if isinstance(out.dtype, pd.DatetimeTZDtype):
        out = out.dt.tz_convert("UTC").dt.tz_localize(None)
    return out


def time_decay_weights(
    timestamps: pd.Series,
    now: datetime | None = None,
    half_life_days: float = 30.0,
) -> pd.Series:
    """Exponential recency weight in (0, 1] for each timestamp.

    A weight of 1.0 means "happened now"; 0.5 means "one half-life ago".

    ``half_life_days <= 0`` disables decay by returning all ones, which makes it
    possible to compare a decayed ranking against an undecayed one without a
    separate code path.
    """
    if half_life_days is None or half_life_days <= 0:
        return pd.Series(1.0, index=timestamps.index, dtype="float64")

    reference = now or utc_now_naive()
    ts = _as_naive_utc(timestamps)
    days_ago = (reference - ts).dt.total_seconds() / 86_400.0

    # Future-dated rows (clock skew between a client and the server) would get a
    # weight above 1 and outrank everything real. Clamp them to "now" instead.
    days_ago = days_ago.clip(lower=0.0).fillna(0.0)
    return np.power(0.5, days_ago / float(half_life_days))


def aggregate_dwell(
    events: pd.DataFrame,
    now: datetime | None = None,
    half_life_days: float = 30.0,
    compress: bool = True,
) -> pd.DataFrame:
    """Collapse a dwell-event log into one row per (user, product).

    Returns columns:

    ``user_id``, ``product_id``
        The pair.
    ``total_seconds``
        Raw summed dwell time. Reported because it is the number a human can
        sanity-check against the database; never used for ranking directly.
    ``view_count``
        How many separate views made up that total.
    ``score``
        The ranking value: summed, optionally log-compressed, recency-decayed.
    ``last_seen``
        Most recent view, used for tie-breaking and for "because you viewed…".

    An empty input returns a correctly-typed empty frame rather than raising, so
    callers serving a brand-new visitor need no special case.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in events.columns]
    if missing:
        raise ValueError(
            f"aggregate_dwell requires columns {REQUIRED_COLUMNS}; missing {missing}"
        )

    if events.empty:
        return pd.DataFrame(
            {
                "user_id": pd.Series(dtype="object"),
                "product_id": pd.Series(dtype="object"),
                "total_seconds": pd.Series(dtype="float64"),
                "view_count": pd.Series(dtype="int64"),
                "score": pd.Series(dtype="float64"),
                "last_seen": pd.Series(dtype="datetime64[ns]"),
            }
        )

    df = events.loc[:, list(REQUIRED_COLUMNS)].copy()
    df["seconds"] = pd.to_numeric(df["seconds"], errors="coerce")
    df = df.dropna(subset=["seconds", "user_id", "product_id"])
    df["timestamp"] = _as_naive_utc(df["timestamp"])

    # Per-event weight, so a decayed sum is a weighted sum rather than the whole
    # (user, product) pair being decayed by its most recent visit alone. This is
    # the difference between "you looked at this a lot, recently" and "you looked
    # at this a lot, and also once recently".
    df["weight"] = time_decay_weights(df["timestamp"], now=now, half_life_days=half_life_days)
    df["contribution"] = np.log1p(df["seconds"]) if compress else df["seconds"]
    df["weighted"] = df["contribution"] * df["weight"]

    grouped = (
        df.groupby(["user_id", "product_id"], sort=False)
        .agg(
            total_seconds=("seconds", "sum"),
            view_count=("seconds", "size"),
            score=("weighted", "sum"),
            last_seen=("timestamp", "max"),
        )
        .reset_index()
    )
    grouped["view_count"] = grouped["view_count"].astype("int64")
    return grouped


def summarise_scores(scored: pd.DataFrame) -> dict[str, float]:
    """Descriptive stats over a scored frame — used by reports and tests."""
    if scored.empty:
        return {"pairs": 0, "users": 0, "products": 0}
    return {
        "pairs": int(len(scored)),
        "users": int(scored["user_id"].nunique()),
        "products": int(scored["product_id"].nunique()),
        "mean_score": float(scored["score"].mean()),
        "median_score": float(scored["score"].median()),
        "max_score": float(scored["score"].max()),
        "mean_total_seconds": float(scored["total_seconds"].mean()),
        "mean_views_per_pair": float(scored["view_count"].mean()),
    }


__all__ = [
    "REQUIRED_COLUMNS",
    "utc_now_naive",
    "time_decay_weights",
    "aggregate_dwell",
    "summarise_scores",
]
