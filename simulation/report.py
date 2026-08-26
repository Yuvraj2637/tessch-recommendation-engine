"""Data-quality report for the simulated event stream.

A synthetic dataset is only worth building if you can show it behaves like the
real thing. This module runs the checks that would catch a broken generator, and
— more importantly — the checks that would catch a *too easy* generator:

Realism checks
    funnel drop-off, position bias, popularity concentration, matrix sparsity,
    activity heavy tail, seasonality, sale-day lift.

Learnability checks
    Does observed behaviour actually correlate with hidden affinity? If not,
    no model can succeed and the whole project is measuring noise.

Difficulty checks
    Is popularity alone almost as good as knowing the truth? If so the dataset
    is trivial and "our model beats popularity" would prove nothing.

Every number printed here is derived from the event log alone, except the block
explicitly labelled ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from backend.core import constants as C


def gini(values: np.ndarray) -> float:
    """Gini coefficient: 0 = perfectly even, 1 = one item takes everything."""
    values = np.sort(np.asarray(values, dtype=float))
    n = values.size
    if n == 0 or values.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * (index * values).sum()) / (n * values.sum()) - (n + 1.0) / n)


def _fmt_table(rows: list[tuple[Any, ...]], headers: tuple[str, ...]) -> str:
    try:
        from tabulate import tabulate

        return tabulate(rows, headers=headers, tablefmt="github", floatfmt=".4f")
    except ImportError:  # pragma: no cover
        lines = ["  ".join(str(h) for h in headers)]
        lines += ["  ".join(str(c) for c in row) for row in rows]
        return "\n".join(lines)


@dataclass
class Report:
    sections: list[tuple[str, str]]
    stats: dict[str, Any]

    def to_text(self) -> str:
        out: list[str] = []
        for title, body in self.sections:
            out.append("")
            out.append("=" * 78)
            out.append(title)
            out.append("=" * 78)
            out.append(body)
        return "\n".join(out).strip()

    def to_markdown(self) -> str:
        out = ["# TESSCH — Simulated Data Quality Report", ""]
        for title, body in self.sections:
            out.append(f"## {title}")
            out.append("")
            out.append(body)
            out.append("")
        return "\n".join(out)


def build_report(
    events: pd.DataFrame,
    sessions: pd.DataFrame,
    users: pd.DataFrame,
    products: pd.DataFrame,
    orders: pd.DataFrame,
    affinity: pd.DataFrame | None = None,
    hidden_users: pd.DataFrame | None = None,
    meta: dict[str, Any] | None = None,
) -> Report:
    sections: list[tuple[str, str]] = []
    stats: dict[str, Any] = {}
    meta = meta or {}

    events = events.copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True)

    product_events = events[events["product_id"].notna()]
    views = events[events["event_type"] == C.EVENT_PRODUCT_VIEW]
    impressions = events[events["event_type"] == C.EVENT_PRODUCT_IMPRESSION]
    clicks = events[events["event_type"] == C.EVENT_PRODUCT_CLICK]
    purchases = events[events["event_type"] == C.EVENT_PURCHASE]

    # ── 1. Volume ────────────────────────────────────────────────────────────
    span_days = (events["timestamp"].max() - events["timestamp"].min()).days
    revenue = float(orders["total_amount"].sum()) if len(orders) else 0.0
    rows = [
        ("Products", f"{len(products):,}"),
        ("Users", f"{len(users):,}"),
        ("Sessions", f"{len(sessions):,}"),
        ("Events", f"{len(events):,}"),
        ("Orders", f"{len(orders):,}"),
        ("Revenue", f"INR {revenue:,.0f}"),
        ("Window", f"{events['timestamp'].min():%Y-%m-%d} to {events['timestamp'].max():%Y-%m-%d} ({span_days} days)"),
        ("Events / user", f"{len(events) / max(1, len(users)):,.1f}"),
        ("Events / session", f"{len(events) / max(1, len(sessions)):,.1f}"),
        ("Sessions / user", f"{len(sessions) / max(1, len(users)):,.1f}"),
    ]
    stats.update(
        n_products=len(products), n_users=len(users), n_sessions=len(sessions),
        n_events=len(events), n_orders=len(orders), revenue=revenue,
    )
    sections.append(("1. Volume", _fmt_table(rows, ("Metric", "Value"))))

    # ── 2. Event mix ─────────────────────────────────────────────────────────
    counts = events["event_type"].value_counts()
    rows = [
        (name, f"{count:,}", f"{100 * count / len(events):.2f}%")
        for name, count in counts.items()
    ]
    sections.append(("2. Event mix", _fmt_table(rows, ("Event type", "Count", "Share"))))
    stats["event_mix"] = {str(k): int(v) for k, v in counts.items()}

    # ── 3. Funnel ────────────────────────────────────────────────────────────
    funnel_counts = {step: int((events["event_type"] == step).sum()) for step in C.FUNNEL_STEPS}
    rows = []
    previous: int | None = None
    for step in C.FUNNEL_STEPS:
        count = funnel_counts[step]
        step_rate = "—" if previous in (None, 0) else f"{100 * count / previous:.2f}%"
        overall = (
            "—"
            if funnel_counts[C.EVENT_PRODUCT_IMPRESSION] == 0
            else f"{100 * count / funnel_counts[C.EVENT_PRODUCT_IMPRESSION]:.3f}%"
        )
        rows.append((step, f"{count:,}", step_rate, overall))
        previous = count
    body = _fmt_table(rows, ("Stage", "Count", "vs previous", "vs impression"))
    body += (
        "\n\nA real footwear funnel loses roughly an order of magnitude at each"
        "\nof click, cart and purchase. Impression-to-purchase in the low tenths"
        "\nof a percent is the shape to expect."
    )
    sections.append(("3. Conversion funnel", body))
    stats["funnel"] = funnel_counts

    # ── 4. Position bias ─────────────────────────────────────────────────────
    if len(impressions) and "position" in impressions.columns:
        imp_by_pos = impressions.groupby("position").size()
        click_by_pos = clicks.groupby("position").size().reindex(imp_by_pos.index, fill_value=0)
        ctr = (click_by_pos / imp_by_pos).dropna()
        rows = [
            (int(pos), f"{int(imp_by_pos[pos]):,}", f"{int(click_by_pos[pos]):,}",
             f"{100 * ctr[pos]:.2f}%")
            for pos in ctr.index[:10]
        ]
        first = float(ctr.iloc[0]) if len(ctr) else 0.0
        last = float(ctr.iloc[min(7, len(ctr) - 1)]) if len(ctr) else 0.0
        body = _fmt_table(rows, ("Grid slot", "Impressions", "Clicks", "CTR"))
        body += (
            f"\n\nSlot 1 CTR / slot {min(8, len(ctr))} CTR = "
            f"{(first / last) if last else float('nan'):.2f}x."
            "\nThis is the exposure bias any honest offline evaluation has to"
            "\ncontend with: a product in slot 8 is not unpopular, it is unseen."
        )
        sections.append(("4. Position bias", body))
        stats["ctr_by_position"] = {int(k): float(v) for k, v in ctr.items()}

    # ── 5. Surface performance ───────────────────────────────────────────────
    if len(impressions):
        imp = impressions.groupby("surface").size().rename("impressions")
        clk = clicks.groupby("surface").size().rename("clicks")
        surface = pd.concat([imp, clk], axis=1).fillna(0)
        surface["ctr"] = surface["clicks"] / surface["impressions"].replace(0, np.nan)
        surface = surface.sort_values("impressions", ascending=False)
        rows = [
            (name, f"{int(r.impressions):,}", f"{int(r.clicks):,}", f"{100 * r.ctr:.2f}%")
            for name, r in surface.iterrows()
        ]
        body = _fmt_table(rows, ("Surface", "Impressions", "Clicks", "CTR"))
        body += (
            "\n\n`recommended_for_you` here is the *legacy* popularity module the"
            "\nsite already had. Its CTR is the number the ML system has to beat"
            "\nin Stage 7, and it is logged in recommendation_logs."
        )
        sections.append(("5. Surface performance", body))

    # ── 6. User activity distribution ────────────────────────────────────────
    per_user = product_events.groupby("user_id").size()
    per_user = per_user.reindex(users["user_id"], fill_value=0)
    quantiles = [0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
    rows = [("min", f"{per_user.min():,.0f}")]
    rows += [(f"p{int(q * 100)}", f"{per_user.quantile(q):,.0f}") for q in quantiles]
    rows += [("max", f"{per_user.max():,.0f}"), ("mean", f"{per_user.mean():,.1f}")]
    top_decile_share = (
        per_user.nlargest(max(1, len(per_user) // 10)).sum() / max(1, per_user.sum())
    )
    body = _fmt_table(rows, ("Percentile", "Product events"))
    body += (
        f"\n\nGini = {gini(per_user.to_numpy()):.3f}. "
        f"Top 10% of users generate {100 * top_decile_share:.1f}% of all product events."
        "\nA flat distribution here would be the clearest sign of a fake dataset."
    )
    sections.append(("6. User activity (heavy tail)", body))
    stats["user_activity_gini"] = gini(per_user.to_numpy())
    stats["top_decile_user_share"] = float(top_decile_share)

    # ── 7. Product popularity concentration ──────────────────────────────────
    per_product = views.groupby("product_id").size().reindex(
        products["product_id"], fill_value=0
    )
    top10_share = per_product.nlargest(10).sum() / max(1, per_product.sum())
    zero_view = int((per_product == 0).sum())
    top_names = (
        products.set_index("product_id")
        .loc[per_product.nlargest(8).index, "product_name"]
        .tolist()
    )
    rows = [
        (rank, pid, name, f"{int(per_product[pid]):,}")
        for rank, (pid, name) in enumerate(zip(per_product.nlargest(8).index, top_names), 1)
    ]
    body = _fmt_table(rows, ("#", "Product", "Name", "Views"))
    body += (
        f"\n\nGini = {gini(per_product.to_numpy()):.3f}. "
        f"Top 10 of {len(products)} products take {100 * top10_share:.1f}% of views. "
        f"{zero_view} products have zero views (cold items)."
    )
    sections.append(("7. Product popularity concentration", body))
    stats["product_popularity_gini"] = gini(per_product.to_numpy())
    stats["cold_products"] = zero_view

    # ── 8. Sparsity ──────────────────────────────────────────────────────────
    pairs = product_events.groupby(["user_id", "product_id"]).size()
    view_pairs = views.groupby(["user_id", "product_id"]).size()
    cells = max(1, len(users) * len(products))
    rows = [
        ("Observed (user, product) pairs — any interaction", f"{len(pairs):,}"),
        ("Observed (user, product) pairs — views only", f"{len(view_pairs):,}"),
        ("Matrix cells", f"{cells:,}"),
        ("Density (any interaction)", f"{100 * len(pairs) / cells:.3f}%"),
        ("Density (views only)", f"{100 * len(view_pairs) / cells:.3f}%"),
        ("Sparsity (views only)", f"{100 * (1 - len(view_pairs) / cells):.3f}%"),
        ("Distinct products viewed / user (median)",
         f"{views.groupby('user_id')['product_id'].nunique().median():.0f}"),
    ]
    body = _fmt_table(rows, ("Metric", "Value"))
    body += (
        "\n\nProduction recommenders typically work at 99%+ sparsity. This is why"
        "\nthe catalogue is extended past the 50 curated products: at 50 items the"
        "\nmatrix would be ~50% dense and none of the hard problems would appear."
    )
    sections.append(("8. User-item matrix sparsity", body))
    stats["density_views"] = float(len(view_pairs) / cells)

    # ── 9. Repeat views ──────────────────────────────────────────────────────
    repeat = view_pairs.value_counts().sort_index()
    rows = [
        (f"{int(k)} view(s)", f"{int(v):,}", f"{100 * v / len(view_pairs):.2f}%")
        for k, v in list(repeat.items())[:8]
    ]
    multi = int((view_pairs > 1).sum())
    body = _fmt_table(rows, ("Times viewed", "Pairs", "Share"))
    body += (
        f"\n\n{multi:,} pairs ({100 * multi / max(1, len(view_pairs)):.1f}%) were viewed"
        " more than once."
        "\nRepeat viewing is the behavioural signal Phase 4 singles out, and it only"
        "\nexists because the simulator models comparison shopping inside a session"
        "\nand deep-linking back across sessions."
    )
    sections.append(("9. Repeat viewing", body))

    # ── 10. Dwell time ───────────────────────────────────────────────────────
    dwell = views["duration_seconds"].dropna().astype(float)
    if len(dwell):
        rows = [
            ("count", f"{len(dwell):,}"),
            ("mean", f"{dwell.mean():.1f}s"),
            ("median", f"{dwell.median():.1f}s"),
            ("p90", f"{dwell.quantile(0.90):.1f}s"),
            ("p99", f"{dwell.quantile(0.99):.1f}s"),
            ("max", f"{dwell.max():.1f}s"),
            ("< 5s (bounces)", f"{100 * (dwell < 5).mean():.1f}%"),
            ("> 60s (deep reads)", f"{100 * (dwell > 60).mean():.1f}%"),
        ]
        sections.append(("10. Dwell time distribution",
                         _fmt_table(rows, ("Statistic", "Value"))))

    # ── 11. Sessions ─────────────────────────────────────────────────────────
    if len(sessions):
        s = sessions.copy()
        rows = [
            ("Median duration", f"{s['duration_seconds'].median():.0f}s"),
            ("Mean duration", f"{s['duration_seconds'].mean():.0f}s"),
            ("Median products viewed", f"{s['products_viewed'].median():.0f}"),
            ("Sessions with a wishlist add", f"{100 * (s['wishlist_count'] > 0).mean():.1f}%"),
            ("Sessions with an add-to-cart", f"{100 * (s['cart_count'] > 0).mean():.1f}%"),
            ("Sessions with a purchase", f"{100 * (s['purchase_count'] > 0).mean():.1f}%"),
            ("Single-product sessions (bounce)",
             f"{100 * (s['unique_products'] <= 1).mean():.1f}%"),
        ]
        body = _fmt_table(rows, ("Metric", "Value"))
        if "intent" in s.columns:
            intent = s["intent"].value_counts(normalize=True)
            body += "\n\nSession intent mix: " + ", ".join(
                f"{k} {100 * v:.1f}%" for k, v in intent.items()
            )
        if "device_type" in s.columns:
            device = s["device_type"].value_counts(normalize=True)
            body += "\nDevice mix: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in device.items())
        sections.append(("11. Session shape", body))

    # ── 12. Temporal structure ───────────────────────────────────────────────
    weekly = events.set_index("timestamp").resample("W").size()
    peak_week = weekly.idxmax()
    daily = events.set_index("timestamp").resample("D").size()
    spikes = daily[daily > daily.mean() + 2 * daily.std()]
    body = (
        f"Weekly event volume: min {weekly.min():,} / median {weekly.median():,.0f} / "
        f"max {weekly.max():,} (peak week of {peak_week:%Y-%m-%d}).\n"
        f"Days more than 2 sd above the mean: {len(spikes)} "
        f"({', '.join(f'{d:%Y-%m-%d}' for d in spikes.index[:8])})."
    )
    monthly_cat = (
        views.merge(products[["product_id", "category"]], on="product_id", how="left")
        .assign(month=lambda d: d["timestamp"].dt.strftime("%Y-%m"))
        .pivot_table(index="month", columns="category", values="event_type",
                     aggfunc="size", fill_value=0)
    )
    share = monthly_cat.div(monthly_cat.sum(axis=1), axis=0)
    body += "\n\nCategory share of product views by month (seasonality):\n\n"
    body += _fmt_table(
        [(month, *[f"{100 * share.loc[month, c]:.1f}%" for c in share.columns])
         for month in share.index],
        ("Month", *share.columns),
    )
    body += (
        "\n\nSandal share should climb into summer and boot share should fall."
        "\nIf those two lines are flat, the seasonality code is not firing."
    )
    sections.append(("12. Temporal structure and seasonality", body))

    # ── 13. Cold start ───────────────────────────────────────────────────────
    thresholds = (0, 1, 3, 5, 10)
    rows = []
    for t in thresholds:
        n = int((per_user <= t).sum())
        rows.append((f"<= {t} product events", f"{n:,}", f"{100 * n / max(1, len(users)):.1f}%"))
    body = _fmt_table(rows, ("Cohort", "Users", "Share"))
    if "signup_at" in users.columns:
        signup = pd.to_datetime(users["signup_at"], utc=True, errors="coerce")
        recent = int((signup > events["timestamp"].max() - pd.Timedelta(days=30)).sum())
        body += f"\n\n{recent:,} users signed up in the final 30 days — genuine cold-start cases."
    sections.append(("13. Cold-start population", body))

    # ── 14. Legacy recommender baseline ──────────────────────────────────────
    rec_imps = impressions[impressions["surface"] == C.SURFACE_RECOMMENDED]
    rec_clicks = clicks[clicks["surface"] == C.SURFACE_RECOMMENDED]
    if len(rec_imps):
        ctr = len(rec_clicks) / len(rec_imps)
        body = (
            f"The pre-existing on-site module served {len(rec_imps):,} impressions and "
            f"earned {len(rec_clicks):,} clicks — CTR {100 * ctr:.2f}%.\n"
            "That is the online number Stage 8 has to improve on, and it comes from "
            "logged data rather than an assumption."
        )
        stats["legacy_ctr"] = float(ctr)
        sections.append(("14. Legacy recommender baseline", body))

    # ── 15. Ground truth: is the data learnable? ──────────────────────────────
    if affinity is not None and len(affinity):
        engagement = (
            product_events.assign(
                weight=product_events["event_type"].map(
                    {
                        C.EVENT_PRODUCT_IMPRESSION: 0.0,
                        C.EVENT_PRODUCT_HOVER: 0.5,
                        C.EVENT_PRODUCT_CLICK: 1.0,
                        C.EVENT_PRODUCT_VIEW: 1.0,
                        C.EVENT_WISHLIST_ADD: 4.0,
                        C.EVENT_ADD_TO_CART: 7.0,
                        C.EVENT_PURCHASE: 12.0,
                    }
                ).fillna(0.0)
            )
            .groupby(["user_id", "product_id"], as_index=False)["weight"]
            .sum()
        )
        merged = affinity.merge(engagement, on=["user_id", "product_id"], how="left")
        merged["weight"] = merged["weight"].fillna(0.0)

        merged["decile"] = pd.qcut(merged["affinity"], 10, labels=False, duplicates="drop")
        by_decile = merged.groupby("decile").agg(
            affinity=("affinity", "mean"),
            engaged_share=("weight", lambda s: float((s > 0).mean())),
            mean_score=("weight", "mean"),
        )
        rows = [
            (int(d) + 1, f"{r.affinity:.3f}", f"{100 * r.engaged_share:.2f}%", f"{r.mean_score:.2f}")
            for d, r in by_decile.iterrows()
        ]
        body = _fmt_table(rows, ("Affinity decile", "Mean affinity", "Any interaction", "Mean score"))

        try:
            from scipy.stats import spearmanr

            rho = float(spearmanr(merged["affinity"], merged["weight"]).statistic)
            body += f"\n\nSpearman(hidden affinity, observed engagement) = {rho:.3f}."
            stats["affinity_engagement_spearman"] = rho
        except Exception:  # pragma: no cover
            pass

        # Dwell time should rise monotonically with hidden affinity.
        dwell_join = (
            views.groupby(["user_id", "product_id"], as_index=False)["duration_seconds"]
            .mean()
            .merge(affinity, on=["user_id", "product_id"], how="inner")
        )
        if len(dwell_join):
            dwell_join["decile"] = pd.qcut(
                dwell_join["affinity"], 5, labels=False, duplicates="drop"
            )
            dwell_by = dwell_join.groupby("decile")["duration_seconds"].mean()
            body += "\n\nMean dwell time by hidden-affinity quintile: " + ", ".join(
                f"Q{int(k) + 1} {v:.1f}s" for k, v in dwell_by.items()
            )

        body += (
            "\n\nBoth relationships must be strongly monotonic: that is the proof that"
            "\nbehaviour encodes preference and therefore that a model can recover it."
            "\nThe relationship must also be well short of perfect — the gap is the"
            "\nnoise, exposure bias and exploration that make the task non-trivial."
        )
        sections.append(("15. GROUND TRUTH — is behaviour predictive of preference?", body))

    # ── 16. Difficulty: can popularity alone win? ────────────────────────────
    if affinity is not None and len(affinity):
        popularity_rank = per_product.rank(ascending=False)
        truth_rank = (
            affinity.groupby("product_id")["affinity"].mean().rank(ascending=False)
        )
        joined = pd.concat(
            [popularity_rank.rename("popularity"), truth_rank.rename("appeal")], axis=1
        ).dropna()
        try:
            from scipy.stats import spearmanr

            rho = float(spearmanr(joined["popularity"], joined["appeal"]).statistic)
            body = (
                f"Spearman(observed popularity rank, mean hidden appeal rank) = {rho:.3f}.\n\n"
                "This is the dataset's difficulty dial. Near 1.0 and popularity is\n"
                "already optimal, so beating it would prove nothing. Near 0.0 and\n"
                "popularity is noise, which is equally unrealistic. A strong but\n"
                "imperfect correlation is what leaves genuine room for\n"
                "personalisation to add measurable value."
            )
            stats["popularity_vs_appeal_spearman"] = rho
            sections.append(("16. DIFFICULTY — how much headroom is there?", body))
        except Exception:  # pragma: no cover
            pass

    # ── 17. Archetype separability ───────────────────────────────────────────
    if hidden_users is not None and len(hidden_users):
        joined = (
            views.merge(products[["product_id", "category"]], on="product_id", how="left")
            .merge(hidden_users[["user_id", "archetype_primary"]], on="user_id", how="left")
        )
        pivot = joined.pivot_table(
            index="archetype_primary", columns="category", values="event_type",
            aggfunc="size", fill_value=0,
        )
        share = pivot.div(pivot.sum(axis=1), axis=0)
        rows = [
            (arch, *[f"{100 * share.loc[arch, c]:.0f}%" for c in share.columns])
            for arch in share.index
        ]
        body = _fmt_table(rows, ("Archetype", *share.columns))
        body += (
            "\n\nEach archetype should concentrate on a different category. This is the"
            "\ncluster structure collaborative filtering exploits — if these rows were"
            "\nidentical, user-based CF could not work at all."
        )
        sections.append(("17. GROUND TRUTH — archetype separability", body))

    return Report(sections=sections, stats=stats)


__all__ = ["build_report", "Report", "gini"]
