"""Phase 4 — does the similarity model actually produce sensible neighbours?

The model is trained on behaviour alone: it never sees a product's category,
colour or style. So catalogue attributes are available as an *independent* check
on it. If products the model calls similar turn out to share a category far more
often than chance would predict, the co-view signal is carrying real information
about the catalogue. That is a genuine validation rather than a circular one,
precisely because category was never an input.

**The number is meaningless without a baseline.** "68% of top neighbours share a
category" sounds like a result, but with five categories and an uneven catalogue
a random pair already shares a category surprisingly often — in this catalogue
roughly 30%. Every coherence figure here is therefore reported next to the rate
expected from random pairs drawn from the same catalogue, and it is the *lift*
between them that means anything.

This is a sanity check, not an accuracy metric. It cannot tell you whether the
recommendations drive clicks; only a live A/B test can. What it can do is catch
the failure mode where the pipeline silently produces noise, which is the thing
most worth catching before showing the work to anyone.

**Coherence is also not the metric that justifies normalisation**, and it took
measuring to find that out. Turning row normalisation off *raises* category
coherence slightly on this dataset, because the heaviest user browses
consistently within a few categories — so letting them dominate the matrix
reinforces category agreement while making the model largely about one person.
What normalisation actually buys is measured by
:func:`leave_one_user_out_stability`: how much the recommendations move when a
single heavy user is removed. That is the honest argument for it, and it is a
different question from whether neighbours look plausible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recommendation.similarity import ItemSimilarityModel


def attribute_coherence(
    model: ItemSimilarityModel,
    products: pd.DataFrame,
    attribute: str = "category",
    top_k: int = 6,
    seed: int = 20260825,
) -> dict[str, float | int | str]:
    """Share of recommended neighbours matching the source product's attribute.

    Returns the observed rate at rank 1 and across the whole top-``top_k`` list,
    the random-pair baseline for the same attribute, and the lift over it.

    ``products`` needs ``product_id`` plus ``attribute``.
    """
    if attribute not in products.columns:
        raise ValueError(f"products has no column {attribute!r}")

    lookup = dict(
        zip(products["product_id"].astype(str), products[attribute].astype(str))
    )
    table = model.neighbour_table(top_k=top_k)

    if table.empty:
        return {
            "attribute": attribute,
            "evaluated_products": 0,
            "pairs": 0,
            "top1_match_rate": 0.0,
            "topk_match_rate": 0.0,
            "random_baseline": 0.0,
            "lift": 0.0,
        }

    src = table["source_product_id"].map(lookup)
    dst = table["product_id"].map(lookup)
    match = (src == dst) & src.notna() & dst.notna()

    at_rank1 = table["rank"] == 1
    baseline = _random_pair_baseline(model, lookup, seed=seed)
    topk_rate = float(match.mean())

    return {
        "attribute": attribute,
        "evaluated_products": int(table["source_product_id"].nunique()),
        "pairs": int(len(table)),
        "top1_match_rate": float(match[at_rank1].mean()) if at_rank1.any() else 0.0,
        "topk_match_rate": topk_rate,
        "random_baseline": baseline,
        "lift": (topk_rate / baseline) if baseline > 0 else float("nan"),
    }


def _random_pair_baseline(
    model: ItemSimilarityModel, lookup: dict[str, str], seed: int = 20260825
) -> float:
    """Probability two distinct products share an attribute value, by chance.

    Computed analytically over the products the model can actually recommend —
    not the whole catalogue — because cold products never appear as neighbours
    and including them would bias the comparison. For group sizes n_g summing to
    N, the probability that an ordered pair of distinct products matches is
    ``sum(n_g * (n_g - 1)) / (N * (N - 1))``.
    """
    recommendable = [
        pid for pid in model.product_ids if model.has_neighbours(pid) and pid in lookup
    ]
    total = len(recommendable)
    if total < 2:
        return 0.0

    counts = pd.Series([lookup[p] for p in recommendable]).value_counts()
    same = float((counts * (counts - 1)).sum())
    return same / (total * (total - 1))


def coherence_report(
    model: ItemSimilarityModel,
    products: pd.DataFrame,
    attributes: tuple[str, ...] = ("category", "style", "color_family"),
    top_k: int = 6,
) -> pd.DataFrame:
    """Coherence across several attributes at once.

    Reporting more than one attribute guards against a single flattering number.
    Category agreement is expected to be strong; colour agreement should be much
    weaker, because there is no reason co-viewing should track colour. A model
    that scored high on *everything* would be evidence of a bug — most likely
    that the neighbour lists are dominated by a handful of globally popular
    products rather than reflecting per-item structure.
    """
    rows = [
        attribute_coherence(model, products, attribute=attr, top_k=top_k)
        for attr in attributes
        if attr in products.columns
    ]
    return pd.DataFrame(rows)


def similarity_distribution(model: ItemSimilarityModel) -> dict[str, float]:
    """Spread of off-diagonal similarity scores.

    A healthy matrix is mostly near zero with a thin tail of strong pairs. If the
    mean sits high, the geometry has collapsed — usually the signature of a
    missing normalisation step letting one heavy user relate everything to
    everything.
    """
    sim = model.similarity
    if sim.size == 0:
        return {}
    off = sim[~np.eye(sim.shape[0], dtype=bool)]
    nonzero = off[off > 0]
    return {
        "pairs": int(off.size),
        "mean": float(off.mean()),
        "median": float(np.median(off)),
        "p90": float(np.quantile(off, 0.90)),
        "p99": float(np.quantile(off, 0.99)),
        "max": float(off.max()),
        "zero_share": float((off <= 0).mean()),
        "mean_nonzero": float(nonzero.mean()) if nonzero.size else 0.0,
    }


def popularity_concentration(
    model: ItemSimilarityModel, top_k: int = 6
) -> dict[str, float | int]:
    """How concentrated the recommendations are across the catalogue.

    Item-item collaborative filtering has a well-known failure mode where a few
    popular products appear in almost every neighbour list. This measures it: how
    many distinct products ever get recommended, and what share of all
    recommendation slots the top 10 products occupy.
    """
    table = model.neighbour_table(top_k=top_k)
    if table.empty:
        return {"distinct_recommended": 0, "top10_slot_share": 0.0, "slots": 0}

    counts = table["product_id"].value_counts()
    return {
        "slots": int(len(table)),
        "distinct_recommended": int(counts.size),
        "catalogue_size": int(len(model.product_ids)),
        "recommendable_share": float(counts.size / len(model.product_ids)),
        "top10_slot_share": float(counts.head(10).sum() / len(table)),
    }


def example_neighbours(
    model: ItemSimilarityModel,
    products: pd.DataFrame,
    product_ids: list[str],
    top_k: int = 4,
) -> pd.DataFrame:
    """Readable neighbour lists for a handful of products.

    The write-up quotes these directly. Worked examples are what make a
    similarity claim checkable by a reader who will not run the code.
    """
    meta = products.set_index("product_id")
    rows: list[dict[str, object]] = []
    for pid in product_ids:
        if pid not in meta.index:
            continue
        for r in model.similar_to(pid, top_k=top_k).itertuples():
            rows.append(
                {
                    "source_product_id": pid,
                    "source_name": meta.at[pid, "product_name"],
                    "source_category": meta.at[pid, "category"],
                    "rank": r.rank,
                    "product_id": r.product_id,
                    "product_name": meta.at[r.product_id, "product_name"],
                    "category": meta.at[r.product_id, "category"],
                    "style": meta.at[r.product_id, "style"],
                    "same_category": bool(
                        meta.at[r.product_id, "category"] == meta.at[pid, "category"]
                    ),
                    "score": round(float(r.score), 4),
                }
            )
    return pd.DataFrame(rows)


def user_activity_concentration(events: pd.DataFrame) -> dict[str, float | int | str]:
    """How unevenly dwell time is spread across users.

    This is the diagnostic that decides whether normalisation is load-bearing at
    all. On a dataset where every user contributes about the same amount of
    attention, row scaling changes almost nothing and arguing about it is wasted
    effort. The moment one user holds a large share, that user is quietly
    co-authoring every recommendation.

    ``events`` needs ``user_id`` and ``seconds``.
    """
    if events.empty:
        return {"users": 0, "top_user": "", "top_user_share": 0.0, "gini": 0.0}

    totals = events.groupby("user_id")["seconds"].sum().sort_values(ascending=False)
    values = np.sort(totals.to_numpy(dtype="float64"))
    n = len(values)
    total = float(values.sum())

    # Gini over user activity. 0 = everyone browses equally, 1 = one user is the
    # entire dataset.
    if n < 2 or total <= 0:
        gini = 0.0
    else:
        ranks = np.arange(1, n + 1)
        gini = float((2.0 * (ranks * values).sum()) / (n * total) - (n + 1) / n)

    median = float(np.median(values))
    return {
        "users": int(n),
        "top_user": str(totals.index[0]),
        "top_user_seconds": round(float(totals.iloc[0]), 1),
        "top_user_share": round(float(totals.iloc[0] / total), 4) if total else 0.0,
        "median_user_seconds": round(median, 1),
        "top_to_median_ratio": round(float(totals.iloc[0] / median), 2) if median else 0.0,
        "gini": round(gini, 4),
    }


def leave_one_user_out_stability(
    events: pd.DataFrame,
    product_ids: list[str],
    normalise: str = "l2",
    top_k: int = 6,
    user_id: str | None = None,
    min_user_products: int = 2,
) -> dict[str, float | int | str]:
    """Retrain without the heaviest user and measure how far the model moved.

    This is the metric that justifies per-user row normalisation, and it is a
    different question from :func:`attribute_coherence`. Coherence asks whether
    neighbours look plausible; this asks whether they are a property of the
    *population* or of one unusually active person. A recommender that rearranges
    itself when a single visitor is removed has learned that visitor, not the
    catalogue — and on a young site with few users that is a live risk rather than
    a theoretical one.

    Reported as Jaccard overlap of each product's top-``top_k`` list before versus
    after removal, so 1.0 means the removal changed nothing. Only products that
    have neighbours in *both* fits are compared; a product that goes cold because
    its only viewer was the removed user has no meaningful overlap to report and
    is counted separately.

    Deliberately recomputes from raw events rather than perturbing the fitted
    matrix, because normalisation happens during matrix construction — reusing the
    already-normalised matrix would measure a different, easier thing.
    """
    from recommendation.matrix import build_user_item_matrix
    from recommendation.similarity import fit_item_similarity

    if events.empty:
        return {"removed_user": "", "compared_products": 0, "topk_jaccard": 1.0}

    totals = events.groupby("user_id")["seconds"].sum().sort_values(ascending=False)
    removed = user_id or str(totals.index[0])

    def _neighbours(frame: pd.DataFrame) -> dict[str, list[str]]:
        matrix = build_user_item_matrix(
            frame,
            product_ids=product_ids,
            normalise=normalise,
            min_user_products=min_user_products,
        )
        model = fit_item_similarity(matrix)
        return {
            pid: list(model.similar_to(pid, top_k=top_k)["product_id"])
            for pid in model.product_ids
            if model.has_neighbours(pid)
        }

    before = _neighbours(events)
    after = _neighbours(events[events["user_id"] != removed])

    shared = [pid for pid in before if pid in after]
    if not shared:
        return {
            "removed_user": removed,
            "compared_products": 0,
            "topk_jaccard": 0.0,
            "top1_unchanged": 0.0,
            "went_cold": len(before),
        }

    overlaps = []
    top1 = []
    for pid in shared:
        a, b = set(before[pid]), set(after[pid])
        union = a | b
        overlaps.append(len(a & b) / len(union) if union else 1.0)
        top1.append(before[pid][0] == after[pid][0])

    return {
        "removed_user": removed,
        "removed_user_share": round(float(totals.loc[removed] / totals.sum()), 4),
        "normalisation": normalise,
        "compared_products": len(shared),
        "went_cold": len(before) - len(shared),
        "topk_jaccard": round(float(np.mean(overlaps)), 4),
        "topk_slots_changed": round(1.0 - float(np.mean(overlaps)), 4),
        "top1_unchanged": round(float(np.mean(top1)), 4),
    }


def normalisation_comparison(
    events: pd.DataFrame,
    products: pd.DataFrame,
    schemes: tuple[str, ...] = ("none", "l2", "user_share"),
    top_k: int = 6,
) -> pd.DataFrame:
    """Score every normalisation scheme on both coherence and stability.

    Kept as one function because the two columns have to be read together. Judging
    normalisation on coherence alone points the wrong way on this dataset — the
    unnormalised matrix wins that column while being markedly less stable. Showing
    them side by side is what makes the trade-off visible instead of letting a
    single number decide.
    """
    from recommendation.matrix import build_user_item_matrix
    from recommendation.similarity import fit_item_similarity

    product_ids = [str(p) for p in products["product_id"]]
    rows = []
    for scheme in schemes:
        matrix = build_user_item_matrix(
            events, product_ids=product_ids, normalise=scheme, min_user_products=2
        )
        model = fit_item_similarity(matrix)
        coherence = attribute_coherence(model, products, "category", top_k=top_k)
        stability = leave_one_user_out_stability(
            events, product_ids, normalise=scheme, top_k=top_k
        )
        concentration = popularity_concentration(model, top_k=top_k)
        rows.append(
            {
                "normalisation": scheme,
                "category_topk_match": coherence["topk_match_rate"],
                "category_lift": coherence["lift"],
                "topk_jaccard_stability": stability["topk_jaccard"],
                "top1_unchanged": stability["top1_unchanged"],
                "top10_slot_share": concentration["top10_slot_share"],
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "attribute_coherence",
    "coherence_report",
    "similarity_distribution",
    "popularity_concentration",
    "example_neighbours",
    "user_activity_concentration",
    "leave_one_user_out_stability",
    "normalisation_comparison",
]
