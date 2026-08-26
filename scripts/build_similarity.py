"""Train the item-item similarity model, evaluate it, and save both.

    python -m scripts.build_similarity
    python -m scripts.build_similarity --top-k 6 --report

Produces:

* ``artifacts/item_similarity.npz`` — the trained matrix, loadable without
  refitting (and without executing pickled code).
* ``reports/phase3_similarity.json`` — every metric, machine-readable.
* ``reports/phase3_similarity.md`` — the same numbers written out, with worked
  neighbour examples.

The report is generated rather than hand-written for one specific reason: the
Phase 4 write-up has to state which numbers came from real traffic and which from
simulation, and a hand-copied figure goes stale the moment the data changes. Every
number quoted in the documentation traces back to a run of this script.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from backend.core.config import ARTIFACTS_DIR, REPORTS_DIR, get_settings
from backend.database import queries
from recommendation import evaluate as ev
from recommendation.matrix import build_user_item_matrix
from recommendation.similarity import SKLEARN_AVAILABLE, fit_item_similarity

ARTIFACT_PATH = ARTIFACTS_DIR / "item_similarity.npz"
JSON_PATH = REPORTS_DIR / "phase3_similarity.json"
MD_PATH = REPORTS_DIR / "phase3_similarity.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=6, help="neighbours per product")
    parser.add_argument(
        "--min-user-products",
        type=int,
        default=2,
        help="drop users who viewed fewer distinct products than this",
    )
    parser.add_argument(
        "--normalise",
        default="l2",
        choices=["l2", "user_share", "none"],
        help="per-user row scaling; 'none' shows why normalisation matters",
    )
    parser.add_argument(
        "--report", action="store_true", help="also write the markdown report"
    )
    parser.add_argument(
        "--compare-normalisation",
        action="store_true",
        help="score every normalisation scheme on coherence and stability together",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="skip writing the .npz artifact"
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    print(f"database: {settings.database_url}")
    if not queries.database_exists():
        print("No database. Run: python -m scripts.init_db && python -m scripts.generate_events")
        return 1

    conn = queries.connect()
    try:
        products = queries.load_products(conn)
        events = queries.load_dwell_events(conn, min_seconds=1.0)
        provenance = queries.data_provenance(conn)

        if events.empty:
            print("No dwell events recorded — nothing to train on.")
            return 1

        print(f"events: {len(events):,}  products: {len(products)}")
        matrix = build_user_item_matrix(
            events,
            product_ids=list(products["product_id"]),
            normalise=args.normalise,
            min_user_products=args.min_user_products,
        )
        print(f"matrix: {matrix.n_users} x {matrix.n_products}, density {matrix.density:.4f}")

        if matrix.n_users < 2:
            print("Need at least two qualifying users to compute co-view similarity.")
            return 1

        model = fit_item_similarity(matrix)
        coverage = model.coverage()
        print(
            f"model: backend={model.backend}, coverage "
            f"{coverage['products_with_neighbours']}/{coverage['n_products']}"
        )

        coherence = ev.coherence_report(model, products, top_k=args.top_k)
        distribution = ev.similarity_distribution(model)
        concentration = ev.popularity_concentration(model, top_k=args.top_k)

        print("\ncoherence vs random baseline:")
        for row in coherence.itertuples():
            print(
                f"  {row.attribute:14} top1={row.top1_match_rate:6.1%}  "
                f"top{args.top_k}={row.topk_match_rate:6.1%}  "
                f"baseline={row.random_baseline:6.1%}  lift={row.lift:.2f}x"
            )

        examples = _example_products(conn, model, products, limit=6)
        example_table = ev.example_neighbours(
            model, products, examples, top_k=min(4, args.top_k)
        )

        activity = ev.user_activity_concentration(events)
        print(
            f"\nactivity: {activity['users']} users, Gini {activity['gini']:.2f}, "
            f"heaviest ({activity['top_user']}) holds "
            f"{activity['top_user_share']:.1%} of all dwell time"
        )

        comparison = None
        stability = ev.leave_one_user_out_stability(
            events, list(products["product_id"]), normalise=args.normalise,
            top_k=args.top_k,
        )
        print(
            f"stability: removing {stability['removed_user']} leaves top-"
            f"{args.top_k} Jaccard {stability['topk_jaccard']:.3f} "
            f"({stability['topk_slots_changed']:.1%} of slots move)"
        )

        if args.compare_normalisation:
            comparison = ev.normalisation_comparison(events, products, top_k=args.top_k)
            print("\nnormalisation trade-off:")
            print(comparison.to_string(index=False))

        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sklearn_installed": SKLEARN_AVAILABLE,
            "backend": model.backend,
            "settings": {
                "normalisation": args.normalise,
                "top_k": args.top_k,
                "min_user_products": args.min_user_products,
            },
            "data": {
                "events_used": int(len(events)),
                "catalogue_size": int(len(products)),
                "n_categories": int(products["category"].nunique()),
                "provenance": provenance,
            },
            "matrix": matrix.summary(),
            "coverage": coverage,
            "coherence": coherence.to_dict(orient="records"),
            "similarity_distribution": distribution,
            "popularity_concentration": concentration,
            "activity_concentration": activity,
            "stability": stability,
            "normalisation_comparison": (
                comparison.to_dict(orient="records") if comparison is not None else None
            ),
            "examples": example_table.to_dict(orient="records"),
        }

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {JSON_PATH.relative_to(JSON_PATH.parents[1])}")

        if not args.no_save:
            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            model.save(ARTIFACT_PATH)
            size_kb = ARTIFACT_PATH.stat().st_size / 1024
            print(f"wrote artifacts/{ARTIFACT_PATH.name} ({size_kb:.0f} KB)")

        if args.report:
            MD_PATH.write_text(_markdown(payload, example_table), encoding="utf-8")
            print(f"wrote reports/{MD_PATH.name}")
    finally:
        conn.close()

    return 0


def _example_products(conn, model, products, limit: int = 6) -> list[str]:
    """Pick the most-viewed products that actually have neighbours.

    Most-viewed rather than random: these are the products whose neighbour lists
    rest on the most evidence, so they are the fair examples to show. Cherry-picking
    would be showing the best-looking ones instead.
    """
    popular = queries.popular_products(conn, limit=limit * 4)
    if popular.empty:
        return []
    ids = [
        str(p)
        for p in popular["product_id"]
        if model.knows(str(p)) and model.has_neighbours(str(p))
    ]
    return ids[:limit]


def _markdown(payload: dict, examples: pd.DataFrame) -> str:
    coherence = pd.DataFrame(payload["coherence"])
    prov = payload["data"]["provenance"]
    matrix = payload["matrix"]
    cov = payload["coverage"]
    dist = payload["similarity_distribution"]
    conc = payload["popularity_concentration"]
    act = payload["activity_concentration"]
    stab = payload["stability"]
    comparison = (
        pd.DataFrame(payload["normalisation_comparison"])
        if payload.get("normalisation_comparison")
        else None
    )

    lines: list[str] = []
    add = lines.append

    add("# Phase 3 — item-item similarity results")
    add("")
    add(f"Generated {payload['generated_at']} · "
        f"cosine backend: `{payload['backend']}`"
        f"{'' if payload['sklearn_installed'] else ' (scikit-learn not installed; NumPy fallback)'}")
    add("")
    add("## Data")
    add("")
    add(f"The model was trained on {payload['data']['events_used']:,} product views "
        f"across a catalogue of {payload['data']['catalogue_size']} products. "
        f"Of those views, {prov['simulated']['views']:,} are simulated and "
        f"{prov['live']['views']:,} came from real browsing "
        f"({prov['live_share']:.1%} of the total).")
    add("")
    add(f"The resulting user-item matrix is {matrix['n_users']} users by "
        f"{matrix['n_products']} products with {matrix['observed_cells']:,} filled cells "
        f"— a density of {matrix['density']:.2%}, which is normal for behavioural data "
        f"and is the reason cosine similarity is computed over columns rather than "
        f"anything requiring dense overlap. Row normalisation: "
        f"`{matrix['normalisation']}`.")
    add("")
    add(f"{cov['products_with_neighbours']} of {cov['n_products']} products have at "
        f"least one learned neighbour. The remaining {cov['cold_products']} were never "
        f"viewed, so they have no column to compare and fall back to popularity — the "
        f"cold-start limitation, stated as a number rather than a caveat.")
    add("")

    add("## Does the model produce sensible neighbours?")
    add("")
    add("The model never sees a product's category, style or colour — it is trained "
        "purely on who viewed what, for how long. That makes those attributes an "
        "independent check: if behaviourally-similar products share a category far more "
        "often than chance predicts, the co-view signal carries real information.")
    add("")
    add("Chance is the important part. With "
        f"{payload['data'].get('n_categories', 5)} categories and an uneven catalogue, "
        "two random products already share a category surprisingly often, so each rate "
        "is reported against the random-pair baseline for the same attribute.")
    add("")
    add("| attribute | top-1 match | top-k match | random baseline | lift |")
    add("| --- | --- | --- | --- | --- |")
    for row in coherence.itertuples():
        add(f"| {row.attribute} | {row.top1_match_rate:.1%} | {row.topk_match_rate:.1%} "
            f"| {row.random_baseline:.1%} | {row.lift:.2f}x |")
    add("")
    add("The colour row is the one that validates the rest. There is no reason "
        "co-viewing should track colour, and it doesn't — colour agreement sits "
        "essentially at chance. A model scoring highly on *every* attribute would be "
        "evidence of a bug, most likely neighbour lists dominated by a few globally "
        "popular products rather than genuine per-item structure.")
    add("")
    add(f"Concentration confirms that: {conc['distinct_recommended']} distinct products "
        f"appear across all neighbour lists, and the ten most-recommended occupy only "
        f"{conc['top10_slot_share']:.1%} of the slots. Recommendations are spread across "
        f"the catalogue rather than collapsing onto bestsellers.")
    add("")
    add(f"The similarity distribution is the expected shape for a healthy matrix: "
        f"{dist['zero_share']:.0%} of product pairs score zero, the median is "
        f"{dist['median']:.3f}, and there is a thin strong tail (p99 = {dist['p99']:.3f}, "
        f"max = {dist['max']:.3f}). A high mean would indicate the geometry had "
        f"collapsed — the signature of a missing normalisation step.")
    add("")

    add("## What normalisation actually buys")
    add("")
    add(f"Attention is very unevenly distributed. Across {act['users']} users the "
        f"activity Gini is {act['gini']:.2f}: the heaviest browser "
        f"(`{act['top_user']}`) accounts for {act['top_user_share']:.1%} of all "
        f"recorded dwell time, {act['top_to_median_ratio']:.0f}x the median user. "
        f"Without per-user row scaling that one person's taste is a large fraction of "
        f"every similarity score in the model.")
    add("")
    add("The obvious way to justify normalisation would be to show it improves "
        "category coherence. On this dataset it does not — and reporting that is more "
        "useful than quietly picking the metric that agrees. The unnormalised matrix "
        "scores *higher* on category coherence, because the heaviest user browses "
        "consistently within a few categories, so letting them dominate reinforces "
        "category agreement while making the model mostly about one person.")
    add("")
    add("The right question is whether the recommendations belong to the population or "
        "to an individual. Removing the single heaviest user and refitting answers it "
        "directly:")
    add("")
    if comparison is not None:
        add("| row scaling | category top-k | lift | top-k stability | top-1 unchanged |")
        add("| --- | --- | --- | --- | --- |")
        for row in comparison.itertuples():
            add(f"| `{row.normalisation}` | {row.category_topk_match:.1%} "
                f"| {row.category_lift:.2f}x | {row.topk_jaccard_stability:.3f} "
                f"| {row.top1_unchanged:.1%} |")
        add("")
        add("Stability is Jaccard overlap of each product's neighbour list before "
            "versus after that user is removed, so 1.000 would mean their departure "
            "changed nothing.")
        add("")
        base = comparison[comparison.normalisation == "none"]
        l2row = comparison[comparison.normalisation == "l2"]
        if not base.empty and not l2row.empty:
            b, l = base.iloc[0], l2row.iloc[0]
            add(f"L2 was chosen on that trade: it gives up "
                f"{(b.category_topk_match - l.category_topk_match) * 100:.1f} points of "
                f"category coherence to gain "
                f"{(l.topk_jaccard_stability - b.topk_jaccard_stability) * 100:.1f} "
                f"points of stability, cutting the share of neighbour slots that move "
                f"when one user leaves from "
                f"{(1 - b.topk_jaccard_stability):.1%} to "
                f"{(1 - l.topk_jaccard_stability):.1%}. `user_share` is more stable "
                f"still but costs noticeably more coherence, which reads as "
                f"over-flattening genuine differences in engagement.")
    else:
        add(f"Removing `{stab['removed_user']}` "
            f"({stab['removed_user_share']:.1%} of all dwell time) leaves a top-k "
            f"Jaccard overlap of {stab['topk_jaccard']:.3f} under "
            f"`{stab['normalisation']}` scaling — {stab['topk_slots_changed']:.1%} of "
            f"neighbour slots move, and {stab['top1_unchanged']:.1%} of products keep "
            f"the same nearest neighbour. Run with `--compare-normalisation` for the "
            f"scheme-by-scheme table.")
    add("")
    add("Cosine already normalises each product column to unit length, so the column "
        "side needs nothing further. Row scaling is the part that has to be a decision.")
    add("")

    if not examples.empty:
        add("## Worked examples")
        add("")
        add("The most-viewed products with neighbours, so these lists rest on the most "
            "evidence available rather than being chosen for how good they look.")
        add("")
        for pid, group in examples.groupby("source_product_id", sort=False):
            head = group.iloc[0]
            add(f"**{pid} · {head['source_name']}** — category "
                f"{head['source_category']}")
            add("")
            for row in group.itertuples():
                # Spelled out rather than shortened because "Athletic" is both a
                # category and a style in this catalogue, and a bare
                # "(Athletic, Athletic)" is unreadable.
                mark = (
                    "same category"
                    if row.same_category
                    else f"category {row.category} — differs"
                )
                add(f"- `{row.score:.3f}` {row.product_id} {row.product_name} "
                    f"· style {row.style} · {mark}")
            add("")

    add("## Reproducing this")
    add("")
    add("```")
    add("python -m scripts.build_similarity --report --compare-normalisation")
    add("```")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
