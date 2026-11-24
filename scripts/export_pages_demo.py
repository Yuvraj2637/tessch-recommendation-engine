"""Export the static storefront's catalogue and anonymous synthetic dwell rows."""

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def export(database: Path, output: Path) -> None:
    with sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        products = conn.execute(
            """
            SELECT product_id, product_name, category, color, style, price,
                   material, collection, description
            FROM products WHERE is_core = 1 AND availability = 1
            ORDER BY product_id
            """
        ).fetchall()
        views = conn.execute(
            """
            SELECT user_id, product_id, duration_seconds
            FROM user_events
            WHERE source = 'simulated' AND event_type = 'product_view'
                  AND duration_seconds > 1 AND product_id IS NOT NULL
            ORDER BY user_id, event_id
            """
        ).fetchall()

    if not products or not views:
        raise ValueError("Generate the catalogue and simulated dwell events first.")

    by_visitor = defaultdict(list)
    for view in views:
        by_visitor[view["user_id"]].append(
            [view["product_id"], round(view["duration_seconds"], 3)]
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"products": [dict(product) for product in products],
             "sessions": list(by_visitor.values())},
            ensure_ascii=False,
            separators=(",", ":"),
        ) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=ROOT / "data/tessch.db")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/demo-data.json")
    args = parser.parse_args()
    export(args.database, args.output)
