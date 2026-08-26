"""Drive browsing sessions against a running server over HTTP.

    python -m scripts.devserver                      # in one terminal
    python -m scripts.simulate_browsing --personas 3 # in another

This exists to verify the one thing offline tests cannot: that the whole loop
works over the wire. It posts real dwell events to ``/api/events`` exactly as the
browser tracker does, then reads ``/api/recommendations/user/{id}`` back and
checks that each persona is shown a *different* list -- the Phase 2 acceptance
criterion.

The personas are deliberately given non-overlapping taste (one browses athletic
shoes, another boots, another sandals) because the interesting failure is not a
crash. It is a system that runs perfectly and returns the same popular products to
everybody, which looks like success until you compare two users side by side. So
this script asserts the lists *differ* rather than merely that the calls return
200.

Events are posted with ``source='live'``, so they are distinguishable from the
simulated dataset afterwards and can be removed with ``--cleanup``.

Uses only ``urllib`` from the standard library -- nothing here should depend on the
requests package being installed.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

DEFAULT_BASE = "http://127.0.0.1:8000"


@dataclass
class Persona:
    """A fake visitor with a coherent taste, so their picks should be predictable."""

    visitor_id: str
    label: str
    category: str
    viewed: list[tuple[str, float]] = field(default_factory=list)

    @property
    def total_seconds(self) -> float:
        return sum(s for _, s in self.viewed)


def call(base: str, path: str, payload: dict | None = None, timeout: float = 10.0):
    url = f"{base}{path}"
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body) if body else None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, {"detail": body[:200]}
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"Cannot reach {url}: {exc.reason}\n"
            f"Start the server first:  python -m scripts.devserver"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE, help="server base URL")
    parser.add_argument("--personas", type=int, default=3, help="how many visitors")
    parser.add_argument(
        "--views", type=int, default=6, help="products each persona opens"
    )
    parser.add_argument(
        "--seed", type=int, default=20260826, help="reproducible dwell times"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="delete the live events this run created, then exit",
    )
    parser.add_argument(
        "--keep", action="store_true", help="do not offer to clean up afterwards"
    )
    args = parser.parse_args(argv)
    rng = random.Random(args.seed)

    status, health = call(args.base, "/api/health")
    if status != 200:
        print(f"health check failed ({status}): {health}")
        return 1
    print(
        f"server up - status={health['status']} - products={health['products']} "
        f"- events={health['view_events']}"
    )

    if args.cleanup:
        return _cleanup(args.base)

    status, catalogue = call(args.base, "/api/products?limit=200")
    if status != 200 or not catalogue.get("items"):
        print(f"could not load catalogue ({status})")
        return 1
    products = catalogue["items"]

    by_category: dict[str, list[dict]] = {}
    for product in products:
        by_category.setdefault(product.get("category") or "Other", []).append(product)

    categories = sorted(by_category, key=lambda c: -len(by_category[c]))
    personas = [
        Persona(
            visitor_id=f"anon_{args.seed:x}{index:04x}"[:17],
            label=f"the {categories[index % len(categories)].lower()} shopper",
            category=categories[index % len(categories)],
        )
        for index in range(args.personas)
    ]

    print(f"\n-- Browsing as {len(personas)} visitors " + "-" * 30)
    for persona in personas:
        pool = by_category[persona.category][:]
        rng.shuffle(pool)
        others = [p for p in products if p.get("category") != persona.category]
        rng.shuffle(others)
        plan = pool[: max(1, args.views - 1)] + others[:1]

        for product in plan:
            seconds = round(rng.uniform(6.0, 95.0), 2)
            status, body = call(
                args.base,
                "/api/events",
                {
                    "user_id": persona.visitor_id,
                    "product_id": product["product_id"],
                    "duration_seconds": seconds,
                    "surface": "product_page",
                },
            )
            if status != 201:
                print(f"  ! rejected {product['product_id']}: {body}")
                continue
            persona.viewed.append((product["product_id"], seconds))

        print(
            f"  {persona.visitor_id}  {persona.label:28} "
            f"{len(persona.viewed)} views, {persona.total_seconds:.0f}s"
        )

    # Retrain so the new live events are reflected
    status, rebuilt = call(args.base, "/api/recommendations/rebuild", payload={})
    print(f"\nrebuild -> {status} {rebuilt.get('detail') or rebuilt}")

    print("\n-- Personalised lists " + "-" * 40)
    lists: dict[str, list[str]] = {}
    for persona in personas:
        status, response = call(
            args.base,
            f"/api/recommendations/user/{persona.visitor_id}?top_k=5",
        )
        if status != 200:
            print(f"  ! {persona.visitor_id}: {status} {response}")
            continue
        ids = [item["product_id"] for item in response["items"]]
        lists[persona.visitor_id] = ids
        print(f"\n  {persona.visitor_id} - {persona.label}")
        print(f"    strategy: {response['strategy']}")
        print(f"    reason:   {response.get('reason', '')}")
        for item in response["items"]:
            seen = round(item.get("total_seconds") or 0)
            print(
                f"      {item['product_id']} {item['product_name'][:26]:26} "
                f"{item.get('category', ''):9} {seen:>4}s"
            )

    ok = _report_distinctness(lists)

    if personas and personas[0].viewed:
        pid = personas[0].viewed[0][0]
        status, similar = call(args.base, f"/api/recommendations/similar/{pid}?top_k=4")
        print(f"\n-- Similar to {pid} " + "-" * 44)
        print(f"    strategy: {similar.get('strategy')}")
        for item in similar.get("items", []):
            score = item.get("score")
            score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "-"
            print(
                f"      {score_text}  {item['product_id']} "
                f"{item['product_name'][:26]:26} {item.get('category', '')}"
            )

    if not args.keep:
        print(
            "\nThese events are tagged source='live'. Remove them with:\n"
            "  python -m scripts.simulate_browsing --cleanup"
        )
    return 0 if ok else 1


def _report_distinctness(lists: dict[str, list[str]]) -> bool:
    """The actual Phase 2 assertion: different visitors get different lists."""
    print("\n-- Are the lists actually different? " + "-" * 26)
    if len(lists) < 2:
        print("  need at least two personas to compare")
        return False

    ids = list(lists)
    worst = 1.0
    identical = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = set(lists[ids[i]]), set(lists[ids[j]])
            union = a | b
            overlap = len(a & b) / len(union) if union else 1.0
            worst = min(worst, 1.0 - overlap)
            flag = "IDENTICAL" if a == b else f"{overlap:.0%} overlap"
            if a == b:
                identical.append((ids[i], ids[j]))
            print(f"  {ids[i]} vs {ids[j]}: {flag}")

    if identical:
        print(
            "\n  FAIL - some visitors received identical lists. That is the "
            "failure mode this check exists for: the pipeline runs, but "
            "personalisation is not happening."
        )
        return False
    print("\n  PASS - every visitor received a distinguishable list.")
    return True


def _cleanup(base: str) -> int:
    status, before = call(base, "/api/provenance")
    print(f"live events before: {before['live']['views']}")
    print(
        "The API has no destructive endpoint by design, so cleanup runs directly "
        "against the database rather than over HTTP."
    )
    from backend.database import queries, writes

    conn = queries.connect()
    try:
        removed = writes.delete_live_events(conn)
    finally:
        conn.close()
    print(f"deleted {removed} live events (simulated data untouched)")

    status, after = call(base, "/api/provenance")
    print(
        f"live events after: {after['live']['views']} - "
        f"simulated still {after['simulated']['views']}"
    )
    call(base, "/api/recommendations/rebuild", payload={})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
