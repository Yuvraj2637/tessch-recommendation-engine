"""A stdlib-only mirror of the FastAPI app, for running without dependencies.

    python -m scripts.devserver          # http://127.0.0.1:8000

**Why this exists.** The real API is ``backend/api/app.py`` and that is what you
should run. But it needs FastAPI, uvicorn and Pydantic installed, and there are
situations — a locked-down machine, a fresh clone, a CI container with no network
— where you want to confirm the system genuinely works end to end before sorting
out a virtualenv. This file does that using nothing but the standard library and
pandas.

**The important property:** it calls exactly the same functions the FastAPI
routers call — ``services.ingest.record_view``, ``services.recommend.*``. It does
not reimplement any logic. So a route working here means the service layer works;
the only thing it does not exercise is FastAPI's own request parsing. Routes are
kept in the same order and shape as the routers so the two can be diffed by eye.

It is single-threaded and has no reload. It is a verification tool, not a
deployment.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from backend.core.config import FRONTEND_DIR, get_settings
from backend.database import queries, writes
from backend.services import ingest, recommend
from recommendation.similarity import SKLEARN_AVAILABLE

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}


class Router:
    """Minimal path-pattern router: ``/api/products/{product_id}``."""

    def __init__(self) -> None:
        self._routes: list[tuple[str, re.Pattern[str], Callable[..., Any]]] = []

    def add(self, method: str, pattern: str, handler: Callable[..., Any]) -> None:
        regex = re.sub(r"\{(\w+)\}", r"(?P<\1>[^/]+)", pattern)
        self._routes.append((method, re.compile(f"^{regex}$"), handler))

    def match(
        self, method: str, path: str
    ) -> tuple[Callable[..., Any], dict[str, str]] | None:
        for route_method, regex, handler in self._routes:
            match = regex.match(path)
            if match and route_method == method:
                return handler, match.groupdict()
        return None


router = Router()


# ── Handlers: these mirror backend/api/routers/*.py ──────────────────────────
def h_health(conn: sqlite3.Connection, **_: Any) -> dict[str, Any]:
    counts = queries.table_counts(conn)
    model = recommend.get_similarity_model(conn)
    products = int(counts.get("products", 0))
    events = int(counts.get("user_events", 0))
    return {
        "status": "ok" if (products and events) else "degraded",
        "database": True,
        "products": products,
        "view_events": events,
        "similarity_model": model is not None,
        "sklearn": SKLEARN_AVAILABLE,
    }


def h_provenance(conn: sqlite3.Connection, **_: Any) -> dict[str, Any]:
    return queries.data_provenance(conn)


def h_products(conn: sqlite3.Connection, query: dict[str, list[str]], **_: Any) -> dict[str, Any]:
    limit = _int_param(query, "limit", 100, 1, 500)
    offset = _int_param(query, "offset", 0, 0, 10_000)
    available_only = _bool_param(query, "available_only", False)
    category = query.get("category", [None])[0]

    products = queries.load_products(conn, available_only=available_only)
    if category:
        products = products[products["category"].str.lower() == category.lower()]
    total = int(len(products))
    page = products.iloc[offset : offset + limit]
    cards = recommend.product_cards(page, [str(p) for p in page["product_id"]])
    return {"items": cards, "count": len(cards), "total": total}


def h_product(conn: sqlite3.Connection, product_id: str, **_: Any) -> dict[str, Any]:
    products = queries.load_products(conn)
    cards = recommend.product_cards(products, [product_id.upper()])
    if not cards:
        raise HttpError(404, f"No product with id {product_id.upper()!r}")
    return cards[0]


def h_rec_user(
    conn: sqlite3.Connection, user_id: str, query: dict[str, list[str]], **_: Any
) -> dict[str, Any]:
    top_k = _top_k(query)
    response = recommend.recommend_for_user(conn, user_id, top_k=top_k)
    if _bool_param(query, "log", False) and response.items:
        writes.log_recommendations(
            conn,
            user_id,
            [
                (i["product_id"], i.get("rank") or n, i.get("score") or 0.0)
                for n, i in enumerate(response.items, start=1)
            ],
            response.model_name,
            response.model_version,
        )
    return response.as_dict()


def h_rec_similar(
    conn: sqlite3.Connection, product_id: str, query: dict[str, list[str]], **_: Any
) -> dict[str, Any]:
    return recommend.recommend_similar(
        conn,
        product_id.upper(),
        top_k=_top_k(query),
        exclude_seen_by=query.get("exclude_seen_by", [None])[0],
    ).as_dict()


def h_rec_popular(
    conn: sqlite3.Connection, query: dict[str, list[str]], **_: Any
) -> dict[str, Any]:
    return recommend._popular_fallback(
        conn, _top_k(query), reason="Most-viewed products across all visitors."
    ).as_dict()


def h_model(conn: sqlite3.Connection, **_: Any) -> dict[str, Any]:
    model = recommend.get_similarity_model(conn)
    if model is None:
        return {"trained": False}
    return _model_payload(model)


def h_rebuild(conn: sqlite3.Connection, **_: Any) -> dict[str, Any]:
    recommend.invalidate_model_cache()
    model = recommend.get_similarity_model(conn, force_rebuild=True)
    if model is None:
        raise HttpError(409, "Not enough viewing data to fit a similarity model.")
    return _model_payload(model)


def _model_payload(model: Any) -> dict[str, Any]:
    coverage = model.coverage()
    return {
        "trained": True,
        "backend": model.backend,
        "normalisation": model.normalisation,
        "n_users": model.n_users,
        "n_products": int(coverage["n_products"]),
        "observed_cells": model.observed_cells,
        "products_with_neighbours": int(coverage["products_with_neighbours"]),
        "cold_products": int(coverage["cold_products"]),
        "coverage": float(coverage["coverage"]),
    }


def h_event(conn: sqlite3.Connection, body: dict[str, Any], **_: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise HttpError(422, "expected a JSON object")
    try:
        result = ingest.record_view(
            conn,
            body.get("user_id"),
            body.get("product_id"),
            body.get("duration_seconds"),
            timestamp=body.get("timestamp"),
            surface=body.get("surface"),
            position=body.get("position"),
        )
    except ingest.ValidationError as exc:
        raise HttpError(422, str(exc)) from exc
    return {
        "status": "recorded",
        "event_id": result.event_id,
        "user_id": result.user_id,
        "product_id": result.product_id,
        "duration_seconds": result.duration_seconds,
        "clamped": result.clamped,
        "user_created": result.user_created,
    }


def h_event_batch(
    conn: sqlite3.Connection, body: dict[str, Any], **_: Any
) -> dict[str, Any]:
    events = (body or {}).get("events")
    if not isinstance(events, list) or not events:
        raise HttpError(422, "events: expected a non-empty list")

    stats = ingest.IngestStats()
    errors: list[str] = []
    for index, event in enumerate(events[:100]):
        try:
            ingest.record_view(
                conn,
                (event or {}).get("user_id"),
                (event or {}).get("product_id"),
                (event or {}).get("duration_seconds"),
                timestamp=(event or {}).get("timestamp"),
                surface=(event or {}).get("surface"),
                position=(event or {}).get("position"),
                stats=stats,
            )
        except ingest.ValidationError as exc:
            errors.append(f"event[{index}]: {exc}")
    return {
        "accepted": stats.accepted,
        "rejected": stats.rejected,
        "clamped": stats.clamped,
        "errors": errors,
    }


router.add("GET", "/api/health", h_health)
router.add("GET", "/api/provenance", h_provenance)
router.add("GET", "/api/products", h_products)
router.add("GET", "/api/products/{product_id}", h_product)
router.add("GET", "/api/recommendations/popular", h_rec_popular)
router.add("GET", "/api/recommendations/model", h_model)
router.add("GET", "/api/recommendations/user/{user_id}", h_rec_user)
router.add("GET", "/api/recommendations/similar/{product_id}", h_rec_similar)
router.add("POST", "/api/events", h_event)
router.add("POST", "/api/events/batch", h_event_batch)
router.add("POST", "/api/recommendations/rebuild", h_rebuild)


# ── Plumbing ────────────────────────────────────────────────────────────────
class HttpError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _int_param(
    query: dict[str, list[str]], key: str, default: int, low: int, high: int
) -> int:
    try:
        value = int(query.get(key, [default])[0])
    except (TypeError, ValueError):
        return default
    return max(low, min(value, high))


def _bool_param(query: dict[str, list[str]], key: str, default: bool) -> bool:
    raw = query.get(key, [None])[0]
    if raw is None:
        return default
    return str(raw).lower() in {"1", "true", "yes", "on"}


def _top_k(query: dict[str, list[str]]) -> int:
    return _int_param(query, "top_k", get_settings().default_top_k, 1, 50)


class Handler(BaseHTTPRequestHandler):
    server_version = "TesschDevServer/1.0"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path == "/" or path == "/index.html":
            return self._send_file(FRONTEND_DIR / "index.html")
        if path.startswith("/product/"):
            # Mirrors the FastAPI route: one template, id read client-side.
            return self._send_file(FRONTEND_DIR / "product.html")
        if path.startswith("/static/"):
            return self._send_static(path[len("/static/") :])
        if path == "/docs":
            return self._send_json(
                200,
                {
                    "detail": (
                        "Interactive docs require the FastAPI app. Run: "
                        "uvicorn backend.api.app:app --reload"
                    ),
                    "routes": sorted(
                        p.pattern for _, p, _ in router._routes  # noqa: SLF001
                    ),
                },
            )
        return self._dispatch("GET", path, parse_qs(parsed.query), None)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        try:
            body = self._read_json()
        except ValueError as exc:
            return self._send_json(400, {"detail": str(exc)})
        return self._dispatch("POST", path, parse_qs(parsed.query), body)

    def _dispatch(
        self,
        method: str,
        path: str,
        query: dict[str, list[str]],
        body: Any,
    ) -> None:
        matched = router.match(method, path)
        if matched is None:
            return self._send_json(404, {"detail": f"No route for {method} {path}"})

        handler, params = matched
        if not queries.database_exists():
            return self._send_json(
                503,
                {
                    "detail": (
                        "Database not found or empty. Create it with: "
                        "python -m scripts.init_db && python -m scripts.generate_events"
                    )
                },
            )

        conn = queries.connect()
        try:
            payload = handler(conn, query=query, body=body, **params)
            self._send_json(201 if path == "/api/events" else 200, payload)
        except HttpError as exc:
            self._send_json(exc.status, {"detail": exc.detail})
        except Exception as exc:  # pragma: no cover - dev tool
            import traceback

            traceback.print_exc()
            self._send_json(500, {"detail": f"{type(exc).__name__}: {exc}"})
        finally:
            conn.close()

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length") or 0)
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid JSON body: {exc}") from exc

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, relative: str) -> None:
        # Resolve and confirm the result is inside FRONTEND_DIR. Without this a
        # request for /static/../../.env would read arbitrary files.
        target = (FRONTEND_DIR / relative).resolve()
        try:
            target.relative_to(FRONTEND_DIR.resolve())
        except ValueError:
            return self._send_json(403, {"detail": "path outside static root"})
        self._send_file(target)

    def _send_file(self, path: Path) -> None:
        if not path.is_file():
            return self._send_json(404, {"detail": f"Not found: {path.name}"})
        data = path.read_bytes()
        self.send_response(200)
        self.send_header(
            "Content-Type", CONTENT_TYPES.get(path.suffix, "application/octet-stream")
        )
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[dev] {fmt % args}")


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=settings.api_host)
    parser.add_argument("--port", type=int, default=settings.api_port)
    args = parser.parse_args()

    print(f"[dev] database: {settings.database_url}")
    print(f"[dev] storefront: http://{args.host}:{args.port}/")
    print("[dev] NOTE: this is the dependency-free mirror of backend/api/app.py.")
    print("[dev]       For the real API and its /docs page, run:")
    print("[dev]       uvicorn backend.api.app:app --reload")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[dev] stopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
