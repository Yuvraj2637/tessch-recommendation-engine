"""The FastAPI application.

Run it with::

    uvicorn backend.api.app:app --reload

The interactive docs at http://127.0.0.1:8000/docs are worth pointing out: they
are generated from the Pydantic schemas, so they cannot describe an endpoint that
does not exist or omit a field that does. For a demo that means the API is
explorable without a client, and for the recording it means the whole system can
be exercised from a browser.

The storefront is mounted from this same app rather than served separately. That
is not laziness — it makes the tracker's requests same-origin, which sidesteps the
class of CORS and cookie problems that otherwise eat an afternoon. CORS is still
configured, for the case where someone opens the HTML from the filesystem.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from backend.api.routers import events, ops, products, recommendations
from backend.core.config import FRONTEND_DIR, get_settings
from backend.database import queries

DESCRIPTION = """
Behaviour-based product recommendations for the TESSCH modular sneaker store.

The system records how long visitors spend on each product page and turns that
dwell time into two recommendation surfaces:

* **Picked for you** (`/api/recommendations/user/{user_id}`) — ranks the products
  a single visitor gave the most attention to, weighted toward recent views.
* **You might also like** (`/api/recommendations/similar/{product_id}`) — pools
  dwell time across all visitors into a user-item matrix and computes item-to-item
  cosine similarity, so viewing one product surfaces others that attract the same
  kind of attention.

Every response carries a `strategy` field identifying which model produced it,
including when it fell back to popularity — so a personalised list is never
confused with a generic one.

Most of the training data is simulated; `/api/provenance` reports the exact split.
"""

TAGS_METADATA = [
    {"name": "tracking", "description": "Phase 1 — recording dwell time."},
    {
        "name": "recommendations",
        "description": "Phases 2 and 3 — personalised and cross-user recommendations.",
    },
    {"name": "catalogue", "description": "Products, for the storefront to render."},
    {"name": "ops", "description": "Health and data provenance."},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the similarity model at boot, and report the journal mode.

    Fitting on first request would make one unlucky visitor wait for a full scan of
    the event log. Doing it here moves that cost to startup, where nobody is
    watching, and surfaces a data problem at boot rather than mid-demo.

    Failures are caught and logged rather than raised: a server that will not start
    because it has no data yet is less useful than one that starts and says so
    through ``/api/health``.
    """
    settings = get_settings()
    print(f"[tessch] database: {settings.database_url}")

    if queries.database_exists():
        try:
            conn = queries.connect()
            try:
                print(f"[tessch] journal_mode: {queries.journal_mode(conn)}")
                counts = queries.table_counts(conn)
                print(
                    f"[tessch] catalogue: {counts.get('products', 0)} products, "
                    f"{counts.get('user_events', 0)} events"
                )
                from backend.services.recommend import get_similarity_model

                model = get_similarity_model(conn)
                if model is None:
                    print("[tessch] similarity model: not enough data yet")
                else:
                    cov = model.coverage()
                    print(
                        f"[tessch] similarity model: {model.backend}, "
                        f"{model.n_users} users, coverage "
                        f"{cov['products_with_neighbours']}/{cov['n_products']}"
                    )
            finally:
                conn.close()
        except Exception as exc:  # pragma: no cover - startup diagnostics only
            print(f"[tessch] warm-up skipped: {exc}")
    else:
        print("[tessch] no database yet — run: python -m scripts.init_db")

    yield


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="TESSCH Recommendation Engine",
        description=DESCRIPTION,
        version="1.0.0",
        openapi_tags=TAGS_METADATA,
        lifespan=lifespan,
    )

    # Permissive by default because the only client is a local mock storefront
    # that may be opened straight from the filesystem (origin `null`). A real
    # deployment would pin this to the storefront's domain — the setting exists so
    # that is a config change rather than a code change.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(events.router)
    app.include_router(recommendations.router)
    app.include_router(products.router)
    app.include_router(ops.router)

    _mount_storefront(app)
    return app


def _mount_storefront(app: FastAPI) -> None:
    """Serve the mock storefront from the API process, if it is present.

    Guarded so the API still boots in an environment where only the backend was
    deployed — a missing frontend directory would otherwise crash ``StaticFiles``
    at import time.
    """
    if not FRONTEND_DIR.is_dir():
        @app.get("/", include_in_schema=False)
        def _docs_redirect() -> RedirectResponse:
            return RedirectResponse("/docs")

        return

    app.mount(
        "/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static"
    )

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/product/{product_id}", include_in_schema=False)
    def product_page(product_id: str) -> FileResponse:
        """Serve one HTML file for every product URL.

        The product id is read from the path by client-side JavaScript. A single
        template keeps the storefront honest about being a mock: there is no
        server-side rendering to confuse with the part of the system that matters.
        """
        return FileResponse(FRONTEND_DIR / "product.html")


app = create_app()
