"""Operational endpoints — is this thing working, and where did its data come from?

``/api/health`` is the first thing to hit when the demo misbehaves, and it answers
the three questions that actually go wrong: is there a database, is there any data
in it, and did a model get fitted.

``/api/provenance`` exists because of a specific claim this project has to make
honestly. Phase 3 is trained mostly on simulated data, and the write-up must say
so. Computing that split by hand invites it drifting out of date, so it is an
endpoint whose numbers come straight from the ``source`` column on every event.
"""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends

from backend.api import schemas
from backend.api.deps import get_db
from backend.database import queries
from backend.services import recommend
from recommendation.similarity import SKLEARN_AVAILABLE

router = APIRouter(prefix="/api", tags=["ops"])


@router.get("/health", response_model=schemas.HealthOut, summary="Service health")
def health(conn: sqlite3.Connection = Depends(get_db)) -> schemas.HealthOut:
    """Report whether the system can actually serve recommendations.

    Returns ``degraded`` rather than an error when the catalogue or the event log is
    empty. The process is up and correct; it just has nothing to recommend yet, and
    conflating "not populated" with "broken" would send someone debugging the wrong
    thing.
    """
    counts = queries.table_counts(conn)
    products = int(counts.get("products", 0))
    events = int(counts.get("user_events", 0))
    model = recommend.get_similarity_model(conn)

    healthy = products > 0 and events > 0
    return schemas.HealthOut(
        status="ok" if healthy else "degraded",
        database=True,
        products=products,
        view_events=events,
        similarity_model=model is not None,
        sklearn=SKLEARN_AVAILABLE,
    )


@router.get(
    "/provenance",
    response_model=schemas.ProvenanceOut,
    summary="How much of the data is real browser traffic vs simulated",
)
def provenance(conn: sqlite3.Connection = Depends(get_db)) -> schemas.ProvenanceOut:
    """The real-vs-simulated split, straight from the ``source`` column.

    ``live_share`` will be small, and that is the honest state of a project built
    in three weeks without production traffic. It is reported rather than buried
    because a similarity model trained on synthetic co-views is a different claim
    from one trained on real ones, and anyone evaluating this work is entitled to
    know which they are looking at.
    """
    return schemas.ProvenanceOut(**queries.data_provenance(conn))
