"""Phase 1 — the tracking endpoint.

Thin by design: parse, hand to ``services.ingest``, translate the outcome into a
status code. All of the judgement about what counts as a valid view lives in the
service, so the identical rules apply whether an event arrives here, through the
batch endpoint, or through the zero-dependency dev server.
"""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, status

from backend.api import schemas
from backend.api.deps import get_db
from backend.services import ingest

router = APIRouter(prefix="/api/events", tags=["tracking"])


@router.post(
    "",
    response_model=schemas.ViewEventOut,
    status_code=status.HTTP_201_CREATED,
    summary="Record how long a visitor spent on a product page",
    responses={422: {"description": "The event failed validation and was not stored."}},
)
def record_view(
    payload: schemas.ViewEventIn, conn: sqlite3.Connection = Depends(get_db)
) -> schemas.ViewEventOut:
    """Store one dwell-time observation.

    Returns 422 with the specific reason when an event is rejected. That detail is
    deliberate: the tracker is fire-and-forget, so without a readable reason a
    silently-dropped third of all events would be indistinguishable from a working
    system.
    """
    try:
        result = ingest.record_view(
            conn,
            payload.user_id,
            payload.product_id,
            payload.duration_seconds,
            timestamp=payload.timestamp,
            surface=payload.surface,
            position=payload.position,
        )
    except ingest.ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    return schemas.ViewEventOut(
        event_id=result.event_id,
        user_id=result.user_id,
        product_id=result.product_id,
        duration_seconds=result.duration_seconds,
        clamped=result.clamped,
        user_created=result.user_created,
    )


@router.post(
    "/batch",
    response_model=schemas.ViewEventBatchOut,
    summary="Record several views at once",
)
def record_batch(
    payload: schemas.ViewEventBatchIn, conn: sqlite3.Connection = Depends(get_db)
) -> schemas.ViewEventBatchOut:
    """Store a queue of views, keeping the good ones when some are bad.

    Deliberately **not** all-or-nothing. A batch is whatever the client had queued,
    so one malformed event should not discard the visitor's other perfectly good
    views. Rejections are reported per-event instead, and the response always
    returns 200 — the request itself was well-formed even if some of its contents
    were not.
    """
    stats = ingest.IngestStats()
    errors: list[str] = []

    for index, event in enumerate(payload.events):
        try:
            ingest.record_view(
                conn,
                event.user_id,
                event.product_id,
                event.duration_seconds,
                timestamp=event.timestamp,
                surface=event.surface,
                position=event.position,
                stats=stats,
            )
        except ingest.ValidationError as exc:
            errors.append(f"event[{index}]: {exc}")

    return schemas.ViewEventBatchOut(
        accepted=stats.accepted,
        rejected=stats.rejected,
        clamped=stats.clamped,
        errors=errors,
    )
