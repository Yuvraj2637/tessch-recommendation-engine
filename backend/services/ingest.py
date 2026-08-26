"""Phase 1 — accepting tracked views, and rejecting the ones that would poison the data.

The plan asks for "basic validation so junk data or negative values don't corrupt
the database". That framing understates the stakes slightly, so it is worth being
explicit about why each rule exists: this table is the *only* input to both
recommenders. There is no second source to cross-check against and no labelled
ground truth to correct against. A bad row is not noise that averages out — it is
a permanent, silent distortion of every ranking computed afterwards.

The rules, and what each one is actually defending against:

``duration_seconds`` **must be a finite number.** ``NaN`` and ``Infinity`` are
valid JSON-ish values that some clients will happily send, and either one
propagates through a sum to make a user's entire profile ``NaN``.

**Below the floor (default 1s), the view is discarded.** A sub-second page view
is a mis-click or a bounce through a listing page. It carries no preference
information, and admitting it inflates ``view_count`` for products that merely sit
in a high-traffic position.

**Above the ceiling (default 1800s), the view is clamped, not discarded.** This is
the one asymmetry, and it is deliberate. A 3-hour reading almost certainly means
a tab was left open, but the visitor *did* open that product deliberately — the
signal is real, only the magnitude is wrong. Discarding it throws away a genuine
interaction; clamping keeps it while capping its influence. Since Phase 2 sums
dwell time, a single unclamped 3-hour reading would outrank a genuine hour of
browsing across a dozen products.

**The product must exist in the catalogue.** Client-supplied ids reach this
endpoint, so this check is both a data-integrity guard and the thing standing
between a typo'd URL and a foreign-key crash on insert.

**The user id must match the expected anonymous format.** Without this, anything
a client sends becomes a row in ``users``. Constraining the shape keeps live
visitors distinguishable from simulated ids and bounds what can be written.

Rejections are reported back to the caller with a reason, and counted. A tracker
that silently drops a third of its events looks identical to one that works.
"""

from __future__ import annotations

import math
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from backend.core.config import IngestSettings
from backend.core.constants import ANON_USER_PREFIX, MAX_USER_ID_LENGTH, SOURCE_LIVE
from backend.database import queries, writes

#: Anonymous ids the tracker generates: `anon_` plus 12 lowercase hex chars.
#: Matches the generator in frontend/js/tracker.js — if you change one, change both.
ANON_ID_PATTERN = re.compile(rf"^{re.escape(ANON_USER_PREFIX)}[0-9a-f]{{8,16}}$")

#: How far ahead of server time a client timestamp may sit before it is replaced.
#: Browser clocks drift and are user-settable; a few minutes is normal skew,
#: anything beyond it would let a client inject events that permanently rank
#: first under time decay.
MAX_CLOCK_SKEW = timedelta(minutes=5)


class ValidationError(ValueError):
    """A tracked event was rejected. The message is safe to return to a client."""


@dataclass
class IngestResult:
    """Outcome of one accepted event."""

    event_id: str
    user_id: str
    product_id: str
    duration_seconds: float
    clamped: bool = False
    user_created: bool = False


@dataclass
class IngestStats:
    """Running counts, so drop rates are observable rather than invisible."""

    accepted: int = 0
    rejected: int = 0
    clamped: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def record_rejection(self, reason: str) -> None:
        self.rejected += 1
        key = reason.split(":")[0].strip()
        self.reasons[key] = self.reasons.get(key, 0) + 1

    def as_dict(self) -> dict[str, object]:
        total = self.accepted + self.rejected
        return {
            "accepted": self.accepted,
            "rejected": self.rejected,
            "clamped": self.clamped,
            "total": total,
            "reject_rate": round(self.rejected / total, 4) if total else 0.0,
            "reasons": dict(self.reasons),
        }


def validate_user_id(user_id: object) -> str:
    """Check an anonymous visitor id, returning the cleaned value."""
    if not isinstance(user_id, str):
        raise ValidationError("user_id: must be a string")
    candidate = user_id.strip()
    if not candidate:
        raise ValidationError("user_id: must not be empty")
    if len(candidate) > MAX_USER_ID_LENGTH:
        raise ValidationError(
            f"user_id: longer than the {MAX_USER_ID_LENGTH}-character column"
        )
    if not ANON_ID_PATTERN.match(candidate):
        raise ValidationError(
            f"user_id: expected the form {ANON_USER_PREFIX}<8-16 hex chars>"
        )
    return candidate


def validate_duration(
    seconds: object, settings: IngestSettings
) -> tuple[float, bool]:
    """Check a dwell reading. Returns ``(value, was_clamped)``."""
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise ValidationError("duration_seconds: must be a number")

    value = float(seconds)
    # Checked before the range comparisons because NaN compares False against
    # everything, so `NaN < min` would not catch it.
    if not math.isfinite(value):
        raise ValidationError("duration_seconds: must be finite")
    if value < 0:
        raise ValidationError("duration_seconds: must not be negative")
    if value < settings.min_dwell_seconds:
        raise ValidationError(
            f"duration_seconds: below the {settings.min_dwell_seconds}s floor "
            "(treated as a mis-click)"
        )
    if value > settings.max_dwell_seconds:
        return settings.max_dwell_seconds, True
    return value, False


def validate_timestamp(value: object) -> datetime:
    """Parse a client timestamp, falling back to server time when unusable.

    Client clocks are not trustworthy, but they are usually right, and using them
    keeps the ordering of events within one session correct even if the beacon
    arrives late. The compromise: accept the client's value unless it is
    unparseable or implausibly far in the future, in which case use server time.
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    if value is None:
        return now
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return now
    else:
        return now

    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    if parsed > now + MAX_CLOCK_SKEW:
        return now
    return parsed


def record_view(
    conn: sqlite3.Connection,
    user_id: object,
    product_id: object,
    duration_seconds: object,
    *,
    timestamp: object = None,
    surface: str | None = None,
    position: int | None = None,
    settings: IngestSettings | None = None,
    stats: IngestStats | None = None,
    source: str = SOURCE_LIVE,
) -> IngestResult:
    """Validate and store one tracked product view.

    Raises ``ValidationError`` with a client-safe reason on rejection. The caller
    (the API router, or the dev server) turns that into a 422.
    """
    cfg = settings or IngestSettings.from_env()

    try:
        clean_user = validate_user_id(user_id)
        clean_product = _validate_product_id(product_id)
        seconds, clamped = validate_duration(duration_seconds, cfg)
        when = validate_timestamp(timestamp)

        # Checked last: it is the only rule requiring a database round-trip, so
        # the cheap in-memory rules get to reject malformed input first.
        if not queries.product_exists(conn, clean_product):
            raise ValidationError(f"product_id: {clean_product!r} is not in the catalogue")
    except ValidationError as exc:
        if stats is not None:
            stats.record_rejection(str(exc))
        raise

    # One transaction for the user upsert and the event, so a failure cannot
    # leave a placeholder user with no event explaining why it exists.
    created = writes.ensure_user(conn, clean_user, now=when, commit=False)
    event_id = writes.insert_view_event(
        conn,
        clean_user,
        clean_product,
        seconds,
        timestamp=when,
        surface=surface,
        position=position,
        source=source,
        commit=False,
    )
    conn.commit()

    if stats is not None:
        stats.accepted += 1
        if clamped:
            stats.clamped += 1

    return IngestResult(
        event_id=event_id,
        user_id=clean_user,
        product_id=clean_product,
        duration_seconds=seconds,
        clamped=clamped,
        user_created=created,
    )


def _validate_product_id(product_id: object) -> str:
    if not isinstance(product_id, str):
        raise ValidationError("product_id: must be a string")
    candidate = product_id.strip().upper()
    if not candidate:
        raise ValidationError("product_id: must not be empty")
    # Shape-checked before the database lookup so a hostile or malformed value
    # never reaches a query, and so the error explains the format.
    if not re.match(r"^T\d{3}$", candidate):
        raise ValidationError("product_id: expected the form T### (e.g. T042)")
    return candidate


__all__ = [
    "ANON_ID_PATTERN",
    "ValidationError",
    "IngestResult",
    "IngestStats",
    "validate_user_id",
    "validate_duration",
    "validate_timestamp",
    "record_view",
]
