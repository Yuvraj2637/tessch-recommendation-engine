"""Website event simulator.

Produces the raw event stream that everything downstream is built from. The
guiding principle is that the output should have the *same pathologies as real
clickstream data*, because a recommender that only works on clean data is
worthless:

* **Exposure bias.** A user can only interact with something the site showed
  them, and the site shows popular things. Every impression is logged, so
  "not clicked" is distinguishable from "never seen".
* **Position bias.** Slot 1 in a grid gets clicked far more than slot 8, at
  equal preference.
* **A funnel, not a coin flip.** impression -> hover -> click -> view ->
  wishlist -> cart -> purchase, each with its own drop-off.
* **Heavy tails.** A few users generate a large share of events; a few products
  absorb a large share of impressions.
* **Seasonality and campaigns.** Sandals in summer, boots in winter, two sale
  spikes with elevated conversion.
* **Intra-session repetition.** Users return to the same product several times
  in one visit before committing, which is the signal Phase 4 calls out.
* **Cross-session memory.** Wishlists and carts persist; people come back to
  them days later.
* **A legacy recommender.** The "Recommended for you" slot already exists and is
  popularity-driven, and its impressions are logged to ``recommendation_logs``.
  That gives Stage 7 a real historical baseline CTR to beat rather than a
  hypothetical one.
"""

from __future__ import annotations

import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable

import numpy as np

from backend.core import constants as C
from simulation.catalog import Catalogue
from simulation.personas import ARCHETYPES, UserProfile, build_user_population, mass_appeal

# ─────────────────────────────────────────────────────────────────────────────
# Tunables. Every probability in the simulator is named here rather than being
# an inline magic number, so the generative assumptions can be audited.
# ─────────────────────────────────────────────────────────────────────────────

GRID_SIZES: dict[str, int] = {
    C.SURFACE_HOME_GRID: 8,
    C.SURFACE_CATEGORY: 8,
    C.SURFACE_SEARCH: 6,
    C.SURFACE_RELATED: 6,
    C.SURFACE_RECOMMENDED: 6,
    C.SURFACE_WISHLIST: 6,
}

#: Click-through rate for a perfectly-matched product in slot 1.
CTR_CEILING = 0.42
#: Position bias: slot n gets 1 / (1 + POSITION_DECAY * (n - 1)) of slot 1.
POSITION_DECAY = 0.28
#: Sharpness of the affinity -> click relationship.
CLICK_AFFINITY_POWER = 1.6
#: Maximum clicks a user will make from a single grid render.
MAX_CLICKS_PER_GRID = 2
#: Chance a click is preceded by a logged hover (desktop only, mostly).
HOVER_BEFORE_CLICK = 0.62
#: Standalone hovers with no click — real grids generate plenty of these.
IDLE_HOVER_RATE = 0.10

DWELL_FLOOR_SECONDS = 1.5
DWELL_CEILING_SECONDS = 900.0
DWELL_BASE_SECONDS = 4.0
DWELL_AFFINITY_GAIN = 58.0
DWELL_AFFINITY_POWER = 1.6
DWELL_SIGMA = 0.55
#: Probability of an instant bounce back out of a PDP, given low affinity.
BOUNCE_RATE_AT_ZERO_AFFINITY = 0.38

WISHLIST_CEILING = 0.14
WISHLIST_POWER = 2.6
CART_CEILING = 0.09
CART_POWER = 3.0
SHARE_CEILING = 0.006
CHECKOUT_CEILING = 0.68
#: Once a product is bought, interest in that exact item collapses.
POST_PURCHASE_CLICK_DAMPING = 0.35
#: Fraction of a cart that survives to checkout when the user does convert.
CART_ITEM_CONVERSION = 0.80
#: Chance of pruning the cart instead of buying.
CART_ABANDON_PRUNE_RATE = 0.55

INTENT_WEIGHTS: dict[str, float] = {
    "quick_bounce": 0.27,
    "casual_browse": 0.45,
    "focused_research": 0.19,
    "purchase_intent": 0.09,
}
INTENT_STEPS: dict[str, tuple[int, int]] = {
    "quick_bounce": (1, 2),
    "casual_browse": (2, 4),
    "focused_research": (3, 7),
    "purchase_intent": (2, 5),
}
#: Multiplier applied to wishlist / cart / checkout probabilities.
INTENT_COMMITMENT: dict[str, float] = {
    "quick_bounce": 0.25,
    "casual_browse": 0.75,
    "focused_research": 1.25,
    "purchase_intent": 2.3,
}
#: Chance of re-viewing a product already seen this session.
INTENT_REVISIT: dict[str, float] = {
    "quick_bounce": 0.01,
    "casual_browse": 0.09,
    "focused_research": 0.26,
    "purchase_intent": 0.22,
}

#: Traffic by hour of day (local), bimodal around lunch and evening.
HOUR_WEIGHTS = np.array(
    [0.6, 0.35, 0.22, 0.16, 0.14, 0.2, 0.5, 1.0, 1.6, 2.1, 2.4, 2.6,
     2.9, 2.7, 2.4, 2.3, 2.5, 3.0, 3.6, 4.2, 4.6, 4.0, 2.6, 1.4],
    dtype=float,
)
#: Monday..Sunday traffic multiplier.
WEEKDAY_WEIGHTS = np.array([0.95, 0.93, 0.97, 1.02, 1.15, 1.28, 1.18], dtype=float)

#: Days before ``end_date`` at which sale campaigns run, and their effect.
SALE_OFFSETS_DAYS: tuple[tuple[int, int], ...] = ((128, 4), (46, 5))
SALE_TRAFFIC_MULTIPLIER = 2.3
SALE_CONVERSION_MULTIPLIER = 1.85

#: Merchandising boosts for the curated hero products named in the brief.
HERO_BOOSTS: dict[str, float] = {"T001": 2.3, "T002": 1.9, "T003": 1.55}
CORE_CATALOGUE_BOOST = 1.5


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class UserState:
    """Cross-session memory for one shopper."""

    wishlist: set[str] = field(default_factory=set)
    cart: set[str] = field(default_factory=set)
    purchased: set[str] = field(default_factory=set)
    recent: deque[str] = field(default_factory=lambda: deque(maxlen=12))
    lifetime_views: Counter = field(default_factory=Counter)
    last_seen_at: datetime | None = None
    sessions: int = 0


@dataclass
class SimulationOutput:
    """Everything one simulation run produced."""

    products: list[dict[str, Any]]
    users: list[dict[str, Any]]
    sessions: list[dict[str, Any]]
    events: list[dict[str, Any]]
    orders: list[dict[str, Any]]
    recommendation_logs: list[dict[str, Any]]
    hidden_users: list[dict[str, Any]]
    affinity_long: dict[str, np.ndarray]
    meta: dict[str, Any]


def _weighted_sample(
    rng: np.random.Generator, weights: np.ndarray, k: int, candidates: np.ndarray | None = None
) -> np.ndarray:
    """Sample ``k`` distinct indices with probability proportional to weights.

    Uses the Gumbel top-k trick, which is O(n) and far faster than
    ``rng.choice(replace=False, p=...)`` when called hundreds of thousands of
    times. Returns indices ordered best-first, which doubles as the grid order.
    """
    if candidates is None:
        candidates = np.arange(weights.size)
    else:
        weights = weights[candidates]

    positive = weights > 0
    if not positive.any():
        return candidates[: min(k, candidates.size)]

    candidates = candidates[positive]
    weights = weights[positive]
    k = int(min(k, candidates.size))
    if k == 0:
        return np.empty(0, dtype=int)

    keys = np.log(weights) + rng.gumbel(size=weights.size)
    top = np.argpartition(-keys, k - 1)[:k]
    top = top[np.argsort(-keys[top])]
    return candidates[top]


class EventSimulator:
    """Turns a catalogue plus a population into a realistic clickstream."""

    def __init__(
        self,
        catalogue: Catalogue,
        n_users: int = 1000,
        history_days: int = 180,
        end_date: date = date(2026, 8, 25),
        seed: int = 20260825,
    ) -> None:
        self.catalogue = catalogue
        self.n_users = n_users
        self.history_days = history_days
        self.end_date = end_date
        self.seed = seed

        self.rng = np.random.default_rng(seed)
        self.n_products = len(catalogue)
        self.product_ids = np.array(catalogue.ids)

        self.start_dt = datetime.combine(
            end_date - timedelta(days=history_days), datetime.min.time(), tzinfo=timezone.utc
        )

        self.profiles = build_user_population(
            catalogue, n_users, history_days, np.random.default_rng(seed + 101)
        )

        self._build_exposure_priors()
        self._build_related_index()
        self._build_attribute_indices()

        self._events: list[dict[str, Any]] = []
        self._sessions: list[dict[str, Any]] = []
        self._orders: list[dict[str, Any]] = []
        self._rec_logs: list[dict[str, Any]] = []
        self._session_counter = 0

    # ── setup ────────────────────────────────────────────────────────────────
    def _build_exposure_priors(self) -> None:
        """Merchandising prior: what the site pushes, independent of any user.

        Deliberately *correlated with but not equal to* mean population appeal.
        If popularity were exactly appeal, the popularity baseline would be
        near-optimal and nothing could beat it; if it were pure noise, popularity
        would be useless. Real merchandising sits in between.
        """
        appeal = mass_appeal(self.profiles)
        merch_rng = np.random.default_rng(self.seed + 202)
        prior = np.power(np.clip(appeal, 1e-6, None), 1.4) * merch_rng.lognormal(0.0, 0.75,
                                                                                self.n_products)

        for i, product in enumerate(self.catalogue):
            if product.get("is_core"):
                prior[i] *= CORE_CATALOGUE_BOOST
            boost = HERO_BOOSTS.get(product["product_id"])
            if boost:
                prior[i] *= boost

        self.popularity_prior = prior / prior.sum()

        # Seasonality: a smooth annual cycle rather than month-boundary steps.
        categories = self.catalogue.attribute_array("category")
        collections = self.catalogue.attribute_array("collection")
        is_warm = (categories == "Sandal") | (collections == "Summer")
        is_cold = (categories == "Boot") | (collections == "Terrain")
        is_active = collections == "Active"

        season = np.ones((self.history_days, self.n_products), dtype=float)
        traffic = np.ones(self.history_days, dtype=float)
        for day in range(self.history_days):
            current = self.start_dt + timedelta(days=day)
            doy = current.timetuple().tm_yday
            # +1 at peak summer (~day 150), -1 at peak winter.
            summer = float(np.cos(2 * np.pi * (doy - 150) / 365.0))
            factor = np.ones(self.n_products, dtype=float)
            factor[is_warm] *= 1.0 + 0.55 * summer
            factor[is_cold] *= 1.0 - 0.50 * summer
            # New-year resolution bump for performance product.
            jan_bump = float(np.exp(-(((doy - 8) % 365) ** 2) / (2 * 18.0**2)))
            factor[is_active] *= 1.0 + 0.35 * jan_bump
            season[day] = np.clip(factor, 0.15, None)

            traffic[day] = WEEKDAY_WEIGHTS[current.weekday()] * (1.0 + 0.25 * jan_bump)

        # Sale campaigns.
        self.sale_days: set[int] = set()
        for offset, length in SALE_OFFSETS_DAYS:
            first = self.history_days - offset
            for d in range(first, min(first + length, self.history_days)):
                if d >= 0:
                    self.sale_days.add(d)
                    traffic[d] *= SALE_TRAFFIC_MULTIPLIER

        self.season = season
        self.day_traffic = traffic

        # Precomputed per-day exposure weights (popularity x season).
        exposure = self.popularity_prior[None, :] * season
        self.exposure_weights = exposure / exposure.sum(axis=1, keepdims=True)

    def _build_related_index(self) -> None:
        """"You might also like" neighbours, by attribute overlap.

        This is the site's merchandising rule, not our content model — it is
        what creates the item-item co-occurrence that collaborative filtering
        will later pick up on.
        """
        cat = self.catalogue.attribute_array("category")
        coll = self.catalogue.attribute_array("collection")
        style = self.catalogue.attribute_array("style")
        fam = self.catalogue.attribute_array("color_family")
        prices = self.catalogue.prices

        def match(values: np.ndarray) -> np.ndarray:
            return (values[:, None] == values[None, :]).astype(float)

        price_scale = max(1.0, float(prices.std()))
        price_close = np.exp(-np.abs(prices[:, None] - prices[None, :]) / (1.2 * price_scale))

        sim = (
            0.34 * match(cat)
            + 0.20 * match(coll)
            + 0.16 * match(style)
            + 0.14 * match(fam)
            + 0.16 * price_close
        )
        np.fill_diagonal(sim, -np.inf)

        neighbours = min(24, self.n_products - 1)
        self.related_index = np.argsort(-sim, axis=1)[:, :neighbours]

    def _build_attribute_indices(self) -> None:
        self.category_index: dict[str, np.ndarray] = {}
        for value in set(self.catalogue.attribute_array("category")):
            self.category_index[str(value)] = self.catalogue.indices_where("category", value)

        self.searchable = [
            {
                "category": str(p["category"]).lower(),
                "color": str(p.get("color", "")).lower(),
                "style": str(p.get("style", "")).lower(),
                "material": str(p.get("material", "")).lower(),
                "name": str(p["product_name"]).lower(),
            }
            for p in self.catalogue
        ]

    # ── helpers ──────────────────────────────────────────────────────────────
    def _event_id(self, session_id: str, sequence: int) -> uuid.UUID:
        """Deterministic event id: same seed and sequence -> same UUID.

        Kept as a ``UUID`` object rather than a string because that is what the
        ``Uuid`` column type expects; it renders as native UUID on PostgreSQL
        and CHAR(32) on SQLite.
        """
        return uuid.uuid5(uuid.NAMESPACE_URL, f"tessch/{session_id}/{sequence}")

    def _record(
        self,
        session_id: str,
        sequence: int,
        user_id: str,
        event_type: str,
        timestamp: datetime,
        product_index: int | None = None,
        duration: float | None = None,
        surface: str | None = None,
        position: int | None = None,
        search_query: str | None = None,
        value: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._events.append(
            {
                "event_id": self._event_id(session_id, sequence),
                "user_id": user_id,
                "product_id": None if product_index is None else str(self.product_ids[product_index]),
                "session_id": session_id,
                "event_type": event_type,
                "duration_seconds": None if duration is None else round(float(duration), 2),
                "surface": surface,
                "position": position,
                "search_query": search_query,
                "event_value": None if value is None else round(float(value), 2),
                "timestamp": timestamp,
                "metadata": metadata or {},
            }
        )

    def _top_category(self, affinity: np.ndarray, rng: np.random.Generator,
                      explore_rate: float) -> str:
        """The category this user gravitates to, with an exploration chance."""
        names = list(self.category_index)
        if rng.random() < explore_rate:
            return str(rng.choice(names))
        means = [float(affinity[self.category_index[n]].mean()) for n in names]
        return names[int(np.argmax(means))]

    def _build_query(self, affinity: np.ndarray, rng: np.random.Generator) -> tuple[str, np.ndarray]:
        """Compose a plausible search query and the result set it returns."""
        best = int(np.argmax(affinity + rng.normal(0, 0.05, self.n_products)))
        product = self.catalogue.products[best]

        tokens: list[str] = []
        if rng.random() < 0.55:
            tokens.append(str(product["color"]).lower())
        if rng.random() < 0.75:
            tokens.append(str(product["category"]).lower())
        elif rng.random() < 0.6:
            tokens.append(str(product["style"]).lower())
        else:
            tokens.append(str(product["material"]).lower())
        if not tokens:
            tokens = [str(product["category"]).lower()]

        query = " ".join(tokens)
        hits = [
            i
            for i, fields in enumerate(self.searchable)
            if all(
                token in fields["category"]
                or token in fields["color"]
                or token in fields["style"]
                or token in fields["material"]
                or token in fields["name"]
                for token in tokens
            )
        ]
        if not hits:  # fall back to a single-token match
            token = tokens[-1]
            hits = [
                i
                for i, fields in enumerate(self.searchable)
                if token in fields["category"] or token in fields["style"]
            ]
        return query, np.array(hits, dtype=int)

    def _render_grid(
        self,
        surface: str,
        profile: UserProfile,
        state: UserState,
        affinity: np.ndarray,
        day: int,
        rng: np.random.Generator,
        anchor: int | None,
    ) -> tuple[np.ndarray, str | None]:
        """Pick which products the site puts in front of the user, in order."""
        size = GRID_SIZES.get(surface, 6)
        exposure = self.exposure_weights[day]
        query: str | None = None

        if surface == C.SURFACE_HOME_GRID:
            chosen = _weighted_sample(rng, exposure, size)

        elif surface == C.SURFACE_CATEGORY:
            category = self._top_category(affinity, rng, profile.explore_rate)
            chosen = _weighted_sample(rng, exposure, size, self.category_index[category])

        elif surface == C.SURFACE_SEARCH:
            query, hits = self._build_query(affinity, rng)
            if hits.size == 0:
                chosen = _weighted_sample(rng, exposure, size)
            else:
                # Search ranks by relevance, which we proxy as popularity.
                chosen = _weighted_sample(rng, exposure, size, hits)

        elif surface == C.SURFACE_RELATED and anchor is not None:
            pool = self.related_index[anchor]
            chosen = _weighted_sample(rng, exposure, size, pool)

        elif surface == C.SURFACE_RECOMMENDED:
            # The legacy on-site module: popularity inside the user's dominant
            # category. Weak personalisation — exactly what we want to beat.
            category = self._top_category(affinity, rng, explore_rate=0.15)
            chosen = _weighted_sample(rng, exposure, size, self.category_index[category])

        elif surface == C.SURFACE_WISHLIST and state.wishlist:
            pool = np.array(
                [self.catalogue.position(pid) for pid in sorted(state.wishlist)], dtype=int
            )
            chosen = pool[: min(size, pool.size)]

        else:
            chosen = _weighted_sample(rng, exposure, size)

        return chosen, query

    def _dwell_time(self, affinity: float, profile: UserProfile,
                    rng: np.random.Generator) -> float:
        """Dwell time in seconds: rises with affinity, heavy right tail."""
        if rng.random() < BOUNCE_RATE_AT_ZERO_AFFINITY * (1.0 - affinity):
            return float(rng.uniform(1.5, 6.5))

        base = DWELL_BASE_SECONDS + DWELL_AFFINITY_GAIN * affinity**DWELL_AFFINITY_POWER
        value = base * profile.dwell_scale * rng.lognormal(0.0, DWELL_SIGMA)
        return float(np.clip(value, DWELL_FLOOR_SECONDS, DWELL_CEILING_SECONDS))

    # ── session ──────────────────────────────────────────────────────────────
    def _simulate_session(
        self,
        profile: UserProfile,
        state: UserState,
        day: int,
        start: datetime,
        rng: np.random.Generator,
    ) -> None:
        self._session_counter += 1
        session_id = f"S{self._session_counter:07d}"
        day_fraction = day / max(1, self.history_days - 1)
        affinity = profile.affinity_at(day_fraction) * self.season[day]
        affinity = np.clip(affinity, 0.0, 1.0)

        # Intent: a standing cart or wishlist pulls the visit toward converting.
        if state.cart and rng.random() < 0.30:
            intent = "purchase_intent"
        elif state.wishlist and rng.random() < 0.20:
            intent = "focused_research"
        else:
            names = list(INTENT_WEIGHTS)
            probs = np.array([INTENT_WEIGHTS[n] for n in names])
            intent = str(rng.choice(names, p=probs / probs.sum()))

        low, high = INTENT_STEPS[intent]
        steps = int(np.clip(round(rng.integers(low, high + 1) * profile.attention_span), 1, 16))
        commitment = INTENT_COMMITMENT[intent]
        sale = day in self.sale_days

        device = (
            profile.device_preference
            if rng.random() < 0.78
            else str(rng.choice(C.DEVICE_TYPES))
        )

        clock = start
        sequence = 0
        anchor: int | None = None
        viewed_in_session: Counter = Counter()
        stats = Counter()

        def advance(seconds: float) -> None:
            nonlocal clock
            clock = clock + timedelta(seconds=float(max(0.3, seconds)))

        # ── entry ────────────────────────────────────────────────────────────
        # Returning users often deep-link straight back to something they were
        # looking at. This is the main source of cross-session repeat views.
        direct_entry: int | None = None
        if state.recent and rng.random() < 0.32:
            pool = np.array([self.catalogue.position(p) for p in state.recent], dtype=int)
            weights = affinity[pool] ** 2 + 1e-6
            direct_entry = int(rng.choice(pool, p=weights / weights.sum()))

        if direct_entry is None:
            sequence += 1
            self._record(session_id, sequence, profile.user_id, C.EVENT_HOME_VIEW, clock,
                         surface=C.SURFACE_HOME_GRID, metadata={"device": device})
            stats["page"] += 1
            advance(rng.uniform(1.0, 5.0))

        surfaces_first = [
            (C.SURFACE_HOME_GRID, 0.50),
            (C.SURFACE_SEARCH, 0.16),
            (C.SURFACE_CATEGORY, 0.16),
            (C.SURFACE_RECOMMENDED, 0.10),
            (C.SURFACE_WISHLIST, 0.08 if state.wishlist else 0.0),
        ]
        surfaces_next = [
            (C.SURFACE_RELATED, 0.42),
            (C.SURFACE_CATEGORY, 0.19),
            (C.SURFACE_HOME_GRID, 0.14),
            (C.SURFACE_SEARCH, 0.11),
            (C.SURFACE_RECOMMENDED, 0.09),
            (C.SURFACE_WISHLIST, 0.05 if state.wishlist else 0.0),
        ]

        for step in range(steps):
            # ── a direct product view short-circuits the first grid ──────────
            if step == 0 and direct_entry is not None:
                sequence = self._visit_product(
                    session_id, sequence, profile, state, affinity, direct_entry,
                    C.SURFACE_DIRECT, None, clock, rng, commitment, sale,
                    viewed_in_session, stats, device,
                )
                dwell = self._events[-1]["duration_seconds"] or 5.0
                advance(dwell + rng.uniform(0.5, 4.0))
                anchor = direct_entry
                continue

            options = surfaces_first if step == 0 else surfaces_next
            if anchor is None:
                options = [(s, w) for s, w in options if s != C.SURFACE_RELATED]
            names = [s for s, w in options if w > 0]
            probs = np.array([w for _, w in options if w > 0], dtype=float)
            surface = str(rng.choice(names, p=probs / probs.sum()))

            grid, query = self._render_grid(
                surface, profile, state, affinity, day, rng, anchor
            )
            if grid.size == 0:
                continue

            if surface == C.SURFACE_SEARCH:
                sequence += 1
                self._record(session_id, sequence, profile.user_id, C.EVENT_SEARCH, clock,
                             surface=surface, search_query=query,
                             metadata={"results": int(grid.size)})
                stats["search"] += 1
                advance(rng.uniform(1.5, 6.0))
            elif surface in (C.SURFACE_CATEGORY, C.SURFACE_WISHLIST):
                sequence += 1
                self._record(session_id, sequence, profile.user_id, C.EVENT_PAGE_VIEW, clock,
                             surface=surface, metadata={"device": device})
                stats["page"] += 1
                advance(rng.uniform(0.8, 3.5))

            # ── impressions ─────────────────────────────────────────────────
            clicks: list[tuple[int, int]] = []
            for position, product_index in enumerate(grid, start=1):
                product_index = int(product_index)
                sequence += 1
                self._record(session_id, sequence, profile.user_id,
                             C.EVENT_PRODUCT_IMPRESSION, clock, product_index=product_index,
                             surface=surface, position=position,
                             metadata={"device": device})
                stats["impression"] += 1

                if surface == C.SURFACE_RECOMMENDED:
                    self._rec_logs.append(
                        {
                            "user_id": profile.user_id,
                            "product_id": str(self.product_ids[product_index]),
                            "session_id": session_id,
                            "recommendation_rank": position,
                            "score": float(round(self.popularity_prior[product_index] * 1000, 6)),
                            "model_name": "legacy_popularity",
                            "model_version": "0.1",
                            "experiment_group": "control",
                            "served_at": clock,
                        }
                    )

                aff = float(affinity[product_index])
                pid = str(self.product_ids[product_index])
                damping = POST_PURCHASE_CLICK_DAMPING if pid in state.purchased else 1.0

                # Exploration: sometimes attention is not preference-driven.
                effective = aff if rng.random() > profile.explore_rate else float(rng.random())
                position_weight = 1.0 / (1.0 + POSITION_DECAY * (position - 1))
                p_click = (
                    CTR_CEILING
                    * position_weight
                    * effective**CLICK_AFFINITY_POWER
                    * damping
                )
                if surface == C.SURFACE_WISHLIST:
                    p_click = min(0.95, p_click * 2.4)  # already-saved intent

                if rng.random() < p_click and len(clicks) < MAX_CLICKS_PER_GRID:
                    clicks.append((product_index, position))
                elif rng.random() < IDLE_HOVER_RATE * position_weight:
                    sequence += 1
                    self._record(session_id, sequence, profile.user_id,
                                 C.EVENT_PRODUCT_HOVER, clock, product_index=product_index,
                                 surface=surface, position=position,
                                 duration=float(rng.uniform(0.4, 3.5)))
                    stats["hover"] += 1

                advance(rng.uniform(0.2, 1.4))

            # ── clicks -> product views ─────────────────────────────────────
            for product_index, position in clicks:
                if rng.random() < HOVER_BEFORE_CLICK and device != "mobile":
                    sequence += 1
                    self._record(session_id, sequence, profile.user_id,
                                 C.EVENT_PRODUCT_HOVER, clock, product_index=product_index,
                                 surface=surface, position=position,
                                 duration=float(rng.uniform(0.5, 4.0)))
                    stats["hover"] += 1
                    advance(rng.uniform(0.3, 1.5))

                sequence += 1
                self._record(session_id, sequence, profile.user_id, C.EVENT_PRODUCT_CLICK,
                             clock, product_index=product_index, surface=surface,
                             position=position, metadata={"device": device})
                stats["click"] += 1
                advance(rng.uniform(0.4, 2.0))

                sequence = self._visit_product(
                    session_id, sequence, profile, state, affinity, product_index,
                    surface, position, clock, rng, commitment, sale,
                    viewed_in_session, stats, device,
                )
                dwell = self._events[-1]["duration_seconds"] or 5.0
                advance(dwell + rng.uniform(0.4, 3.0))
                anchor = product_index

                # ── intra-session revisit ───────────────────────────────────
                # "Viewed Arc Black, looked at two others, came back to Arc
                # Black" — the comparison-shopping pattern that makes repeat
                # views such a strong signal.
                if viewed_in_session and rng.random() < INTENT_REVISIT[intent]:
                    candidates = np.array(list(viewed_in_session), dtype=int)
                    weights = affinity[candidates] ** 2.5 + 1e-6
                    revisit = int(rng.choice(candidates, p=weights / weights.sum()))
                    sequence = self._visit_product(
                        session_id, sequence, profile, state, affinity, revisit,
                        C.SURFACE_DIRECT, None, clock, rng, commitment, sale,
                        viewed_in_session, stats, device,
                    )
                    stats["repeat"] += 1
                    advance((self._events[-1]["duration_seconds"] or 5.0) + rng.uniform(0.3, 2.0))
                    anchor = revisit

        # ── checkout ─────────────────────────────────────────────────────────
        if state.cart:
            cart_indices = [self.catalogue.position(p) for p in state.cart]
            mean_aff = float(np.mean(affinity[cart_indices])) if cart_indices else 0.0
            p_checkout = min(
                CHECKOUT_CEILING,
                0.26
                * profile.conversion_propensity
                * (0.35 + mean_aff)
                * commitment
                * (SALE_CONVERSION_MULTIPLIER if sale else 1.0),
            )
            if rng.random() < p_checkout:
                advance(rng.uniform(15.0, 90.0))  # checkout flow
                for product_id in sorted(state.cart):
                    if rng.random() > CART_ITEM_CONVERSION:
                        continue
                    index = self.catalogue.position(product_id)
                    price = float(self.catalogue.prices[index])
                    discount = 0.8 if sale else 1.0
                    total = round(price * discount, 2)

                    sequence += 1
                    self._record(session_id, sequence, profile.user_id, C.EVENT_PURCHASE,
                                 clock, product_index=index, surface=C.SURFACE_DIRECT,
                                 value=total, metadata={"on_sale": sale})
                    stats["purchase"] += 1
                    self._orders.append(
                        {
                            "order_id": uuid.uuid5(
                                uuid.NAMESPACE_URL,
                                f"tessch/order/{session_id}/{product_id}",
                            ),
                            "user_id": profile.user_id,
                            "session_id": session_id,
                            "product_id": product_id,
                            "quantity": 1,
                            "unit_price": price,
                            "total_amount": total,
                            "status": "completed",
                            "ordered_at": clock,
                        }
                    )
                    state.purchased.add(product_id)
                    state.wishlist.discard(product_id)
                    advance(rng.uniform(1.0, 5.0))
                state.cart.clear()
            elif rng.random() < CART_ABANDON_PRUNE_RATE:
                for product_id in sorted(state.cart):
                    if rng.random() < 0.5:
                        index = self.catalogue.position(product_id)
                        sequence += 1
                        self._record(session_id, sequence, profile.user_id,
                                     C.EVENT_REMOVE_FROM_CART, clock, product_index=index,
                                     surface=C.SURFACE_DIRECT)
                        state.cart.discard(product_id)
                        advance(rng.uniform(0.5, 3.0))

        # ── session rollup ───────────────────────────────────────────────────
        duration = max(1, int((clock - start).total_seconds()))
        self._sessions.append(
            {
                "session_id": session_id,
                "user_id": profile.user_id,
                "started_at": start,
                "ended_at": clock,
                "duration_seconds": duration,
                "events_count": sequence,
                "products_viewed": int(sum(viewed_in_session.values())),
                "unique_products": int(len(viewed_in_session)),
                "repeat_views": int(sum(v - 1 for v in viewed_in_session.values() if v > 1)),
                "wishlist_count": int(stats["wishlist"]),
                "cart_count": int(stats["cart"]),
                "purchase_count": int(stats["purchase"]),
                "search_count": int(stats["search"]),
                "device_type": device,
                "intent": intent,
            }
        )
        state.last_seen_at = clock
        state.sessions += 1

    def _visit_product(
        self,
        session_id: str,
        sequence: int,
        profile: UserProfile,
        state: UserState,
        affinity: np.ndarray,
        product_index: int,
        surface: str,
        position: int | None,
        clock: datetime,
        rng: np.random.Generator,
        commitment: float,
        sale: bool,
        viewed_in_session: Counter,
        stats: Counter,
        device: str,
    ) -> int:
        """Emit a product_view plus any wishlist / cart / share it triggers."""
        product_id = str(self.product_ids[product_index])
        aff = float(affinity[product_index])
        available = bool(self.catalogue.available_mask[product_index])

        dwell = self._dwell_time(aff, profile, rng)
        sequence += 1
        self._record(session_id, sequence, profile.user_id, C.EVENT_PRODUCT_VIEW, clock,
                     product_index=product_index, duration=dwell, surface=surface,
                     position=position,
                     metadata={"device": device,
                               "view_number": int(viewed_in_session[product_index]) + 1})
        viewed_in_session[product_index] += 1
        state.lifetime_views[product_id] += 1
        if product_id in state.recent:
            state.recent.remove(product_id)
        state.recent.appendleft(product_id)

        # A long read is itself evidence of intent, over and above affinity.
        engagement = aff * (0.65 + 0.35 * min(1.0, dwell / 45.0))

        if product_id not in state.wishlist and product_id not in state.purchased:
            if rng.random() < min(0.9, WISHLIST_CEILING * engagement**WISHLIST_POWER * commitment):
                sequence += 1
                self._record(session_id, sequence, profile.user_id, C.EVENT_WISHLIST_ADD,
                             clock, product_index=product_index, surface=surface)
                state.wishlist.add(product_id)
                stats["wishlist"] += 1
        elif product_id in state.wishlist and rng.random() < 0.015:
            sequence += 1
            self._record(session_id, sequence, profile.user_id, C.EVENT_WISHLIST_REMOVE,
                         clock, product_index=product_index, surface=surface)
            state.wishlist.discard(product_id)

        if available and product_id not in state.cart and product_id not in state.purchased:
            p_cart = (
                CART_CEILING
                * engagement**CART_POWER
                * commitment
                * min(2.0, profile.conversion_propensity)
                * (1.35 if sale else 1.0)
            )
            if rng.random() < min(0.85, p_cart):
                sequence += 1
                self._record(session_id, sequence, profile.user_id, C.EVENT_ADD_TO_CART,
                             clock, product_index=product_index, surface=surface,
                             value=float(self.catalogue.prices[product_index]))
                state.cart.add(product_id)
                stats["cart"] += 1

        if rng.random() < SHARE_CEILING * engagement**2:
            sequence += 1
            self._record(session_id, sequence, profile.user_id, C.EVENT_PRODUCT_SHARE,
                         clock, product_index=product_index, surface=surface)

        return sequence

    # ── driver ───────────────────────────────────────────────────────────────
    def run(self, progress: bool = True) -> SimulationOutput:
        users: list[dict[str, Any]] = []
        hidden: list[dict[str, Any]] = []

        for n, profile in enumerate(self.profiles):
            rng = np.random.default_rng(self.seed + 1_000_003 * (n + 1))
            state = UserState()

            first_day = max(0, profile.signup_day)
            if first_day >= self.history_days:
                first_day = self.history_days - 1

            days = np.arange(first_day, self.history_days)
            lam = profile.sessions_per_day * self.day_traffic[days]
            counts = np.minimum(rng.poisson(lam), 4)

            for day, count in zip(days, counts):
                if count == 0:
                    continue
                hours = rng.choice(24, size=int(count), p=HOUR_WEIGHTS / HOUR_WEIGHTS.sum())
                for hour in np.sort(hours):
                    start = (
                        self.start_dt
                        + timedelta(days=int(day))
                        + timedelta(hours=int(hour), minutes=int(rng.integers(0, 60)),
                                    seconds=int(rng.integers(0, 60)))
                    )
                    self._simulate_session(profile, state, int(day), start, rng)

            signup_at = self.start_dt + timedelta(days=profile.signup_day)
            record = profile.public_record()
            record.update(
                {
                    "is_new_user": state.sessions <= 2,
                    "signup_at": signup_at,
                    "created_at": signup_at,
                    "last_active_at": state.last_seen_at,
                }
            )
            users.append(record)
            hidden_record = profile.hidden_record()
            hidden_record["sessions_simulated"] = state.sessions
            hidden_record["archetype_label"] = ARCHETYPES[profile.primary].label
            hidden.append(hidden_record)

            if progress and (n + 1) % 200 == 0:
                print(
                    f"  simulated {n + 1:>5}/{len(self.profiles)} users  "
                    f"({len(self._events):>9,} events, {len(self._sessions):>7,} sessions)",
                    flush=True,
                )

        # Ground-truth affinity, flattened. Analysis only.
        n_users = len(self.profiles)
        affinity_matrix = np.vstack([p.affinity_base for p in self.profiles]).astype(np.float32)
        affinity_long = {
            "user_id": np.repeat([p.user_id for p in self.profiles], self.n_products),
            "product_id": np.tile(self.product_ids, n_users),
            "affinity": affinity_matrix.reshape(-1),
        }

        meta = {
            "seed": self.seed,
            "n_users": n_users,
            "n_products": self.n_products,
            "history_days": self.history_days,
            "start_date": self.start_dt.date().isoformat(),
            "end_date": self.end_date.isoformat(),
            "n_events": len(self._events),
            "n_sessions": len(self._sessions),
            "n_orders": len(self._orders),
            "sale_days": sorted(self.sale_days),
        }

        return SimulationOutput(
            products=list(self.catalogue.products),
            users=users,
            sessions=self._sessions,
            events=self._events,
            orders=self._orders,
            recommendation_logs=self._rec_logs,
            hidden_users=hidden,
            affinity_long=affinity_long,
            meta=meta,
        )


__all__ = ["EventSimulator", "SimulationOutput", "UserState"]
