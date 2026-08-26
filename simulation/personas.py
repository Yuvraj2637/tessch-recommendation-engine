"""Latent user preference model.

This is the "truth" the recommendation engine is trying to recover. None of it
is ever written into the training data — the models only see behaviour.

The design goal is that a naive recommender should *fail* on this data and a
good one should succeed. That requires three properties:

1. **Cluster structure.** Users are mixtures of archetypes, so genuinely similar
   users exist. Without this, collaborative filtering has nothing to find.
2. **Attribute structure.** Affinity is driven by category / colour / style /
   collection / price, so a content model has something to learn. Without this,
   content-based recommendation and cold start are unsolvable.
3. **Idiosyncrasy.** Every user gets private per-product noise, so archetype
   alone can never fully explain behaviour. Without this, the problem is too
   easy and every model saturates at the same score.

A fourth property — preference *drift* — gives Phase 14 (short-term vs
long-term intent) something real to model: a subset of users migrate toward a
different archetype partway through the window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from backend.core.constants import AGE_GROUPS, DEVICE_TYPES
from simulation.catalog import Catalogue

# Relative importance of each attribute in the affinity function. These sum to
# 1.0 so that affinity stays on a comparable scale across users.
W_CATEGORY = 0.28
W_COLOR = 0.16
W_STYLE = 0.26
W_COLLECTION = 0.16
W_PRICE = 0.14

#: Preference for an attribute value the archetype does not name.
INDIFFERENT = 0.12


@dataclass(frozen=True)
class Archetype:
    """A coherent taste profile. Users are weighted mixtures of two of these."""

    name: str
    label: str
    category: dict[str, float]
    color: dict[str, float]
    style: dict[str, float]
    collection: dict[str, float]
    price_center: float
    price_tolerance: float
    #: Multiplier on baseline purchase probability. A "browser" is ~0.15.
    conversion_multiplier: float
    #: What this person would tick on a signup survey.
    declared_style: str
    #: Relative share of the population.
    share: float


ARCHETYPES: dict[str, Archetype] = {
    "minimalist": Archetype(
        name="minimalist",
        label="Minimalist everyday sneaker buyer",
        category={"Sneaker": 1.0, "Loafer": 0.35},
        color={"dark": 0.85, "light": 0.95, "neutral": 0.7, "earth": 0.3, "bright": 0.05},
        style={"Casual": 1.0, "Smart Casual": 0.6, "Streetwear": 0.2},
        collection={"Core": 1.0, "Refined": 0.45, "Summer": 0.3},
        price_center=2950.0,
        price_tolerance=700.0,
        conversion_multiplier=1.0,
        declared_style="Casual",
        share=0.19,
    ),
    "athlete": Archetype(
        name="athlete",
        label="Performance and training focused",
        category={"Athletic": 1.0, "Sneaker": 0.3},
        color={"bright": 0.8, "dark": 0.75, "neutral": 0.45, "light": 0.5, "earth": 0.2},
        style={"Athletic": 1.0, "Outdoor": 0.45, "Casual": 0.2},
        collection={"Active": 1.0, "Terrain": 0.4, "Core": 0.2},
        price_center=3650.0,
        price_tolerance=800.0,
        conversion_multiplier=1.15,
        declared_style="Athletic",
        share=0.17,
    ),
    "streetwear": Archetype(
        name="streetwear",
        label="Street style, bold silhouettes",
        category={"Sneaker": 0.9, "Athletic": 0.6, "Boot": 0.5},
        color={"dark": 0.8, "bright": 0.7, "neutral": 0.6, "earth": 0.5, "light": 0.55},
        style={"Streetwear": 1.0, "Casual": 0.4, "Athletic": 0.3},
        collection={"Street": 1.0, "Active": 0.3, "Core": 0.25},
        price_center=3550.0,
        price_tolerance=900.0,
        conversion_multiplier=0.85,
        declared_style="Streetwear",
        share=0.15,
    ),
    "heritage": Archetype(
        name="heritage",
        label="Heritage boots and leather goods",
        category={"Boot": 1.0, "Loafer": 0.45},
        color={"earth": 1.0, "dark": 0.7, "neutral": 0.35, "light": 0.1, "bright": 0.03},
        style={"Heritage": 1.0, "Outdoor": 0.45, "Smart Casual": 0.3},
        collection={"Heritage": 1.0, "Terrain": 0.4, "Refined": 0.3},
        price_center=4600.0,
        price_tolerance=750.0,
        conversion_multiplier=0.95,
        declared_style="Heritage",
        share=0.12,
    ),
    "outdoors": Archetype(
        name="outdoors",
        label="Trail, terrain and weather",
        category={"Boot": 0.95, "Athletic": 0.7},
        color={"earth": 0.95, "dark": 0.7, "neutral": 0.5, "bright": 0.25, "light": 0.1},
        style={"Outdoor": 1.0, "Heritage": 0.45, "Athletic": 0.4},
        collection={"Terrain": 1.0, "Heritage": 0.4, "Active": 0.35},
        price_center=4500.0,
        price_tolerance=900.0,
        conversion_multiplier=1.05,
        declared_style="Outdoor",
        share=0.10,
    ),
    "refined": Archetype(
        name="refined",
        label="Formal and smart-casual",
        category={"Loafer": 1.0, "Boot": 0.4, "Sneaker": 0.25},
        color={"dark": 0.95, "earth": 0.7, "neutral": 0.5, "light": 0.4, "bright": 0.05},
        style={"Formal": 1.0, "Smart Casual": 0.9, "Heritage": 0.35},
        collection={"Refined": 1.0, "Heritage": 0.35, "Core": 0.25},
        price_center=3550.0,
        price_tolerance=700.0,
        conversion_multiplier=1.1,
        declared_style="Formal",
        share=0.11,
    ),
    "summer": Archetype(
        name="summer",
        label="Warm-weather, sandals and light canvas",
        category={"Sandal": 1.0, "Sneaker": 0.55},
        color={"light": 0.85, "neutral": 0.8, "earth": 0.7, "bright": 0.5, "dark": 0.3},
        style={"Casual": 1.0, "Smart Casual": 0.5},
        collection={"Summer": 1.0, "Core": 0.35},
        price_center=2250.0,
        price_tolerance=700.0,
        conversion_multiplier=1.0,
        declared_style="Casual",
        share=0.09,
    ),
    "value": Archetype(
        name="value",
        label="Price-led, cross-category",
        category={"Sandal": 0.8, "Sneaker": 0.8, "Athletic": 0.6, "Loafer": 0.4, "Boot": 0.3},
        color={"dark": 0.6, "neutral": 0.6, "light": 0.55, "earth": 0.5, "bright": 0.4},
        style={"Casual": 0.8, "Athletic": 0.6, "Smart Casual": 0.5, "Streetwear": 0.4},
        collection={"Core": 0.7, "Summer": 0.7, "Active": 0.5, "Street": 0.4},
        price_center=2150.0,
        # Tight tolerance: this is the defining trait — price dominates fit.
        price_tolerance=450.0,
        conversion_multiplier=0.8,
        declared_style="Casual",
        share=0.09,
    ),
    "window_shopper": Archetype(
        name="window_shopper",
        label="High browsing volume, rarely converts",
        category={"Sneaker": 0.85, "Athletic": 0.7, "Boot": 0.65, "Loafer": 0.5, "Sandal": 0.5},
        color={"dark": 0.7, "light": 0.7, "neutral": 0.65, "earth": 0.6, "bright": 0.6},
        style={"Casual": 0.8, "Streetwear": 0.7, "Athletic": 0.6, "Heritage": 0.5, "Formal": 0.4},
        collection={"Core": 0.7, "Street": 0.6, "Active": 0.6, "Heritage": 0.5, "Refined": 0.4},
        price_center=3300.0,
        price_tolerance=1400.0,
        # The important negative class: lots of views, almost no purchases.
        conversion_multiplier=0.12,
        declared_style="Casual",
        share=0.08,
    ),
}


@dataclass
class UserProfile:
    """One simulated shopper: public attributes plus hidden preference state."""

    user_id: str
    username: str
    email: str
    age_group: str
    gender: str
    #: Self-reported at signup — deliberately noisy, and a legitimate feature.
    declared_style: str
    device_preference: str
    #: Day index (0 = first day of the window). Negative means pre-existing.
    signup_day: int

    # ── hidden state ─────────────────────────────────────────────────────────
    primary: str
    secondary: str
    secondary_weight: float
    sessions_per_day: float
    explore_rate: float
    conversion_propensity: float
    dwell_scale: float
    attention_span: float
    #: Archetype this user migrates toward, or None for a stable user.
    drift_to: str | None
    #: Fraction of the window at which drift begins.
    drift_start: float

    #: Affinity in [0, 1] over the catalogue, before drift/season. Shape (n_products,).
    affinity_base: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    #: Affinity after full drift. Identical to ``affinity_base`` if not drifting.
    affinity_drift: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))

    def affinity_at(self, day_fraction: float) -> np.ndarray:
        """Affinity vector at a point in the window, interpolating any drift."""
        if self.drift_to is None or day_fraction <= self.drift_start:
            return self.affinity_base
        span = max(1e-6, 1.0 - self.drift_start)
        weight = min(1.0, (day_fraction - self.drift_start) / span)
        return (1.0 - weight) * self.affinity_base + weight * self.affinity_drift

    def public_record(self) -> dict[str, Any]:
        """The fields a real system would legitimately have in its users table."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "age_group": self.age_group,
            "gender": self.gender,
            "style_preference": self.declared_style,
            "device_preference": self.device_preference,
        }

    def hidden_record(self) -> dict[str, Any]:
        """Ground truth. Analysis only — never a model input."""
        return {
            "user_id": self.user_id,
            "archetype_primary": self.primary,
            "archetype_secondary": self.secondary,
            "secondary_weight": round(self.secondary_weight, 4),
            "sessions_per_day": round(self.sessions_per_day, 4),
            "explore_rate": round(self.explore_rate, 4),
            "conversion_propensity": round(self.conversion_propensity, 4),
            "dwell_scale": round(self.dwell_scale, 4),
            "attention_span": round(self.attention_span, 4),
            "drift_to": self.drift_to,
            "drift_start": round(self.drift_start, 4),
            "signup_day": self.signup_day,
        }


# ─────────────────────────────────────────────────────────────────────────────
def _lookup(values: np.ndarray, prefs: dict[str, float]) -> np.ndarray:
    return np.array([prefs.get(v, INDIFFERENT) for v in values], dtype=float)


def _archetype_scores(
    catalogue: Catalogue, archetype: Archetype, price_sensitivity: float
) -> np.ndarray:
    """Unnormalised attribute-match score of one archetype over the catalogue."""
    categories = catalogue.attribute_array("category")
    colors = catalogue.attribute_array("color_family")
    styles = catalogue.attribute_array("style")
    collections = catalogue.attribute_array("collection")
    prices = catalogue.prices

    tolerance = max(150.0, archetype.price_tolerance / max(0.35, price_sensitivity))
    price_fit = np.exp(-0.5 * ((prices - archetype.price_center) / tolerance) ** 2)

    return (
        W_CATEGORY * _lookup(categories, archetype.category)
        + W_COLOR * _lookup(colors, archetype.color)
        + W_STYLE * _lookup(styles, archetype.style)
        + W_COLLECTION * _lookup(collections, archetype.collection)
        + W_PRICE * price_fit
    )


def _to_affinity(scores: np.ndarray, sharpness: float = 2.6) -> np.ndarray:
    """Standardise then squash into (0, 1).

    Standardising per user is what makes affinity comparable across people: a
    picky user and an easy-going user both end up with a full-range preference
    vector, differing in *what* they like rather than in overall enthusiasm.
    """
    std = scores.std()
    if std < 1e-9:
        return np.full_like(scores, 0.5)
    z = (scores - scores.mean()) / std
    return 1.0 / (1.0 + np.exp(-sharpness * z))


def _sample_archetype(rng: np.random.Generator, exclude: str | None = None) -> str:
    names = [n for n in ARCHETYPES if n != exclude]
    weights = np.array([ARCHETYPES[n].share for n in names], dtype=float)
    return str(rng.choice(names, p=weights / weights.sum()))


def build_user_population(
    catalogue: Catalogue,
    n_users: int,
    history_days: int,
    rng: np.random.Generator,
) -> list[UserProfile]:
    """Create ``n_users`` shoppers with hidden preferences over ``catalogue``."""
    profiles: list[UserProfile] = []

    for i in range(1, n_users + 1):
        primary = _sample_archetype(rng)
        secondary = _sample_archetype(rng, exclude=primary)
        # Most people have one dominant taste with a minor second influence.
        secondary_weight = float(np.clip(rng.beta(2.0, 5.0) * 0.65, 0.0, 0.5))

        price_sensitivity = float(np.clip(rng.lognormal(0.0, 0.35), 0.4, 2.6))

        # Sessions/day is heavy-tailed: a few power users, a long casual tail.
        sessions_per_day = float(np.clip(rng.lognormal(np.log(0.10), 0.85), 0.012, 0.85))

        arch_primary = ARCHETYPES[primary]
        arch_secondary = ARCHETYPES[secondary]

        base_scores = (1.0 - secondary_weight) * _archetype_scores(
            catalogue, arch_primary, price_sensitivity
        ) + secondary_weight * _archetype_scores(catalogue, arch_secondary, price_sensitivity)

        # Private taste: two users of the same archetype still disagree.
        idiosyncrasy = rng.normal(0.0, 0.055, size=len(catalogue))
        base_scores = base_scores + idiosyncrasy

        # ~18% of users drift toward a different archetype during the window.
        drift_to: str | None = None
        drift_start = 1.0
        drift_scores = base_scores
        if rng.random() < 0.18:
            drift_to = _sample_archetype(rng, exclude=primary)
            drift_start = float(rng.uniform(0.35, 0.75))
            drift_scores = (
                0.35 * base_scores
                + 0.65 * _archetype_scores(catalogue, ARCHETYPES[drift_to], price_sensitivity)
                + idiosyncrasy
            )

        conversion = float(
            np.clip(
                arch_primary.conversion_multiplier * rng.lognormal(0.0, 0.45), 0.02, 3.0
            )
        )

        # Signup timing. 78% predate the window; the rest arrive during it, with
        # a deliberate cluster in the final 30 days so evaluation has genuinely
        # cold users rather than only users who happen to be quiet.
        roll = rng.random()
        if roll < 0.78:
            signup_day = -int(rng.integers(1, 720))
        elif roll < 0.90:
            signup_day = int(rng.integers(0, max(1, history_days - 30)))
        else:
            signup_day = int(rng.integers(max(0, history_days - 30), history_days))

        # Declared style agrees with real taste ~68% of the time. Self-reported
        # preference being unreliable is the whole reason behavioural data wins.
        if rng.random() < 0.68:
            declared = arch_primary.declared_style
        else:
            declared = ARCHETYPES[_sample_archetype(rng)].declared_style

        profiles.append(
            UserProfile(
                user_id=f"U{i:05d}",
                username=f"tessch_user_{i:05d}",
                email=f"user{i:05d}@example.com",
                age_group=str(rng.choice(AGE_GROUPS, p=[0.26, 0.34, 0.24, 0.16])),
                # Gender is sampled independently of archetype on purpose: the
                # simulator should not bake demographic stereotypes into the
                # data and then have the model "discover" them.
                gender=str(rng.choice(["female", "male", "other", "undisclosed"],
                                      p=[0.44, 0.44, 0.04, 0.08])),
                declared_style=declared,
                device_preference=str(rng.choice(DEVICE_TYPES, p=[0.42, 0.50, 0.08])),
                signup_day=signup_day,
                primary=primary,
                secondary=secondary,
                secondary_weight=secondary_weight,
                sessions_per_day=sessions_per_day,
                explore_rate=float(np.clip(rng.beta(2.0, 6.0), 0.02, 0.6)),
                conversion_propensity=conversion,
                dwell_scale=float(np.clip(rng.lognormal(0.0, 0.32), 0.35, 3.0)),
                attention_span=float(np.clip(rng.lognormal(0.0, 0.45), 0.3, 3.2)),
                drift_to=drift_to,
                drift_start=drift_start,
                affinity_base=_to_affinity(base_scores),
                affinity_drift=_to_affinity(drift_scores),
            )
        )

    return profiles


def mass_appeal(profiles: list[UserProfile]) -> np.ndarray:
    """Mean base affinity per product across the population.

    Used to seed the merchandising popularity prior, so that the products the
    site pushes hardest are broadly (but not perfectly) the ones people like.
    """
    if not profiles:
        return np.zeros(0)
    return np.mean([p.affinity_base for p in profiles], axis=0)


__all__ = [
    "Archetype",
    "ARCHETYPES",
    "UserProfile",
    "build_user_population",
    "mass_appeal",
]
