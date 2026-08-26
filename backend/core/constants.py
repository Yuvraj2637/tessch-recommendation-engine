"""Domain constants shared by the simulator, database, API and frontend tracker.

Keeping the event vocabulary in one place means the tracking script, the
validation layer and the feature engineering can never silently disagree about
what "wishlist_add" is called.
"""

from __future__ import annotations

from typing import Final

# ── Event taxonomy ───────────────────────────────────────────────────────────
EVENT_HOME_VIEW: Final = "home_view"
EVENT_PAGE_VIEW: Final = "page_view"
EVENT_PRODUCT_IMPRESSION: Final = "product_impression"
EVENT_PRODUCT_HOVER: Final = "product_hover"
EVENT_PRODUCT_CLICK: Final = "product_click"
EVENT_PRODUCT_VIEW: Final = "product_view"
EVENT_WISHLIST_ADD: Final = "wishlist_add"
EVENT_WISHLIST_REMOVE: Final = "wishlist_remove"
EVENT_ADD_TO_CART: Final = "add_to_cart"
EVENT_REMOVE_FROM_CART: Final = "remove_from_cart"
EVENT_PURCHASE: Final = "purchase"
EVENT_SEARCH: Final = "search"
EVENT_PRODUCT_SHARE: Final = "product_share"

EVENT_TYPES: Final[tuple[str, ...]] = (
    EVENT_HOME_VIEW,
    EVENT_PAGE_VIEW,
    EVENT_PRODUCT_IMPRESSION,
    EVENT_PRODUCT_HOVER,
    EVENT_PRODUCT_CLICK,
    EVENT_PRODUCT_VIEW,
    EVENT_WISHLIST_ADD,
    EVENT_WISHLIST_REMOVE,
    EVENT_ADD_TO_CART,
    EVENT_REMOVE_FROM_CART,
    EVENT_PURCHASE,
    EVENT_SEARCH,
    EVENT_PRODUCT_SHARE,
)

#: Events that name a specific product (i.e. carry a usable product_id).
PRODUCT_EVENT_TYPES: Final[frozenset[str]] = frozenset(
    {
        EVENT_PRODUCT_IMPRESSION,
        EVENT_PRODUCT_HOVER,
        EVENT_PRODUCT_CLICK,
        EVENT_PRODUCT_VIEW,
        EVENT_WISHLIST_ADD,
        EVENT_WISHLIST_REMOVE,
        EVENT_ADD_TO_CART,
        EVENT_REMOVE_FROM_CART,
        EVENT_PURCHASE,
        EVENT_PRODUCT_SHARE,
    }
)

#: An impression means "the product was rendered on screen", not "the user
#: engaged with it". It is the denominator for CTR and — crucially — the record
#: of what the user *could* have chosen. Without it, every un-clicked product
#: looks identical to a product the user never saw, and the ranking model learns
#: exposure bias instead of preference.
EXPOSURE_EVENT_TYPES: Final[frozenset[str]] = frozenset({EVENT_PRODUCT_IMPRESSION})

#: Events that count as a positive engagement signal in the interest score.
ENGAGEMENT_EVENT_TYPES: Final[frozenset[str]] = frozenset(
    {
        EVENT_PRODUCT_HOVER,
        EVENT_PRODUCT_CLICK,
        EVENT_PRODUCT_VIEW,
        EVENT_WISHLIST_ADD,
        EVENT_ADD_TO_CART,
        EVENT_PURCHASE,
        EVENT_PRODUCT_SHARE,
    }
)

#: Ordered strongest-signal-last. Used for funnel reporting.
FUNNEL_STEPS: Final[tuple[str, ...]] = (
    EVENT_PRODUCT_IMPRESSION,
    EVENT_PRODUCT_CLICK,
    EVENT_PRODUCT_VIEW,
    EVENT_WISHLIST_ADD,
    EVENT_ADD_TO_CART,
    EVENT_PURCHASE,
)

# ── Surfaces (where on the site the interaction happened) ────────────────────
SURFACE_HOME_GRID: Final = "home_grid"
SURFACE_CATEGORY: Final = "category_page"
SURFACE_SEARCH: Final = "search_results"
SURFACE_RELATED: Final = "related_products"
SURFACE_RECOMMENDED: Final = "recommended_for_you"
SURFACE_WISHLIST: Final = "wishlist_page"
SURFACE_DIRECT: Final = "direct"

SURFACES: Final[tuple[str, ...]] = (
    SURFACE_HOME_GRID,
    SURFACE_CATEGORY,
    SURFACE_SEARCH,
    SURFACE_RELATED,
    SURFACE_RECOMMENDED,
    SURFACE_WISHLIST,
    SURFACE_DIRECT,
)

# ── Devices ──────────────────────────────────────────────────────────────────
DEVICE_TYPES: Final[tuple[str, ...]] = ("desktop", "mobile", "tablet")

# ── Catalogue vocabulary (mirrors data/products/products.json) ───────────────
CATEGORIES: Final[tuple[str, ...]] = ("Sneaker", "Athletic", "Boot", "Sandal", "Loafer")

STYLES: Final[tuple[str, ...]] = (
    "Casual",
    "Athletic",
    "Streetwear",
    "Heritage",
    "Outdoor",
    "Smart Casual",
    "Formal",
)

COLLECTIONS: Final[tuple[str, ...]] = (
    "Core",
    "Active",
    "Street",
    "Heritage",
    "Terrain",
    "Refined",
    "Summer",
)

#: Colours collapsed into families. Individual colours are too sparse to learn
#: from with a 50-product catalogue, but "does this user like dark shoes?" is a
#: real and learnable preference.
COLOR_FAMILIES: Final[dict[str, str]] = {
    "Black": "dark",
    "Charcoal": "dark",
    "Navy": "dark",
    "Dark Green": "dark",
    "Burgundy": "dark",
    "White": "light",
    "Off-White": "light",
    "Cream": "light",
    "Beige": "neutral",
    "Sand": "neutral",
    "Grey": "neutral",
    "Steel Blue": "neutral",
    "Khaki": "earth",
    "Olive": "earth",
    "Tan": "earth",
    "Brown": "earth",
    "Cognac": "earth",
    "Rust": "earth",
    "Red": "bright",
    "Yellow": "bright",
    "Neon Green": "bright",
    "Coral": "bright",
    "Pink": "bright",
    "Purple": "bright",
    "Teal": "bright",
    "Blue": "bright",
    "Sky Blue": "bright",
    "Ice Blue": "light",
}

COLOR_FAMILY_NAMES: Final[tuple[str, ...]] = ("dark", "light", "neutral", "earth", "bright")

AGE_GROUPS: Final[tuple[str, ...]] = ("18-24", "25-34", "35-44", "45+")

# ── Event provenance ─────────────────────────────────────────────────────────
#: Tracked from a real browser by frontend/js/tracker.js.
SOURCE_LIVE: Final = "live"
#: Produced by scripts/generate_events.py.
SOURCE_SIMULATED: Final = "simulated"

EVENT_SOURCES: Final[tuple[str, ...]] = (SOURCE_LIVE, SOURCE_SIMULATED)

# ── Anonymous visitor identifiers ────────────────────────────────────────────
#: Prefix for IDs minted by the browser tracker for visitors with no account.
#: Keeping a fixed prefix means "how much of this table is real traffic?" is a
#: one-line query, and it cannot collide with simulated users (``U00001``).
ANON_USER_PREFIX: Final = "anon_"

#: Max length of a user_id, matching ``users.user_id VARCHAR(20)``. The tracker
#: mints ``anon_`` + 12 hex chars = 17 characters, leaving headroom.
MAX_USER_ID_LENGTH: Final = 20
MAX_PRODUCT_ID_LENGTH: Final = 10


def is_anonymous_user(user_id: str | None) -> bool:
    """True if ``user_id`` was minted by the browser tracker rather than signup."""
    return bool(user_id) and str(user_id).startswith(ANON_USER_PREFIX)


def color_family(color: str | None) -> str:
    """Map a catalogue colour to its family, defaulting to ``neutral``."""
    if not color:
        return "neutral"
    return COLOR_FAMILIES.get(color, "neutral")


__all__ = [name for name in dir() if not name.startswith("_")]
