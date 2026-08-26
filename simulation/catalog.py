"""Product catalogue: load the curated core and extend it deterministically.

Why extend at all?
------------------
`data/products/products.json` holds 50 hand-written TESSCH products. That is a
good catalogue to *read*, but a bad catalogue to *model*: with 50 items, an
average user touches enough of them that the user-item matrix ends up ~50%
dense. Real e-commerce matrices are 99%+ sparse, and every interesting problem
in recommendation — cold items, candidate generation, popularity bias — only
shows up when the catalogue is much larger than any one user's history.

So we keep all 50 curated products verbatim (``is_core = True``) and generate
additional variants from the same design language to reach ``target_size``.
The generator is seeded, so the catalogue is byte-identical on every run.

Descriptions are composed from attribute-keyed clause banks rather than a single
template. That matters for Stage 3's TF-IDF content model: if every generated
description shared one skeleton, TF-IDF would separate core from generated items
instead of learning anything about style.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from backend.core.config import PRODUCTS_JSON
from backend.core.constants import color_family

# ─────────────────────────────────────────────────────────────────────────────
# Design language
# ─────────────────────────────────────────────────────────────────────────────

NAME_PREFIXES: tuple[str, ...] = (
    "Alder", "Amber", "Anvil", "Ash", "Aster", "Basalt", "Beacon", "Bevel",
    "Birch", "Bluff", "Brisk", "Cairn", "Cove", "Crag", "Crux", "Dale",
    "Dart", "Dawn", "Delta", "Ember", "Fable", "Fathom", "Fern", "Flare",
    "Flint", "Fjord", "Gale", "Glade", "Grain", "Granite", "Grove", "Harbor",
    "Haze", "Heath", "Helm", "Hollow", "Ivory", "Junction", "Kiln", "Knoll",
    "Lattice", "Ledger", "Loft", "Lumen", "Marsh", "Meridian", "Mica", "Moor",
    "Nectar", "Nimbus", "North", "Oxide", "Palisade", "Pebble", "Pier",
    "Pivot", "Plume", "Quarry", "Quill", "Rally", "Ramble", "Ravine", "Relay",
    "Rift", "Rove", "Sable", "Salt", "Scout", "Shale", "Shard", "Sierra",
    "Silt", "Slate", "Sonder", "Spruce", "Stark", "Stone", "Summit", "Tally",
    "Thicket", "Timber", "Torrent", "Trace", "Trellis", "Tundra", "Vale",
    "Vantage", "Verge", "Vessel", "Warden", "Weald", "Wharf", "Willow",
    "Wren", "Zephyr",
)

# Category -> plausible attribute space. Constraining this per-category is what
# keeps generated products coherent (no Gore-Tex sandals, no formal high-tops).
CATEGORY_SPEC: dict[str, dict[str, Any]] = {
    "Sneaker": {
        "upper_types": ("Low-top", "Low-top", "Low-top", "Mid-top", "High-top"),
        "materials": ("Canvas", "Leather", "Suede", "Knit", "Nubuck"),
        "styles": ("Casual", "Casual", "Casual", "Streetwear", "Smart Casual"),
        "collections": ("Core", "Street", "Summer"),
        "price_range": (2499, 3699),
        "colors": (
            "Black", "White", "Off-White", "Cream", "Grey", "Navy", "Charcoal",
            "Olive", "Beige", "Sand", "Sky Blue", "Ice Blue", "Rust", "Pink",
            "Teal", "Dark Green", "Burgundy", "Tan",
        ),
    },
    "Athletic": {
        "upper_types": ("Low-top", "Low-top", "Low-top", "Mid-top", "High-top"),
        "materials": ("Mesh", "Knit", "Mesh", "Ripstop", "Knit"),
        "styles": ("Athletic", "Athletic", "Athletic", "Outdoor", "Streetwear"),
        "collections": ("Active", "Active", "Terrain", "Street"),
        "price_range": (3199, 4399),
        "colors": (
            "Black", "White", "Grey", "Blue", "Steel Blue", "Red", "Coral",
            "Neon Green", "Yellow", "Teal", "Olive", "Khaki", "Navy", "Purple",
        ),
    },
    "Boot": {
        "upper_types": ("Ankle", "Ankle", "Mid", "Mid"),
        "materials": (
            "Full-grain Leather", "Nubuck", "Suede", "Waxed Canvas", "Gore-Tex",
        ),
        "styles": ("Heritage", "Heritage", "Outdoor", "Streetwear"),
        "collections": ("Heritage", "Terrain", "Street"),
        "price_range": (3999, 5599),
        "colors": (
            "Black", "Brown", "Tan", "Cognac", "Grey", "Olive", "Khaki",
            "Charcoal", "White", "Navy", "Rust", "Burgundy",
        ),
    },
    "Sandal": {
        "upper_types": ("Open", "Open", "Slip-on"),
        "materials": ("Leather", "Leather", "Synthetic", "Suede"),
        "styles": ("Casual", "Casual", "Smart Casual"),
        "collections": ("Summer",),
        "price_range": (1599, 2599),
        "colors": (
            "Tan", "Brown", "Black", "White", "Beige", "Sand", "Cognac",
            "Olive", "Grey", "Pink",
        ),
    },
    "Loafer": {
        "upper_types": ("Slip-on",),
        "materials": (
            "Suede", "Full-grain Leather", "Leather", "Patent Leather", "Nubuck",
        ),
        "styles": ("Smart Casual", "Smart Casual", "Formal", "Formal"),
        "collections": ("Refined",),
        "price_range": (2899, 4199),
        "colors": (
            "Black", "Brown", "Cognac", "Burgundy", "Beige", "Grey", "Navy",
            "White", "Tan", "Off-White",
        ),
    },
}

#: Sampling weights over categories for generated products, roughly matching a
#: real footwear brand's assortment (sneakers dominate shelf space).
CATEGORY_MIX: dict[str, float] = {
    "Sneaker": 0.34,
    "Athletic": 0.26,
    "Boot": 0.18,
    "Sandal": 0.11,
    "Loafer": 0.11,
}

# ── Description clause banks ─────────────────────────────────────────────────
# Each clause is picked independently, so descriptions vary lexically even when
# two products share every structured attribute.

OPENERS: dict[str, tuple[str, ...]] = {
    "Sneaker": (
        "A considered take on the everyday low-profile silhouette.",
        "Built around a clean last and an unfussy upper.",
        "Our everyday sneaker, refined over three seasons.",
        "Quiet proportions, careful stitching, nothing shouted.",
        "A wear-anywhere shoe that earns its place in the rotation.",
    ),
    "Athletic": (
        "Engineered for output, not for show.",
        "A performance shoe tuned for repeat sessions.",
        "Built to be run in, trained in, and run in again.",
        "Every gram here is doing a job.",
        "A trainer designed around ground feel and rebound.",
    ),
    "Boot": (
        "A boot made to be resoled, not replaced.",
        "Heavier in the hand, and deliberately so.",
        "Constructed the slow way, for the long run.",
        "A cold-weather workhorse with proper bones.",
        "Rugged construction under a considered exterior.",
    ),
    "Sandal": (
        "Warm-weather footwear that respects your arches.",
        "Minimal on the foot, generous underfoot.",
        "A sandal built for long days on hot pavement.",
        "Pared back to the parts that matter.",
        "Summer comfort without the plastic feel.",
    ),
    "Loafer": (
        "A slip-on that reads as considered rather than casual.",
        "Sharp lines, soft break-in.",
        "Dressed up or down without complaint.",
        "The shoe you keep by the door on meeting days.",
        "Refined construction in a low-effort package.",
    ),
}

UPPER_CLAUSES: dict[str, tuple[str, ...]] = {
    "Canvas": (
        "The cotton canvas upper breathes freely and softens with wear.",
        "Tightly woven canvas keeps the shape crisp through the season.",
        "A washed canvas upper gives it a lived-in hand from day one.",
    ),
    "Leather": (
        "Smooth leather panels are cut to minimise seams across the vamp.",
        "The leather upper is lined edge to edge for a clean interior.",
        "Supple leather takes a polish and holds a crease well.",
    ),
    "Full-grain Leather": (
        "Full-grain leather develops a patina unique to the wearer.",
        "Thick full-grain hide resists scuffing and burnishes beautifully.",
        "The full-grain upper is vegetable-tanned and lightly waxed.",
    ),
    "Patent Leather": (
        "High-shine patent leather needs nothing more than a wipe.",
        "The patent finish holds a mirror gloss under evening light.",
        "A patent upper keeps the silhouette sharp and formal.",
    ),
    "Suede": (
        "Brushed suede gives the upper a soft, matte depth.",
        "The suede nap catches light differently from every angle.",
        "Water-resistant treated suede keeps its colour through drizzle.",
    ),
    "Nubuck": (
        "Sanded nubuck offers suede's texture with more structure.",
        "The nubuck upper resists abrasion better than it looks.",
        "Oiled nubuck darkens attractively where it flexes.",
    ),
    "Mesh": (
        "An engineered mesh upper vents heat where it builds.",
        "Zoned mesh density holds the midfoot while opening the forefoot.",
        "The seamless mesh upper eliminates hotspots on longer efforts.",
    ),
    "Knit": (
        "A precision knit upper wraps the foot without pressure points.",
        "The one-piece knit stretches with the foot and springs back.",
        "Circular-knit construction means no seams and no break-in.",
    ),
    "Ripstop": (
        "Ripstop fabric shrugs off scrapes on rock and root.",
        "The ripstop shell adds tear resistance at almost no weight.",
        "A ripstop upper dries quickly after stream crossings.",
    ),
    "Waxed Canvas": (
        "Waxed canvas sheds light rain and stiffens into shape.",
        "The waxed shell can be re-proofed season after season.",
        "Hand-waxed canvas ages into something entirely your own.",
    ),
    "Synthetic": (
        "The quick-dry synthetic upper is built for water and back again.",
        "Closed-cell synthetic straps stay light even when soaked.",
        "A wipe-clean synthetic upper handles sand and salt without complaint.",
    ),
    "Gore-Tex": (
        "A Gore-Tex membrane keeps water out while letting vapour escape.",
        "The Gore-Tex bootie is seam-sealed throughout.",
        "Gore-Tex lining handles sustained wet without turning clammy.",
    ),
}

SOLE_CLAUSES: dict[str, tuple[str, ...]] = {
    "Casual": (
        "A vulcanised rubber sole keeps the profile low and flexible.",
        "The gum rubber outsole grips smooth floors without squeaking.",
        "A slim cupsole adds structure without adding stack height.",
    ),
    "Smart Casual": (
        "A lightweight rubber sole makes it quieter than it looks.",
        "The leather-wrapped midsole keeps the lines clean.",
        "A thin flexible sole reads formal but walks casual.",
    ),
    "Formal": (
        "Blake-stitched construction keeps the sole slim and resoleable.",
        "A leather sole with a rubber heel tap balances polish and grip.",
        "Goodyear welting means this can be rebuilt rather than binned.",
    ),
    "Athletic": (
        "A dual-density foam midsole returns energy on repeat strides.",
        "The carbon-rubber outsole holds up to treadmill and track alike.",
        "A herringbone tread pattern bites on multi-directional work.",
    ),
    "Outdoor": (
        "Deep lugs clear mud rather than packing it in.",
        "A Vibram outsole and nylon shank steady the foot on uneven ground.",
        "The multi-directional lug pattern grips on descent as well as climb.",
    ),
    "Streetwear": (
        "An exaggerated stacked sole gives it presence on the street.",
        "The chunky EVA midsole adds height and absorbs kerb impact.",
        "A sculpted platform sole keeps the proportions deliberately bold.",
    ),
    "Heritage": (
        "A stacked leather heel and crepe sole nod to the original pattern.",
        "The welted sole is built to be resoled several times over.",
        "A cork filler bed moulds to the foot over the first month.",
    ),
}

COMFORT_CLAUSES: tuple[str, ...] = (
    "A moulded footbed supports the arch without forcing it.",
    "The removable insole means you can drop in your own orthotic.",
    "An OrthoLite insole keeps its loft far longer than standard foam.",
    "A padded collar stops the heel from lifting on long walks.",
    "The cushioned heel pocket takes the sting out of hard surfaces.",
    "A wider toe box lets the foot splay naturally under load.",
)

CLOSERS: dict[str, tuple[str, ...]] = {
    "dark": (
        "The darker colourway hides city wear better than most.",
        "A tonal palette keeps it easy to pair with almost anything.",
        "Deep colour, minimal branding, maximum flexibility.",
    ),
    "light": (
        "The pale finish rewards a little care and looks sharp for it.",
        "A bright upper lifts an otherwise plain outfit.",
        "Clean and crisp, with a stain-resistant treatment to help it stay that way.",
    ),
    "neutral": (
        "A mid-tone neutral that works across the whole wardrobe.",
        "Understated colour that never fights what you are wearing.",
        "Neutral enough for daily wear, textured enough to stay interesting.",
    ),
    "earth": (
        "Earth tones that only look better as they weather.",
        "A warm natural shade that suits autumn layering.",
        "The muted earth colourway ages rather than fades.",
    ),
    "bright": (
        "A high-visibility colourway that is easy to spot and hard to ignore.",
        "Saturated colour for people who would rather not blend in.",
        "The bold shade is the point — wear it accordingly.",
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Catalogue:
    """An immutable view over the product catalogue.

    Exposes the lookups the simulator and the models both need, computed once.
    """

    products: list[dict[str, Any]]
    _index: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_index",
            {p["product_id"]: i for i, p in enumerate(self.products)},
        )

    def __len__(self) -> int:
        return len(self.products)

    def __iter__(self):
        return iter(self.products)

    @property
    def ids(self) -> list[str]:
        return [p["product_id"] for p in self.products]

    def position(self, product_id: str) -> int:
        """Row index of a product in the catalogue arrays."""
        return self._index[product_id]

    def get(self, product_id: str) -> dict[str, Any]:
        return self.products[self._index[product_id]]

    def attribute_array(self, key: str) -> np.ndarray:
        return np.array([p.get(key) for p in self.products], dtype=object)

    @property
    def prices(self) -> np.ndarray:
        return np.array([float(p["price"]) for p in self.products], dtype=float)

    @property
    def available_mask(self) -> np.ndarray:
        return np.array([bool(p["availability"]) for p in self.products], dtype=bool)

    def indices_where(self, key: str, value: Any) -> np.ndarray:
        return np.array(
            [i for i, p in enumerate(self.products) if p.get(key) == value], dtype=int
        )


# ─────────────────────────────────────────────────────────────────────────────
def load_core_products(path: Path | None = None) -> list[dict[str, Any]]:
    """Read the 50 curated products and normalise their fields."""
    path = path or PRODUCTS_JSON
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    products: list[dict[str, Any]] = []
    for item in raw:
        product = dict(item)
        product["price"] = float(product["price"])
        product["availability"] = bool(product.get("availability", True))
        product["color_family"] = color_family(product.get("color"))
        product["is_core"] = True
        products.append(product)
    return products


def _sentence_join(parts: list[str]) -> str:
    return " ".join(part.strip() for part in parts if part)


def _generate_product(
    index: int,
    rng: np.random.Generator,
    used_names: set[str],
) -> dict[str, Any]:
    categories = list(CATEGORY_MIX)
    weights = np.array([CATEGORY_MIX[c] for c in categories], dtype=float)
    category = str(rng.choice(categories, p=weights / weights.sum()))
    spec = CATEGORY_SPEC[category]

    upper_type = str(rng.choice(spec["upper_types"]))
    material = str(rng.choice(spec["materials"]))
    style = str(rng.choice(spec["styles"]))
    collection = str(rng.choice(spec["collections"]))
    color = str(rng.choice(spec["colors"]))
    family = color_family(color)

    low, high = spec["price_range"]
    # Prices cluster at the ...99 / ...98 marks the way retail prices do.
    price = float(int(rng.uniform(low, high) / 100) * 100 + rng.choice([98.0, 99.0]))

    # Name: <Prefix> <Colour> is the house convention; fall back to adding the
    # category word if that collides with something already in the catalogue.
    for _ in range(60):
        prefix = str(rng.choice(NAME_PREFIXES))
        candidate = f"{prefix} {color}"
        if candidate not in used_names:
            break
        candidate = f"{prefix} {color} {category}"
        if candidate not in used_names:
            break
    else:  # pragma: no cover - 96 prefixes x 28 colours makes this unreachable
        candidate = f"{prefix} {color} {index:03d}"
    used_names.add(candidate)

    description = _sentence_join(
        [
            str(rng.choice(OPENERS[category])),
            str(rng.choice(UPPER_CLAUSES[material])),
            str(rng.choice(SOLE_CLAUSES[style])),
            str(rng.choice(COMFORT_CLAUSES)),
            str(rng.choice(CLOSERS[family])),
        ]
    )

    product_id = f"T{index:03d}"
    return {
        "product_id": product_id,
        "product_name": candidate,
        "category": category,
        "upper_type": upper_type,
        "color": color,
        "color_family": family,
        "style": style,
        "price": price,
        "material": material,
        "collection": collection,
        "description": description,
        # ~6% of the catalogue is out of stock at any time, as in real retail.
        "availability": bool(rng.random() > 0.06),
        "image_url": f"/images/products/{product_id}.jpg",
        "is_core": False,
    }


def build_catalogue(target_size: int = 300, seed: int = 20260825) -> Catalogue:
    """Curated 50 products, extended deterministically to ``target_size``."""
    products = load_core_products()
    if target_size <= len(products):
        return Catalogue(products[:target_size])

    rng = np.random.default_rng(seed)
    used_names = {p["product_name"] for p in products}
    for index in range(len(products) + 1, target_size + 1):
        products.append(_generate_product(index, rng, used_names))

    return Catalogue(products)


__all__ = [
    "Catalogue",
    "build_catalogue",
    "load_core_products",
    "CATEGORY_SPEC",
    "CATEGORY_MIX",
]
