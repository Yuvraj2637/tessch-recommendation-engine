"""Central configuration for the TESSCH recommendation engine.

Every tunable lives here so that no magic number is buried in a model file.
Values are read from the environment (optionally via a `.env` file) and fall
back to the defaults documented in `.env.example`.

Paths are resolved relative to the repository root, not the current working
directory, so scripts behave identically no matter where they are launched from.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

try:  # python-dotenv is optional at runtime
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*_args, **_kwargs):  # type: ignore[misc]
        return False


# ── Paths ────────────────────────────────────────────────────────────────────
# backend/core/config.py -> backend/core -> backend -> <repo root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
PRODUCTS_DIR = DATA_DIR / "products"
RAW_EVENTS_DIR = DATA_DIR / "raw_events"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

PRODUCTS_JSON = PRODUCTS_DIR / "products.json"

load_dotenv(PROJECT_ROOT / ".env")


# ── Helpers ──────────────────────────────────────────────────────────────────
def _env_str(key: str, default: str) -> str:
    value = os.getenv(key)
    return default if value is None or value.strip() == "" else value.strip()


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env_str(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env_str(key, str(default)))
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    return _env_str(key, str(default)).lower() in {"1", "true", "yes", "on"}


def _env_date(key: str, default: date) -> date:
    try:
        return date.fromisoformat(_env_str(key, default.isoformat()))
    except ValueError:
        return default


def _resolve_sqlite_url(url: str) -> str:
    """Make relative SQLite paths absolute so cwd never matters.

    ``sqlite:///data/tessch.db`` -> ``sqlite:///C:/.../repo/data/tessch.db``
    An absolute path or a non-SQLite URL is returned untouched.
    """
    prefix = "sqlite:///"
    if not url.startswith(prefix):
        return url

    raw_path = url[len(prefix) :]
    if raw_path in {"", ":memory:"}:
        return url

    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return prefix + path.as_posix()


# ── Interest-score weights (Phase 5) ─────────────────────────────────────────
@dataclass(frozen=True)
class ScoreWeights:
    """Weights turning raw behaviour into a single implicit-preference score.

    These are a *starting hypothesis*, not a tuned optimum. Stage 3 sweeps them
    and reports which combination actually maximises NDCG@10 on held-out data.
    """

    view: float = 1.0
    dwell: float = 1.5
    repeat: float = 2.0
    click: float = 1.2
    wishlist: float = 4.0
    cart: float = 7.0
    purchase: float = 12.0

    @classmethod
    def from_env(cls) -> "ScoreWeights":
        return cls(
            view=_env_float("W_VIEW", 1.0),
            dwell=_env_float("W_DWELL", 1.5),
            repeat=_env_float("W_REPEAT", 2.0),
            click=_env_float("W_CLICK", 1.2),
            wishlist=_env_float("W_WISHLIST", 4.0),
            cart=_env_float("W_CART", 7.0),
            purchase=_env_float("W_PURCHASE", 12.0),
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "view": self.view,
            "dwell": self.dwell,
            "repeat": self.repeat,
            "click": self.click,
            "wishlist": self.wishlist,
            "cart": self.cart,
            "purchase": self.purchase,
        }


# ── Simulation settings (Stage 1) ────────────────────────────────────────────
@dataclass(frozen=True)
class SimulationSettings:
    seed: int = 20260825
    n_users: int = 80
    history_days: int = 30
    end_date: date = date(2026, 8, 25)

    @classmethod
    def from_env(cls) -> "SimulationSettings":
        return cls(
            seed=_env_int("SIM_SEED", 20260825),
            n_users=_env_int("SIM_N_USERS", 80),
            history_days=_env_int("SIM_HISTORY_DAYS", 30),
            end_date=_env_date("SIM_END_DATE", date(2026, 8, 25)),
        )


# ── Ingest validation (Phase 1) ──────────────────────────────────────────────
@dataclass(frozen=True)
class IngestSettings:
    """Bounds a dwell-time reading has to clear to be stored.

    Both ends of the range matter. Below ``min_dwell_seconds`` a "view" is a
    mis-click or a bounce, and counting it as interest would pull noise into
    every downstream score. Above ``max_dwell_seconds`` the number almost
    certainly measures an abandoned tab rather than attention, and because the
    Phase 2 ranking sums dwell time, one such reading would otherwise dominate
    a user's entire profile.
    """

    min_dwell_seconds: float = 1.0
    max_dwell_seconds: float = 1800.0

    @classmethod
    def from_env(cls) -> "IngestSettings":
        return cls(
            min_dwell_seconds=_env_float("MIN_DWELL_SECONDS", 1.0),
            max_dwell_seconds=_env_float("MAX_DWELL_SECONDS", 1800.0),
        )


# ── Top-level settings ───────────────────────────────────────────────────────
@dataclass(frozen=True)
class Settings:
    database_url: str = "sqlite:///data/tessch.db"
    sql_echo: bool = False
    #: Opt into write-ahead logging. Off by default because it breaks on
    #: cloud-synced folders — see backend/database/queries.connect().
    sqlite_wal: bool = False

    api_host: str = "127.0.0.1"
    api_port: int = 8000
    default_top_k: int = 8
    cors_origins: tuple[str, ...] = ("*",)

    decay_half_life_days: float = 30.0

    weights: ScoreWeights = field(default_factory=ScoreWeights)
    simulation: SimulationSettings = field(default_factory=SimulationSettings)
    ingest: IngestSettings = field(default_factory=IngestSettings)

    @classmethod
    def from_env(cls) -> "Settings":
        raw_origins = _env_str("CORS_ORIGINS", "*")
        origins = tuple(o.strip() for o in raw_origins.split(",") if o.strip())
        return cls(
            database_url=_resolve_sqlite_url(
                _env_str("DATABASE_URL", "sqlite:///data/tessch.db")
            ),
            sql_echo=_env_bool("SQL_ECHO", False),
            sqlite_wal=_env_bool("SQLITE_WAL", False),
            api_host=_env_str("API_HOST", "127.0.0.1"),
            api_port=_env_int("API_PORT", 8000),
            default_top_k=_env_int("DEFAULT_TOP_K", 8),
            cors_origins=origins or ("*",),
            decay_half_life_days=_env_float("DECAY_HALF_LIFE_DAYS", 30.0),
            weights=ScoreWeights.from_env(),
            simulation=SimulationSettings.from_env(),
            ingest=IngestSettings.from_env(),
        )

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def sqlite_path(self) -> Path:
        """Filesystem path behind a ``sqlite:///`` URL.

        The serving layer opens the database with stdlib ``sqlite3``, which wants
        a path rather than a URL.
        """
        if not self.is_sqlite:
            raise ValueError(
                f"sqlite_path is only meaningful for a SQLite DATABASE_URL, got {self.database_url!r}"
            )
        return Path(self.database_url[len("sqlite:///") :])


#: Module-level singleton, resolved once at import. Convenient for scripts.
settings = Settings.from_env()


def get_settings() -> Settings:
    """The active settings object.

    Prefer this over importing ``settings`` directly. The indirection exists so a
    test can call ``reload_settings()`` after changing an environment variable and
    have the whole application see the new value — with a direct import, whichever
    module imported first would hold a stale binding forever, and the resulting
    failures look like unrelated logic bugs.
    """
    return settings


def reload_settings() -> Settings:
    """Re-read the environment. For tests and for scripts that set env vars late."""
    global settings
    settings = Settings.from_env()
    return settings


def ensure_directories() -> None:
    """Create every directory the pipeline writes to."""
    for directory in (
        DATA_DIR,
        PRODUCTS_DIR,
        RAW_EVENTS_DIR,
        PROCESSED_DIR,
        ARTIFACTS_DIR,
        REPORTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


__all__ = [
    "get_settings",
    "reload_settings",
    "PROJECT_ROOT",
    "DATA_DIR",
    "PRODUCTS_DIR",
    "RAW_EVENTS_DIR",
    "PROCESSED_DIR",
    "ARTIFACTS_DIR",
    "REPORTS_DIR",
    "PRODUCTS_JSON",
    "ScoreWeights",
    "SimulationSettings",
    "Settings",
    "settings",
    "ensure_directories",
]
