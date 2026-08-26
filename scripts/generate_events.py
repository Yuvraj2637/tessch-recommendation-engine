"""Stage 1 entry point: build the world.

    python -m scripts.generate_events

Builds the catalogue, simulates the population's browsing history, loads
everything into the database, writes parquet copies, and prints the data-quality
report that says whether the result is worth modelling.

Everything is a deterministic function of ``SIM_SEED``, so two runs with the same
configuration produce byte-identical data.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow `python scripts/generate_events.py` as well as `python -m scripts...`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from backend.core.config import (
    PROCESSED_DIR,
    PRODUCTS_DIR,
    RAW_EVENTS_DIR,
    REPORTS_DIR,
    ensure_directories,
    settings,
)
from backend.database.session import create_db_engine, table_counts
from simulation.catalog import build_catalogue
from simulation.generator import EventSimulator
from simulation.loader import load_simulation, write_parquet
from simulation.report import build_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    sim = settings.simulation
    parser = argparse.ArgumentParser(description="Generate the TESSCH interaction dataset.")
    parser.add_argument("--users", type=int, default=sim.n_users,
                        help=f"number of simulated shoppers (default {sim.n_users})")
    parser.add_argument("--days", type=int, default=sim.history_days,
                        help=f"days of history (default {sim.history_days})")
    parser.add_argument("--catalog-size", type=int, default=300,
                        help="total products; the first 50 are the curated core (default 300)")
    parser.add_argument("--seed", type=int, default=sim.seed,
                        help=f"random seed (default {sim.seed})")
    parser.add_argument("--end-date", type=str, default=sim.end_date.isoformat(),
                        help="last day of the simulated window (YYYY-MM-DD)")
    parser.add_argument("--database-url", type=str, default=settings.database_url,
                        help="override DATABASE_URL for this run")
    parser.add_argument("--no-db", action="store_true",
                        help="skip the database load and only write parquet")
    parser.add_argument("--no-parquet", action="store_true",
                        help="skip the parquet export")
    parser.add_argument("--quiet", action="store_true", help="suppress per-user progress")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_directories()

    from datetime import date

    end_date = date.fromisoformat(args.end_date)

    print("TESSCH recommendation engine — Stage 1: build the world")
    print("-" * 78)
    print(f"  seed            {args.seed}")
    print(f"  users           {args.users:,}")
    print(f"  history         {args.days} days ending {end_date}")
    print(f"  catalogue       {args.catalog_size} products (50 curated + generated)")
    print(f"  database        {'(skipped)' if args.no_db else args.database_url}")
    print("-" * 78)

    # ── catalogue ────────────────────────────────────────────────────────────
    started = time.perf_counter()
    catalogue = build_catalogue(target_size=args.catalog_size, seed=args.seed)
    catalogue_path = PRODUCTS_DIR / "catalog_full.json"
    catalogue_path.write_text(
        json.dumps(catalogue.products, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[1/5] catalogue    {len(catalogue):>4} products -> {catalogue_path.name}"
          f"  ({time.perf_counter() - started:.1f}s)")

    # ── simulation ───────────────────────────────────────────────────────────
    started = time.perf_counter()
    simulator = EventSimulator(
        catalogue=catalogue,
        n_users=args.users,
        history_days=args.days,
        end_date=end_date,
        seed=args.seed,
    )
    output = simulator.run(progress=not args.quiet)
    elapsed = time.perf_counter() - started
    print(f"[2/5] simulation   {output.meta['n_events']:>9,} events, "
          f"{output.meta['n_sessions']:,} sessions, {output.meta['n_orders']:,} orders"
          f"  ({elapsed:.1f}s)")

    # ── parquet ──────────────────────────────────────────────────────────────
    if not args.no_parquet:
        started = time.perf_counter()
        paths = write_parquet(output)
        print(f"[3/5] parquet      {len(paths)} files -> {PROCESSED_DIR.name}/ and "
              f"{RAW_EVENTS_DIR.name}/ground_truth/  ({time.perf_counter() - started:.1f}s)")
    else:
        print("[3/5] parquet      skipped")

    # ── database ─────────────────────────────────────────────────────────────
    if not args.no_db:
        started = time.perf_counter()
        engine = create_db_engine(args.database_url)
        counts = load_simulation(output, bind=engine, reset=True)
        elapsed = time.perf_counter() - started
        print(f"[4/5] database     loaded in {elapsed:.1f}s")
        for table, count in table_counts(engine).items():
            print(f"        {table:<24} {count:>10,}")
    else:
        print("[4/5] database     skipped")

    # ── report ───────────────────────────────────────────────────────────────
    started = time.perf_counter()
    report = build_report(
        events=pd.DataFrame(output.events),
        sessions=pd.DataFrame(output.sessions),
        users=pd.DataFrame(output.users),
        products=pd.DataFrame(output.products),
        orders=pd.DataFrame(output.orders),
        affinity=pd.DataFrame(output.affinity_long),
        hidden_users=pd.DataFrame(output.hidden_users),
        meta=output.meta,
    )
    report_path = REPORTS_DIR / "stage1_data_quality.md"
    report_path.write_text(report.to_markdown(), encoding="utf-8")
    stats_path = REPORTS_DIR / "stage1_stats.json"
    stats_path.write_text(json.dumps(report.stats, indent=2, default=str), encoding="utf-8")
    print(f"[5/5] report       -> reports/{report_path.name}  "
          f"({time.perf_counter() - started:.1f}s)")

    print(report.to_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
