"""Test runner executing all test modules without external dependencies."""

from __future__ import annotations

import gc
import inspect
import sys
import tempfile
import time
import traceback
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import tests.test_api as test_api
import tests.test_evaluation as test_evaluation
import tests.test_ingest as test_ingest
import tests.test_matrix as test_matrix
import tests.test_personal as test_personal
import tests.test_schema as test_schema
import tests.test_similarity as test_similarity


def run_all_tests() -> int:
    modules = [
        ("Database Schema", test_schema),
        ("Phase 1 Ingestion & Validation", test_ingest),
        ("Phase 2 Personalized Ranking", test_personal),
        ("User-Item Matrix", test_matrix),
        ("Phase 3 Item-Item Cosine Similarity", test_similarity),
        ("Evaluation Metrics", test_evaluation),
        ("FastAPI Backend Routers", test_api),
    ]

    total_passed = 0
    total_failed = 0
    start_time = time.perf_counter()

    print("=" * 70)
    print("TESSCH Recommendation System -- Test Suite")
    print("=" * 70)

    for group_name, module in modules:
        print(f"\n[Suite] {group_name}")
        functions = [
            (name, fn)
            for name, fn in inspect.getmembers(module, inspect.isfunction)
            if name.startswith("test_")
        ]

        for test_name, test_fn in functions:
            sig = inspect.signature(test_fn)
            params = sig.parameters

            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
                tmp_path = Path(tmp_dir)
                kwargs = {}

                try:
                    if "tmp_path" in params:
                        kwargs["tmp_path"] = tmp_path
                    if "sample_events" in params:
                        kwargs["sample_events"] = test_matrix.sample_events()
                    if "dummy_matrix" in params:
                        kwargs["dummy_matrix"] = test_similarity.dummy_matrix()
                    if "eval_catalog" in params:
                        kwargs["eval_catalog"] = test_evaluation.eval_catalog()
                    if "eval_model" in params:
                        kwargs["eval_model"] = test_evaluation.eval_model()

                    test_fn(**kwargs)
                    print(f"  [PASS] {test_name}")
                    total_passed += 1
                except Exception as exc:
                    print(f"  [FAIL] {test_name}: FAILED")
                    traceback.print_exc()
                    total_failed += 1
                finally:
                    gc.collect()

    elapsed = time.perf_counter() - start_time
    print("\n" + "=" * 70)
    print(f"Results: {total_passed} passed, {total_failed} failed in {elapsed:.2f}s")
    print("=" * 70)

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(run_all_tests())
