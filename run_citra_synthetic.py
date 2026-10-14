"""CPU correctness/system smoke experiment. No prediction accuracy claims."""

import argparse
from dataclasses import asdict
from datetime import timedelta
import json
from pathlib import Path
from time import perf_counter
import numpy as np

from citra import CITRAIndex
from citra.kernel import exact_aggregate
from citra.prediction import FixedLinearForecaster
from citra.synthetic import EPOCH, synthetic_store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dimensions", type=int, default=4)
    parser.add_argument("--radius", type=float, default=0.05)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.0, 0.01, 0.1, 1.0])
    parser.add_argument("--cutoff-days", type=int, nargs="+", default=[0, 10, 30])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("repeats must be positive")
    start = perf_counter()
    store = synthetic_store(args.documents, args.dimensions, seed=args.seed, radius=args.radius)
    insert_seconds = perf_counter() - start
    model = FixedLinearForecaster(12, 2, 3, store.impact_dim, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    history = rng.normal(size=(1, 2, 12))
    rows = []
    for day in args.cutoff_days:
        cutoff = EPOCH + timedelta(days=day)
        q = rng.normal(size=args.dimensions) * args.radius
        start = perf_counter()
        oracle = store.aggregate_exact(q, cutoff)
        full_seconds = perf_counter() - start
        snapshot = store.snapshot(cutoff)
        start = perf_counter()
        index = CITRAIndex(snapshot, store.embedding_dim, store.impact_dim,
                           store.kernel, store.leaf_size)
        prepare_seconds = perf_counter() - start
        timings = []
        for _ in range(args.repeats):
            start = perf_counter()
            exact_aggregate(snapshot, q, store.kernel, store.impact_dim)
            timings.append(perf_counter() - start)
        baseline = {"full_scan_seconds": full_seconds,
                    "full_scan_metadata_records": len(store.documents),
                    "full_scan_kernel_evaluations": len(snapshot),
                    "filtered_exact_p50_seconds": float(np.percentile(timings, 50)),
                    "filtered_exact_p95_seconds": float(np.percentile(timings, 95)),
                    "filtered_exact_documents_scanned": len(snapshot),
                    "index_prepare_seconds": prepare_seconds,
                    "index_document_reads": index.build_document_reads,
                    "index_summary_bytes": index.summary_bytes,
                    "index_resident_estimate_bytes": index.resident_bytes}
        if index.root:
            coarse, _ = index.approximate_node(index.root, q)
        else:
            coarse = np.zeros(store.impact_dim)
        baseline["uncertified_root_aggregate_error"] = float(np.linalg.norm(coarse - oracle))
        for epsilon in args.epsilons:
            # Index is maintained on writes, reused even for a new cutoff.
            cold_prediction = model.predict_certified(history, store, q, cutoff, epsilon)
            cold = cold_prediction.aggregation
            timings = []
            for _ in range(args.repeats):
                prediction = model.predict_certified(history, store, q, cutoff, epsilon)
                result = prediction.aggregation
                timings.append(result.stats.elapsed_seconds)
            full = model.predict(history, oracle[None])
            fast = prediction.prediction
            deviation = float(np.linalg.norm(fast - full))
            tolerance = 1e-10 + 1e-10 * float(np.linalg.norm(full))
            if deviation > prediction.certified_bound + tolerance or prediction.certified_bound > epsilon:
                raise AssertionError("certificate violation")
            rows.append({"cutoff_days": day, "epsilon": epsilon,
                         "visible_documents": len(snapshot),
                         "total_versions": len(store.documents),
                         "cold_stats": asdict(cold.stats), "warm_stats": asdict(result.stats),
                         "p50_seconds": float(np.percentile(timings, 50)),
                         "p95_seconds": float(np.percentile(timings, 95)),
                         "certified_bound": prediction.certified_bound,
                         "actual_prediction_deviation": deviation,
                         "bound_tightness": deviation / prediction.certified_bound
                         if prediction.certified_bound else None,
                         "warm_scan_fraction": result.stats.documents_scanned / len(snapshot)
                         if snapshot else 0.0, "baselines": baseline})
    report = {"seed": args.seed, "numpy": np.__version__, "device": "CPU",
              "timestamp_provenance": "simulated", "store_insert_seconds": insert_seconds,
              "protocol_version": 2,
              "index_write_seconds": store.total_update_seconds,
              "index_write_document_reads": store.index.total_update_document_reads,
              "note": "Fixed untrained forecaster; no MSE/MAE or real-data claims. "
              "Oracle checks are offline. cold_stats means first query on maintained index; "
              "construction cost is in store_insert_seconds/index_write_seconds.",
              "queries": rows}
    payload = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
