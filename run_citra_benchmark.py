"""Rolling-cutoff system benchmarks, local updates, and full memory diagnostics."""
import argparse
from dataclasses import asdict, replace
from datetime import timedelta
import gc
import json
from pathlib import Path
import resource
from time import perf_counter
import tracemalloc
import numpy as np

from citra.index import deep_size
from citra.kernel import exact_aggregate
from citra.synthetic import EPOCH, synthetic_store


def rss_bytes():
    """Linux current RSS; ru_maxrss is separately labelled process lifetime peak."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return None


def measured(call, repeats):
    times = []
    for _ in range(repeats):
        start = perf_counter()
        result = call()
        times.append(perf_counter() - start)
    return result, {"first_seconds": times[0], "p50_seconds": float(np.percentile(times, 50)),
                    "p95_seconds": float(np.percentile(times, 95)), "total_seconds": sum(times)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--dimensions", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--leaf-size", type=int, default=8)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[0, 5, 10, 15, 20, 30])
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.01, 0.1, 1.])
    parser.add_argument("--distances", type=float, nargs="+", default=[0., 0.2, 50.])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats <= 0 or args.updates < 0 or any(n <= 0 for n in args.documents):
        parser.error("invalid repeat/update/document count")
    sizes = []
    for n in args.documents:
        gc.collect()
        before_rss = rss_bytes()
        tracemalloc.start()
        start = perf_counter()
        store = synthetic_store(n, args.dimensions, seed=args.seed, leaf_size=args.leaf_size)
        construction_seconds = perf_counter() - start
        traced_current, traced_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        traced_write_seconds = store.total_update_seconds
        resident_rss = rss_bytes()
        memory = {"rss_before_bytes": before_rss, "rss_after_bytes": resident_rss,
                  "rss_delta_bytes": resident_rss - before_rss if before_rss and resident_rss else None,
                  "tracemalloc_resident_bytes": traced_current, "tracemalloc_construction_peak_bytes": traced_peak,
                  "process_lifetime_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  "index_with_documents_resident_estimate_bytes": store.index.resident_bytes,
                  "store_and_index_resident_estimate_bytes": deep_size(store._versions, store._intervals, store.index.root),
                  "numeric_summary_bytes": store.index.summary_bytes,
                  "leaf_document_references": sum(len(node.document_refs) for node in store.index.nodes),
                  "internal_document_references": sum(len(node.document_refs) for node in store.index.nodes if node.children)}
        # Measure execution costs in a separate untraced replay. Both systems
        # independently generate the identical seeded input and charge writes.
        del store
        gc.collect()
        start = perf_counter()
        scan_store = synthetic_store(n, args.dimensions, seed=args.seed, leaf_size=args.leaf_size, indexed=False)
        scan_construction_seconds = perf_counter() - start
        start = perf_counter()
        store = synthetic_store(n, args.dimensions, seed=args.seed, leaf_size=args.leaf_size)
        untraced_construction_seconds = perf_counter() - start
        query_rows, totals = [], {}
        snapshot_prepare_total = 0.
        retained_root = store.index.root
        for day in args.cutoffs:
            cutoff = EPOCH + timedelta(days=day)
            # Cached snapshot baseline has separately charged preparation, never hidden.
            start = perf_counter()
            snapshot = scan_store.snapshot(cutoff)
            snapshot_seconds = perf_counter() - start
            snapshot_prepare_total += snapshot_seconds
            for distance in args.distances:
                q = np.full(args.dimensions, distance)
                for epsilon in args.epsilons:
                    methods = {
                        "exact_full_scan": lambda: scan_store.aggregate_exact(q, cutoff),
                        "temporal_filter_exact": lambda: exact_aggregate(snapshot, q, store.kernel, store.impact_dim),
                        "uncertified_hierarchy": lambda: store.index.query(q, epsilon, cutoff=cutoff, uncertified=True),
                        "citra_global": lambda: store.query_certified(q, cutoff, epsilon, bound_mode="global"),
                        "citra_local": lambda: store.query_certified(q, cutoff, epsilon),
                    }
                    results = {}
                    for name, call in methods.items():
                        value, timing = measured(call, args.repeats)
                        a = value.aggregate if hasattr(value, "aggregate") else value
                        results[name] = {"aggregate": a, "timing": timing, "value": value}
                        totals[name] = totals.get(name, 0.) + timing["total_seconds"]
                    oracle = results["exact_full_scan"]["aggregate"]
                    row = {"cutoff_days": day, "query_coordinate": distance, "epsilon_aggregate": epsilon,
                           "visible_documents": len(snapshot), "snapshot_prepare_seconds": snapshot_seconds,
                           "methods": {}}
                    for name, result in results.items():
                        value = result["value"]
                        error = float(np.linalg.norm(result["aggregate"] - oracle))
                        record = dict(result["timing"], actual_aggregate_error=error)
                        if hasattr(value, "stats"):
                            record.update(stats=asdict(value.stats), aggregate_bound=value.aggregate_bound,
                                          certificate_scope=value.certificate_scope)
                            if name.startswith("citra") and (error > value.aggregate_bound + 1e-10*(1+np.linalg.norm(oracle))
                                                             or value.aggregate_bound > epsilon):
                                raise AssertionError("certified aggregation violation")
                        else:
                            record.update(documents_scanned=len(snapshot), exact_kernel_evaluations=len(snapshot),
                                          metadata_records_checked=len(store.documents) if name == "exact_full_scan" else 0)
                        row["methods"][name] = record
                    query_rows.append(row)
            if store.index.root is not retained_root:
                raise AssertionError("cutoff queries must not rebuild or modify the index")
        updates = []
        for i in range(args.updates):
            d = store.documents[i % len(store.documents)]
            revised = replace(d, version_id="benchmark_revision_" + str(i),
                              available_at=EPOCH + timedelta(days=35 + i), issued_at=EPOCH + timedelta(days=35 + i),
                              ingested_at=None, impact_vector=-d.impact_vector)
            store.append_revision(revised)
            stats = dict(store.last_update_stats)
            scan_store.append_revision(revised)
            stats["scan_store_update_seconds"] = scan_store.last_update_seconds
            start = perf_counter()
            verification_query = store.query_certified(np.zeros(args.dimensions), EPOCH + timedelta(days=60), 0.)
            stats["post_update_verification_seconds"] = perf_counter() - start
            np.testing.assert_array_equal(verification_query.aggregate,
                                          scan_store.aggregate_exact(np.zeros(args.dimensions), EPOCH + timedelta(days=60)))
            updates.append(stats)
        update_total = sum(u["elapsed_seconds"] for u in updates)
        sizes.append({"documents": n, "initial_versions": n + (n + 12)//13,
                      "construction_seconds_with_tracemalloc": construction_seconds,
                      "construction_seconds_untraced_replay": untraced_construction_seconds,
                      "scan_construction_seconds_untraced_replay": scan_construction_seconds,
                      "construction_index_write_seconds_with_tracemalloc": traced_write_seconds,
                      "construction_index_write_seconds_untraced_replay": store.total_update_seconds - update_total,
                      "construction_note": "Memory is a separate traced pass. Timing replays seeded generation+writes without tracing, in the same warmed process; not independent fresh-process cold samples.",
                      "memory": memory, "rolling_query_total_seconds": totals,
                      "filtered_exact_snapshot_prepare_total_seconds": snapshot_prepare_total,
                      "filtered_exact_queries_plus_snapshot_prepare_seconds": snapshot_prepare_total + totals.get("temporal_filter_exact", 0.),
                      "initial_write_plus_rolling_citra_seconds": untraced_construction_seconds + totals.get("citra_local", 0.),
                      "updates": updates, "update_total_seconds": update_total,
                      "write_query_update_citra_total_seconds": untraced_construction_seconds + totals.get("citra_local", 0.) + update_total,
                      "write_query_update_fullscan_total_seconds": scan_construction_seconds + totals.get("exact_full_scan", 0.) + sum(u["scan_store_update_seconds"] for u in updates),
                      "queries": query_rows})
    report = {"protocol_version": 2, "device": "CPU", "numpy": np.__version__, "seed": args.seed,
              "certificate_scope": "analytic aggregation remainder, explicit float64 tolerance; no prediction/label accuracy claim",
              "baseline_note": "All methods share legal sum semantics. temporal_filter_exact reuses a charged snapshot. "
              "uncertified_hierarchy ignores epsilon. Compare total write+rolling queries, not query scans alone. "
              "RSS includes interpreter/allocator and is not isolated index memory.", "sizes": sizes}
    payload = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
