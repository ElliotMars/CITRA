"""Audit prepared data without upgrading unknown/simulated availability to verified."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_provider.citra_weather import WeatherData, UNKNOWN_TIME, TextCorpus, load_jsonl


def audit(directory, allow_simulated=False, seq_len=48, horizon=6, availability_protocol=None, text_delay_hours=None, stride=1):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    rows = load_jsonl(directory / "documents.jsonl")
    corpus = TextCorpus(rows, manifest["base_dim"], manifest["metadata_dim"], allow_simulated=allow_simulated,
                        base_encoder_version=manifest["base_encoder_version"])
    counts = {p: sum(d.availability_provenance == p for d in corpus.documents) for p in ("unknown", "verified", "simulated")}
    report = dict(manifest=manifest, text_provenance_counts=counts, strict_backtest_ready=False,
                  allow_simulated=allow_simulated, stride=stride, errors=[])
    try:
        data = WeatherData(directory, allow_simulated=allow_simulated, availability_protocol=availability_protocol,
                           text_delay_hours=text_delay_hours)
        report["availability_protocol"] = data.availability_protocol
        windows = {split: data.windows(split, seq_len, horizon, list(range(data.values.shape[1])), stride)
                   for split in ("train", "validation", "test")}
        coverage = {}
        for split, ds in windows.items():
            mask = data.corpus.visibility([ds[i][2] for i in range(len(ds))])
            coverage[split] = dict(windows=len(ds), windows_with_visible_text=int(mask.any(axis=1).sum()),
                                   dropped_missing=ds.dropped_missing)
        report.update(coverage=coverage, unknown_numeric_cells=int(np.sum(data.available == UNKNOWN_TIME)))
        report["strict_backtest_ready"] = (not manifest["is_fixture"] and not allow_simulated
            and manifest["numeric_availability_provenance"] == "verified"
            and all(row["windows_with_visible_text"] for row in coverage.values()))
        report["scenario_ready"] = all(row["windows_with_visible_text"] for row in coverage.values())
        report["ready_for_assumption_backtest"] = (data.availability_protocol == "issuance_asof_assumption"
                                                   and report["scenario_ready"])
    except ValueError as error:
        report["errors"].append(str(error))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--allow-simulated", action="store_true")
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--availability-protocol", choices=["verified_availability", "issuance_asof_assumption"])
    parser.add_argument("--text-delay-hours", type=float)
    args = parser.parse_args()
    report = audit(args.data, args.allow_simulated, args.seq_len, args.horizon,
                   args.availability_protocol, args.text_delay_hours, args.stride)
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(payload + "\n")
    print(payload)
    if not (report.get("ready_for_assumption_backtest", False) if args.availability_protocol == "issuance_asof_assumption"
            else report["scenario_ready"] if args.allow_simulated and "scenario_ready" in report else report["strict_backtest_ready"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
