"""Prepare one region with explicit availability evidence or simulated delays.

Nothing infers historical availability from issuance/observation/file mtime.
Run from the repository root. Unknown availability remains unknown.
"""
import argparse
import csv
from datetime import timedelta
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import zipfile
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_provider.citra_weather import UNKNOWN_TIME, parse_time, epoch_seconds, availability_seconds, datetime_utc, load_jsonl
from citra.text_features import hash_text
from citra.availability import VERIFIED, ASSUMED, issuance_records


def afd_records(path, office, entity, ledger=None, simulated_delay=None, quarantine=None):
    ledger = ledger or {}
    records = []
    source = json.loads(Path(str(path) + ".metadata.json").read_text())
    if source["sha256"] != hashlib.sha256(Path(path).read_bytes()).hexdigest():
        raise ValueError("raw AFD archive differs from saved download hash")
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            raw = archive.read(member)
            text = raw.decode("utf-8", errors="replace")
            # IEM archive filenames include the product timestamp; ambiguous WMO
            # day-only headers are deliberately not used to guess month/year.
            match = re.search(r"(\d{12})(?:\d{2})?-K([A-Z]{3})-[A-Z0-9]{6}-AFD([A-Z]{3})", member.filename)
            simple = re.fullmatch(r"AFD([A-Z]{3})_(\d{12})(?:_[A-Z0-9]+)?\.txt", Path(member.filename).name)
            if match and match[2] == office and match[3] == office:
                stamp = match[1]
            elif simple and simple[1] == office:
                stamp = simple[2]
            else:
                if quarantine is not None:
                    quarantine.append(dict(member=member.filename,sha256=hashlib.sha256(raw).hexdigest(),
                                           reason="unrecognized product identity/timestamp"))
                    continue
                raise ValueError(f"unrecognized product identity/timestamp: {member.filename}")
            # Real IEM retrieve.py ZIPs currently use AFDxxx_YYYYmmddHHMM.txt.
            # Cross-check the WMO header and PIL, rather than treating the file
            # name as independently verified issuance evidence.
            normalized = text.replace("\r", "")
            header = re.search(r"^[A-Z]{4}\d{2}[ \t]+K" + office + r"[ \t]+(\d{6})", normalized, re.MULTILINE)
            if (not header or header[1] != stamp[6:12]
                    or not re.search(r"^AFD" + office + r"[ \t]*$", normalized, re.MULTILINE)):
                if quarantine is not None:
                    quarantine.append(dict(member=member.filename,sha256=hashlib.sha256(raw).hexdigest(),
                        archive_timestamp=stamp,wmo_day_time=header[1] if header else None,
                        reason="filename/WMO header/PIL timestamp mismatch"))
                    continue
                raise ValueError(f"filename/WMO header/PIL timestamp mismatch: {member.filename}")
            issued = parse_time(f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}T{stamp[8:10]}:{stamp[10:12]}:00Z")
            sha = hashlib.sha256(raw).hexdigest()
            evidence = ledger.get(sha)
            available, provenance = None, "unknown"
            if evidence:
                if not evidence.get("evidence_url"):
                    raise ValueError("verified availability ledger requires evidence_url")
                available, provenance = parse_time(evidence["available_at"]), "verified"
            elif simulated_delay is not None:
                available, provenance = issued + timedelta(minutes=simulated_delay), "simulated"
            if available is not None and available < issued:
                raise ValueError("availability precedes issuance")
            row = dict(doc_id=f"{office}:{stamp}", version_id=sha, source="IEM-NWS-AFD", entity_id=entity,
                       source_url=source["source_url"],
                       office_id=office, issued_at=issued.isoformat(), available_at=available.isoformat() if available else None,
                       valid_from=issued.isoformat(), valid_to=None, ingested_at=None, event_start=None, event_end=None,
                       text=text, raw_member=member.filename, retrieved_at=source["retrieved_at"],
                       sha256=sha, availability_provenance=provenance,
                       availability_evidence=evidence, metadata=[np.sin(issued.month * np.pi/6), np.cos(issued.month * np.pi/6)])
            records.append(row)
    # Archive duplicate copies of identical content are not distinct versions.
    unique = {(row["doc_id"], row["version_id"]): row for row in records}
    return sorted(unique.values(), key=lambda row: (row["issued_at"], row["raw_member"]))


def hourly_asos(path, station, ledger=None, simulated_delay=None,excluded_timestamps=()):
    ledger = ledger or {}
    source = json.loads(Path(str(path) + ".metadata.json").read_text())
    if source.get("timestamp_timezone") != "UTC" or source.get("station_id") != station:
        raise ValueError("raw observations need downloader metadata confirming UTC and station identity")
    payload = Path(path).read_bytes()
    if source["sha256"] != hashlib.sha256(payload).hexdigest():
        raise ValueError("raw ASOS file differs from saved download hash")
    lines = [line for line in payload.decode().splitlines() if line and not line.startswith("#")]
    buckets = {}
    for row in csv.DictReader(lines):
        if row["station"] != station:
            raise ValueError("mixed stations in a single-region series")
        if (row["station"],row["valid"]) in excluded_timestamps:
            continue
        observed = parse_time(row["valid"].replace(" ", "T") + ("Z" if not row["valid"].endswith("Z") else ""))
        end = epoch_seconds(observed) // 3600 * 3600 + 3600
        record_id = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        evidence = ledger.get(record_id)
        available = UNKNOWN_TIME
        if evidence:
            if not evidence.get("evidence_url"):
                raise ValueError("numeric ledger requires evidence_url")
            evidence_time = parse_time(evidence["available_at"])
            if evidence_time < observed:
                raise ValueError("numeric availability precedes observation")
            available = availability_seconds(evidence_time)
        elif simulated_delay is not None:
            available = availability_seconds(observed + timedelta(minutes=simulated_delay))
        values = []
        for field in ("tmpf", "sknt"):
            try:
                values.append(float(row[field]))
            except (ValueError, TypeError):
                values.append(np.nan)
        buckets.setdefault(end, []).append((values, available))
    if not buckets:
        raise ValueError("ASOS archive has no usable observations")
    times = np.arange(min(buckets), max(buckets) + 3600, 3600, dtype=np.int64)
    values = np.full((len(times), 2), np.nan)
    availability = np.full((len(times), 2), UNKNOWN_TIME, dtype=np.int64)
    for i, end in enumerate(times):
        rows = buckets.get(end, [])
        for c in range(2):
            finite = [(r[c], a) for r, a in rows if np.isfinite(r[c])]
            if finite:
                values[i, c] = np.mean([v for v, _ in finite])
                latest = max(a for _, a in finite)
                availability[i, c] = max(int(end), latest)
    return times, values, availability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--afd", type=Path, required=True)
    parser.add_argument("--asos", type=Path, required=True)
    parser.add_argument("--office", choices=["DMX", "LOT", "HGX"], required=True)
    parser.add_argument("--station", choices=["DSM", "ORD", "IAH"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--validation-end", required=True)
    parser.add_argument("--test-end", required=True)
    parser.add_argument("--availability-ledger", type=Path, help="JSONL: sha256,available_at,evidence_url")
    parser.add_argument("--numeric-ledger", type=Path)
    parser.add_argument("--availability-protocol", choices=[VERIFIED, ASSUMED])
    parser.add_argument("--text-delay-hours", type=float)
    parser.add_argument("--numeric-delay-hours", type=float)
    parser.add_argument("--simulate-text-delay-minutes", type=float)
    parser.add_argument("--simulate-numeric-delay-minutes", type=float)
    parser.add_argument("--embeddings", type=Path, help="frozen offline JSONL doc_id/version_id/embedding")
    parser.add_argument("--embedding-version")
    parser.add_argument("--hash-features", action="store_true", help="explicit lexical baseline, not pretrained semantic embeddings")
    parser.add_argument("--hash-dimensions", type=int, default=32)
    args = parser.parse_args()
    if args.availability_protocol == ASSUMED:
        if args.availability_ledger or args.numeric_ledger:
            parser.error("keep verified-ledger data separate from explicit assumption scenarios")
        args.simulate_text_delay_minutes = 60 * (args.text_delay_hours if args.text_delay_hours is not None else 0.)
        args.simulate_numeric_delay_minutes = 60 * (args.numeric_delay_hours if args.numeric_delay_hours is not None else 1.)
    elif args.availability_protocol == VERIFIED and any(x is not None for x in (
            args.text_delay_hours, args.numeric_delay_hours, args.simulate_text_delay_minutes, args.simulate_numeric_delay_minutes)):
        parser.error("verified_availability cannot create simulated release times")
    elif args.availability_protocol is None and (args.text_delay_hours is not None or args.numeric_delay_hours is not None):
        parser.error("hour-based sensitivity requires explicit issuance_asof_assumption")
    if args.output.exists():
        raise FileExistsError("preserving existing prepared data")
    if {"DSM": "DMX", "ORD": "LOT", "IAH": "HGX"}[args.station] != args.office:
        parser.error("station and forecast office pairing is inconsistent")
    for delay in (args.simulate_text_delay_minutes, args.simulate_numeric_delay_minutes):
        if delay is not None and (not np.isfinite(delay) or delay < 0):
            parser.error("simulated delays must be finite and nonnegative")
    if bool(args.embeddings) == bool(args.hash_features) or (args.embeddings and not args.embedding_version):
        parser.error("choose external embeddings with version, or explicitly choose --hash-features")
    ledger = {r["sha256"]: r for r in load_jsonl(args.availability_ledger)} if args.availability_ledger else {}
    numeric_ledger = {r["sha256"]: r for r in load_jsonl(args.numeric_ledger)} if args.numeric_ledger else {}
    records = afd_records(args.afd, args.office, args.station, ledger, args.simulate_text_delay_minutes)
    assumption_audit = None
    if args.availability_protocol == ASSUMED:
        records, assumption_audit = issuance_records(records, args.simulate_text_delay_minutes / 60, parse_time)
    external = {(r["doc_id"], r["version_id"]): r["embedding"] for r in load_jsonl(args.embeddings)} if args.embeddings else {}
    dimensions = set()
    for row in records:
        row["embedding"] = external[row["doc_id"], row["version_id"]] if external else hash_text(row["text"], args.hash_dimensions).tolist()
        dimensions.add(len(row["embedding"]))
    if len(dimensions) != 1:
        raise ValueError("empty or inconsistent text embeddings/archive")
    times, values, available = hourly_asos(args.asos, args.station, numeric_ledger, args.simulate_numeric_delay_minutes)
    boundaries = [parse_time(getattr(args, field)) for field in ("train_end", "validation_end", "test_end")]
    if not datetime_utc(times[0]) < boundaries[0] < boundaries[1] < boundaries[2]:
        raise ValueError("invalid chronological split boundaries")
    manifest = dict(schema_version=1, is_fixture=False, station=args.station, office=args.office,
                    availability_protocol=args.availability_protocol or ("legacy_simulated" if
                        args.simulate_text_delay_minutes is not None or args.simulate_numeric_delay_minutes is not None else VERIFIED),
                    text_delay_hours=args.simulate_text_delay_minutes/60 if args.simulate_text_delay_minutes is not None else None,
                    numeric_delay_hours=args.simulate_numeric_delay_minutes/60 if args.simulate_numeric_delay_minutes is not None else None,
                    numeric_availability_protocol="observation_asof_assumption" if args.availability_protocol == ASSUMED else
                        (VERIFIED if numeric_ledger else "unknown"), collection_assumption_audit=assumption_audit,
                    channels=["temperature_F", "wind_speed_knots"], base_dim=dimensions.pop(), metadata_dim=2,
                    base_encoder_version=args.embedding_version or "lexical-sha256-hash-v1",
                    numeric_availability_provenance="simulated" if args.simulate_numeric_delay_minutes is not None else
                        "verified" if numeric_ledger else "unknown",
                    train_end=boundaries[0].isoformat(), validation_end=boundaries[1].isoformat(), test_end=boundaries[2].isoformat(),
                    unknown_numeric_cells=int(np.sum(available == UNKNOWN_TIME)),
                    finite_numeric_cells=int(np.isfinite(values).sum()),
                    text_availability_counts={p: sum(r["availability_provenance"] == p for r in records)
                                              for p in ("verified", "simulated", "unknown")},
                    hourly_rule="mean of finite observations in [hour,hour+1), timestamp at hour end; no backward fill",
                    source_files={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.afd, args.asos)},
                    coverage=dict(first_hour=datetime_utc(times[0]).isoformat(), last_hour=datetime_utc(times[-1]).isoformat(),
                                  grid_hours=len(times), finite_fraction=float(np.isfinite(values).mean())),
                    mos_extension="asof_forecast_covariates join available; MOS download/model baseline not evaluated")
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "numeric.npz", times=times, values=values, available=available)
    (args.output / "documents.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in records))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
