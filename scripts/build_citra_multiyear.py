"""Collect yearly archives, audit coverage, then attach separately frozen features."""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile
import csv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_citra_weather import afd_records, hourly_asos, hash_text
from scripts.encode_citra_text import load_features
from data_provider.citra_weather import parse_time, epoch_seconds, datetime_utc, load_jsonl
from citra.availability import VERIFIED, ASSUMED, issuance_records


def observation_audit(path):
    lines=[line for line in Path(path).read_text().splitlines() if line and not line.startswith("#")]
    rows=list(csv.DictReader(lines))
    timestamps=Counter((row["station"],row["valid"]) for row in rows)
    exact=len(rows)-len({json.dumps(row,sort_keys=True) for row in rows})
    duplicates=sum(n>1 for n in timestamps.values())
    return dict(raw_observation_rows=len(rows),duplicate_observation_timestamp_groups=duplicates,
                exact_duplicate_observation_rows=exact)


def collect(raw_dir, output, years, protocol, numeric_delay_hours=None):
    if protocol not in (VERIFIED, ASSUMED):
        raise ValueError("choose an explicit availability protocol")
    if protocol == VERIFIED and numeric_delay_hours is not None:
        raise ValueError("verified_availability cannot invent numerical availability")
    if protocol == ASSUMED:
        numeric_delay_hours = 1. if numeric_delay_hours is None else numeric_delay_hours
        if not np.isfinite(numeric_delay_hours) or numeric_delay_hours < 0:
            raise ValueError("invalid numerical observation delay")
    if sorted(set(years)) != list(range(min(years), max(years) + 1)) or len(years) < 3:
        raise ValueError("at least three consecutive years are required")
    output, raw_dir = Path(output), Path(raw_dir)
    if output.exists():
        raise FileExistsError("preserving existing collection")
    rows, numeric, audits, source_files = [], [], [], {}
    for year in years:
        afd, asos = raw_dir / f"AFDDMX_{year}.zip", raw_dir / f"DSM_{year}.csv"
        start = datetime(year, 1, 1, tzinfo=timezone.utc)
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        quarantine = []
        products = afd_records(afd, "DMX", "DSM", quarantine=quarantine if protocol == ASSUMED else None)
        if any(not start <= parse_time(r["issued_at"]) < end for r in products):
            raise ValueError("archive returned products outside requested year")
        with zipfile.ZipFile(afd) as archive:
            members = sum(not m.is_dir() for m in archive.infolist())
        if members >= 9999:
            raise ValueError("potentially truncated yearly AFD archive")
        numeric_audit=observation_audit(asos)
        if numeric_audit["duplicate_observation_timestamp_groups"]:
            raise ValueError("ambiguous repeated numerical observations need explicit revision handling before replay")
        times, values, available = hourly_asos(asos, "DSM", simulated_delay=
            numeric_delay_hours*60 if protocol == ASSUMED else None)
        expected = np.arange(epoch_seconds(start)+3600, epoch_seconds(end)+3600, 3600, dtype=np.int64)
        if np.any(times < expected[0]) or np.any(times > expected[-1]):
            raise ValueError("numeric timestamps outside requested year")
        full_values = np.full((len(expected), 2), np.nan)
        full_available = np.full((len(expected), 2), np.iinfo(np.int64).max, dtype=np.int64)
        positions = np.searchsorted(expected, times)
        full_values[positions], full_available[positions] = values, available
        days = Counter(parse_time(r["issued_at"]).date().isoformat() for r in products)
        missing_days = [(start+timedelta(days=i)).date().isoformat() for i in range((end-start).days)
                        if (start+timedelta(days=i)).date().isoformat() not in days]
        audits.append(dict(year=year, raw_afd_members=members, unique_afd_versions=len(products),**numeric_audit,
            exact_duplicate_products=members-len(products)-len(quarantine), timestamp_quarantine=quarantine,
            days_with_afd=len(days), days_without_afd=missing_days,
            first_issued_at=products[0]["issued_at"] if products else None,
            last_issued_at=products[-1]["issued_at"] if products else None,
            expected_hours=len(expected), finite_hourly_cells=int(np.isfinite(full_values).sum()),
            missing_fraction_by_channel=np.mean(~np.isfinite(full_values), axis=0).tolist(),
            assumption_first_publication_verified=False))
        rows.extend(products)
        numeric.append((expected, full_values, full_available))
        for p in (afd, asos, Path(str(afd)+".metadata.json"), Path(str(asos)+".metadata.json")):
            source_files[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    original_count = len(rows)
    unique = {(r["doc_id"], r["version_id"]): r for r in rows}
    rows = sorted(unique.values(), key=lambda r: (r["issued_at"], r["raw_member"]))
    assumption_audit = None
    if protocol == ASSUMED:
        rows, assumption_audit = issuance_records(rows, 0., parse_time)
    output.mkdir(parents=True)
    np.savez_compressed(output / "numeric.npz", times=np.concatenate([a[0] for a in numeric]),
        values=np.concatenate([a[1] for a in numeric]), available=np.concatenate([a[2] for a in numeric]))
    (output / "documents.jsonl").write_text("".join(json.dumps(r, allow_nan=False)+"\n" for r in rows))
    manifest = dict(schema_version=2, is_fixture=False, station="DSM", office="DMX", years=years,
        availability_protocol=protocol, text_delay_hours=0., numeric_delay_hours=numeric_delay_hours,
        numeric_availability_protocol="observation_asof_assumption" if protocol == ASSUMED else VERIFIED,
        numeric_availability_provenance="simulated" if protocol == ASSUMED else "unknown",
        numeric_delay_origin="each observation's event valid time; hourly aggregate waits for all its contributors",
        text_delay_origin="WMO-checked product issuance, not archive retrieval time",
        historical_first_availability_verified=False, channels=["temperature_F", "wind_speed_knots"],
        train_end=f"{years[1]}-01-01T00:00:00+00:00", validation_end=f"{years[2]}-01-01T00:00:00+00:00",
        test_end=f"{years[-1]+1}-01-01T00:00:00+00:00", metadata_dim=2,
        collection_assumption_audit=assumption_audit, source_files=source_files, yearly_audits=audits,
        total_products_before_deduplication=original_count, products_after_protocol_filter=len(rows),
        raw_product_files=sum(y["raw_afd_members"] for y in audits),
        timestamp_quarantined_products=sum(len(y["timestamp_quarantine"]) for y in audits))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def prepare(collection, features, output):
    collection, output = Path(collection), Path(output)
    if output.exists():
        raise FileExistsError("preserving existing prepared data")
    manifest = json.loads((collection / "manifest.json").read_text())
    records, encoder = load_features(features, load_jsonl(collection / "documents.jsonl"))
    output.mkdir(parents=True)
    shutil.copy2(collection / "numeric.npz", output / "numeric.npz")
    (output / "documents.jsonl").write_text("".join(json.dumps(r, allow_nan=False)+"\n" for r in records))
    manifest.update(base_dim=encoder["dimensions"], base_encoder_version=encoder["encoder_name"]+"@"+encoder["revision"],
                    frozen_text_encoder=encoder)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def prepare_baseline(collection, output, *, numeric_only=False, dimensions=384):
    """Build a reproducible numeric-only bundle or the retained hash ablation."""
    collection, output = Path(collection), Path(output)
    if output.exists():
        raise FileExistsError("preserving existing baseline data")
    manifest = json.loads((collection / "manifest.json").read_text())
    records = [] if numeric_only else load_jsonl(collection / "documents.jsonl")
    for row in records:
        row["embedding"] = hash_text(row["text"], dimensions).tolist()
    manifest.update(base_dim=1 if numeric_only else dimensions,
                    base_encoder_version="unused-in-ts-only" if numeric_only else f"signed-sha256-bow-{dimensions}-v1",
                    input_scope="numeric_only_no_text_features" if numeric_only else "hash_text_ablation")
    output.mkdir(parents=True)
    shutil.copy2(collection / "numeric.npz", output / "numeric.npz")
    (output / "documents.jsonl").write_text("".join(json.dumps(r, allow_nan=False)+"\n" for r in records))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["collect", "prepare", "numeric", "hash"])
    p.add_argument("--raw", type=Path)
    p.add_argument("--years", nargs="+", type=int, default=[2020,2021,2022])
    p.add_argument("--protocol", choices=[VERIFIED,ASSUMED], default=VERIFIED)
    p.add_argument("--numeric-delay-hours", type=float)
    p.add_argument("--collection", type=Path)
    p.add_argument("--features", type=Path)
    p.add_argument("--hash-dimensions", type=int, default=384)
    p.add_argument("--output", type=Path, required=True)
    a=p.parse_args()
    if a.stage=="collect":
        if a.raw is None:
            p.error("collect requires --raw")
        result=collect(a.raw,a.output,a.years,a.protocol,a.numeric_delay_hours)
    elif a.stage=="prepare":
        if a.collection is None or a.features is None:
            p.error("prepare requires --collection and --features")
        result=prepare(a.collection,a.features,a.output)
    else:
        if a.collection is None:
            p.error("numeric/hash requires --collection")
        result=prepare_baseline(a.collection,a.output,numeric_only=a.stage=="numeric",dimensions=a.hash_dimensions)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
