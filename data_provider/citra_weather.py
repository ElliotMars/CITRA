"""UTC hourly, point-in-time weather windows and independently versioned text."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

from citra import Document, TemporalDocumentStore
from citra.kernel import immutable
from citra.temporal import utc
from citra.availability import VERIFIED, ASSUMED, issuance_records


UNKNOWN_TIME = np.iinfo(np.int64).max


def parse_time(value):
    return utc(datetime.fromisoformat(value.replace("Z", "+00:00")))


def epoch_seconds(value):
    delta = utc(value) - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return delta.days * 86400 + delta.seconds


def availability_seconds(value):
    """Ceil microsecond availability; never round a future arrival into cutoff."""
    value = utc(value)
    return epoch_seconds(value) + bool(value.microsecond)


def hourly_boundary(value):
    value = parse_time(value)
    if value.minute or value.second or value.microsecond:
        raise ValueError("hourly split boundaries must be whole UTC hours")
    return epoch_seconds(value)


def datetime_utc(seconds):
    return datetime.fromtimestamp(int(seconds), timezone.utc)


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def asof_forecast_covariates(records, cutoff, valid_times, fields, allow_simulated=False):
    """Extension join for external MOS forecasts, not future observations.

    Expected rows: issued_at, available_at, valid_time, availability_provenance,
    values={field:value}. Missing forecasts remain NaN, never future-filled.
    """
    cutoff = utc(cutoff)
    latest = {}
    for i, row in enumerate(records):
        if not row.get("available_at") or not row.get("issued_at"):
            continue
        if row.get("availability_provenance") != "verified" and not (
                allow_simulated and row.get("availability_provenance") == "simulated"):
            continue
        issued, available = parse_time(row["issued_at"]), parse_time(row["available_at"])
        if available < issued:
            raise ValueError("forecast availability precedes issuance")
        if max(issued, available) > cutoff:
            continue
        valid = parse_time(row["valid_time"])
        key = (available, issued, i)
        if valid not in latest or key > latest[valid][0]:
            latest[valid] = (key, row["values"])
    return np.array([[latest.get(utc(t), (None, {}))[1].get(field, np.nan) for field in fields]
                     for t in valid_times], dtype=np.float64)


class TextCorpus:
    def __init__(self, records, base_dim, metadata_dim=2, allow_simulated=False,
                 base_encoder_version="external-frozen-embedding"):
        self.records = tuple(records)
        self.base_dim, self.metadata_dim = base_dim, metadata_dim
        self.allow_simulated = allow_simulated
        self.base_encoder_version = base_encoder_version
        self.store = TemporalDocumentStore(base_dim, 1, indexed=False, allow_simulated=allow_simulated)
        documents, embeddings, metadata = [], [], []
        seen = set()
        for row in records:
            dates = {key: parse_time(row[key]) if row.get(key) else None for key in (
                "issued_at", "available_at", "valid_from", "valid_to", "event_start", "event_end", "ingested_at")}
            if dates["valid_from"] is None:
                raise ValueError("documents require explicit valid_from")
            d = Document(row["doc_id"], row["version_id"], row["source"], row["entity_id"],
                         **dates, embedding=row["embedding"], impact_vector=[0.],
                         availability_provenance=row["availability_provenance"])
            if len(d.embedding) != base_dim or len(row["metadata"]) != metadata_dim:
                raise ValueError("text feature dimensions differ from manifest")
            (self.store.append_revision if d.doc_id in seen else self.store.insert_document)(d)
            seen.add(d.doc_id)
            documents.append(d)
            embeddings.append(d.embedding)
            metadata.append(row["metadata"])
        self.documents = tuple(documents)
        self.embeddings = immutable(np.asarray(embeddings).reshape(len(records), base_dim))
        self.metadata = immutable(np.asarray(metadata).reshape(len(records), metadata_dim))
        self._position = {(d.doc_id, d.version_id): i for i, d in enumerate(documents)}
        self._compile_visibility()
        payload = json.dumps(dict(records=records, base_encoder_version=base_encoder_version,
                                  allow_simulated=allow_simulated), sort_keys=True, allow_nan=False)
        self.fingerprint = hashlib.sha256(payload.encode()).hexdigest()

    def _compile_visibility(self):
        self._starts = np.full(len(self.documents), np.iinfo(np.int64).max, dtype=np.int64)
        self._ends = np.full(len(self.documents), np.iinfo(np.int64).min, dtype=np.int64)
        for versions in self.store._versions.values():
            for entry in self.store._visibility_intervals(versions):
                j = self._position[entry.document.doc_id, entry.document.version_id]
                self._starts[j] = epoch_seconds(entry.start) * 1000000 + entry.start.microsecond
                self._ends[j] = (epoch_seconds(entry.end) * 1000000 + entry.end.microsecond
                                 if entry.end is not None else np.iinfo(np.int64).max)
        self._visibility_generation = self.store.generation

    def visibility(self, cutoffs):
        if self.store.generation != self._visibility_generation:
            raise RuntimeError("text metadata changed; explicitly reconstruct the corpus/feature table")
        micros = []
        for cutoff in cutoffs:
            if isinstance(cutoff, (int, np.integer)):
                micros.append(int(cutoff) * 1000000)
            else:
                cutoff = utc(cutoff)
                micros.append(epoch_seconds(cutoff) * 1000000 + cutoff.microsecond)
        t = np.asarray(micros, dtype=np.int64)[:, None]
        return (t >= self._starts[None]) & (t < self._ends[None])

    def causal_shuffle(self, seed):
        """Prefix resampling control; a global permutation could leak future text.

        Permuting paired (z,v) inside a full sum would leave the answer identical.
        Here content is reassigned to original evidence slots from strictly
        earlier legal donor slots, with replacement, without reading labels.
        """
        rng, rows = np.random.default_rng(seed), []
        donors_available = np.array([epoch_seconds(d.available_at)*1000000+d.available_at.microsecond
                                    if d.available_at else np.iinfo(np.int64).max for d in self.documents])
        for i, (d, original) in enumerate(zip(self.documents, self.records)):
            row = dict(original)
            activation = max([t for t in (d.available_at, d.valid_from, d.ingested_at) if t is not None])
            visible = self.visibility([activation])[0]
            visible &= donors_available < (epoch_seconds(activation)*1000000 + activation.microsecond)
            visible[i] = False
            candidates = np.flatnonzero(visible)
            donor = int(rng.choice(candidates)) if len(candidates) else i
            row.update(embedding=self.records[donor]["embedding"], text=self.records[donor].get("text", ""),
                       control_kind="causal_prefix_resampling_with_replacement",
                       control_donor_doc_id=self.documents[donor].doc_id,
                       control_donor_version_id=self.documents[donor].version_id,
                       control_donor_available_at=self.documents[donor].available_at.isoformat()
                       if self.documents[donor].available_at else None)
            rows.append(row)
        return TextCorpus(rows, self.base_dim, self.metadata_dim, self.allow_simulated,
                          self.base_encoder_version + f":causal-shuffle-seed-{seed}")

    def tensors(self, device, dtype):
        import torch
        return (torch.tensor(self.embeddings.copy(), dtype=dtype, device=device),
                torch.tensor(self.metadata.copy(), dtype=dtype, device=device))


class WeatherData:
    def __init__(self, directory, *, allow_simulated=False, availability_protocol=None,
                 text_delay_hours=None, load_text=True):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        requested = availability_protocol or self.manifest.get("availability_protocol")
        if requested not in (None, VERIFIED, ASSUMED, "legacy_simulated"):
            raise ValueError("unknown availability protocol")
        if requested == ASSUMED and availability_protocol is None and not allow_simulated:
            raise ValueError("explicitly opt into issuance_asof_assumption; it is not verified availability")
        if availability_protocol == VERIFIED and self.manifest["numeric_availability_provenance"] != "verified":
            raise ValueError("verified_availability requires verified numeric availability")
        if requested == VERIFIED:
            allow_simulated = False
        if requested == ASSUMED:
            allow_simulated = True
        self.availability_protocol = requested or ("legacy_simulated" if allow_simulated else VERIFIED)
        if text_delay_hours is not None and requested != ASSUMED:
            raise ValueError("text-delay sensitivity is only defined for issuance_asof_assumption")
        with np.load(self.directory / "numeric.npz", allow_pickle=False) as data:
            self.times = data["times"].astype(np.int64)
            self.values = data["values"].astype(np.float64)
            self.available = data["available"].astype(np.int64)
        if (self.values.ndim != 2 or self.available.shape != self.values.shape
                or self.times.shape != (len(self.values),) or len(self.times) < 2
                or not np.all(np.diff(self.times) == 3600) or np.any(self.times % 3600)):
            raise ValueError("expected strictly hourly UTC series, per-cell availability and [T,C] values")
        if not allow_simulated and self.manifest["numeric_availability_provenance"] != "verified":
            raise ValueError("numeric historical availability is not verified; use explicit simulated mode only for scenarios")
        known = self.available != UNKNOWN_TIME
        if np.any(known & (self.available < self.times[:, None])):
            raise ValueError("hour aggregates cannot be available before the hour ends")
        self.train_end = hourly_boundary(self.manifest["train_end"])
        self.validation_end = hourly_boundary(self.manifest["validation_end"])
        self.test_end = hourly_boundary(self.manifest["test_end"])
        if not self.times[0] < self.train_end < self.validation_end < self.test_end:
            raise ValueError("splits must be ordered, nonempty chronological intervals")
        self.fit_scaler()
        records = load_jsonl(self.directory / "documents.jsonl") if load_text else []
        if requested == ASSUMED:
            delay = float(self.manifest.get("text_delay_hours", 0.) if text_delay_hours is None else text_delay_hours)
            records, audit = issuance_records(records, delay, parse_time)
            self.manifest = dict(self.manifest, text_delay_hours=delay, text_assumption_audit=audit)
        self.corpus = TextCorpus(records,
                                 self.manifest["base_dim"], self.manifest["metadata_dim"],
                                 allow_simulated=allow_simulated,
                                 base_encoder_version=self.manifest["base_encoder_version"])
        digest = hashlib.sha256(json.dumps(self.manifest, sort_keys=True).encode())
        for array in (self.times, self.values, self.available):
            digest.update(array.tobytes())
        digest.update(self.corpus.fingerprint.encode())
        self.fingerprint = digest.hexdigest()
        numeric = hashlib.sha256(json.dumps(dict(train_end=self.manifest["train_end"],
            validation_end=self.manifest["validation_end"], test_end=self.manifest["test_end"],
            channels=self.manifest["channels"], numeric_availability_protocol=self.manifest.get("numeric_availability_protocol"),
            numeric_delay_hours=self.manifest.get("numeric_delay_hours")), sort_keys=True).encode())
        for array in (self.times, self.values, self.available):
            numeric.update(array.tobytes())
        self.numeric_fingerprint = numeric.hexdigest()

    def with_causal_shuffle(self, seed):
        import copy
        result = copy.copy(self)
        result.corpus = self.corpus.causal_shuffle(seed)
        result.control_kind = "causal_prefix_resampling_with_replacement"
        result.fingerprint = hashlib.sha256((self.fingerprint + result.corpus.fingerprint).encode()).hexdigest()
        return result

    def fit_scaler(self):
        # Future validation/test values and observations unavailable at train_end
        # cannot contribute to fitted preprocessing.
        legal = ((self.times[:, None] < self.train_end) & (self.available < self.train_end)
                 & np.isfinite(self.values))
        if np.any(legal.sum(axis=0) == 0):
            raise ValueError("no historically available training values for one or more channels")
        self.mean = np.array([self.values[legal[:, c], c].mean() for c in range(self.values.shape[1])])
        self.scale = np.array([self.values[legal[:, c], c].std() for c in range(self.values.shape[1])])
        self.scale[self.scale < 1e-8] = 1.

    def windows(self, split, seq_len, horizon, targets, stride=1):
        return WeatherWindows(self, split, seq_len, horizon, targets, stride)


class WeatherWindows:
    def __init__(self, data, split, seq_len, horizon, targets, stride=1):
        if min(seq_len, horizon, stride) <= 0 or not targets:
            raise ValueError("invalid window dimensions")
        intervals = {"train": (int(data.times[0]), data.train_end),
                     "validation": (data.train_end, data.validation_end),
                     "test": (data.validation_end, data.test_end)}
        if split not in intervals:
            raise ValueError("unknown split")
        self.data, self.seq_len, self.horizon, self.targets = data, seq_len, horizon, tuple(targets)
        lower, upper = intervals[split]
        self.starts, self.dropped_missing = [], 0
        for pos in range(seq_len, len(data.times) - horizon + 1, stride):
            if data.times[pos] < lower or data.times[pos + horizon - 1] >= upper:
                continue
            # Model fitting/selection has its own historical availability:
            # validation forecasts start after train_end; test forecasts start
            # after validation_end. Their preceding labels may arrive late.
            if split != "train" and data.times[pos - 1] < lower:
                continue
            if split in ("train", "validation") and np.any(data.available[pos:pos + horizon, self.targets] > upper):
                continue
            x = self.history(pos)
            y = data.values[pos:pos + horizon, self.targets]
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                self.dropped_missing += 1
                continue
            self.starts.append(pos)
        if not self.starts:
            raise ValueError(f"no legal {split} windows; inspect coverage, availability and split boundaries")

    def history(self, pos):
        data, begin = self.data, pos - self.seq_len
        cutoff = data.times[pos - 1]
        x = data.values[begin:pos].copy()
        x[data.available[begin:pos] > cutoff] = np.nan
        # Forward fill ONLY earlier rows already accessible at the forecast
        # cutoff. Leading unknowns stay unknown and cause this window to drop.
        source = np.where(np.isfinite(x), np.arange(len(x))[:, None], -1)
        source = np.maximum.accumulate(source, axis=0)
        x = x[np.maximum(source, 0), np.arange(x.shape[1])[None, :]]
        x[source < 0] = np.nan
        return (x - data.mean) / data.scale

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        pos, data = self.starts[index], self.data
        y = (data.values[pos:pos + self.horizon, self.targets] - data.mean[list(self.targets)]) / data.scale[list(self.targets)]
        return self.history(pos), y, int(data.times[pos - 1])


def make_fixture(directory, hours=300, base_dim=12, seed=0):
    """Explicit simulated integration fixture, never a real weather experiment."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    origin = datetime(2020, 1, 1, tzinfo=timezone.utc)
    times = np.array([epoch_seconds(origin + timedelta(hours=i + 1)) for i in range(hours)])
    external = rng.normal(size=(hours, 2))
    values = np.zeros_like(external)
    for i in range(1, hours):
        values[i] = 0.8 * values[i - 1] + 0.3 * external[i - 1] + rng.normal(scale=0.03, size=2)
    records = []
    for i in range(0, hours, 6):
        issued = origin + timedelta(hours=i)
        z = np.zeros(base_dim)
        z[:2] = external[i]
        z[2:] = rng.normal(scale=0.1, size=base_dim - 2)
        row = dict(doc_id=f"fixture-{i}", version_id="original", source="synthetic-weather-fixture",
                   source_url="fixture://generated", entity_id="fixture", office_id="FIX", issued_at=issued.isoformat(),
                   available_at=issued.isoformat(), ingested_at=None, valid_from=issued.isoformat(), valid_to=None,
                   event_start=(issued + timedelta(hours=1)).isoformat(), event_end=(issued + timedelta(hours=6)).isoformat(),
                   text="Synthetic exogenous fixture; not a historical weather product.",
                   availability_provenance="simulated", embedding=z.tolist(), metadata=[np.sin(i / 24), np.cos(i / 24)])
        records.append(row)
        if i % 24 == 0:
            revision = issued + timedelta(hours=9)
            records.append(dict(row, version_id="revision", issued_at=revision.isoformat(),
                                available_at=revision.isoformat(), embedding=(-z).tolist()))
    np.savez_compressed(directory / "numeric.npz", times=times, values=values,
                        available=np.repeat(times[:, None], 2, axis=1))
    (directory / "documents.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in records))
    manifest = dict(schema_version=1, is_fixture=True, numeric_availability_provenance="simulated",
                    base_dim=base_dim, metadata_dim=2, base_encoder_version="synthetic-fixed-features-v1",
                    channels=["synthetic_temperature", "synthetic_wind"],
                    train_end=datetime_utc(times[int(hours * .6)]).isoformat(),
                    validation_end=datetime_utc(times[int(hours * .8)]).isoformat(),
                    test_end=datetime_utc(times[-1] + 3600).isoformat(),
                    source="generated smoke fixture; no real forecast metrics")
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return directory
