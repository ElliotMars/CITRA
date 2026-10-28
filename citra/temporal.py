"""Append-only versions with explicit point-in-time accessibility semantics."""

from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Optional
from itertools import groupby

from .kernel import RBFKernel, exact_aggregate, immutable, vector
from .index import TemporalCITRAIndex, TemporalEntry


def utc(value):
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware datetime objects")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True, eq=False)
class Document:
    doc_id: str
    version_id: str
    source: str
    entity_id: str
    issued_at: Optional[datetime]
    available_at: Optional[datetime]
    valid_from: datetime
    valid_to: Optional[datetime]
    event_start: Optional[datetime]
    event_end: Optional[datetime]
    embedding: object
    impact_vector: object
    ingested_at: Optional[datetime] = None
    availability_provenance: str = "unknown"

    def __post_init__(self):
        if not all(isinstance(v, str) and v for v in
                   (self.doc_id, self.version_id, self.source, self.entity_id)):
            raise ValueError("document identity fields must be nonempty strings")
        for name in ("issued_at", "available_at", "valid_from", "valid_to",
                     "event_start", "event_end", "ingested_at"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, utc(value))
        if self.valid_from is None:
            raise ValueError("valid_from is required")
        if self.valid_to is not None and self.valid_to <= self.valid_from:
            raise ValueError("validity interval must be nonempty")
        if (self.event_start is not None and self.event_end is not None
                and self.event_end < self.event_start):
            raise ValueError("reversed event interval")
        if (self.issued_at is not None and self.available_at is not None
                and self.available_at < self.issued_at):
            raise ValueError("availability cannot precede issuance")
        if self.availability_provenance not in ("unknown", "verified", "simulated"):
            raise ValueError("invalid availability provenance")
        object.__setattr__(self, "embedding", immutable(vector(self.embedding)))
        object.__setattr__(self, "impact_vector", immutable(vector(self.impact_vector)))

    def accessible(self, cutoff, allow_simulated=False):
        provenance_ok = (self.availability_provenance == "verified"
                         or (allow_simulated and self.availability_provenance == "simulated"))
        return (provenance_ok and self.available_at is not None
                and self.available_at <= cutoff
                and (self.issued_at is None or self.issued_at <= cutoff)
                and (self.ingested_at is None or self.ingested_at <= cutoff))


class TemporalDocumentStore:
    """One authoritative version per doc_id; revisions never mutate old records.

    Unknown availability is excluded. Synthetic availability needs opt-in.
    Ingestion, when known, additionally limits system accessibility.
    The newest accessible version whose valid_from has begun supersedes older
    versions. An expired newest version does not resurrect an older version.
    """

    def __init__(self, embedding_dim, impact_dim, sigma=1.0, leaf_size=8,
                 allow_simulated=False, indexed=True):
        if (not isinstance(embedding_dim, int) or embedding_dim <= 0
                or not isinstance(impact_dim, int) or impact_dim <= 0):
            raise ValueError("dimensions must be positive integers")
        if not isinstance(leaf_size, int) or leaf_size <= 0:
            raise ValueError("leaf_size must be a positive integer")
        self.embedding_dim = embedding_dim
        self.impact_dim = impact_dim
        self._kernel = RBFKernel(sigma)
        self.leaf_size = leaf_size
        self._allow_simulated = bool(allow_simulated)
        self._versions = {}
        self._generation = 0
        self._intervals = {}
        self.index = TemporalCITRAIndex(embedding_dim, impact_dim, self.kernel, leaf_size) if indexed else None
        self.last_update_seconds = 0.0
        self.last_update_stats = {}
        self.total_update_seconds = 0.0

    @property
    def allow_simulated(self):
        return self._allow_simulated

    @property
    def kernel(self):
        return self._kernel

    @property
    def generation(self):
        return self._generation

    @property
    def documents(self):
        return tuple(d for versions in self._versions.values() for d in versions)

    def _validate(self, document):
        if not isinstance(document, Document):
            raise TypeError("expected Document")
        vector(document.embedding, self.embedding_dim)
        vector(document.impact_vector, self.impact_dim)

    def _visibility_intervals(self, versions):
        """Sweep activations; expired winners suppress old versions permanently.

        Versions only enter the eligible set, never leave it. Therefore the
        highest availability/append rank can win at most one interval.
        """
        activations = []
        for i, d in enumerate(versions):
            provenance_ok = d.availability_provenance == "verified" or (
                self.allow_simulated and d.availability_provenance == "simulated")
            if not provenance_ok or d.available_at is None:
                continue
            activation = max(d.available_at, d.valid_from, d.ingested_at or d.available_at)
            activations.append((activation, (d.available_at, i), d))
        activations.sort(key=lambda row: (row[0], row[1]))
        entries, winner = [], None

        def finish(previous, end):
            if previous is None:
                return
            _, start, d = previous
            if d.valid_to is not None:
                end = d.valid_to if end is None else min(end, d.valid_to)
            if end is None or start < end:
                entries.append(TemporalEntry(d, start, end))

        for activation, rows in groupby(activations, key=lambda row: row[0]):
            _, rank, d = max(rows, key=lambda row: row[1])
            if winner is None or rank > winner[0]:
                finish(winner, activation)
                winner = (rank, activation, d)
        finish(winner, None)
        return tuple(entries)

    def _publish(self, doc_id, versions, start):
        entries = self._visibility_intervals(versions) if self.index is not None else ()
        stats = (self.index.replace_intervals(self._intervals.get(doc_id, ()), entries)
                 if self.index is not None else {"nodes_created": 0, "documents_read": 0,
                                                "intervals_removed": 0, "intervals_inserted": 0, "index_seconds": 0.})
        self._versions[doc_id] = versions
        self._intervals[doc_id] = entries
        self._generation += 1
        self.last_update_seconds = perf_counter() - start
        self.total_update_seconds += self.last_update_seconds
        self.last_update_stats = dict(stats, metadata_records_checked=len(versions),
                                      validation_document_reads=1, elapsed_seconds=self.last_update_seconds)

    def insert_document(self, document):
        start = perf_counter()
        self._validate(document)
        if document.doc_id in self._versions:
            raise ValueError("doc_id already exists; use append_revision")
        self._publish(document.doc_id, (document,), start)

    def append_revision(self, document):
        start = perf_counter()
        self._validate(document)
        versions = self._versions.get(document.doc_id)
        if versions is None:
            raise ValueError("unknown doc_id")
        if any(d.version_id == document.version_id for d in versions):
            raise ValueError("duplicate version_id")
        if any((d.source, d.entity_id) != (document.source, document.entity_id)
               for d in versions):
            raise ValueError("revision must retain source and entity")
        # Accept out-of-order ingestion; snapshot orders by known availability,
        # with append order breaking ties. Historical intervals stay immutable.
        self._publish(document.doc_id, versions + (document,), start)

    def snapshot(self, cutoff):
        cutoff = utc(cutoff)
        visible = []
        for versions in self._versions.values():
            eligible = [(d.available_at, i, d) for i, d in enumerate(versions)
                        if d.accessible(cutoff, self.allow_simulated)
                        and d.valid_from <= cutoff]
            if eligible:
                newest = max(eligible, key=lambda row: (row[0], row[1]))[2]
                if newest.valid_to is None or cutoff < newest.valid_to:
                    visible.append(newest)
        return tuple(visible)

    def visible_documents(self, cutoff):
        return self.snapshot(cutoff)

    def aggregate_exact(self, query, cutoff):
        vector(query, self.embedding_dim)
        return exact_aggregate(self.snapshot(cutoff), query, self.kernel, self.impact_dim)

    def build_index(self):
        """Explicit offline construction; exact-only consumers need no AVL."""
        start = perf_counter()
        if self.index is not None:
            return 0.0
        index = TemporalCITRAIndex(self.embedding_dim, self.impact_dim, self.kernel, self.leaf_size)
        intervals = {}
        for doc_id, versions in self._versions.items():
            entries = self._visibility_intervals(versions)
            index.replace_intervals((), entries)
            intervals[doc_id] = entries
        self.index, self._intervals = index, intervals
        return perf_counter() - start

    def query_certified(self, query, cutoff, epsilon, *, bound_mode="local"):
        """Aggregation-only certificate; all cutoffs reuse the same temporal tree.

        Index construction/update is performed and measured on insert/revision,
        not hidden inside an uncounted query-time snapshot build.
        """
        start = perf_counter()
        if self.index is None:
            raise RuntimeError("scan-only store has no certified index")
        cutoff = utc(cutoff)
        result = self.index.query(query, epsilon, cutoff=cutoff, bound_mode=bound_mode)
        result.stats.cache_hit = True  # maintained on every write, reused across cutoffs
        result.stats.elapsed_seconds = perf_counter() - start
        return result
