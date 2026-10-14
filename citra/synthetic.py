"""Reproducible, explicitly simulated timestamps; no future labels as impacts."""

from datetime import datetime, timedelta, timezone
import numpy as np

from .temporal import Document, TemporalDocumentStore


EPOCH = datetime(2020, 1, 1, tzinfo=timezone.utc)


def synthetic_store(n=128, embedding_dim=4, impact_dim=3, seed=0,
                    radius=0.05, leaf_size=8, indexed=True):
    if n < 0 or radius < 0 or not np.isfinite(radius):
        raise ValueError("invalid synthetic size or radius")
    rng = np.random.default_rng(seed)
    store = TemporalDocumentStore(embedding_dim, impact_dim, leaf_size=leaf_size,
                                  allow_simulated=True, indexed=indexed)
    for i in range(n):
        event = EPOCH + timedelta(days=i % 20)
        issued = event - timedelta(days=2)
        available = issued + timedelta(days=4 if i % 7 == 0 else 0)
        z = rng.normal(size=embedding_dim) * radius
        v = rng.normal(size=impact_dim)  # signed, independent of future labels
        if i % 11 == 1:
            previous = store.documents[-1]
            z, v = previous.embedding, -previous.impact_vector
        store.insert_document(Document(
            str(i), "v1", "synthetic", "series", issued, available, issued,
            None, event, event + timedelta(days=1), z, v,
            availability_provenance="simulated"))
        if i % 13 == 0:
            revision_time = available + timedelta(days=8)
            store.append_revision(Document(
                str(i), "v2", "synthetic", "series", revision_time,
                revision_time, issued, None, event, event + timedelta(days=1),
                z + rng.normal(size=embedding_dim) * radius, -v,
                availability_provenance="simulated"))
    return store
