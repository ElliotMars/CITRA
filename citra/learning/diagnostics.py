"""Untimed, label-free diagnostics on the same frozen real-document geometry."""
from dataclasses import replace
from time import perf_counter
import numpy as np
import torch

from citra import TemporalDocumentStore
from citra.kernel import exact_aggregate


def local_bounds(session, history, cutoff, samples=16):
    """Use the independent compensated sum only in offline bound diagnostics."""
    with torch.no_grad():
        q = session.model.encode_query(torch.as_tensor(history, dtype=torch.float64,
            device=next(session.model.parameters()).device)).cpu().numpy()[0]
    index = session.store.index
    legal = [n for n in index.nodes if cutoff >= n.max_start
             and (n.min_end is None or cutoff < n.min_end)]
    legal.sort(key=lambda n: (n.count, n.node_id))
    selected = np.unique(np.linspace(0, len(legal)-1, min(samples, len(legal)), dtype=int))

    def documents(node):
        if node.children:
            return tuple(d for child in node.children for d in documents(child))
        return tuple(e.document for e in node.document_refs)

    rows = []
    for i in selected:
        node = legal[int(i)]
        approximation, bound = index.approximate_node(node, q, "local")
        _, global_bound = index.approximate_node(node, q, "global")
        exact = exact_aggregate(documents(node), q, index.kernel, index.impact_dim)
        error = float(np.linalg.norm(approximation-exact))
        if error > bound + 1e-10 + 1e-10*float(np.linalg.norm(exact)):
            raise AssertionError("sampled local node certificate violation")
        rows.append(dict(node_id=node.node_id, documents=node.count, radius=node.radius,
            query_center_distance=float(np.linalg.norm(q-node.center)),
            local_hessian_bound=index.kernel.local_hessian_bound(q,node.center,node.radius),
            global_hessian_bound=index.kernel.hessian_bound,
            local_aggregate_bound=bound, global_aggregate_bound=global_bound,
            actual_aggregate_deviation=error, bound_over_error=bound/error if error else None))
    return dict(fully_visible_nodes=len(legal), sampled_nodes=rows,
        sampling="uniform ranks by (document count,node id), at most 16 fully visible nodes; no labels",
        scope="offline diagnostics, excluded from online latency; float64 atol/rtol 1e-10")


def maintenance(session, updates=100):
    """Replay actual late-arriving documents into a separate store, no model edits."""
    documents = [replace(d, embedding=z, impact_vector=v) for d,z,v in
                 zip(session.corpus.documents,session.keys_np,session.impacts_np)]
    documents.sort(key=lambda d:(d.available_at,d.doc_id,d.version_id))
    count = min(updates,len(documents))
    c = session.model.config
    store = TemporalDocumentStore(c["embedding_dim"],c["impact_dim"],sigma=c["sigma"],
                                  allow_simulated=session.corpus.allow_simulated,indexed=False)
    seen = set()
    def insert(d):
        (store.append_revision if d.doc_id in seen else store.insert_document)(d)
        seen.add(d.doc_id)
    for d in documents[:-count]:
        insert(d)
    construction = store.build_index()
    rows = []
    for d in documents[-count:]:
        start = perf_counter()
        kind = "revision" if d.doc_id in seen else "new_document"
        insert(d)
        rows.append(dict(kind=kind,seconds=perf_counter()-start,**store.last_update_stats))
    return dict(updates=count,prefix_documents=len(documents)-count,
        prefix_index_construction_seconds=construction,
        total_seconds=sum(r["seconds"] for r in rows),
        index_only_seconds=sum(r["index_seconds"] for r in rows),
        p50_seconds=float(np.percentile([r["seconds"] for r in rows],50)),
        p95_seconds=float(np.percentile([r["seconds"] for r in rows],95)),rows=rows,
        scope="separate store, last 100 actual arrivals; frozen key/impact vectors; no re-encoding; not in online latency")
