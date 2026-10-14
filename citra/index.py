"""Leaf-only summaries and persistent AVL updates; certificates cover aggregation."""
from dataclasses import dataclass, fields, is_dataclass
import heapq
import math
import sys
from time import perf_counter
import numpy as np
from .kernel import exact_aggregate, immutable, vector

@dataclass
class QueryStats:
    nodes_visited: int = 0
    documents_scanned: int = 0
    exact_kernel_evaluations: int = 0
    summary_kernel_evaluations: int = 0
    metadata_records_checked: int = 0
    temporal_nodes_pruned: int = 0
    temporal_partial_nodes: int = 0
    index_document_reads: int = 0
    build_seconds: float = 0.
    elapsed_seconds: float = 0.
    fallback_exact: bool = False
    cache_hit: bool = False

@dataclass(frozen=True)
class QueryResult:
    aggregate: np.ndarray
    aggregate_bound: float
    stats: QueryStats
    certificate_scope: str = "analytic_aggregation_remainder_float64_tolerance"

@dataclass(frozen=True, slots=True, eq=False)
class TemporalEntry:
    document: object
    start: object = None
    end: object = None

    @property
    def key(self):
        return tuple(self.document.embedding) + (self.document.doc_id, self.document.version_id)

    def visible(self, cutoff):
        return self.start is None or (self.start <= cutoff and (self.end is None or cutoff < self.end))

@dataclass(frozen=True, slots=True, eq=False)
class Node:
    node_id: int
    count: int
    center: np.ndarray
    radius: float
    influence_sum: np.ndarray
    first_moment: np.ndarray
    second_residual: float
    norm_sum: float
    weighted_center: np.ndarray
    weighted_scatter: float
    document_refs: tuple
    children: tuple
    height: int
    max_key: tuple
    min_start: object
    max_start: object
    min_end: object
    max_end: object

def end_min(a, b):
    return b if a is None else a if b is None else min(a, b)

def end_max(a, b):
    return None if a is None or b is None else max(a, b)

def deep_size(*objects):
    """Deduplicated Python/NumPy resident estimate including referenced data."""
    seen = set()
    def visit(obj):
        if id(obj) in seen:
            return 0
        seen.add(id(obj))
        size = sys.getsizeof(obj)
        if isinstance(obj, np.ndarray):
            return size + (visit(obj.base) if obj.base is not None else 0)
        if isinstance(obj, dict):
            return size + sum(visit(k) + visit(v) for k, v in obj.items())
        if isinstance(obj, (tuple, list, set, frozenset)):
            return size + sum(visit(v) for v in obj)
        if is_dataclass(obj) and not isinstance(obj, type):
            size += sum(visit(getattr(obj, f.name)) for f in fields(obj))
            if hasattr(obj, "__dict__"):
                size += sys.getsizeof(obj.__dict__)
        return size
    return sum(visit(obj) for obj in objects)

def positive_product(*values):
    result = math.prod(values)
    if not math.isfinite(result):
        raise ValueError("nonfinite error bound; rescale inputs")
    if all(v > 0 for v in values):
        result = max(float(np.nextafter(0., 1.)), float(np.nextafter(result, np.inf)))
    if not math.isfinite(result):
        raise ValueError("outward rounding overflow; rescale inputs")
    return result

class CITRAIndex:
    def __init__(self, documents, embedding_dim, impact_dim, kernel, leaf_size=8):
        start = perf_counter()
        if not isinstance(leaf_size, int) or leaf_size <= 0:
            raise ValueError("leaf_size must be positive integer")
        if any(not isinstance(d, int) or d <= 0 for d in (embedding_dim, impact_dim)):
            raise ValueError("dimensions must be positive integers")
        self.embedding_dim, self.impact_dim = embedding_dim, impact_dim
        self._kernel, self.leaf_size = kernel, leaf_size
        self._next_node_id = self.build_document_reads = 0
        self.total_update_nodes = self.total_update_document_reads = 0
        entries = []
        for d in documents:
            vector(d.embedding, embedding_dim)
            vector(d.impact_vector, impact_dim)
            self.build_document_reads += 1
            entries.append(TemporalEntry(d))
        entries.sort(key=lambda e: e.key)
        self.root = self._build(tuple(entries))
        self.build_seconds = perf_counter() - start

    @property
    def kernel(self):
        return self._kernel

    def _id(self):
        value = self._next_node_id
        self._next_node_id += 1
        return value

    def _leaf(self, entries):
        self.total_update_document_reads += len(entries)
        self.total_update_nodes += 1
        z = np.stack([e.document.embedding for e in entries])
        v = np.stack([e.document.impact_vector for e in entries])
        with np.errstate(over="raise", invalid="raise"):
            c = z.mean(axis=0)
            delta = z - c
            norms = np.linalg.norm(v, axis=1)
            mass = math.fsum(norms)
            wc = (norms[:, None] * z).sum(axis=0) / mass if mass else c
            scatter = float(np.sum(norms * np.sum((z - wc)**2, axis=1)))
            residual = float(np.sum(norms * np.sum(delta**2, axis=1)))
            radius = float(np.linalg.norm(delta, axis=1).max())
            influence, moment = v.sum(axis=0), delta.T @ v
        if not all(math.isfinite(x) for x in (mass, scatter, residual, radius)):
            raise ValueError("nonfinite summaries; rescale inputs")
        starts = [e.start for e in entries]
        if starts[0] is None:
            min_start = max_start = min_end = max_end = None
        else:
            min_start, max_start = min(starts), max(starts)
            min_end = max_end = entries[0].end
            for entry in entries[1:]:
                min_end, max_end = end_min(min_end, entry.end), end_max(max_end, entry.end)
        return Node(self._id(), len(entries), immutable(c), radius, immutable(influence),
                    immutable(moment), residual, mass, immutable(wc), scatter, entries,
                    (), 1, entries[-1].key, min_start, max_start, min_end, max_end)

    def _branch(self, left, right):
        self.total_update_nodes += 1
        count = left.count + right.count
        with np.errstate(over="raise", invalid="raise"):
            c = left.center + (right.center - left.center) * (right.count / count)
            dl, dr = left.center - c, right.center - c
            influence = left.influence_sum + right.influence_sum
            moment = (left.first_moment + right.first_moment
                      + np.outer(dl, left.influence_sum) + np.outer(dr, right.influence_sum))
            mass = left.norm_sum + right.norm_sum
            if mass:
                wc = left.weighted_center + (right.weighted_center - left.weighted_center) * (right.norm_sum / mass)
                separation = float(np.sum((right.weighted_center - left.weighted_center)**2))
                scatter = (left.weighted_scatter + right.weighted_scatter
                           + (left.norm_sum / mass) * right.norm_sum * separation)
            else:
                wc, scatter = c, 0.
            residual = scatter + mass * float(np.sum((wc - c)**2))
            radius = max(left.radius + float(np.linalg.norm(dl)), right.radius + float(np.linalg.norm(dr)))
        if not all(math.isfinite(x) for x in (mass, scatter, residual, radius)):
            raise ValueError("nonfinite summaries; rescale inputs")
        if left.min_start is None:
            min_start = max_start = min_end = max_end = None
        else:
            min_start, max_start = min(left.min_start, right.min_start), max(left.max_start, right.max_start)
            min_end, max_end = end_min(left.min_end, right.min_end), end_max(left.max_end, right.max_end)
        return Node(self._id(), count, immutable(c), radius, immutable(influence), immutable(moment),
                    residual, mass, immutable(wc), scatter, (), (left, right),
                    1 + max(left.height, right.height), right.max_key,
                    min_start, max_start, min_end, max_end)

    def _build(self, entries):
        if not entries:
            return None
        if len(entries) <= self.leaf_size:
            self.build_document_reads += len(entries)
            return self._leaf(entries)
        mid = len(entries) // 2
        return self._branch(self._build(entries[:mid]), self._build(entries[mid:]))

    @property
    def nodes(self):
        def walk(node):
            if node:
                yield node
                for child in node.children:
                    yield from walk(child)
        return tuple(walk(self.root))

    @property
    def documents(self):
        return tuple(e.document for n in self.nodes for e in n.document_refs)

    @property
    def summary_bytes(self):
        return sum(n.center.nbytes + n.influence_sum.nbytes + n.first_moment.nbytes
                   + n.weighted_center.nbytes + 40 for n in self.nodes)

    @property
    def resident_bytes(self):
        return deep_size(self.root)

    def approximate_node(self, node, query, bound_mode="local"):
        q = vector(query, self.embedding_dim)
        if bound_mode not in ("local", "global"):
            raise ValueError("bound_mode must be local or global")
        value = self.kernel.value(q, node.center)
        gradient = value * (q - node.center) / self.kernel.sigma**2
        aggregate = immutable(value * node.influence_sum + node.first_moment.T @ gradient)
        hessian = (self.kernel.local_hessian_bound(q, node.center, node.radius)
                   if bound_mode == "local" else self.kernel.hessian_bound)
        bound = positive_product(0.5, hessian, node.second_residual)
        # A proved alternative: omit the node with ||sum k*v|| <= k_max*sum||v||.
        if bound_mode == "local":
            zero_bound = positive_product(self.kernel.maximum_on_ball(q, node.center, node.radius), node.norm_sum)
            if zero_bound < bound:
                return immutable(np.zeros(self.impact_dim)), zero_bound
        return aggregate, bound

    def aggregate_exact(self, query):
        vector(query, self.embedding_dim)
        return exact_aggregate(self.documents, query, self.kernel, self.impact_dim)

    def query(self, query, epsilon, *, cutoff=None, bound_mode="local", uncertified=False):
        """epsilon is an aggregation budget; arbitrary model L is not accepted."""
        start = perf_counter()
        q = vector(query, self.embedding_dim)
        if not math.isfinite(epsilon) or epsilon < 0:
            raise ValueError("epsilon must be finite and nonnegative")
        if bound_mode not in ("local", "global"):
            raise ValueError("bound_mode must be local or global")
        stats = QueryStats()
        frontier, queue, visible_docs = {}, [], []

        def visit(node):
            if node is None:
                return
            stats.nodes_visited += 1
            if node.min_start is None:
                legal = True
            else:
                if cutoff is None:
                    raise ValueError("temporal index requires cutoff")
                if cutoff < node.min_start or (node.max_end is not None and cutoff >= node.max_end):
                    stats.temporal_nodes_pruned += 1
                    return
                legal = cutoff >= node.max_start and (node.min_end is None or cutoff < node.min_end)
            if not legal and node.children:
                stats.temporal_partial_nodes += 1
                for child in node.children:
                    visit(child)
                return
            if not node.children:
                if not legal:
                    stats.metadata_records_checked += len(node.document_refs)
                docs = tuple(e.document for e in node.document_refs if legal or e.visible(cutoff))
                if epsilon == 0:
                    visible_docs.extend(docs)
                    return
                aggregate = exact_aggregate(docs, q, self.kernel, self.impact_dim)
                stats.documents_scanned += len(docs)
                stats.exact_kernel_evaluations += len(docs)
                frontier[node.node_id] = (aggregate, 0.)
            elif epsilon == 0:
                for child in node.children:
                    visit(child)
            else:
                aggregate, bound = self.approximate_node(node, q, bound_mode)
                stats.summary_kernel_evaluations += 1
                frontier[node.node_id] = (aggregate, bound)
                heapq.heappush(queue, (-bound, node.node_id, node))

        visit(self.root)
        if epsilon == 0:
            aggregate = exact_aggregate(visible_docs, q, self.kernel, self.impact_dim)
            stats.documents_scanned = stats.exact_kernel_evaluations = len(visible_docs)
            stats.fallback_exact = bool(visible_docs)
            bound = 0.
        else:
            while not uncertified and math.fsum(b for _, b in frontier.values()) > epsilon:
                if not queue:
                    raise ArithmeticError("exact frontier must have zero analytical remainder")
                _, _, node = heapq.heappop(queue)
                del frontier[node.node_id]
                for child in node.children:
                    visit(child)
            bound = math.fsum(b for _, b in frontier.values())
            aggregate = immutable([math.fsum(float(a[j]) for a, _ in frontier.values())
                                   for j in range(self.impact_dim)])
            stats.fallback_exact = bool(frontier) and all(b == 0 for _, b in frontier.values()) and stats.documents_scanned > 0
        stats.elapsed_seconds = perf_counter() - start
        scope = "uncertified_coarse" if uncertified else "analytic_aggregation_remainder_float64_tolerance"
        return QueryResult(aggregate, bound, stats, scope)

class TemporalCITRAIndex(CITRAIndex):
    """Persistent leaf-oriented AVL; interval extrema enforce legal summaries."""
    def __init__(self, embedding_dim, impact_dim, kernel, leaf_size=8):
        super().__init__((), embedding_dim, impact_dim, kernel, leaf_size)

    def aggregate_exact(self, query, *, cutoff):
        """A temporal oracle requires cutoff; never sum all physical versions."""
        from .temporal import utc
        cutoff = utc(cutoff)
        vector(query, self.embedding_dim)
        docs = tuple(e.document for n in self.nodes for e in n.document_refs if e.visible(cutoff))
        return exact_aggregate(docs, query, self.kernel, self.impact_dim)

    def _balance(self, left, right):
        if left.height > right.height + 1:
            a, b = left.children
            if a.height >= b.height:
                return self._branch(a, self._branch(b, right))
            c, d = b.children
            return self._branch(self._branch(a, c), self._branch(d, right))
        if right.height > left.height + 1:
            a, b = right.children
            if b.height >= a.height:
                return self._branch(self._branch(left, a), b)
            c, d = a.children
            return self._branch(self._branch(left, c), self._branch(d, b))
        return self._branch(left, right)

    def _insert(self, node, entry):
        if node is None:
            return self._leaf((entry,))
        if not node.children:
            if any(e.key == entry.key for e in node.document_refs):
                raise ValueError("duplicate interval key")
            entries = tuple(sorted(node.document_refs + (entry,), key=lambda e: e.key))
            if len(entries) <= self.leaf_size:
                return self._leaf(entries)
            mid = len(entries) // 2
            return self._branch(self._leaf(entries[:mid]), self._leaf(entries[mid:]))
        left, right = node.children
        if entry.key <= left.max_key:
            left = self._insert(left, entry)
        else:
            right = self._insert(right, entry)
        return self._balance(left, right)

    def _delete(self, node, key):
        if node is None:
            raise KeyError(key)
        if not node.children:
            entries = tuple(e for e in node.document_refs if e.key != key)
            if len(entries) == len(node.document_refs):
                raise KeyError(key)
            return self._leaf(entries) if entries else None
        left, right = node.children
        if key <= left.max_key:
            left = self._delete(left, key)
        else:
            right = self._delete(right, key)
        if left is None:
            return right
        if right is None:
            return left
        return self._balance(left, right)

    def replace_intervals(self, old_entries, new_entries):
        start = perf_counter()
        nodes, reads = self.total_update_nodes, self.total_update_document_reads
        old = {e.document.version_id: e for e in old_entries}
        new = {e.document.version_id: e for e in new_entries}
        unchanged = {v for v in old.keys() & new.keys()
                     if old[v].start == new[v].start and old[v].end == new[v].end}
        root = self.root
        for version in sorted(old.keys() - unchanged):
            root = self._delete(root, old[version].key)
        for version in sorted(new.keys() - unchanged):
            root = self._insert(root, new[version])
        self.root = root
        return {"nodes_created": self.total_update_nodes - nodes,
                "documents_read": self.total_update_document_reads - reads,
                "intervals_removed": len(old.keys() - unchanged),
                "intervals_inserted": len(new.keys() - unchanged),
                "index_seconds": perf_counter() - start}
