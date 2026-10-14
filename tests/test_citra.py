"""Run with: python -m unittest discover -s tests -v (NumPy only)."""

from dataclasses import replace
from datetime import timedelta
import math
import unittest
import numpy as np

from citra import Document, TemporalDocumentStore, RBFKernel, CITRAIndex
from citra.kernel import exact_aggregate
from citra.prediction import FixedLinearForecaster
from citra.synthetic import EPOCH, synthetic_store


def day(n):
    return EPOCH + timedelta(days=n)


def document(doc_id="d", version_id="v1", available=0, **kwargs):
    fields = dict(doc_id=doc_id, version_id=version_id, source="test", entity_id="series",
                  issued_at=day(0), available_at=day(available) if available is not None else None,
                  valid_from=day(0), valid_to=None, event_start=day(0), event_end=day(1),
                  embedding=[0.0, 0.0], impact_vector=[1.0, -1.0],
                  availability_provenance="verified")
    fields.update(kwargs)
    return Document(**fields)


class Dual:
    """Small forward-mode autodiff engine for an independent gradient check."""
    def __init__(self, value, derivative):
        self.value = value
        self.derivative = np.asarray(derivative)

    def lift(self, other):
        return other if isinstance(other, Dual) else Dual(other, np.zeros_like(self.derivative))

    def __add__(self, other):
        other = self.lift(other)
        return Dual(self.value + other.value, self.derivative + other.derivative)

    __radd__ = __add__

    def __sub__(self, other):
        other = self.lift(other)
        return Dual(self.value - other.value, self.derivative - other.derivative)

    def __mul__(self, other):
        other = self.lift(other)
        return Dual(self.value * other.value,
                    self.derivative * other.value + self.value * other.derivative)

    __rmul__ = __mul__

    def exp(self):
        value = math.exp(self.value)
        return Dual(value, value * self.derivative)


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.store = TemporalDocumentStore(2, 2, leaf_size=2)

    def test_temporal_visibility(self):
        self.store.insert_document(document(available=2, valid_from=day(1), valid_to=day(5)))
        self.assertEqual(self.store.snapshot(day(1)), ())
        self.assertEqual(len(self.store.snapshot(day(2))), 1)
        self.assertEqual(self.store.snapshot(day(5)), ())

    def test_historical_revision(self):
        old = document()
        self.store.insert_document(old)
        self.store.append_revision(document(version_id="v2", available=5,
                                             impact_vector=[100, 0], valid_from=day(0)))
        self.assertEqual(self.store.snapshot(day(4))[0].version_id, "v1")
        self.assertEqual(self.store.snapshot(day(5))[0].version_id, "v2")
        self.assertIsNone(old.valid_to)
        np.testing.assert_array_equal(self.store.aggregate_exact([0, 0], day(4)), [1, -1])

    def test_no_future_document(self):
        self.store.insert_document(document("past"))
        self.store.insert_document(document("future", available=10))
        self.store.insert_document(document("unknown", available=None))
        self.store.insert_document(document("unverified", availability_provenance="unknown"))
        self.store.insert_document(document("simulated", availability_provenance="simulated"))
        self.store.insert_document(document("late", ingested_at=day(8)))
        self.assertEqual([d.doc_id for d in self.store.snapshot(day(3))], ["past"])
        np.testing.assert_array_equal(self.store.query_certified([0, 0], day(3), 0).aggregate, [1, -1])

    def test_future_event_can_be_known(self):
        self.store.insert_document(document(event_start=day(30), event_end=day(31)))
        self.assertEqual(len(self.store.snapshot(day(0))), 1)

    def test_unknown_revision_does_not_replace_history(self):
        self.store.insert_document(document())
        self.store.append_revision(document(version_id="unknown", available=None))
        self.assertEqual(self.store.snapshot(day(10))[0].version_id, "v1")

    def test_expired_revision_does_not_resurrect_old_version(self):
        self.store.insert_document(document())
        self.store.append_revision(document(version_id="v2", available=2, valid_to=day(4)))
        self.assertEqual(self.store.snapshot(day(5)), ())

    def test_out_of_order_revision(self):
        self.store.insert_document(document())
        self.store.append_revision(document(version_id="v3", available=6))
        self.store.append_revision(document(version_id="v2", available=3))
        self.assertEqual(self.store.snapshot(day(4))[0].version_id, "v2")
        self.assertEqual(self.store.snapshot(day(8))[0].version_id, "v3")

    def test_late_revision(self):
        self.store.insert_document(document())
        self.store.append_revision(document(version_id="v2", available=2, ingested_at=day(6)))
        self.assertEqual(self.store.snapshot(day(5))[0].version_id, "v1")
        self.assertEqual(self.store.snapshot(day(6))[0].version_id, "v2")

    def test_immutable_documents(self):
        original = np.array([1.0, 2.0])
        d = document(embedding=original)
        original[0] = 99
        self.assertEqual(d.embedding[0], 1)
        with self.assertRaises(ValueError):
            d.embedding.setflags(write=True)

    def test_invalid_metadata_and_dimensions(self):
        with self.assertRaises(ValueError):
            document(available=-1)
        with self.assertRaises(ValueError):
            document(valid_from=EPOCH.replace(tzinfo=None))
        with self.assertRaises(ValueError):
            document(valid_to=day(0))
        with self.assertRaises(ValueError):
            self.store.insert_document(document(embedding=[1]))
        self.store.insert_document(document())
        with self.assertRaises(ValueError):
            self.store.insert_document(document())
        with self.assertRaises(ValueError):
            self.store.append_revision(document())
        with self.assertRaises(ValueError):
            self.store.append_revision(document("missing", "v2"))

    def test_incremental_update(self):
        self.store.insert_document(document())
        old_snapshot = self.store.snapshot(day(1))
        result = self.store.query_certified([0, 0], day(1), 1)
        root = self.store.index.root
        self.assertTrue(result.stats.cache_hit)
        self.assertTrue(self.store.query_certified([0, 0], day(1), 1).stats.cache_hit)
        self.assertIs(root, self.store.index.root)
        self.store.insert_document(document("second", impact_vector=[2, 3]))
        new = self.store.query_certified([0, 0], day(1), 0)
        self.assertTrue(new.stats.cache_hit)
        self.assertIsNot(root, self.store.index.root)
        np.testing.assert_array_equal(new.aggregate, [3, 2])
        self.store.append_revision(document(version_id="v2", available=4, impact_vector=[9, 9]))
        np.testing.assert_array_equal(self.store.aggregate_exact([0, 0], day(1)), [3, 2])
        np.testing.assert_array_equal(self.store.aggregate_exact([0, 0], day(4)), [11, 12])
        self.assertEqual(len(old_snapshot), 1)
        self.assertEqual(old_snapshot[0].version_id, "v1")


class AggregationTests(unittest.TestCase):
    def test_exact_aggregation(self):
        store = TemporalDocumentStore(2, 2)
        store.insert_document(document(embedding=[0, 0], impact_vector=[2, -3]))
        store.insert_document(document("other", embedding=[1, 0], impact_vector=[-4, 5]))
        expected = np.array([2, -3]) + math.exp(-0.5) * np.array([-4, 5])
        np.testing.assert_allclose(store.aggregate_exact([0, 0], day(0)), expected)

    def test_gradient_finite_difference_and_autodiff(self):
        rng = np.random.default_rng(17)
        for p in (1, 2, 9):
            q, z = rng.normal(size=(2, p))
            kernel = RBFKernel(1.7)
            derivatives = np.eye(p)
            dual_z = [Dual(z[j], derivatives[j]) for j in range(p)]
            delta = [dual_z[j] - q[j] for j in range(p)]
            automatic = (sum(d * d for d in delta) * (-0.5 / kernel.sigma**2)).exp()
            finite = []
            for j in range(p):
                step = derivatives[j] * 1e-5
                finite.append((kernel.value(q, z + step) - kernel.value(q, z - step)) / 2e-5)
            np.testing.assert_allclose(kernel.gradient_z(q, z), automatic.derivative, atol=1e-13)
            np.testing.assert_allclose(kernel.gradient_z(q, z), finite, atol=1e-9)

    def test_hessian_bound(self):
        rng = np.random.default_rng(8)
        for p in (1, 2, 12):
            for sigma in (0.1, 1.0, 10.0):
                kernel = RBFKernel(sigma)
                for _ in range(20):
                    delta = rng.normal(size=p) * sigma
                    hessian = math.exp(-float(delta @ delta) / (2 * sigma**2)) * (
                        np.outer(delta, delta) / sigma**4 - np.eye(p) / sigma**2)
                    self.assertLessEqual(np.linalg.norm(hessian, 2), kernel.hessian_bound + 1e-10)

    def test_taylor_approximation(self):
        store = synthetic_store(31, 3, 5, seed=1, radius=0.01, leaf_size=2)
        index = CITRAIndex(store.snapshot(day(30)), 3, 5, store.kernel, leaf_size=2)
        q = np.array([0.5, -0.4, 0.1])
        root = index.root
        docs = index.documents
        z = np.stack([d.embedding for d in docs])
        v = np.stack([d.impact_vector for d in docs])
        delta = z - z.mean(axis=0)
        self.assertEqual(root.first_moment.shape, (3, 5))
        np.testing.assert_allclose(root.first_moment, delta.T @ v)
        self.assertAlmostEqual(root.second_residual,
                               float(np.sum(np.sum(delta**2, axis=1) * np.linalg.norm(v, axis=1))))
        approximate, bound = index.approximate_node(root, q)
        self.assertLessEqual(np.linalg.norm(approximate - index.aggregate_exact(q)), bound + 1e-10)

    def test_certified_bound(self):
        cases = 0
        for seed in range(4):
            for p in (1, 3, 16):
                for leaf_size in (1, 4, 16):
                    for radius in (1e-6, 0.05, 100.0):
                        store = synthetic_store(37, p, 4, seed, radius, leaf_size)
                        model = FixedLinearForecaster(5, 2, 3, 4, seed)
                        q = np.random.default_rng(seed + 77).normal(size=p) * radius
                        for cutoff in (day(0), day(10), day(30)):
                            full = store.aggregate_exact(q, cutoff)
                            history = np.zeros((1, 2, 5))
                            for epsilon in (0.0, 1e-5, 0.05, 100.0):
                                with self.subTest(seed=seed, p=p, leaf=leaf_size, radius=radius,
                                                  cutoff=cutoff, epsilon=epsilon):
                                    prediction = model.predict_certified(history, store, q, cutoff, epsilon)
                                    result = prediction.aggregation
                                    tolerance = 1e-10 + 1e-10 * np.linalg.norm(full)
                                    error = np.linalg.norm(result.aggregate - full)
                                    deviation = np.linalg.norm(prediction.prediction - model.predict(history, full[None]))
                                    self.assertLessEqual(error, result.aggregate_bound + tolerance)
                                    self.assertLessEqual(deviation, prediction.certified_bound + tolerance)
                                    self.assertLessEqual(prediction.certified_bound, epsilon)
                                    if epsilon == 0:
                                        np.testing.assert_array_equal(result.aggregate, full)
                                cases += 1
        self.assertEqual(cases, 1296)

    def test_epsilon_monotonicity(self):
        store = synthetic_store(127, seed=8, leaf_size=2)
        results = [store.query_certified(np.zeros(4), day(30), epsilon)
                   for epsilon in (100, 1, 0.1, 0.01, 0.001, 0)]
        for loose, tight in zip(results, results[1:]):
            self.assertLessEqual(loose.stats.documents_scanned, tight.stats.documents_scanned)
            self.assertLessEqual(tight.aggregate_bound, loose.aggregate_bound + 1e-12)
        # Actual errors and measured latency need not be monotone.

    def test_empty_database(self):
        store = TemporalDocumentStore(3, 2)
        for epsilon in (0, 1):
            result = store.query_certified([0, 0, 0], day(0), epsilon)
            np.testing.assert_array_equal(result.aggregate, [0, 0])
            self.assertEqual(result.aggregate_bound, 0)
            self.assertEqual(result.stats.documents_scanned, 0)
            self.assertFalse(result.stats.fallback_exact)
        np.testing.assert_array_equal(store.aggregate_exact([0, 0, 0], day(0)), [0, 0])

    def test_signed_cancellation_and_duplicate_content(self):
        store = TemporalDocumentStore(2, 2, leaf_size=1)
        for i in range(10):
            store.insert_document(document(str(i), impact_vector=[(-1)**i, 0]))
        result = store.query_certified([0.5, 0], day(0), 1)
        np.testing.assert_array_equal(result.aggregate, [0, 0])
        self.assertEqual(result.aggregate_bound, 0)
        self.assertEqual(result.stats.documents_scanned, 0)
        self.assertEqual(store.index.root.count, 10)

    def test_partition_has_no_omission_or_duplicate(self):
        store = synthetic_store(51, seed=3, leaf_size=3)
        index = CITRAIndex(store.snapshot(day(30)), 4, 3, store.kernel, 3)
        leaf_refs = [e.document.doc_id for n in index.nodes for e in n.document_refs]
        self.assertEqual(sorted(leaf_refs), sorted(d.doc_id for d in index.documents))
        for node in index.nodes:
            if node.children:
                self.assertEqual(node.document_refs, ())
                self.assertEqual(sum(c.count for c in node.children), node.count)

    def test_query_stats_include_cold_build(self):
        store = synthetic_store(31, leaf_size=2)
        cold = store.query_certified([0, 0, 0, 0], day(30), 100)
        self.assertEqual(cold.stats.index_document_reads, 0)
        self.assertEqual(cold.stats.build_seconds, 0)
        self.assertGreater(store.index.total_update_document_reads, 0)
        warm = store.query_certified([0, 0, 0, 0], day(30), 100)
        self.assertTrue(warm.stats.cache_hit)
        self.assertEqual(warm.stats.index_document_reads, 0)
        exact = store.query_certified([0, 0, 0, 0], day(30), 0)
        self.assertEqual(exact.stats.documents_scanned, 31)
        self.assertEqual(exact.stats.exact_kernel_evaluations, 31)

    def test_invalid_query(self):
        store = TemporalDocumentStore(2, 2)
        for epsilon in (-1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                store.query_certified([0, 0], day(0), epsilon)
        for lipschitz in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(TypeError):
                store.query_certified([0, 0], day(0), 1, lipschitz=lipschitz)
        with self.assertRaises(ValueError):
            store.query_certified([float("nan"), 0], day(0), 1)
        for sigma in (0, -1, float("inf"), 1e-300):
            with self.assertRaises(ValueError):
                RBFKernel(sigma)

    def test_prediction_shape(self):
        model = FixedLinearForecaster(12, 3, 5, 7)
        history = np.zeros((4, 3, 12))
        self.assertEqual(model.predict(history, np.zeros((4, 7))).shape, (4, 3, 5))
        self.assertGreaterEqual(model.lipschitz_bound, np.linalg.norm(model.weight, 2) - 1e-12)
        self.assertLessEqual(model.lipschitz_bound, 1.0 + 1e-12)


if __name__ == "__main__":
    unittest.main()
