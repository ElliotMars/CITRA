"""Regression tests for prompt.md's review, including adversarial version order."""
from copy import copy
from dataclasses import replace
from datetime import timedelta
import math
import unittest
import numpy as np

from citra import CITRAIndex, RBFKernel, TemporalDocumentStore
from citra.kernel import exact_aggregate
from citra.prediction import FixedLinearForecaster
from citra.synthetic import synthetic_store
from test_citra import day, document


class CertificateAPITests(unittest.TestCase):
    def test_wrong_lipschitz_is_rejected(self):
        store = synthetic_store(128, radius=2.)
        for invalid in (0., 1., 100.):
            with self.assertRaises(TypeError):
                store.query_certified(np.zeros(4), day(30), 1., lipschitz=invalid)
            with self.assertRaises(TypeError):
                store.index.query(np.zeros(4), 1., lipschitz=invalid)
        result = store.query_certified(np.zeros(4), day(30), 1.)
        self.assertFalse(hasattr(result, "certified_bound"))

    def test_prediction_derives_actual_weight_norm(self):
        store = synthetic_store(128, seed=8)
        for scale in (0., 1., 20.):
            model = FixedLinearForecaster(5, 2, 3, 3)
            model.set_parameters(weight=model.weight * scale)
            history, q = np.zeros((1, 2, 5)), np.zeros(4)
            for epsilon in (0., 0.01, 1.):
                result = model.predict_certified(history, store, q, day(30), epsilon)
                full = model.predict(history, store.aggregate_exact(q, day(30))[None])
                self.assertEqual(result.lipschitz_bound, float(np.linalg.norm(model.weight)))
                self.assertLessEqual(np.linalg.norm(result.prediction - full), result.certified_bound + 1e-10)
                self.assertLessEqual(result.certified_bound, epsilon)
            with self.assertRaises(TypeError):
                model.predict_certified(history, store, q, day(30), 1., lipschitz=0.)

    def test_certificates_invalidated_on_parameters_or_impacts(self):
        store = synthetic_store(16)
        model = FixedLinearForecaster(5, 2, 3, 3)
        history = np.zeros((1, 2, 5))
        result = model.predict_certified(history, store, np.zeros(4), day(30), 1.)
        result.assert_current()
        with self.assertRaises(AttributeError):
            model.weight = model.weight * 2
        model.set_parameters(weight=model.weight * 2)
        with self.assertRaises(RuntimeError):
            result.assert_current()
        result = model.predict_certified(history, store, np.zeros(4), day(30), 1.)
        d = store.documents[0]
        store.append_revision(replace(d, version_id="impact_update", impact_vector=-d.impact_vector))
        with self.assertRaises(RuntimeError):
            result.assert_current()


class LocalBoundTests(unittest.TestCase):
    def test_ball_hessian_supremum(self):
        rng = np.random.default_rng(42)
        for p in (1, 2, 8):
            for sigma in (0.1, 1., 10.):
                k = RBFKernel(sigma)
                for radius in (0., 1e-8, 0.5 * sigma, 10 * sigma):
                    for distance in (0., sigma, math.sqrt(3)*sigma, 50*sigma):
                        c, q = np.zeros(p), np.zeros(p)
                        q[0] = distance
                        bound = k.local_hessian_bound(q, c, radius)
                        self.assertLessEqual(bound, k.hessian_bound)
                        for _ in range(12):
                            direction = rng.normal(size=p)
                            z = c + direction / np.linalg.norm(direction) * radius * rng.random()
                            delta = (q - z) / sigma
                            t = float(delta @ delta)
                            h = math.exp(-t/2) * (np.outer(delta, delta) - np.eye(p)) / sigma**2
                            self.assertLessEqual(np.linalg.norm(h, 2), bound + 1e-10*k.hessian_bound)

    def test_queries_change_work_and_bound(self):
        store = synthetic_store(256)
        index = CITRAIndex(store.snapshot(day(30)), 4, 3, store.kernel)
        near = index.query(np.zeros(4), 0.1)
        far = index.query(np.full(4, 50.), 0.1)
        global_far = index.query(np.full(4, 50.), 0.1, bound_mode="global")
        self.assertLess(far.stats.documents_scanned, near.stats.documents_scanned)
        self.assertLess(far.stats.nodes_visited, global_far.stats.nodes_visited)
        self.assertGreater(far.aggregate_bound, 0.)  # no false zero due to exp underflow
        for q in (np.zeros(4), np.full(4, 0.2), np.full(4, 50.)):
            result = index.query(q, 0.1)
            self.assertLessEqual(np.linalg.norm(result.aggregate - index.aggregate_exact(q)), result.aggregate_bound + 1e-10)

    def test_extreme_radius_offset_and_heterogeneous_influences(self):
        rng = np.random.default_rng(12)
        for sigma in (0.1, 1., 10.):
            for offset in (0., 1e5):
                for radius in (1e-7, 0.1, 100.):
                    docs = [document(str(i), embedding=rng.normal(size=2)*radius + offset,
                                     impact_vector=rng.normal(size=2)*10**(i % 5 - 2)) for i in range(43)]
                    index = CITRAIndex(docs, 2, 2, RBFKernel(sigma), leaf_size=2)
                    for query in (np.full(2, offset), np.full(2, offset + 50*sigma)):
                        for epsilon in (0., 1e-4, 0.1, 10.):
                            result = index.query(query, epsilon)
                            exact = index.aggregate_exact(query)
                            self.assertLessEqual(np.linalg.norm(result.aggregate - exact), result.aggregate_bound + 1e-8*(1+np.linalg.norm(exact)))


class TemporalIndexTests(unittest.TestCase):
    def test_temporal_index_oracle_requires_cutoff(self):
        store = synthetic_store(31, seed=7)
        with self.assertRaises(TypeError):
            store.index.aggregate_exact(np.zeros(4))
        for t in (0, 10, 30):
            np.testing.assert_array_equal(store.index.aggregate_exact(np.zeros(4), cutoff=day(t)),
                                          store.aggregate_exact(np.zeros(4), day(t)))

    def test_kernel_and_visibility_policy_cannot_silently_change(self):
        store = synthetic_store(16)
        with self.assertRaises(AttributeError):
            store.kernel = RBFKernel(2.)
        with self.assertRaises(AttributeError):
            store.index.kernel = RBFKernel(2.)
        with self.assertRaises(AttributeError):
            store.allow_simulated = False

    def test_scan_only_baseline_matches_indexed_store(self):
        indexed = synthetic_store(31, seed=7)
        scan = synthetic_store(31, seed=7, indexed=False)
        for t in (0, 10, 30):
            np.testing.assert_array_equal(indexed.aggregate_exact(np.zeros(4), day(t)), scan.aggregate_exact(np.zeros(4), day(t)))
        with self.assertRaises(RuntimeError):
            scan.query_certified(np.zeros(4), day(30), 1.)

    def verify_node_summaries(self, node):
        if node.children:
            left, right = node.children
            self.assertEqual(node.document_refs, ())
            self.assertLessEqual(abs(left.height - right.height), 1)
            self.assertLess(left.max_key, right.max_key)
            refs = self.verify_node_summaries(left) + self.verify_node_summaries(right)
        else:
            refs = node.document_refs
        z = np.stack([e.document.embedding for e in refs])
        v = np.stack([e.document.impact_vector for e in refs])
        delta = z - node.center
        np.testing.assert_allclose(node.center, z.mean(axis=0), atol=1e-10)
        np.testing.assert_allclose(node.influence_sum, v.sum(axis=0), atol=1e-10)
        np.testing.assert_allclose(node.first_moment, delta.T @ v, atol=1e-10)
        self.assertAlmostEqual(node.second_residual, float(np.sum(np.sum(delta**2, axis=1)*np.linalg.norm(v, axis=1))), places=8)
        self.assertLessEqual(float(np.linalg.norm(delta, axis=1).max()), node.radius + 1e-10)
        self.assertEqual(node.count, len(refs))
        return refs

    def test_rolling_cutoffs_reuse_index(self):
        store = synthetic_store(128, seed=4)
        index, root = store.index, store.index.root
        for t in (30, -3, 0, 4, 20, 10, 30):
            for eps in (0., 0.01, 1.):
                result = store.query_certified(np.zeros(4), day(t), eps)
                exact = store.aggregate_exact(np.zeros(4), day(t))
                self.assertLessEqual(np.linalg.norm(result.aggregate - exact), result.aggregate_bound + 1e-10)
                self.assertIs(store.index, index)
                self.assertIs(store.index.root, root)
                self.assertEqual(result.stats.index_document_reads, 0)
                self.assertEqual(result.stats.build_seconds, 0.)
        self.verify_node_summaries(root)

    def test_insert_and_revision_update_locally_with_shared_subtrees(self):
        store = synthetic_store(256, seed=5, leaf_size=2)
        held = copy(store.index)
        old_ids = {id(n) for n in held.nodes}
        store.insert_document(document("new", embedding=[0.1]*4, impact_vector=[1., 2., 3.],
                                       availability_provenance="simulated"))
        self.assertGreater(len(old_ids & {id(n) for n in store.index.nodes}), len(old_ids)//2)
        self.assertLess(store.last_update_stats["documents_read"], 20)
        self.assertLess(store.last_update_stats["nodes_created"], 50)
        previous_exact = held.query(np.zeros(4), 0., cutoff=day(30)).aggregate
        current_exact = store.aggregate_exact(np.zeros(4), day(30))
        self.assertGreater(np.linalg.norm(previous_exact - current_exact), 0.)
        self.verify_node_summaries(store.index.root)

    def test_arbitrary_versions_match_independent_snapshot(self):
        for seed in range(3):
            rng = np.random.default_rng(seed)
            store = TemporalDocumentStore(2, 2, leaf_size=3)
            for i in range(30):
                store.insert_document(document(str(i), available=int(rng.integers(0, 8)),
                                               embedding=rng.normal(size=2), impact_vector=rng.normal(size=2)))
            for iteration in range(90):
                d = store.documents[int(rng.integers(len(store.documents)))]
                available, valid_from = int(rng.integers(0, 18)), int(rng.integers(0, 18))
                end = valid_from + int(rng.integers(1, 8)) if iteration % 3 == 0 else None
                store.append_revision(document(d.doc_id, "revision_" + str(iteration), available,
                    valid_from=day(valid_from), valid_to=day(end) if end is not None else None,
                    ingested_at=day(int(rng.integers(0, 20))) if iteration % 4 == 0 else None,
                    availability_provenance="unknown" if iteration % 11 == 0 else "verified",
                    embedding=rng.normal(size=2), impact_vector=rng.normal(size=2)))
                for t in (0, 5, 10, 17, 25):
                    query = np.array([0.2, -0.4])
                    result = store.query_certified(query, day(t), 0.)
                    np.testing.assert_array_equal(result.aggregate, store.aggregate_exact(query, day(t)))
                    approximate = store.query_certified(query, day(t), 0.1)
                    self.assertLessEqual(np.linalg.norm(approximate.aggregate - result.aggregate), approximate.aggregate_bound + 1e-10)
            refs = self.verify_node_summaries(store.index.root)
            for t in range(26):
                visible = [e.document for e in refs if e.visible(day(t))]
                self.assertEqual(len({d.doc_id for d in visible}), len(visible))
                self.assertEqual({(d.doc_id, d.version_id) for d in visible},
                                 {(d.doc_id, d.version_id) for d in store.snapshot(day(t))})

    def test_expired_activation_tombstone_and_same_timestamp(self):
        store = TemporalDocumentStore(2, 2, leaf_size=1)
        store.insert_document(document())
        store.append_revision(document(version_id="v2", available=2, ingested_at=day(10), valid_to=day(5)))
        np.testing.assert_array_equal(store.query_certified([0, 0], day(9), 0).aggregate, [1, -1])
        np.testing.assert_array_equal(store.query_certified([0, 0], day(10), 0).aggregate, [0, 0])
        store.append_revision(document(version_id="v3", available=2, ingested_at=day(10), impact_vector=[3, 4]))
        np.testing.assert_array_equal(store.query_certified([0, 0], day(10), 0).aggregate, [3, 4])

    def test_leaf_only_storage_and_linear_build_reads(self):
        store = synthetic_store(4096)
        index = CITRAIndex(store.snapshot(day(30)), 4, 3, store.kernel)
        self.assertEqual(sum(len(n.document_refs) for n in index.nodes), 4096)
        self.assertEqual(index.build_document_reads, 8192)
        self.assertTrue(all(not n.document_refs for n in index.nodes if n.children))
        self.assertGreater(index.resident_bytes, index.summary_bytes)
