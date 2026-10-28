"""Trainable exact aggregation, strict temporal windows, and frozen-index regression."""
import copy
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

try:
    import torch
except ImportError as error:
    raise unittest.SkipTest(f"trainable CITRA tests require working PyTorch: {error}")

from data_provider.citra_weather import WeatherData, TextCorpus, make_fixture, datetime_utc, parse_time
from citra.learning.inference import FrozenForecastSession, parameter_digest
from citra.learning.training import seed_everything, fit, load_checkpoint
from models.CITRA import Model


class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="citra-learning-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        make_fixture(self.root / "data", hours=180, base_dim=8, seed=3)
        self.data = WeatherData(self.root / "data", allow_simulated=True)
        seed_everything(3)
        self.model = Model(seq_len=12, channels=2, horizon=3, base_dim=8,
                           embedding_dim=4, impact_dim=3, hidden_dim=8).double()
        self.windows = self.data.windows("train", 12, 3, [0, 1], stride=3)
        self.x = torch.tensor(np.stack([self.windows[i][0] for i in range(3)]))
        self.base, self.metadata = self.data.corpus.tensors("cpu", torch.float64)
        self.mask = torch.tensor(self.data.corpus.visibility([self.windows[i][2] for i in range(3)]))

    def forward_loss(self):
        y = torch.tensor(np.stack([self.windows[i][1] for i in range(3)]))
        prediction = self.model(self.x, self.base, self.metadata, self.mask)
        return (prediction - y).square().mean()

    def frozen(self):
        return FrozenForecastSession(self.model.eval().requires_grad_(False), self.data.corpus)

    def test_trainable_query_encoder(self):
        self.forward_loss().backward()
        for encoder in (self.model.query_encoder, self.model.key_encoder):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in encoder.parameters() if p.grad is not None), 0.)
        q = self.model.encode_query(self.x)
        torch.testing.assert_close(torch.linalg.vector_norm(q, dim=1), torch.ones(3, dtype=torch.float64))
        targets = torch.tensor([1])
        changed_target = self.model.query_encoder(self.x, targets, 3)
        changed_horizon = self.model.query_encoder(self.x, self.model.target_ids, 2)
        self.assertFalse(torch.allclose(q, changed_target))
        self.assertFalse(torch.allclose(q, changed_horizon))

    def test_trainable_impact_encoder(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        old = parameter_digest(self.model, "impact_encoder.")
        loss = self.forward_loss()
        loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in self.model.impact_encoder.parameters()), 0.)
        optimizer.step()
        self.assertNotEqual(old, parameter_digest(self.model, "impact_encoder."))
        self.assertTrue(torch.isfinite(loss))

    def test_end_to_end_forecast_shape(self):
        for mode in ("ts_only", "semantic_topk", "full_asof_text"):
            pred = self.model(self.x, self.base, self.metadata, self.mask, mode=mode, topk=2)
            self.assertEqual(tuple(pred.shape), (3, 3, 2))
        empty = torch.zeros_like(self.mask)
        pred = self.model(self.x, self.base, self.metadata, empty)
        torch.testing.assert_close(pred, self.model(self.x, mode="ts_only"))

    def test_frozen_index_consistency(self):
        with self.assertRaises(ValueError):
            FrozenForecastSession(self.model, self.data.corpus)
        session = self.frozen()
        for i in (0, 5, 15):
            x, _, t = self.windows[i]
            mask = torch.tensor(self.data.corpus.visibility([t]))
            reference = self.model(torch.tensor(x[None]), self.base, self.metadata, mask).detach().numpy()
            indexed = session.predict(x[None], datetime_utc(t), execution="full_asof_text")
            np.testing.assert_allclose(indexed.prediction, reference, atol=1e-10, rtol=1e-10)
        self.assertEqual(session.index_generation, session.store.generation)

    def test_full_vs_certified_prediction(self):
        session = self.frozen()
        for i in (0, 8, 20):
            x, _, t = self.windows[i]
            full = session.predict(x[None], datetime_utc(t), execution="full_asof_text")
            for epsilon in (0., .01, .1, 1.):
                fast = session.predict(x[None], datetime_utc(t), epsilon)
                self.assertLessEqual(np.linalg.norm(fast.prediction - full.prediction), fast.certified_bound + 1e-10)
                self.assertLessEqual(fast.certified_bound, epsilon)
                fast.assert_current()
            topk = session.predict(x[None], datetime_utc(t), execution="semantic_topk", topk=2)
            self.assertIsNone(topk.certified_bound)

    def test_encoder_update_invalidates_index(self):
        for module in ("query_encoder", "key_encoder", "impact_encoder", "fusion", "backbone"):
            model = copy.deepcopy(self.model).eval().requires_grad_(False)
            session = FrozenForecastSession(model, self.data.corpus)
            x, _, t = self.windows[0]
            result = session.predict(x[None], datetime_utc(t), .1)
            # .data writes can bypass torch's version counters; content hashes
            # must still catch them.
            next(getattr(model, module).parameters()).data.add_(0.1)
            with self.assertRaises(RuntimeError):
                session.predict(x[None], datetime_utc(t), .1)
            with self.assertRaises(RuntimeError):
                result.assert_current()
            refreshed = FrozenForecastSession(model, self.data.corpus)
            refreshed.predict(x[None], datetime_utc(t), .1).assert_current()

    def test_uncertified_fusion_is_rejected(self):
        class SquaredCorrection(torch.nn.Module):
            def __init__(self, projection):
                super().__init__()
                self.projection = projection

            def forward(self, value):
                return self.projection(value).square()

        session = self.frozen()
        projection = self.model.fusion.projection
        self.model.fusion = SquaredCorrection(projection)
        # The state_dict bytes can remain identical while the computation is
        # nonlinear; checking weight hashes alone is insufficient.
        with self.assertRaises(ValueError):
            session.assert_current()
        with self.assertRaises(ValueError):
            FrozenForecastSession(self.model, self.data.corpus)

    def test_no_future_text_leakage(self):
        x, _, cutoff = self.windows[0]
        mask = torch.tensor(self.data.corpus.visibility([cutoff]))
        self.assertTrue((~mask).any())
        base = self.base.clone()
        base[~mask[0]] = 1e6
        metadata = self.metadata.clone()
        metadata[~mask[0]] = -1e6
        with torch.no_grad():
            expected = self.model(torch.tensor(x[None]), self.base, self.metadata, mask)
            altered = self.model(torch.tensor(x[None]), base, metadata, mask)
        torch.testing.assert_close(expected, altered, atol=0., rtol=0.)
        # A revision may describe a future event, but cannot replace the old
        # version until its own publication/availability cutoff.
        original = self.data.corpus.records[0]
        version_indices = [i for i, d in enumerate(self.data.corpus.documents) if d.doc_id == original["doc_id"]]
        early = self.data.corpus.visibility([parse_time(original["issued_at"]) + timedelta(hours=1)])
        self.assertEqual(int(early[0, version_indices].sum()), 1)
        self.assertTrue(early[0, version_indices[0]])

    def test_no_future_numeric_leakage(self):
        test = self.data.windows("test", 12, 3, [0, 1])
        pos = test.starts[0]
        expected_history = test[0][0].copy()
        expected_mean, expected_scale = self.data.mean.copy(), self.data.scale.copy()
        self.data.values[pos:] = 1e9
        self.data.fit_scaler()
        np.testing.assert_array_equal(self.data.mean, expected_mean)
        np.testing.assert_array_equal(self.data.scale, expected_scale)
        np.testing.assert_array_equal(test.history(pos), expected_history)
        # A history record that arrives after cutoff must also be excluded.
        self.data.available[pos - 1, 0] = self.data.times[pos - 1] + 3600
        self.data.values[pos - 1, 0] = 1e12
        actual = test.history(pos)
        self.assertEqual(actual[-1, 0], actual[-2, 0])

    def test_strict_data_rejects_simulated_availability(self):
        with self.assertRaises(ValueError):
            WeatherData(self.root / "data")

    def test_training_checkpoint_roundtrip(self):
        config = dict(seed=3, stride=6, learning_rate=.001, weight_decay=.0001, epochs=2,
                      batch_size=8, topk=2, gradient_clip=1., fusion_norm_limit=1.)
        self.root.joinpath("output").mkdir()
        old = parameter_digest(self.model)
        trained, report = fit(self.model, self.data, config, "full_asof_text", self.root / "output")
        self.assertNotEqual(old, parameter_digest(trained))
        loaded, state = load_checkpoint(report["checkpoint"], self.data, "cpu")
        self.assertEqual(parameter_digest(loaded), parameter_digest(trained))
        self.assertEqual(state["training_mode"], "full_asof_text")
        self.assertEqual(len(report["epochs"]), 2)
        with self.assertRaises(ValueError):
            fit(trained, self.data, config, "citra_certified", self.root / "output")

    def test_empty_corpus_has_zero_text_correction(self):
        corpus = TextCorpus([], 8, 2)
        session = FrozenForecastSession(self.model.eval().requires_grad_(False), corpus)
        x, _, t = self.windows[0]
        full = session.predict(x[None], datetime_utc(t), execution="full_asof_text")
        fast = session.predict(x[None], datetime_utc(t), .1)
        np.testing.assert_array_equal(full.prediction, fast.prediction)
        self.assertEqual(fast.certified_bound, 0.)
