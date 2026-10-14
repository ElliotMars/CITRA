"""Fixed NumPy linear forecaster for output-certificate correctness tests.

This is not the trainable PyTorch integration planned for Milestone 3.
"""

import math
from dataclasses import dataclass
from time import perf_counter
import numpy as np

from .kernel import immutable


@dataclass(frozen=True)
class PredictionResult:
    prediction: np.ndarray
    aggregation: object
    certified_bound: float
    lipschitz_bound: float
    model_revision: int
    store_generation: int
    elapsed_seconds: float
    model: object
    store: object
    certificate_scope: str = "analytic_fixed_model_output_remainder_float64_tolerance"

    def assert_current(self):
        if self.model.revision != self.model_revision or self.store.generation != self.store_generation:
            raise RuntimeError("stale certificate: model parameters or document impacts/versions changed")


@dataclass(frozen=True)
class LinearParameters:
    backbone: np.ndarray
    weight: np.ndarray
    revision: int


class FixedLinearForecaster:
    def __init__(self, seq_len, channels, horizon, impact_dim, seed=0,
                 max_lipschitz=1.0):
        if any(not isinstance(x, int) or x <= 0
               for x in (seq_len, channels, horizon, impact_dim)):
            raise ValueError("model dimensions must be positive integers")
        if not math.isfinite(max_lipschitz) or max_lipschitz < 0:
            raise ValueError("invalid Lipschitz limit")
        self.seq_len, self.channels, self.horizon = seq_len, channels, horizon
        self.impact_dim = impact_dim
        rng = np.random.default_rng(seed)
        backbone = rng.normal(size=(horizon, seq_len)) / seq_len
        weight = rng.normal(size=(channels * horizon, impact_dim))
        weight *= max_lipschitz / max(float(np.linalg.norm(weight)), 1.0)
        self._parameters = LinearParameters(immutable(backbone), immutable(weight), 0)

    @property
    def backbone(self):
        return self._parameters.backbone

    @property
    def weight(self):
        return self._parameters.weight

    @property
    def revision(self):
        return self._parameters.revision

    def set_parameters(self, *, backbone=None, weight=None):
        """Replace a complete frozen snapshot; previous certificates become stale."""
        old = self._parameters
        b = immutable(old.backbone if backbone is None else backbone)
        w = immutable(old.weight if weight is None else weight)
        if b.shape != (self.horizon, self.seq_len) or w.shape != (self.channels * self.horizon, self.impact_dim):
            raise ValueError("invalid parameter shapes")
        if not math.isfinite(float(np.linalg.norm(w))):
            raise ValueError("nonfinite parameter norm")
        self._parameters = LinearParameters(b, w, old.revision + 1)

    @property
    def lipschitz_bound(self):
        # Frobenius >= spectral norm; fixed weights cannot change after query.
        return float(np.linalg.norm(self.weight))

    def predict(self, history, aggregate):
        return self._predict(self._parameters, history, aggregate)

    def _predict(self, parameters, history, aggregate):
        x = np.asarray(history, dtype=np.float64)
        a = np.asarray(aggregate, dtype=np.float64)
        if x.ndim != 3 or x.shape[1:] != (self.channels, self.seq_len):
            raise ValueError("history shape must be [B,C,L]")
        if a.shape != (x.shape[0], self.impact_dim):
            raise ValueError("aggregate shape must be [B,m]")
        if not np.isfinite(x).all() or not np.isfinite(a).all():
            raise ValueError("prediction inputs must be finite")
        numerical = x @ parameters.backbone.T
        correction = (a @ parameters.weight.T).reshape(x.shape[0], self.channels, self.horizon)
        return immutable(numerical + correction)

    def predict_certified(self, history, store, query, cutoff, epsilon, *, bound_mode="local"):
        """Single-query output certificate bound to actual fixed parameter arrays.

        Deliberately accepts no caller-supplied Lipschitz constant. A certificate
        references the exact snapshot used to produce the returned prediction.
        """
        start = perf_counter()
        if not math.isfinite(epsilon) or epsilon < 0:
            raise ValueError("epsilon must be finite and nonnegative")
        if np.asarray(history).shape != (1, self.channels, self.seq_len):
            raise ValueError("one certified query requires history shape [1,C,L]")
        parameters, generation = self._parameters, store.generation
        amplification = float(np.linalg.norm(parameters.weight))
        if not math.isfinite(amplification):
            raise ValueError("nonfinite model norm")
        if epsilon == 0:
            budget = 0.0
        elif amplification == 0:
            budget = float(np.finfo(np.float64).max)
        else:
            budget = min(epsilon / amplification, float(np.finfo(np.float64).max))
            budget = max(0., float(np.nextafter(budget, 0.)))
        aggregation = store.query_certified(query, cutoff, budget, bound_mode=bound_mode)
        bound = amplification * aggregation.aggregate_bound
        if bound > epsilon:  # rounded budget conversion can require exact execution
            aggregation = store.query_certified(query, cutoff, 0.)
            bound = 0.
        prediction = self._predict(parameters, history, aggregation.aggregate[None])
        result = PredictionResult(prediction, aggregation, bound, amplification, parameters.revision,
                                  generation, perf_counter() - start, self, store)
        result.assert_current()
        return result
