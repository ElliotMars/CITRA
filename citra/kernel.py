"""RBF kernel and an independent full-sum oracle (never semantic Top-K)."""

from dataclasses import dataclass
import math
import numpy as np


def vector(value, size=None):
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1 or (size is not None and arr.size != size):
        raise ValueError("expected a vector of the configured dimension")
    if not np.all(np.isfinite(arr)):
        raise ValueError("vectors must be finite")
    return arr


def immutable(value):
    arr = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("nonfinite arithmetic; rescale inputs")
    # Backed by immutable bytes: callers cannot re-enable writes.
    return np.frombuffer(arr.tobytes(), dtype=np.float64).reshape(arr.shape)


@dataclass(frozen=True)
class RBFKernel:
    sigma: float = 1.0

    def __post_init__(self):
        squared = self.sigma * self.sigma
        if (not math.isfinite(self.sigma) or self.sigma <= 0
                or not math.isfinite(squared) or squared <= 0
                or not math.isfinite(1.0 / squared)):
            raise ValueError("sigma must have a finite positive squared inverse")

    @property
    def hessian_bound(self):
        """Global spectral norm bound, valid along every Taylor segment."""
        return 1.0 / (self.sigma * self.sigma)

    def distance_interval(self, query, center, radius):
        """Includes the entire node ball and every Taylor segment."""
        q, c = vector(query), vector(center)
        if q.shape != c.shape or not math.isfinite(radius) or radius < 0:
            raise ValueError("invalid ball dimensions/radius")
        with np.errstate(over="raise", invalid="raise"):
            distance = float(np.linalg.norm(q - c))
        upper = distance + radius
        if not math.isfinite(upper):
            raise ValueError("nonfinite distance interval; rescale inputs")
        return max(0., float(np.nextafter(distance - radius, -np.inf))), float(np.nextafter(upper, np.inf))

    def local_hessian_bound(self, query, center, radius):
        """Supremum of exp(-t/2), |t-1|exp(-t/2) on the ball's t interval.

        The only interior radial maximum is t=3. Dimension one has no
        tangential eigenvalues, so only the radial envelope is used there.
        """
        lower, upper = self.distance_interval(query, center, radius)
        a, b = (lower / self.sigma)**2, (upper / self.sigma)**2
        if not math.isfinite(b):
            raise ValueError("distance/sigma is too large; rescale inputs")
        logs = [-a / 2] if vector(query).size > 1 else []
        for t in (a, b):
            if t != 1:
                logs.append(math.log(abs(t - 1)) - t / 2)
        if a <= 3 <= b:
            logs.append(math.log(2) - 1.5)
        if not logs:
            return 0.0
        value = math.exp(max(logs) - 2 * math.log(self.sigma))
        return min(self.hessian_bound, max(float(np.nextafter(0., 1.)), float(np.nextafter(value, np.inf))))

    def maximum_on_ball(self, query, center, radius):
        lower, _ = self.distance_interval(query, center, radius)
        value = math.exp(-0.5 * (lower / self.sigma)**2)
        return min(1., max(float(np.nextafter(0., 1.)), float(np.nextafter(value, np.inf))))

    def value(self, query, embedding):
        q = vector(query)
        z = vector(embedding, q.size)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            delta = (q - z) / self.sigma
            distance = float(delta @ delta)
        return math.exp(-0.5 * distance)

    def gradient_z(self, query, embedding):
        q = vector(query)
        z = vector(embedding, q.size)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return immutable(self.value(q, z) * ((q - z) / self.sigma) / self.sigma)


def exact_aggregate(documents, query, kernel, impact_dim):
    """Full legal-document sum with compensated componentwise reduction."""
    q = vector(query)
    rows = [kernel.value(q, d.embedding) * vector(d.impact_vector, impact_dim)
            for d in documents]
    return immutable([math.fsum(float(row[j]) for row in rows)
                      for j in range(impact_dim)])
