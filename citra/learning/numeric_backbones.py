"""Additional decomposition-linear numeric baseline, not a novel algorithm."""
import torch
from torch import nn
from torch.nn import functional as F


class DecompositionLinear(nn.Module):
    def __init__(self, seq_len, horizon, targets, kernel=25):
        super().__init__()
        self.targets, self.kernel = tuple(targets), kernel
        self.seasonal = nn.ModuleList([nn.Linear(seq_len, horizon) for _ in targets])
        self.trend = nn.ModuleList([nn.Linear(seq_len, horizon) for _ in targets])

    def forward(self, history):
        # This filter only sees the supplied historical window. Symmetric
        # smoothing inside that window does not read post-cutoff observations.
        x = history.transpose(1, 2)
        pad = self.kernel // 2
        trend = F.avg_pool1d(F.pad(x, (pad, pad), mode="replicate"), self.kernel, stride=1)
        seasonal = x - trend
        return torch.stack([s(seasonal[:, c]) + t(trend[:, c])
                            for c, s, t in zip(self.targets, self.seasonal, self.trend)], dim=-1)
