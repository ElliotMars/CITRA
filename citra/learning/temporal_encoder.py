"""History-only, target- and horizon-conditioned forecast queries."""
import torch
from torch import nn
from torch.nn import functional as F


class ForecastQueryEncoder(nn.Module):
    def __init__(self, seq_len, channels, embedding_dim, hidden_dim, max_horizon):
        super().__init__()
        self.channels, self.max_horizon = channels, max_horizon
        self.history = nn.Sequential(nn.Flatten(), nn.LayerNorm(seq_len * channels),
                                     nn.Linear(seq_len * channels, hidden_dim), nn.Tanh(),
                                     nn.Linear(hidden_dim, embedding_dim))
        self.target = nn.Embedding(channels, embedding_dim)
        self.horizon = nn.Embedding(max_horizon + 1, embedding_dim)

    def forward(self, history, targets, horizon):
        if not 1 <= horizon <= self.max_horizon:
            raise ValueError("horizon outside configured range")
        if targets.ndim == 1:
            targets = targets[None].expand(history.shape[0], -1)
        if targets.shape[0] != history.shape[0] or targets.numel() == 0:
            raise ValueError("expected target IDs [T] or [B,T]")
        target = self.target(targets).mean(dim=1)
        horizon_ids = torch.full((history.shape[0],), horizon, dtype=torch.long, device=history.device)
        return F.normalize(self.history(history) + target + self.horizon(horizon_ids), dim=-1, eps=1e-8)
