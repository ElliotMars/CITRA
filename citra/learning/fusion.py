"""Only the aggregate input needs a certified amplification bound."""
import torch
from torch import nn


class LinearCorrection(nn.Module):
    def __init__(self, impact_dim, horizon, targets):
        super().__init__()
        self.horizon, self.targets = horizon, targets
        self.projection = nn.Linear(impact_dim, horizon * targets, bias=False)
        nn.init.normal_(self.projection.weight, std=0.02)

    def forward(self, aggregate):
        return self.projection(aggregate).reshape(-1, self.horizon, self.targets)

    def lipschitz_bound(self):
        return torch.linalg.vector_norm(self.projection.weight)

    @torch.no_grad()
    def constrain(self, maximum):
        if maximum <= 0:
            raise ValueError("correction norm limit must be positive")
        norm = self.lipschitz_bound()
        if norm > maximum:
            self.projection.weight.mul_(maximum / norm)
