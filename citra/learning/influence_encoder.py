"""Signed, query-independent impacts; contains no target/forecast labels."""
import torch
from torch import nn


class TextImpactEncoder(nn.Module):
    def __init__(self, base_dim, metadata_dim, impact_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(base_dim + metadata_dim, hidden_dim),
                                     nn.Tanh(), nn.Linear(hidden_dim, impact_dim))

    def forward(self, embeddings, metadata):
        return self.network(torch.cat((embeddings, metadata), dim=-1))
