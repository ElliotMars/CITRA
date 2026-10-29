"""Bounded scalar gate; its only input is the pre-cutoff numerical history."""
import torch
from torch import nn


class HistoryGate(nn.Module):
    def __init__(self,seq_len,channels):
        super().__init__()
        self.projection=nn.Linear(seq_len*channels,1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self,history):
        return torch.sigmoid(self.projection(history.flatten(1)))
