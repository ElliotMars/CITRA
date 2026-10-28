"""Trainable CITRA Full As-of model; NumPy indices are never used in training."""
import math
import torch
from torch import nn
from citra.learning.temporal_encoder import ForecastQueryEncoder
from citra.learning.text_encoder import TextKeyEncoder
from citra.learning.influence_encoder import TextImpactEncoder
from citra.learning.fusion import LinearCorrection


class Model(nn.Module):
    def __init__(self, seq_len=48, channels=2, horizon=6, target_ids=None, base_dim=32,
                 embedding_dim=8, impact_dim=8, metadata_dim=2, hidden_dim=32, sigma=0.7):
        super().__init__()
        target_ids = tuple(range(channels)) if target_ids is None else tuple(target_ids)
        if (any(not isinstance(x, int) or x <= 0 for x in
                (seq_len, channels, horizon, base_dim, embedding_dim, impact_dim, hidden_dim))
                or metadata_dim < 0 or not target_ids or len(set(target_ids)) != len(target_ids)
                or any(i < 0 or i >= channels for i in target_ids)
                or not math.isfinite(sigma) or sigma <= 0):
            raise ValueError("invalid model dimensions/targets/kernel")
        self.config = dict(seq_len=seq_len, channels=channels, horizon=horizon, target_ids=list(target_ids),
                           base_dim=base_dim, embedding_dim=embedding_dim, impact_dim=impact_dim,
                           metadata_dim=metadata_dim, hidden_dim=hidden_dim, sigma=sigma)
        self.register_buffer("target_ids", torch.tensor(target_ids, dtype=torch.long))
        self.backbone = nn.Linear(seq_len * channels, horizon * len(target_ids))
        self.query_encoder = ForecastQueryEncoder(seq_len, channels, embedding_dim, hidden_dim, horizon)
        self.key_encoder = TextKeyEncoder(base_dim, embedding_dim)
        self.impact_encoder = TextImpactEncoder(base_dim, metadata_dim, impact_dim, hidden_dim)
        self.fusion = LinearCorrection(impact_dim, horizon, len(target_ids))

    def numeric_forecast(self, history):
        c = self.config
        if history.ndim != 3 or history.shape[1:] != (c["seq_len"], c["channels"]):
            raise ValueError("history must have shape [B,L,C]")
        return self.backbone(history.flatten(1)).reshape(-1, c["horizon"], len(c["target_ids"]))

    def encode_query(self, history):
        return self.query_encoder(history, self.target_ids, self.config["horizon"])

    def encode_documents(self, base_embeddings, metadata):
        return self.key_encoder(base_embeddings), self.impact_encoder(base_embeddings, metadata)

    def aggregate(self, query, keys, impacts, visible, mode="full_asof_text", topk=8):
        if visible.shape != (query.shape[0], keys.shape[0]) or visible.dtype != torch.bool:
            raise ValueError("visibility mask must be bool [B,N]")
        if mode not in ("full_asof_text", "semantic_topk"):
            raise ValueError("training aggregation must be exact full or legal Top-K")
        distances = ((query[:, None, :] - keys[None, :, :])**2).sum(dim=-1)
        selected = visible
        if mode == "semantic_topk":
            if topk <= 0:
                raise ValueError("topk must be positive")
            selected = torch.zeros_like(visible)
            if keys.shape[0]:
                indices = distances.masked_fill(~visible, torch.inf).topk(min(topk, keys.shape[0]), largest=False).indices
                selected.scatter_(1, indices, True)
                selected &= visible
        weights = torch.exp(-distances / (2 * self.config["sigma"]**2))
        return torch.where(selected, weights, 0.) @ impacts

    def forward(self, history, base_embeddings=None, metadata=None, visible=None,
                mode="full_asof_text", topk=8, return_details=False):
        forecast = self.numeric_forecast(history)
        if mode == "ts_only":
            return forecast
        query = self.encode_query(history)
        keys, impacts = self.encode_documents(base_embeddings, metadata)
        aggregate = self.aggregate(query, keys, impacts, visible, mode, topk)
        prediction = forecast + self.fusion(aggregate)
        return (prediction, query, aggregate) if return_details else prediction
