"""Offline fixed base embeddings and a trainable normalized retrieval projection."""
from torch import nn
from torch.nn import functional as F
from citra.text_features import hash_text


class TextKeyEncoder(nn.Module):
    def __init__(self, base_dim, embedding_dim):
        super().__init__()
        self.projection = nn.Linear(base_dim, embedding_dim)

    def forward(self, embeddings):
        return F.normalize(self.projection(embeddings), dim=-1, eps=1e-8)
