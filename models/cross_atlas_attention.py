import math

import torch
import torch.nn as nn


class CrossAtlasAttention(nn.Module):
    """
    Atlas-level attention enhancement.

    Each atlas embedding attends only to other atlases. The diagonal mask is
    important: the proposal explicitly aggregates complementary information
    from j != i rather than allowing a trivial self-attention shortcut.
    """

    def __init__(self, embedding_dim=128, dropout=0.5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

    def forward(self, atlas_embeddings):
        if atlas_embeddings.ndim != 3:
            raise ValueError("Expected atlas embeddings with shape B x M x D")

        num_atlases = atlas_embeddings.shape[1]
        if num_atlases < 2:
            raise ValueError("Cross-atlas attention requires at least two atlases")

        query = self.query(atlas_embeddings)
        key = self.key(atlas_embeddings)
        value = self.value(atlas_embeddings)

        scores = torch.matmul(query, key.transpose(-1, -2))
        scores = scores / math.sqrt(self.embedding_dim)

        self_mask = torch.eye(
            num_atlases,
            device=atlas_embeddings.device,
            dtype=torch.bool,
        ).unsqueeze(0)
        scores = scores.masked_fill(self_mask, float("-inf"))

        attention_weight = torch.softmax(scores, dim=-1)
        enhanced_context = torch.matmul(attention_weight, value)
        enhanced_embedding = self.ffn(enhanced_context + atlas_embeddings)

        return enhanced_embedding, attention_weight
