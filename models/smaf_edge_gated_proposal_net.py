from collections import OrderedDict

import torch
import torch.nn as nn

from models.cross_atlas_attention import CrossAtlasAttention
from models.edge_encoder import EdgeBranchEncoder


class AtlasGatedFusion(nn.Module):
    """Feature-level atlas gating used after cross-atlas enhancement."""

    def __init__(self, num_atlases, embedding_dim=128, hidden_dim=256, dropout=0.5):
        super().__init__()
        self.num_atlases = num_atlases
        self.gate_mlp = nn.Sequential(
            nn.Linear(embedding_dim * num_atlases, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_atlases),
        )

    def forward(self, atlas_embeddings):
        if atlas_embeddings.ndim != 3:
            raise ValueError("Expected atlas embeddings with shape B x M x D")

        batch_size, num_atlases, embedding_dim = atlas_embeddings.shape
        if num_atlases != self.num_atlases:
            raise ValueError(
                f"Expected {self.num_atlases} atlases, received {num_atlases}"
            )

        flattened = atlas_embeddings.reshape(batch_size, num_atlases * embedding_dim)
        gate_logits = self.gate_mlp(flattened)
        gate_weight = torch.softmax(gate_logits, dim=-1)
        gated_embedding = torch.sum(gate_weight.unsqueeze(-1) * atlas_embeddings, dim=1)
        return gated_embedding, gate_weight


class SMAFEdgeGatedProposalNet(nn.Module):
    """
    v2.1 hybrid model:

    1. Use the SMAF-Net v5 signed edge branch encoder per atlas.
    2. Apply proposal-style cross-atlas attention.
    3. Use v5-style feature-level gated fusion and an MLP classifier.
    """

    def __init__(
        self,
        atlas_specs,
        hidden_dim=256,
        embedding_dim=128,
        dropout=0.5,
    ):
        super().__init__()
        self.atlas_specs = OrderedDict(atlas_specs)
        self.atlas_names = list(self.atlas_specs.keys())
        self.num_atlases = len(self.atlas_names)

        self.encoders = nn.ModuleDict()
        for atlas_name, spec in self.atlas_specs.items():
            num_nodes = int(spec["num_nodes"])
            input_dim = num_nodes * (num_nodes - 1)
            self.encoders[atlas_name] = EdgeBranchEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
                dropout=dropout,
            )

        self.cross_atlas_attention = CrossAtlasAttention(
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
        self.gated_fusion = AtlasGatedFusion(
            num_atlases=self.num_atlases,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * (self.num_atlases + 1), hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, batch):
        graph_embeddings = []
        branch_details = {}
        for atlas_name in self.atlas_names:
            graph_embedding = self.encoders[atlas_name](batch[atlas_name])
            graph_embeddings.append(graph_embedding)
            branch_details[atlas_name] = {"graph_embedding": graph_embedding}

        stacked_embeddings = torch.stack(graph_embeddings, dim=1)
        enhanced_embeddings, attention_weight = self.cross_atlas_attention(
            stacked_embeddings
        )
        gated_embedding, gate_weight = self.gated_fusion(enhanced_embeddings)

        flattened_embeddings = enhanced_embeddings.reshape(
            enhanced_embeddings.shape[0],
            -1,
        )
        final_embedding = torch.cat([flattened_embeddings, gated_embedding], dim=-1)
        fusion_logits = self.classifier(final_embedding)

        return {
            "fusion_logits": fusion_logits,
            "atlas_weight": gate_weight,
            "attention_weight": attention_weight,
            "branch_details": branch_details,
        }
