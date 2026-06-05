from collections import OrderedDict

import torch
import torch.nn as nn

from models.cross_atlas_attention import CrossAtlasAttention
from models.edge_encoder import EdgeBranchEncoder


class SMAFEdgeProposalNet(nn.Module):
    """
    v2.0 hybrid model:

    1. Use the stronger SMAF-Net v5 signed edge branch encoder per atlas.
    2. Apply proposal-style cross-atlas attention.
    3. Apply proposal-style energy-aware decision fusion and branch losses.
    """

    def __init__(
        self,
        atlas_specs,
        hidden_dim=256,
        embedding_dim=128,
        dropout=0.5,
        temperature=1.0,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")

        self.atlas_specs = OrderedDict(atlas_specs)
        self.atlas_names = list(self.atlas_specs.keys())
        self.temperature = temperature

        self.encoders = nn.ModuleDict()
        self.classifiers = nn.ModuleDict()
        for atlas_name, spec in self.atlas_specs.items():
            num_nodes = int(spec["num_nodes"])
            input_dim = num_nodes * (num_nodes - 1)
            self.encoders[atlas_name] = EdgeBranchEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
                dropout=dropout,
            )
            self.classifiers[atlas_name] = nn.Linear(embedding_dim, 2)

        self.cross_atlas_attention = CrossAtlasAttention(
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

    def compute_energy(self, logits):
        return -self.temperature * torch.logsumexp(
            logits / self.temperature,
            dim=-1,
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

        branch_logits = []
        for atlas_index, atlas_name in enumerate(self.atlas_names):
            branch_logits.append(
                self.classifiers[atlas_name](enhanced_embeddings[:, atlas_index])
            )

        stacked_logits = torch.stack(branch_logits, dim=1)
        energy = self.compute_energy(stacked_logits)
        atlas_weight = torch.softmax(-energy, dim=1)
        fusion_logits = torch.sum(
            atlas_weight.unsqueeze(-1) * stacked_logits,
            dim=1,
        )

        return {
            "fusion_logits": fusion_logits,
            "branch_logits": stacked_logits,
            "energy": energy,
            "atlas_weight": atlas_weight,
            "attention_weight": attention_weight,
            "branch_details": branch_details,
        }
