from collections import OrderedDict

import torch
import torch.nn as nn

from models.edge_encoder import EdgeBranchEncoder


class SMAFEdgeEnergyNet(nn.Module):
    """
    v2.2 hybrid model:

    1. Use the SMAF-Net v5 signed edge branch encoder per atlas.
    2. Produce branch logits directly from each atlas embedding.
    3. Fuse branch decisions with proposal-style energy weights.

    This intentionally removes cross-atlas attention to test whether attention
    perturbs the strong edge-branch representation.
    """

    def __init__(
        self,
        atlas_specs,
        hidden_dim=256,
        embedding_dim=128,
        dropout=0.5,
        temperature=1.0,
        use_atlas_prior=False,
        use_sample_gate=False,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")

        self.atlas_specs = OrderedDict(atlas_specs)
        self.atlas_names = list(self.atlas_specs.keys())
        self.num_atlases = len(self.atlas_names)
        self.temperature = temperature
        self.use_atlas_prior = use_atlas_prior
        self.use_sample_gate = use_sample_gate

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

        if self.use_atlas_prior:
            self.atlas_prior = nn.Parameter(torch.zeros(self.num_atlases))
        else:
            self.register_parameter("atlas_prior", None)

        if self.use_sample_gate:
            self.sample_gate = nn.Sequential(
                nn.Linear(embedding_dim * self.num_atlases, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.num_atlases),
            )
            nn.init.zeros_(self.sample_gate[-1].weight)
            nn.init.zeros_(self.sample_gate[-1].bias)
        else:
            self.sample_gate = None

    def compute_energy(self, logits):
        return -self.temperature * torch.logsumexp(
            logits / self.temperature,
            dim=-1,
        )

    def forward(self, batch):
        branch_logits = []
        branch_details = {}
        graph_embeddings = []
        for atlas_name in self.atlas_names:
            graph_embedding = self.encoders[atlas_name](batch[atlas_name])
            branch_details[atlas_name] = {"graph_embedding": graph_embedding}
            graph_embeddings.append(graph_embedding)
            branch_logits.append(self.classifiers[atlas_name](graph_embedding))

        stacked_logits = torch.stack(branch_logits, dim=1)
        energy = self.compute_energy(stacked_logits)
        energy_score = -energy
        if self.atlas_prior is not None:
            energy_score = energy_score + self.atlas_prior.unsqueeze(0)
        sample_gate_logits = None
        if self.sample_gate is not None:
            sample_gate_input = torch.cat(graph_embeddings, dim=-1)
            sample_gate_logits = self.sample_gate(sample_gate_input)
            energy_score = energy_score + sample_gate_logits

        atlas_weight = torch.softmax(energy_score, dim=1)
        fusion_logits = torch.sum(
            atlas_weight.unsqueeze(-1) * stacked_logits,
            dim=1,
        )

        return {
            "fusion_logits": fusion_logits,
            "branch_logits": stacked_logits,
            "energy": energy,
            "atlas_weight": atlas_weight,
            "atlas_prior": self.atlas_prior,
            "sample_gate_logits": sample_gate_logits,
            "branch_details": branch_details,
        }
