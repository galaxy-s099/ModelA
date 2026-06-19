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
        sample_gate_scale=1.0,
        use_residual_classifier=False,
        residual_classifier_scale=0.5,
        use_dual_energy_blend=False,
        dual_energy_blend_alpha=0.5,
        use_shared_correction=False,
        shared_correction_scale=0.25,
        atlas_overrides=None,
        use_node_summary=False,
        node_summary_hidden_dim=64,
        node_summary_embedding_dim=32,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        if sample_gate_scale < 0:
            raise ValueError("sample_gate_scale must be non-negative")
        if residual_classifier_scale < 0:
            raise ValueError("residual_classifier_scale must be non-negative")
        if not 0.0 <= dual_energy_blend_alpha <= 1.0:
            raise ValueError("dual_energy_blend_alpha must be in [0, 1]")
        if shared_correction_scale < 0:
            raise ValueError("shared_correction_scale must be non-negative")

        self.atlas_specs = OrderedDict(atlas_specs)
        self.atlas_names = list(self.atlas_specs.keys())
        self.num_atlases = len(self.atlas_names)
        self.temperature = temperature
        self.use_atlas_prior = use_atlas_prior
        self.use_sample_gate = use_sample_gate
        self.sample_gate_scale = sample_gate_scale
        self.use_residual_classifier = use_residual_classifier
        self.residual_classifier_scale = residual_classifier_scale
        self.use_dual_energy_blend = use_dual_energy_blend
        self.dual_energy_blend_alpha = dual_energy_blend_alpha
        self.use_shared_correction = use_shared_correction
        self.shared_correction_scale = shared_correction_scale
        self.atlas_overrides = atlas_overrides or {}
        self.use_node_summary = use_node_summary
        self.node_summary_hidden_dim = node_summary_hidden_dim
        self.node_summary_embedding_dim = node_summary_embedding_dim
        self.embedding_dims = {}
        self.total_embedding_dim = 0

        self.encoders = nn.ModuleDict()
        self.classifiers = nn.ModuleDict()
        for atlas_name, spec in self.atlas_specs.items():
            atlas_config = self.atlas_overrides.get(atlas_name, {})
            atlas_hidden_dim = int(atlas_config.get("hidden_dim", hidden_dim))
            atlas_embedding_dim = int(
                atlas_config.get("embedding_dim", embedding_dim)
            )
            atlas_dropout = float(atlas_config.get("dropout", dropout))
            atlas_use_node_summary = bool(
                atlas_config.get("use_node_summary", use_node_summary)
            )
            atlas_node_summary_hidden_dim = int(
                atlas_config.get(
                    "node_summary_hidden_dim",
                    node_summary_hidden_dim,
                )
            )
            atlas_node_summary_embedding_dim = int(
                atlas_config.get(
                    "node_summary_embedding_dim",
                    node_summary_embedding_dim,
                )
            )
            num_nodes = int(spec["num_nodes"])
            input_dim = num_nodes * (num_nodes - 1)
            self.encoders[atlas_name] = EdgeBranchEncoder(
                input_dim=input_dim,
                hidden_dim=atlas_hidden_dim,
                embedding_dim=atlas_embedding_dim,
                dropout=atlas_dropout,
                num_nodes=num_nodes,
                use_node_summary=atlas_use_node_summary,
                node_summary_hidden_dim=atlas_node_summary_hidden_dim,
                node_summary_embedding_dim=atlas_node_summary_embedding_dim,
            )
            self.classifiers[atlas_name] = nn.Linear(atlas_embedding_dim, 2)
            self.embedding_dims[atlas_name] = atlas_embedding_dim
            self.total_embedding_dim += atlas_embedding_dim

        if self.use_atlas_prior:
            self.atlas_prior = nn.Parameter(torch.zeros(self.num_atlases))
        else:
            self.register_parameter("atlas_prior", None)

        if self.use_sample_gate:
            self.sample_gate = nn.Sequential(
                nn.Linear(self.total_embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.num_atlases),
            )
            nn.init.zeros_(self.sample_gate[-1].weight)
            nn.init.zeros_(self.sample_gate[-1].bias)
        else:
            self.sample_gate = None

        if self.use_residual_classifier:
            if len(set(self.embedding_dims.values())) != 1:
                raise ValueError(
                    "use_residual_classifier requires equal atlas embedding dims"
                )
            shared_embedding_dim = next(iter(self.embedding_dims.values()))
            residual_input_dim = shared_embedding_dim * (self.num_atlases + 1)
            self.residual_classifier = nn.Sequential(
                nn.Linear(residual_input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 2),
            )
            nn.init.zeros_(self.residual_classifier[-1].weight)
            nn.init.zeros_(self.residual_classifier[-1].bias)
        else:
            self.residual_classifier = None

        if self.use_shared_correction:
            correction_input_dim = self.total_embedding_dim
            correction_hidden_dim = max(64, embedding_dim)
            self.shared_correction = nn.Sequential(
                nn.LayerNorm(correction_input_dim),
                nn.Linear(correction_input_dim, correction_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(correction_hidden_dim, 2),
            )
            nn.init.zeros_(self.shared_correction[-1].weight)
            nn.init.zeros_(self.shared_correction[-1].bias)
        else:
            self.shared_correction = None

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
        base_energy_score = energy_score
        sample_gate_logits = None
        if self.sample_gate is not None:
            sample_gate_input = torch.cat(graph_embeddings, dim=-1)
            sample_gate_logits = self.sample_gate(sample_gate_input)
            energy_score = energy_score + self.sample_gate_scale * sample_gate_logits

        base_atlas_weight = torch.softmax(base_energy_score, dim=1)
        gated_atlas_weight = torch.softmax(energy_score, dim=1)
        if self.use_dual_energy_blend:
            atlas_weight = (
                self.dual_energy_blend_alpha * gated_atlas_weight
                + (1.0 - self.dual_energy_blend_alpha) * base_atlas_weight
            )
        else:
            atlas_weight = gated_atlas_weight
        weighted_logits = torch.sum(
            atlas_weight.unsqueeze(-1) * stacked_logits,
            dim=1,
        )
        residual_logits = None
        shared_correction_logits = None
        fusion_logits = weighted_logits
        if self.residual_classifier is not None:
            stacked_embeddings = torch.stack(graph_embeddings, dim=1)
            gated_embedding = torch.sum(
                atlas_weight.unsqueeze(-1) * stacked_embeddings,
                dim=1,
            )
            residual_input = torch.cat(
                [
                    stacked_embeddings.reshape(stacked_embeddings.shape[0], -1),
                    gated_embedding,
                ],
                dim=-1,
            )
            residual_logits = self.residual_classifier(residual_input)
            fusion_logits = (
                weighted_logits
                + self.residual_classifier_scale * residual_logits
            )
        if self.shared_correction is not None:
            shared_input = torch.cat(graph_embeddings, dim=-1)
            shared_correction_logits = self.shared_correction(shared_input)
            fusion_logits = (
                fusion_logits
                + self.shared_correction_scale * shared_correction_logits
            )

        return {
            "fusion_logits": fusion_logits,
            "weighted_logits": weighted_logits,
            "residual_logits": residual_logits,
            "shared_correction_logits": shared_correction_logits,
            "branch_logits": stacked_logits,
            "energy": energy,
            "atlas_weight": atlas_weight,
            "base_atlas_weight": base_atlas_weight,
            "gated_atlas_weight": gated_atlas_weight,
            "atlas_prior": self.atlas_prior,
            "sample_gate_logits": sample_gate_logits,
            "branch_details": branch_details,
        }
