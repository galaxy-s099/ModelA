from collections import OrderedDict

import torch
import torch.nn as nn

from models.cross_atlas_attention import CrossAtlasAttention
from models.signed_balance_encoder import SignedBalanceEncoder


class SMAFProposalNet(nn.Module):
    """
    Signed multi-atlas network with controlled fusion ablations:

    1. Two-state signed graph propagation for each atlas.
    2. Optional cross-atlas feature enhancement at graph-embedding level.
    3. Energy-aware, attention-concat, or raw-concat classification.
    """

    def __init__(
        self,
        atlas_specs,
        hidden_dim=128,
        embedding_dim=128,
        num_signed_layers=2,
        dropout=0.5,
        temperature=1.0,
        add_negative_self_loops=True,
        fusion_mode="energy_decision",
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        supported_fusion_modes = {
            "energy_decision",
            "feature_concat",
            "raw_concat",
        }
        if fusion_mode not in supported_fusion_modes:
            raise ValueError(
                f"Unsupported fusion_mode: {fusion_mode}. "
                f"Expected one of {sorted(supported_fusion_modes)}"
            )

        self.atlas_specs = OrderedDict(atlas_specs)
        self.atlas_names = list(self.atlas_specs.keys())
        self.temperature = temperature
        self.fusion_mode = fusion_mode

        self.encoders = nn.ModuleDict()
        self.classifiers = nn.ModuleDict()

        for atlas_name, spec in self.atlas_specs.items():
            num_nodes = int(spec["num_nodes"])
            feature_dim = int(spec.get("feature_dim", num_nodes))
            self.encoders[atlas_name] = SignedBalanceEncoder(
                num_nodes=num_nodes,
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
                num_layers=num_signed_layers,
                dropout=dropout,
                add_negative_self_loops=add_negative_self_loops,
            )
            if self.fusion_mode == "energy_decision":
                self.classifiers[atlas_name] = nn.Linear(embedding_dim, 2)

        if self.fusion_mode in {"energy_decision", "feature_concat"}:
            self.cross_atlas_attention = CrossAtlasAttention(
                embedding_dim=embedding_dim,
                dropout=dropout,
            )
        if self.fusion_mode in {"feature_concat", "raw_concat"}:
            self.fusion_classifier = nn.Sequential(
                nn.Linear(embedding_dim * len(self.atlas_names), hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
            )

    def compute_energy(self, logits):
        return -self.temperature * torch.logsumexp(
            logits / self.temperature,
            dim=-1,
        )

    def forward(self, batch):
        branch_details = {}
        graph_embeddings = []

        for atlas_name in self.atlas_names:
            node_features = batch.get(f"{atlas_name}_features")
            details = self.encoders[atlas_name](
                batch[atlas_name],
                node_features=node_features,
            )
            branch_details[atlas_name] = details
            graph_embeddings.append(details["graph_embedding"])

        # B x M x D, where M is the number of atlases.
        stacked_embeddings = torch.stack(graph_embeddings, dim=1)

        output = {"branch_details": branch_details}
        if self.fusion_mode == "raw_concat":
            flattened_embeddings = stacked_embeddings.reshape(
                stacked_embeddings.shape[0],
                -1,
            )
            output["fusion_logits"] = self.fusion_classifier(flattened_embeddings)
            return output

        enhanced_embeddings, attention_weight = self.cross_atlas_attention(
            stacked_embeddings
        )

        output["attention_weight"] = attention_weight
        if self.fusion_mode == "feature_concat":
            flattened_embeddings = enhanced_embeddings.reshape(
                enhanced_embeddings.shape[0],
                -1,
            )
            output["fusion_logits"] = self.fusion_classifier(flattened_embeddings)
            return output

        branch_logits = []
        for atlas_index, atlas_name in enumerate(self.atlas_names):
            branch_logits.append(
                self.classifiers[atlas_name](enhanced_embeddings[:, atlas_index])
            )

        # B x M x C
        stacked_logits = torch.stack(branch_logits, dim=1)
        energy = self.compute_energy(stacked_logits)
        atlas_weight = torch.softmax(-energy, dim=1)
        output.update(
            {
                "fusion_logits": torch.sum(
                    atlas_weight.unsqueeze(-1) * stacked_logits,
                    dim=1,
                ),
                "branch_logits": stacked_logits,
                "energy": energy,
                "atlas_weight": atlas_weight,
            }
        )
        return output
