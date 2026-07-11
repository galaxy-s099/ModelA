from collections import OrderedDict

import torch
import torch.nn as nn

from models.edge_encoder import EdgeBranchEncoder


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, lambd):
        ctx.lambd = lambd
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambd=1.0):
        super().__init__()
        if lambd < 0:
            raise ValueError("gradient reversal lambda must be non-negative")
        self.lambd = float(lambd)

    def forward(self, inputs):
        return GradientReversalFunction.apply(inputs, self.lambd)


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
        use_branch_residual_correction=False,
        branch_residual_correction_scale=0.1,
        branch_residual_hidden_dim=32,
        use_consensus_gate=False,
        consensus_gate_scale=0.5,
        atlas_overrides=None,
        use_node_summary=False,
        node_summary_hidden_dim=64,
        node_summary_embedding_dim=32,
        use_edge_residual=False,
        edge_residual_hidden_dim=64,
        edge_residual_scale=0.25,
        edge_dropout=0.0,
        edge_topk_ratio=None,
        atlas_dropout=0.0,
        atlas_dropout_mode="single",
        use_logit_meta_fusion=False,
        logit_meta_hidden_dim=16,
        logit_meta_dropout=None,
        use_site_embedding=False,
        num_sites=None,
        site_embedding_dim=8,
        use_site_adversarial=False,
        site_adversarial_hidden_dim=64,
        site_adversarial_grl_lambda=1.0,
        use_tangent_branch=False,
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
        if branch_residual_correction_scale < 0:
            raise ValueError(
                "branch_residual_correction_scale must be non-negative"
            )
        if consensus_gate_scale < 0:
            raise ValueError("consensus_gate_scale must be non-negative")
        if not 0.0 <= atlas_dropout <= 1.0:
            raise ValueError("atlas_dropout must be in [0, 1]")
        if atlas_dropout_mode not in {"single", "independent"}:
            raise ValueError(
                "atlas_dropout_mode must be either single or independent"
            )
        if use_site_embedding:
            if num_sites is None or int(num_sites) <= 0:
                raise ValueError(
                    "num_sites must be provided when use_site_embedding is true"
                )
            if int(site_embedding_dim) <= 0:
                raise ValueError("site_embedding_dim must be positive")
        if use_site_adversarial:
            if num_sites is None or int(num_sites) <= 0:
                raise ValueError(
                    "num_sites must be provided when use_site_adversarial is true"
                )
            if int(site_adversarial_hidden_dim) <= 0:
                raise ValueError("site_adversarial_hidden_dim must be positive")

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
        self.use_branch_residual_correction = use_branch_residual_correction
        self.branch_residual_correction_scale = branch_residual_correction_scale
        self.branch_residual_hidden_dim = branch_residual_hidden_dim
        self.use_consensus_gate = use_consensus_gate
        self.consensus_gate_scale = consensus_gate_scale
        self.atlas_overrides = atlas_overrides or {}
        self.use_node_summary = use_node_summary
        self.node_summary_hidden_dim = node_summary_hidden_dim
        self.node_summary_embedding_dim = node_summary_embedding_dim
        self.use_edge_residual = use_edge_residual
        self.edge_residual_hidden_dim = edge_residual_hidden_dim
        self.edge_residual_scale = edge_residual_scale
        self.edge_dropout = edge_dropout
        self.edge_topk_ratio = edge_topk_ratio
        self.atlas_dropout = atlas_dropout
        self.atlas_dropout_mode = atlas_dropout_mode
        self.use_logit_meta_fusion = use_logit_meta_fusion
        self.logit_meta_hidden_dim = logit_meta_hidden_dim
        self.logit_meta_dropout = (
            dropout
            if logit_meta_dropout is None
            else logit_meta_dropout
        )
        self.use_site_embedding = use_site_embedding
        self.use_site_adversarial = use_site_adversarial
        self.num_sites = None if num_sites is None else int(num_sites)
        self.site_embedding_dim = int(site_embedding_dim)
        self.site_adversarial_hidden_dim = int(site_adversarial_hidden_dim)
        self.site_adversarial_grl_lambda = float(site_adversarial_grl_lambda)
        self.use_tangent_branch = bool(use_tangent_branch)
        self.embedding_dims = {}
        self.total_embedding_dim = 0
        self.total_conditioned_embedding_dim = 0

        if self.use_site_embedding:
            self.site_embedding = nn.Embedding(
                self.num_sites,
                self.site_embedding_dim,
            )
        else:
            self.site_embedding = None
        if self.use_site_adversarial:
            self.gradient_reversal = GradientReversal(
                self.site_adversarial_grl_lambda,
            )
        else:
            self.gradient_reversal = None

        self.encoders = nn.ModuleDict()
        self.tangent_encoders = nn.ModuleDict()
        self.tangent_adapters = nn.ModuleDict()
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
            atlas_use_edge_residual = bool(
                atlas_config.get("use_edge_residual", use_edge_residual)
            )
            atlas_edge_residual_hidden_dim = int(
                atlas_config.get(
                    "edge_residual_hidden_dim",
                    edge_residual_hidden_dim,
                )
            )
            atlas_edge_residual_scale = float(
                atlas_config.get("edge_residual_scale", edge_residual_scale)
            )
            atlas_edge_dropout = float(
                atlas_config.get("edge_dropout", edge_dropout)
            )
            atlas_edge_topk_ratio = atlas_config.get(
                "edge_topk_ratio",
                edge_topk_ratio,
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
                use_edge_residual=atlas_use_edge_residual,
                edge_residual_hidden_dim=atlas_edge_residual_hidden_dim,
                edge_residual_scale=atlas_edge_residual_scale,
                edge_dropout=atlas_edge_dropout,
                edge_topk_ratio=atlas_edge_topk_ratio,
            )
            if self.use_tangent_branch:
                self.tangent_encoders[atlas_name] = EdgeBranchEncoder(
                    input_dim=input_dim,
                    hidden_dim=atlas_hidden_dim,
                    embedding_dim=atlas_embedding_dim,
                    dropout=atlas_dropout,
                    num_nodes=num_nodes,
                    use_node_summary=False,
                    use_edge_residual=False,
                    edge_dropout=0.0,
                    edge_topk_ratio=None,
                )
                adapter = nn.Linear(atlas_embedding_dim * 2, atlas_embedding_dim)
                # Start exactly as the v6.6 raw-FC branch. The tangent pathway
                # is introduced only as its adapter learns non-zero weights.
                with torch.no_grad():
                    adapter.weight.zero_()
                    adapter.bias.zero_()
                    adapter.weight[:, :atlas_embedding_dim].copy_(
                        torch.eye(
                            atlas_embedding_dim,
                            dtype=adapter.weight.dtype,
                        )
                    )
                self.tangent_adapters[atlas_name] = adapter
            classifier_input_dim = atlas_embedding_dim
            if self.use_site_embedding:
                classifier_input_dim += self.site_embedding_dim
            self.classifiers[atlas_name] = nn.Linear(classifier_input_dim, 2)
            self.embedding_dims[atlas_name] = atlas_embedding_dim
            self.total_embedding_dim += atlas_embedding_dim
            self.total_conditioned_embedding_dim += classifier_input_dim

        if self.use_atlas_prior:
            self.atlas_prior = nn.Parameter(torch.zeros(self.num_atlases))
        else:
            self.register_parameter("atlas_prior", None)

        if self.use_sample_gate:
            self.sample_gate = nn.Sequential(
                nn.Linear(self.total_conditioned_embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.num_atlases),
            )
            nn.init.zeros_(self.sample_gate[-1].weight)
            nn.init.zeros_(self.sample_gate[-1].bias)
        else:
            self.sample_gate = None

        if self.use_site_adversarial:
            self.site_classifier = nn.Sequential(
                nn.LayerNorm(self.total_embedding_dim),
                nn.Linear(self.total_embedding_dim, self.site_adversarial_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.site_adversarial_hidden_dim, self.num_sites),
            )
        else:
            self.site_classifier = None

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

        if self.use_branch_residual_correction:
            branch_residual_input_dim = self.num_atlases * 3
            self.branch_residual_correction = nn.Sequential(
                nn.LayerNorm(branch_residual_input_dim),
                nn.Linear(
                    branch_residual_input_dim,
                    branch_residual_hidden_dim,
                ),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(branch_residual_hidden_dim, 2),
            )
            nn.init.zeros_(self.branch_residual_correction[-1].weight)
            nn.init.zeros_(self.branch_residual_correction[-1].bias)
        else:
            self.branch_residual_correction = None

        if self.use_logit_meta_fusion:
            meta_input_dim = self.num_atlases * 3
            self.logit_meta_fusion = nn.Sequential(
                nn.LayerNorm(meta_input_dim),
                nn.Linear(meta_input_dim, logit_meta_hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.logit_meta_dropout),
                nn.Linear(logit_meta_hidden_dim, 2),
            )
        else:
            self.logit_meta_fusion = None

    def compute_energy(self, logits):
        return -self.temperature * torch.logsumexp(
            logits / self.temperature,
            dim=-1,
        )

    def sample_atlas_keep_mask(self, energy_score):
        if not self.training or self.atlas_dropout <= 0:
            return None

        batch_size = energy_score.shape[0]
        device = energy_score.device
        if self.atlas_dropout_mode == "single":
            keep_mask = torch.ones(
                batch_size,
                self.num_atlases,
                dtype=torch.bool,
                device=device,
            )
            drop_sample = torch.rand(batch_size, device=device) < self.atlas_dropout
            drop_index = torch.randint(
                low=0,
                high=self.num_atlases,
                size=(batch_size,),
                device=device,
            )
            keep_mask[drop_sample, drop_index[drop_sample]] = False
            return keep_mask

        keep_mask = (
            torch.rand(
                batch_size,
                self.num_atlases,
                device=device,
            )
            >= self.atlas_dropout
        )
        all_dropped = ~keep_mask.any(dim=1)
        if all_dropped.any():
            restore_index = torch.randint(
                low=0,
                high=self.num_atlases,
                size=(int(all_dropped.sum().item()),),
                device=device,
            )
            keep_mask[all_dropped] = False
            keep_mask[all_dropped, restore_index] = True
        return keep_mask

    def forward(self, batch):
        branch_logits = []
        branch_details = {}
        graph_embeddings = []
        conditioned_graph_embeddings = []
        site_embedding = None
        if self.site_embedding is not None:
            if "site" not in batch:
                raise ValueError(
                    "Batch is missing site ids while use_site_embedding is true"
                )
            site_embedding = self.site_embedding(batch["site"])
        for atlas_name in self.atlas_names:
            raw_graph_embedding = self.encoders[atlas_name](batch[atlas_name])
            graph_embedding = raw_graph_embedding
            branch_details[atlas_name] = {
                "raw_graph_embedding": raw_graph_embedding,
            }
            if self.use_tangent_branch:
                tangent_key = f"{atlas_name}_tangent"
                if tangent_key not in batch:
                    raise ValueError(
                        f"Batch is missing {tangent_key} while use_tangent_branch "
                        "is true"
                    )
                tangent_embedding = self.tangent_encoders[atlas_name](
                    batch[tangent_key]
                )
                graph_embedding = self.tangent_adapters[atlas_name](
                    torch.cat([raw_graph_embedding, tangent_embedding], dim=-1)
                )
                branch_details[atlas_name]["tangent_embedding"] = tangent_embedding
            branch_details[atlas_name]["graph_embedding"] = graph_embedding
            graph_embeddings.append(graph_embedding)
            classifier_input = graph_embedding
            if site_embedding is not None:
                classifier_input = torch.cat(
                    [graph_embedding, site_embedding],
                    dim=-1,
                )
            conditioned_graph_embeddings.append(classifier_input)
            branch_logits.append(self.classifiers[atlas_name](classifier_input))

        stacked_logits = torch.stack(branch_logits, dim=1)
        energy = self.compute_energy(stacked_logits)
        energy_score = -energy
        if self.atlas_prior is not None:
            energy_score = energy_score + self.atlas_prior.unsqueeze(0)
        consensus_disagreement = None
        if self.use_consensus_gate and self.consensus_gate_scale > 0:
            branch_probabilities = torch.softmax(stacked_logits, dim=-1)[..., 1]
            consensus_probability = branch_probabilities.mean(
                dim=1,
                keepdim=True,
            )
            consensus_disagreement = torch.abs(
                branch_probabilities - consensus_probability
            )
            energy_score = (
                energy_score
                - self.consensus_gate_scale * consensus_disagreement
            )
        base_energy_score = energy_score
        sample_gate_logits = None
        if self.sample_gate is not None:
            sample_gate_input = torch.cat(conditioned_graph_embeddings, dim=-1)
            sample_gate_logits = self.sample_gate(sample_gate_input)
            energy_score = energy_score + self.sample_gate_scale * sample_gate_logits

        atlas_keep_mask = self.sample_atlas_keep_mask(energy_score)
        gated_energy_score = energy_score
        if atlas_keep_mask is not None:
            gated_energy_score = energy_score.masked_fill(
                ~atlas_keep_mask,
                -1e9,
            )

        base_atlas_weight = torch.softmax(base_energy_score, dim=1)
        gated_atlas_weight = torch.softmax(gated_energy_score, dim=1)
        if self.use_dual_energy_blend:
            atlas_weight = (
                self.dual_energy_blend_alpha * gated_atlas_weight
                + (1.0 - self.dual_energy_blend_alpha) * base_atlas_weight
            )
            if atlas_keep_mask is not None:
                atlas_weight = atlas_weight * atlas_keep_mask.float()
                atlas_weight = atlas_weight / atlas_weight.sum(
                    dim=1,
                    keepdim=True,
                ).clamp_min(1e-8)
        else:
            atlas_weight = gated_atlas_weight
        weighted_logits = torch.sum(
            atlas_weight.unsqueeze(-1) * stacked_logits,
            dim=1,
        )
        residual_logits = None
        shared_correction_logits = None
        branch_residual_correction_logits = None
        logit_meta_fusion_logits = None
        site_logits = None
        fusion_logits = weighted_logits
        if self.site_classifier is not None:
            if "site" not in batch:
                raise ValueError(
                    "Batch is missing site ids while use_site_adversarial is true"
                )
            site_input = torch.cat(graph_embeddings, dim=-1)
            site_logits = self.site_classifier(self.gradient_reversal(site_input))
        if self.logit_meta_fusion is not None:
            meta_input = torch.cat(
                [
                    stacked_logits.reshape(stacked_logits.shape[0], -1),
                    atlas_weight,
                ],
                dim=-1,
            )
            logit_meta_fusion_logits = self.logit_meta_fusion(meta_input)
            fusion_logits = logit_meta_fusion_logits
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
        if self.branch_residual_correction is not None:
            branch_residual_input = torch.cat(
                [
                    stacked_logits.reshape(stacked_logits.shape[0], -1),
                    atlas_weight,
                ],
                dim=-1,
            )
            branch_residual_correction_logits = self.branch_residual_correction(
                branch_residual_input
            )
            fusion_logits = (
                fusion_logits
                + self.branch_residual_correction_scale
                * branch_residual_correction_logits
            )

        return {
            "fusion_logits": fusion_logits,
            "weighted_logits": weighted_logits,
            "residual_logits": residual_logits,
            "shared_correction_logits": shared_correction_logits,
            "branch_residual_correction_logits": branch_residual_correction_logits,
            "logit_meta_fusion_logits": logit_meta_fusion_logits,
            "site_logits": site_logits,
            "branch_logits": stacked_logits,
            "energy": energy,
            "atlas_weight": atlas_weight,
            "base_atlas_weight": base_atlas_weight,
            "gated_atlas_weight": gated_atlas_weight,
            "atlas_keep_mask": atlas_keep_mask,
            "consensus_disagreement": consensus_disagreement,
            "atlas_prior": self.atlas_prior,
            "sample_gate_logits": sample_gate_logits,
            "branch_details": branch_details,
        }
