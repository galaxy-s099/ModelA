import torch
import torch.nn as nn
import torch.nn.functional as F


def fc_to_signed_edge_vector(fc):
    """
    Convert a batch of FC matrices into signed upper-triangle edge vectors.

    Positive and negative connections are kept as separate non-negative
    channels, following the v5 SMAF-Net edge branch.
    """
    if fc.ndim != 3 or fc.shape[-1] != fc.shape[-2]:
        raise ValueError(f"Expected FC shape B x N x N, received {tuple(fc.shape)}")

    _, num_nodes, _ = fc.shape
    edge_index = torch.triu_indices(
        num_nodes,
        num_nodes,
        offset=1,
        device=fc.device,
    )

    positive_edges = torch.clamp(fc, min=0.0)
    negative_edges = torch.clamp(-fc, min=0.0)

    positive_vector = positive_edges[:, edge_index[0], edge_index[1]]
    negative_vector = negative_edges[:, edge_index[0], edge_index[1]]
    return torch.cat([positive_vector, negative_vector], dim=-1)


def fc_to_node_summary(fc):
    """
    Summarize each ROI by positive, negative, and absolute connection strength.

    This keeps a lightweight node-level view alongside the flattened edge view.
    """
    if fc.ndim != 3 or fc.shape[-1] != fc.shape[-2]:
        raise ValueError(f"Expected FC shape B x N x N, received {tuple(fc.shape)}")

    _, num_nodes, _ = fc.shape
    normalizer = max(num_nodes - 1, 1)
    positive_strength = torch.clamp(fc, min=0.0).sum(dim=-1) / normalizer
    negative_strength = torch.clamp(-fc, min=0.0).sum(dim=-1) / normalizer
    absolute_strength = fc.abs().sum(dim=-1) / normalizer
    return torch.cat(
        [positive_strength, negative_strength, absolute_strength],
        dim=-1,
    )


def apply_fc_topk_sparsity(fc, topk_ratio):
    """
    Keep the strongest absolute FC edges per sample and zero out weaker edges.
    """
    if fc.ndim != 3 or fc.shape[-1] != fc.shape[-2]:
        raise ValueError(f"Expected FC shape B x N x N, received {tuple(fc.shape)}")
    if topk_ratio is None or topk_ratio >= 1.0:
        return fc
    if topk_ratio <= 0.0:
        raise ValueError("edge_topk_ratio must be greater than 0")

    batch_size, num_nodes, _ = fc.shape
    edge_index = torch.triu_indices(
        num_nodes,
        num_nodes,
        offset=1,
        device=fc.device,
    )
    edge_values = fc[:, edge_index[0], edge_index[1]]
    edge_count = edge_values.shape[-1]
    keep_count = max(1, int(round(edge_count * topk_ratio)))
    _, topk_indices = torch.topk(
        edge_values.abs(),
        k=keep_count,
        dim=-1,
    )
    mask_values = torch.zeros_like(edge_values)
    mask_values.scatter_(dim=-1, index=topk_indices, value=1.0)

    sparse_fc = torch.zeros_like(fc)
    sparse_edge_values = edge_values * mask_values
    sparse_fc[:, edge_index[0], edge_index[1]] = sparse_edge_values
    sparse_fc[:, edge_index[1], edge_index[0]] = sparse_edge_values
    return sparse_fc


class EdgeBranchEncoder(nn.Module):
    """Signed edge-vector atlas encoder used by the stronger SMAF-Net v5 branch."""

    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        embedding_dim=128,
        dropout=0.5,
        num_nodes=None,
        use_node_summary=False,
        node_summary_hidden_dim=64,
        node_summary_embedding_dim=32,
        use_edge_residual=False,
        edge_residual_hidden_dim=64,
        edge_residual_scale=0.25,
        edge_dropout=0.0,
        edge_topk_ratio=None,
    ):
        super().__init__()
        if use_node_summary and num_nodes is None:
            raise ValueError("num_nodes is required when use_node_summary is true")
        if edge_residual_scale < 0:
            raise ValueError("edge_residual_scale must be non-negative")
        if not 0.0 <= edge_dropout < 1.0:
            raise ValueError("edge_dropout must be in [0, 1)")
        if edge_topk_ratio is not None and not 0.0 < edge_topk_ratio <= 1.0:
            raise ValueError("edge_topk_ratio must be in (0, 1]")

        self.use_node_summary = use_node_summary
        self.use_edge_residual = use_edge_residual
        self.edge_residual_scale = edge_residual_scale
        self.edge_dropout = edge_dropout
        self.edge_topk_ratio = edge_topk_ratio
        self.edge_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.encoder = self.edge_encoder
        if self.use_node_summary:
            self.node_summary_encoder = nn.Sequential(
                nn.Linear(3 * int(num_nodes), node_summary_hidden_dim),
                nn.BatchNorm1d(node_summary_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(node_summary_hidden_dim, node_summary_embedding_dim),
                nn.BatchNorm1d(node_summary_embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.fusion = nn.Sequential(
                nn.Linear(embedding_dim + node_summary_embedding_dim, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.node_summary_encoder = None
            self.fusion = None

        if self.use_edge_residual:
            self.edge_residual = nn.Sequential(
                nn.Linear(input_dim, edge_residual_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(edge_residual_hidden_dim, embedding_dim),
            )
            nn.init.zeros_(self.edge_residual[-1].weight)
            nn.init.zeros_(self.edge_residual[-1].bias)
        else:
            self.edge_residual = None

    def forward(self, fc):
        fc = apply_fc_topk_sparsity(fc, self.edge_topk_ratio)
        edge_vector = fc_to_signed_edge_vector(fc)
        if self.edge_dropout > 0:
            edge_vector = F.dropout(
                edge_vector,
                p=self.edge_dropout,
                training=self.training,
            )
        edge_embedding = self.edge_encoder(edge_vector)
        if self.edge_residual is not None:
            edge_embedding = (
                edge_embedding
                + self.edge_residual_scale * self.edge_residual(edge_vector)
            )
        if not self.use_node_summary:
            return edge_embedding

        node_embedding = self.node_summary_encoder(fc_to_node_summary(fc))
        return self.fusion(torch.cat([edge_embedding, node_embedding], dim=-1))
