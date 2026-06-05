import torch
import torch.nn as nn


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


class EdgeBranchEncoder(nn.Module):
    """Signed edge-vector atlas encoder used by the stronger SMAF-Net v5 branch."""

    def __init__(self, input_dim, hidden_dim=256, embedding_dim=128, dropout=0.5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, fc):
        return self.encoder(fc_to_signed_edge_vector(fc))
