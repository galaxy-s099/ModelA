import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adjacency(adjacency, eps=1e-6):
    """Compute D^{-1/2} A D^{-1/2} for a batch of non-negative matrices."""
    degree = adjacency.sum(dim=-1)
    degree_inv_sqrt = torch.pow(degree + eps, -0.5)
    return (
        degree_inv_sqrt.unsqueeze(-1)
        * adjacency
        * degree_inv_sqrt.unsqueeze(-2)
    )


def build_signed_adjacencies(fc, add_negative_self_loops=True):
    """
    Split a functional connectivity matrix into normalized positive and
    negative adjacency matrices.

    The proposal describes self-loop normalization for both channels.
    Keep this behavior enabled by default, while exposing a switch for
    controlled ablation experiments.
    """
    batch_size, num_nodes, _ = fc.shape
    identity = torch.eye(num_nodes, device=fc.device, dtype=fc.dtype)
    identity = identity.unsqueeze(0).expand(batch_size, -1, -1)

    # The diagonal of an FC matrix is not an inter-ROI edge. Remove it before
    # inserting the explicit graph self-loops required by the model.
    off_diagonal = fc * (1.0 - identity)
    adjacency_pos = torch.clamp(off_diagonal, min=0.0) + identity
    adjacency_neg = torch.clamp(-off_diagonal, min=0.0)

    if add_negative_self_loops:
        adjacency_neg = adjacency_neg + identity

    return normalize_adjacency(adjacency_pos), normalize_adjacency(adjacency_neg)


class InitialSignedLayer(nn.Module):
    """Initial H_pos^(1) and H_neg^(1) propagation from node features X."""

    def __init__(self, input_dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.pos_linear = nn.Linear(input_dim, hidden_dim)
        self.neg_linear = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features, adjacency_pos, adjacency_neg):
        h_pos = torch.bmm(adjacency_pos, node_features)
        h_neg = torch.bmm(adjacency_neg, node_features)
        h_pos = self.dropout(F.relu(self.pos_linear(h_pos)))
        h_neg = self.dropout(F.relu(self.neg_linear(h_neg)))
        return h_pos, h_neg


class DeepSignedLayer(nn.Module):
    """
    Balanced and unbalanced signed propagation:

    H_pos <- concat(A+ H_pos W_pp, A- H_neg W_nn)
    H_neg <- concat(A+ H_neg W_pn, A- H_pos W_np)
    """

    def __init__(self, hidden_dim, dropout=0.5):
        super().__init__()
        if hidden_dim % 2 != 0:
            raise ValueError("hidden_dim must be even for signed path concatenation")

        path_dim = hidden_dim // 2
        self.weight_pp = nn.Linear(hidden_dim, path_dim)
        self.weight_nn = nn.Linear(hidden_dim, path_dim)
        self.weight_pn = nn.Linear(hidden_dim, path_dim)
        self.weight_np = nn.Linear(hidden_dim, path_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_pos, h_neg, adjacency_pos, adjacency_neg):
        pos_from_pos = self.weight_pp(torch.bmm(adjacency_pos, h_pos))
        pos_from_neg = self.weight_nn(torch.bmm(adjacency_neg, h_neg))
        neg_from_neg = self.weight_pn(torch.bmm(adjacency_pos, h_neg))
        neg_from_pos = self.weight_np(torch.bmm(adjacency_neg, h_pos))

        next_pos = torch.cat([pos_from_pos, pos_from_neg], dim=-1)
        next_neg = torch.cat([neg_from_neg, neg_from_pos], dim=-1)

        return self.dropout(F.relu(next_pos)), self.dropout(F.relu(next_neg))


class SignedBalanceEncoder(nn.Module):
    """Atlas branch encoder that follows the proposal's two-state Signed GCN."""

    def __init__(
        self,
        num_nodes,
        feature_dim=None,
        hidden_dim=128,
        embedding_dim=128,
        num_layers=2,
        dropout=0.5,
        add_negative_self_loops=True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.num_nodes = num_nodes
        self.feature_dim = feature_dim or num_nodes
        self.add_negative_self_loops = add_negative_self_loops

        self.initial_layer = InitialSignedLayer(
            input_dim=self.feature_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.deep_layers = nn.ModuleList(
            [
                DeepSignedLayer(hidden_dim=hidden_dim, dropout=dropout)
                for _ in range(num_layers - 1)
            ]
        )
        self.graph_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, fc, node_features=None):
        if fc.ndim != 3 or fc.shape[1:] != (self.num_nodes, self.num_nodes):
            raise ValueError(
                f"Expected FC shape B x {self.num_nodes} x {self.num_nodes}, "
                f"received {tuple(fc.shape)}"
            )

        features = fc if node_features is None else node_features
        if features.shape[:2] != fc.shape[:2] or features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected node features B x {self.num_nodes} x {self.feature_dim}, "
                f"received {tuple(features.shape)}"
            )

        adjacency_pos, adjacency_neg = build_signed_adjacencies(
            fc,
            add_negative_self_loops=self.add_negative_self_loops,
        )
        h_pos, h_neg = self.initial_layer(features, adjacency_pos, adjacency_neg)

        for layer in self.deep_layers:
            h_pos, h_neg = layer(h_pos, h_neg, adjacency_pos, adjacency_neg)

        node_embedding = torch.cat([h_pos, h_neg], dim=-1)
        graph_embedding = self.graph_projection(node_embedding.mean(dim=1))

        return {
            "graph_embedding": graph_embedding,
            "node_embedding": node_embedding,
            "h_pos": h_pos,
            "h_neg": h_neg,
        }
