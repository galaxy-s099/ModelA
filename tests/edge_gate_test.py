import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.signed_balance_encoder import SignedBalanceEncoder, build_signed_adjacencies


def make_fc(batch_size, num_nodes):
    matrix = torch.randn(batch_size, num_nodes, num_nodes)
    matrix = (matrix + matrix.transpose(-1, -2)) / 2
    matrix = torch.tanh(matrix)
    identity = torch.eye(num_nodes).unsqueeze(0)
    return matrix * (1.0 - identity)


def main():
    torch.manual_seed(17)
    fc = make_fc(batch_size=3, num_nodes=8)
    encoder = SignedBalanceEncoder(
        num_nodes=8,
        hidden_dim=16,
        embedding_dim=12,
        num_layers=2,
        dropout=0.1,
        use_residual=True,
        use_edge_gate=True,
        edge_gate_scale=0.5,
    )

    baseline_pos, baseline_neg = build_signed_adjacencies(fc)
    gated_pos, gated_neg = build_signed_adjacencies(
        fc,
        positive_edge_gate=encoder.positive_edge_gate.detach(),
        negative_edge_gate=encoder.negative_edge_gate.detach(),
        edge_gate_scale=encoder.edge_gate_scale,
    )
    assert torch.allclose(baseline_pos, gated_pos, atol=1e-7)
    assert torch.allclose(baseline_neg, gated_neg, atol=1e-7)

    output = encoder(fc)
    loss = output["graph_embedding"].sum()
    loss.backward()

    assert encoder.positive_edge_gate.grad is not None
    assert encoder.negative_edge_gate.grad is not None
    assert torch.isfinite(encoder.positive_edge_gate.grad).all()
    assert torch.isfinite(encoder.negative_edge_gate.grad).all()
    assert encoder.positive_edge_gate.grad.abs().sum() > 0
    assert encoder.negative_edge_gate.grad.abs().sum() > 0

    print("Edge gate test passed.")
    print("Graph embedding:", tuple(output["graph_embedding"].shape))


if __name__ == "__main__":
    main()
