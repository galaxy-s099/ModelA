import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.edge_encoder import EdgeBranchEncoder, fc_to_node_summary


def main():
    fc = torch.tensor(
        [
            [
                [0.0, 1.0, -2.0],
                [1.0, 0.0, 3.0],
                [-2.0, 3.0, 0.0],
            ]
        ]
    )
    summary = fc_to_node_summary(fc)
    expected = torch.tensor([[0.5, 2.0, 1.5, 1.0, 0.0, 1.0, 1.5, 2.0, 2.5]])
    assert torch.allclose(summary, expected)

    encoder = EdgeBranchEncoder(
        input_dim=6,
        hidden_dim=8,
        embedding_dim=5,
        dropout=0.1,
        num_nodes=3,
        use_node_summary=True,
        node_summary_hidden_dim=6,
        node_summary_embedding_dim=4,
    )
    encoder.train()
    batch = fc.repeat(4, 1, 1)
    embedding = encoder(batch)
    assert embedding.shape == (4, 5)
    assert torch.isfinite(embedding).all()

    baseline_encoder = EdgeBranchEncoder(
        input_dim=6,
        hidden_dim=8,
        embedding_dim=5,
        dropout=0.1,
    )
    residual_encoder = EdgeBranchEncoder(
        input_dim=6,
        hidden_dim=8,
        embedding_dim=5,
        dropout=0.1,
        use_edge_residual=True,
        edge_residual_hidden_dim=4,
        edge_residual_scale=0.25,
    )
    residual_encoder.load_state_dict(baseline_encoder.state_dict(), strict=False)
    baseline_encoder.eval()
    residual_encoder.eval()
    with torch.no_grad():
        baseline_embedding = baseline_encoder(batch)
        residual_embedding = residual_encoder(batch)
    assert torch.allclose(baseline_embedding, residual_embedding, atol=1e-6)

    residual_encoder.train()
    residual_embedding = residual_encoder(batch)
    residual_embedding.sum().backward()
    assert residual_encoder.edge_residual[-1].weight.grad is not None
    assert torch.isfinite(residual_encoder.edge_residual[-1].weight.grad).all()
    assert residual_encoder.edge_residual[-1].weight.grad.abs().sum() > 0

    zero_dropout_encoder = EdgeBranchEncoder(
        input_dim=6,
        hidden_dim=8,
        embedding_dim=5,
        dropout=0.1,
        edge_dropout=0.0,
    )
    zero_dropout_encoder.load_state_dict(baseline_encoder.state_dict())
    baseline_encoder.eval()
    zero_dropout_encoder.eval()
    with torch.no_grad():
        baseline_embedding = baseline_encoder(batch)
        zero_dropout_embedding = zero_dropout_encoder(batch)
    assert torch.allclose(
        baseline_embedding,
        zero_dropout_embedding,
        atol=1e-6,
    )

    edge_dropout_encoder = EdgeBranchEncoder(
        input_dim=6,
        hidden_dim=8,
        embedding_dim=5,
        dropout=0.1,
        edge_dropout=0.2,
    )
    edge_dropout_encoder.train()
    dropout_embedding = edge_dropout_encoder(batch)
    dropout_embedding.sum().backward()
    assert dropout_embedding.shape == (4, 5)
    assert torch.isfinite(dropout_embedding).all()

    print("Edge encoder test passed.")
    print("Node summary:", tuple(summary.shape))
    print("Embedding:", tuple(embedding.shape))


if __name__ == "__main__":
    main()
