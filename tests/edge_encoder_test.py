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

    print("Edge encoder test passed.")
    print("Node summary:", tuple(summary.shape))
    print("Embedding:", tuple(embedding.shape))


if __name__ == "__main__":
    main()
