import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.losses import ProposalLoss
from models.smaf_edge_gated_proposal_net import SMAFEdgeGatedProposalNet


def make_fc(batch_size, num_nodes):
    matrix = torch.randn(batch_size, num_nodes, num_nodes)
    matrix = (matrix + matrix.transpose(-1, -2)) / 2
    matrix = torch.tanh(matrix)
    identity = torch.eye(num_nodes).unsqueeze(0)
    return matrix * (1.0 - identity)


def main():
    torch.manual_seed(29)
    atlas_specs = {
        "aal": {"num_nodes": 8},
        "cc200": {"num_nodes": 10},
        "ho": {"num_nodes": 6},
    }
    model = SMAFEdgeGatedProposalNet(
        atlas_specs=atlas_specs,
        hidden_dim=16,
        embedding_dim=12,
        dropout=0.1,
    )
    batch = {
        atlas_name: make_fc(batch_size=4, num_nodes=spec["num_nodes"])
        for atlas_name, spec in atlas_specs.items()
    }
    output = model(batch)

    assert output["fusion_logits"].shape == (4, 2)
    assert "branch_logits" not in output
    assert output["atlas_weight"].shape == (4, 3)
    assert output["attention_weight"].shape == (4, 3, 3)
    assert torch.allclose(
        output["atlas_weight"].sum(dim=1),
        torch.ones(4),
        atol=1e-6,
    )

    labels = torch.tensor([0, 1, 0, 1])
    criterion = ProposalLoss(lambda_branch=0.0, lambda_reg=0.0, margin=0.1)
    loss_details = criterion(output, labels)
    loss_details["loss"].backward()
    assert torch.isfinite(loss_details["loss"])

    print("Edge gated proposal test passed.")
    print("Fusion logits:", tuple(output["fusion_logits"].shape))
    print("Atlas weights:", tuple(output["atlas_weight"].shape))


if __name__ == "__main__":
    main()
