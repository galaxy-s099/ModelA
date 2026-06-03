import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.losses import ProposalLoss
from models.smaf_proposal_net import SMAFProposalNet


def make_fc(batch_size, num_nodes):
    matrix = torch.randn(batch_size, num_nodes, num_nodes)
    matrix = (matrix + matrix.transpose(-1, -2)) / 2
    matrix = torch.tanh(matrix)
    identity = torch.eye(num_nodes).unsqueeze(0)
    return matrix * (1.0 - identity) + identity


def main():
    torch.manual_seed(17)
    atlas_specs = {
        "aal": {"num_nodes": 8},
        "cc200": {"num_nodes": 10},
        "ho": {"num_nodes": 6},
    }
    model = SMAFProposalNet(
        atlas_specs=atlas_specs,
        hidden_dim=16,
        embedding_dim=12,
        num_signed_layers=2,
        dropout=0.1,
        fusion_mode="raw_concat",
    )
    batch = {
        atlas_name: make_fc(batch_size=4, num_nodes=spec["num_nodes"])
        for atlas_name, spec in atlas_specs.items()
    }
    labels = torch.tensor([0, 1, 0, 1])
    output = model(batch)

    assert output["fusion_logits"].shape == (4, 2)
    assert "attention_weight" not in output
    assert "branch_logits" not in output
    assert "energy" not in output
    assert "atlas_weight" not in output

    criterion = ProposalLoss(lambda_branch=0.0, lambda_reg=0.0, margin=0.1)
    loss_details = criterion(output, labels)
    loss_details["loss"].backward()

    assert torch.isfinite(loss_details["loss"])
    assert loss_details["branch_loss"].item() == 0
    assert loss_details["regularization_loss"].item() == 0

    print("Raw concat test passed.")
    print("Fusion logits:", tuple(output["fusion_logits"].shape))


if __name__ == "__main__":
    main()
