import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.losses import ProposalLoss


def main():
    labels = torch.tensor([0, 1])

    # The fused classifier is clearly better than each atlas classifier.
    output = {
        "fusion_logits": torch.tensor([[4.0, -4.0], [-4.0, 4.0]]),
        "branch_logits": torch.tensor(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        ),
    }

    literal_loss = ProposalLoss(
        lambda_branch=0.2,
        lambda_reg=0.1,
        margin=0.1,
        regularization_mode="proposal_literal",
    )(output, labels)
    fusion_better_loss = ProposalLoss(
        lambda_branch=0.2,
        lambda_reg=0.1,
        margin=0.1,
        regularization_mode="fusion_better",
    )(output, labels)
    weighted_loss = ProposalLoss(
        lambda_branch=0.2,
        lambda_reg=0.1,
        margin=0.1,
        regularization_mode="proposal_literal",
        class_weights=[1.1, 1.0],
    )(output, labels)

    assert literal_loss["regularization_loss"] > 0
    assert torch.isclose(
        fusion_better_loss["regularization_loss"],
        torch.tensor(0.0),
    )
    assert torch.isfinite(weighted_loss["loss"])

    print("Loss mode test passed.")
    print("proposal_literal regularization:", literal_loss["regularization_loss"].item())
    print("fusion_better regularization:", fusion_better_loss["regularization_loss"].item())
    print("weighted loss:", weighted_loss["loss"].item())


if __name__ == "__main__":
    main()
