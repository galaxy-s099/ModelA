import torch
import torch.nn as nn
import torch.nn.functional as F


class ProposalLoss(nn.Module):
    """
    L = L_fusion + lambda_branch * sum(L_i) + lambda_reg * L_reg

    L_reg follows the proposal literally:
    sum_i max(0, L_i - L_fusion + margin)
    """

    def __init__(self, lambda_branch=0.2, lambda_reg=0.1, margin=0.1):
        super().__init__()
        self.lambda_branch = lambda_branch
        self.lambda_reg = lambda_reg
        self.margin = margin

    def forward(self, output, labels):
        fusion_loss = F.cross_entropy(output["fusion_logits"], labels)
        branch_losses = torch.stack(
            [
                F.cross_entropy(output["branch_logits"][:, atlas_index], labels)
                for atlas_index in range(output["branch_logits"].shape[1])
            ]
        )
        regularization_loss = torch.relu(
            branch_losses - fusion_loss + self.margin
        ).sum()
        total_loss = (
            fusion_loss
            + self.lambda_branch * branch_losses.sum()
            + self.lambda_reg * regularization_loss
        )

        return {
            "loss": total_loss,
            "fusion_loss": fusion_loss.detach(),
            "branch_loss": branch_losses.detach().mean(),
            "regularization_loss": regularization_loss.detach(),
        }
