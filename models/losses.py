import torch
import torch.nn as nn
import torch.nn.functional as F


class ProposalLoss(nn.Module):
    """
    L = L_fusion + lambda_branch * sum(L_i) + lambda_reg * L_reg

    Supported regularization modes:

    proposal_literal:
        sum_i max(0, L_i - L_fusion + margin)

    fusion_better:
        sum_i max(0, L_fusion - L_i + margin)
    """

    def __init__(
        self,
        lambda_branch=0.2,
        lambda_reg=0.1,
        margin=0.1,
        regularization_mode="proposal_literal",
        class_weights=None,
    ):
        super().__init__()
        supported_modes = {"proposal_literal", "fusion_better"}
        if regularization_mode not in supported_modes:
            raise ValueError(
                f"Unsupported regularization_mode: {regularization_mode}. "
                f"Expected one of {sorted(supported_modes)}"
            )

        self.lambda_branch = lambda_branch
        self.lambda_reg = lambda_reg
        self.margin = margin
        self.regularization_mode = regularization_mode
        if class_weights is None:
            self.class_weights = None
        else:
            weights = torch.as_tensor(class_weights, dtype=torch.float32)
            if weights.shape != (2,):
                raise ValueError("class_weights must contain exactly two values")
            if torch.any(weights <= 0):
                raise ValueError("class_weights must be positive")
            self.class_weights = weights

    def forward(self, output, labels):
        class_weights = None
        if self.class_weights is not None:
            class_weights = self.class_weights.to(labels.device)

        fusion_loss = F.cross_entropy(
            output["fusion_logits"],
            labels,
            weight=class_weights,
        )
        branch_logits = output.get("branch_logits")
        if branch_logits is None:
            if self.lambda_branch != 0 or self.lambda_reg != 0:
                raise ValueError(
                    "branch_logits are required when lambda_branch or lambda_reg "
                    "is non-zero"
                )
            zero = fusion_loss.detach().new_zeros(())
            return {
                "loss": fusion_loss,
                "fusion_loss": fusion_loss.detach(),
                "branch_loss": zero,
                "regularization_loss": zero,
            }

        branch_losses = torch.stack(
            [
                F.cross_entropy(
                    branch_logits[:, atlas_index],
                    labels,
                    weight=class_weights,
                )
                for atlas_index in range(branch_logits.shape[1])
            ]
        )
        if self.regularization_mode == "proposal_literal":
            regularization_term = branch_losses - fusion_loss + self.margin
        else:
            regularization_term = fusion_loss - branch_losses + self.margin

        regularization_loss = torch.relu(regularization_term).sum()
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
