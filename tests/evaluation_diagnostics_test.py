import sys
from pathlib import Path

try:
    import sklearn  # noqa: F401
except ModuleNotFoundError:
    print("Evaluation diagnostics test skipped: sklearn is not installed.")
    sys.exit(0)

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import evaluate_model


class DiagnosticToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.atlas_names = ["aal", "cc200", "ho"]

    def forward(self, batch):
        labels = batch["label"].float()
        fusion_logits = torch.stack([1.0 - labels, labels], dim=-1)
        weighted_logits = fusion_logits + 0.1
        branch_logits = torch.stack(
            [
                fusion_logits,
                fusion_logits * 0.8,
                torch.flip(fusion_logits, dims=[-1]),
            ],
            dim=1,
        )
        atlas_weight = torch.tensor(
            [[0.2, 0.5, 0.3]],
            dtype=fusion_logits.dtype,
            device=fusion_logits.device,
        ).repeat(fusion_logits.shape[0], 1)

        return {
            "fusion_logits": fusion_logits,
            "weighted_logits": weighted_logits,
            "branch_logits": branch_logits,
            "atlas_weight": atlas_weight,
            "base_atlas_weight": atlas_weight,
            "gated_atlas_weight": atlas_weight,
            "sample_gate_logits": torch.zeros_like(atlas_weight),
        }


def main():
    samples = [
        {"label": torch.tensor(0, dtype=torch.long)},
        {"label": torch.tensor(1, dtype=torch.long)},
        {"label": torch.tensor(0, dtype=torch.long)},
        {"label": torch.tensor(1, dtype=torch.long)},
    ]
    dataloader = DataLoader(samples, batch_size=2, shuffle=False)
    metrics, probabilities, labels = evaluate_model(
        DiagnosticToyModel(),
        dataloader,
        device="cpu",
    )

    assert probabilities.shape == (4,)
    assert labels.shape == (4,)
    assert "Weighted_ACC" in metrics
    assert "Branch_aal_ACC" in metrics
    assert "Branch_cc200_AUC" in metrics
    assert "Branch_ho_F1" in metrics
    assert "Weight_aal" in metrics
    assert "BaseWeight_cc200" in metrics
    assert "GatedWeight_ho" in metrics
    assert "GateLogit_aal" in metrics

    print("Evaluation diagnostics test passed.")
    print("Diagnostic columns:", len(metrics))


if __name__ == "__main__":
    main()
