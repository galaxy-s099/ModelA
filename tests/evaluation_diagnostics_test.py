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

from train import (
    aggregate_checkpoint_diagnostics,
    combine_probability_ensemble,
    evaluate_model,
)


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
        {
            "label": torch.tensor(label, dtype=torch.long),
            "sample_index": torch.tensor(index, dtype=torch.long),
        }
        for index, label in enumerate([0, 1, 0, 1])
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

    _, probabilities, labels, details = evaluate_model(
        DiagnosticToyModel(),
        dataloader,
        device="cpu",
        return_details=True,
    )
    assert details["sample_index"].tolist() == [0, 1, 2, 3]
    assert details["atlas_weight"].shape == (4, 3)
    assert details["branch_probability"].shape == (4, 3)
    _, mean_probabilities, checkpoint_weights = combine_probability_ensemble(
        [probabilities, probabilities],
        labels,
        decision_threshold=0.5,
        return_weights=True,
    )
    aggregated = aggregate_checkpoint_diagnostics(
        [details, details],
        checkpoint_weights,
        mean_probabilities,
        decision_threshold=0.5,
    )
    assert aggregated["atlas_weight"].shape == (4, 3)
    assert torch.allclose(
        torch.from_numpy(aggregated["atlas_weight"].sum(axis=1)),
        torch.ones(4),
    )

    print("Evaluation diagnostics test passed.")
    print("Diagnostic columns:", len(metrics))


if __name__ == "__main__":
    main()
