import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.smaf_edge_energy_net import SMAFEdgeEnergyNet


def build_model(reliability_mode):
    return SMAFEdgeEnergyNet(
        atlas_specs={
            "aal": {"num_nodes": 4},
            "cc200": {"num_nodes": 5},
            "ho": {"num_nodes": 3},
        },
        hidden_dim=8,
        embedding_dim=6,
        dropout=0.0,
        temperature=1.0,
        reliability_mode=reliability_mode,
    )


def main():
    logits = torch.tensor(
        [
            [[1.5, -0.5], [0.2, 0.9], [-1.0, 0.4]],
            [[-0.3, 0.8], [1.1, -0.7], [0.5, 0.6]],
        ],
        dtype=torch.float32,
    )
    branch_shifts = torch.tensor([[[7.0], [-3.5], [2.25]]])
    shifted_logits = logits + branch_shifts

    raw_model = build_model("energy")
    centered_model = build_model("centered_energy")

    raw_score = raw_model.compute_reliability_score(logits)
    shifted_raw_score = raw_model.compute_reliability_score(shifted_logits)
    assert not torch.allclose(raw_score, shifted_raw_score)

    centered_score = centered_model.compute_reliability_score(logits)
    shifted_centered_score = centered_model.compute_reliability_score(
        shifted_logits
    )
    assert torch.allclose(centered_score, shifted_centered_score, atol=1e-6)
    assert torch.allclose(
        torch.softmax(centered_score, dim=1),
        torch.softmax(shifted_centered_score, dim=1),
        atol=1e-6,
    )

    diagnostics = centered_model.compute_reliability_diagnostics(logits)
    assert diagnostics["branch_logit_mean"].shape == (2, 3)
    assert diagnostics["branch_logit_margin"].shape == (2, 3)
    assert diagnostics["raw_energy_score"].shape == (2, 3)
    assert diagnostics["centered_energy_score"].shape == (2, 3)
    assert diagnostics["entropy_confidence"].shape == (2, 3)
    assert torch.all(diagnostics["entropy_confidence"] >= 0)
    assert torch.all(diagnostics["entropy_confidence"] <= 1)

    try:
        build_model("unsupported")
    except ValueError:
        pass
    else:
        raise AssertionError("Unsupported reliability mode was accepted")

    print("Reliability mode test passed.")


if __name__ == "__main__":
    main()
