import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.losses import ProposalLoss
from models.smaf_edge_energy_net import SMAFEdgeEnergyNet


def make_fc(batch_size, num_nodes):
    matrix = torch.randn(batch_size, num_nodes, num_nodes)
    matrix = (matrix + matrix.transpose(-1, -2)) / 2
    matrix = torch.tanh(matrix)
    identity = torch.eye(num_nodes).unsqueeze(0)
    return matrix * (1.0 - identity)


def main():
    torch.manual_seed(31)
    atlas_specs = {
        "aal": {"num_nodes": 8},
        "cc200": {"num_nodes": 10},
        "ho": {"num_nodes": 6},
    }
    model = SMAFEdgeEnergyNet(
        atlas_specs=atlas_specs,
        hidden_dim=16,
        embedding_dim=12,
        dropout=0.1,
        temperature=1.0,
    )
    batch = {
        atlas_name: make_fc(batch_size=4, num_nodes=spec["num_nodes"])
        for atlas_name, spec in atlas_specs.items()
    }
    output = model(batch)

    assert output["fusion_logits"].shape == (4, 2)
    assert output["branch_logits"].shape == (4, 3, 2)
    assert output["energy"].shape == (4, 3)
    assert output["atlas_weight"].shape == (4, 3)
    assert "attention_weight" not in output
    assert torch.allclose(
        output["atlas_weight"].sum(dim=1),
        torch.ones(4),
        atol=1e-6,
    )

    labels = torch.tensor([0, 1, 0, 1])
    criterion = ProposalLoss(lambda_branch=0.2, lambda_reg=0.1, margin=0.1)
    loss_details = criterion(output, labels)
    loss_details["loss"].backward()
    assert torch.isfinite(loss_details["loss"])

    prior_model = SMAFEdgeEnergyNet(
        atlas_specs=atlas_specs,
        hidden_dim=16,
        embedding_dim=12,
        dropout=0.1,
        temperature=1.0,
        use_atlas_prior=True,
    )
    prior_model.load_state_dict(model.state_dict(), strict=False)
    model.eval()
    prior_model.eval()
    with torch.no_grad():
        baseline_output = model(batch)
        prior_output = prior_model(batch)
    assert torch.allclose(
        baseline_output["atlas_weight"],
        prior_output["atlas_weight"],
        atol=1e-6,
    )
    assert torch.allclose(
        baseline_output["fusion_logits"],
        prior_output["fusion_logits"],
        atol=1e-6,
    )

    prior_model.train()
    prior_output = prior_model(batch)
    prior_loss_details = criterion(prior_output, labels)
    prior_loss_details["loss"].backward()
    assert prior_model.atlas_prior.grad is not None
    assert torch.isfinite(prior_model.atlas_prior.grad).all()
    assert prior_model.atlas_prior.grad.abs().sum() > 0

    sample_gate_model = SMAFEdgeEnergyNet(
        atlas_specs=atlas_specs,
        hidden_dim=16,
        embedding_dim=12,
        dropout=0.1,
        temperature=1.0,
        use_sample_gate=True,
        sample_gate_scale=0.5,
    )
    sample_gate_model.load_state_dict(model.state_dict(), strict=False)
    sample_gate_model.eval()
    with torch.no_grad():
        sample_gate_output = sample_gate_model(batch)
    assert torch.allclose(
        baseline_output["atlas_weight"],
        sample_gate_output["atlas_weight"],
        atol=1e-6,
    )
    assert torch.allclose(
        baseline_output["fusion_logits"],
        sample_gate_output["fusion_logits"],
        atol=1e-6,
    )

    sample_gate_model.train()
    sample_gate_output = sample_gate_model(batch)
    sample_gate_loss_details = criterion(sample_gate_output, labels)
    sample_gate_loss_details["loss"].backward()
    assert sample_gate_model.sample_gate[-1].weight.grad is not None
    assert torch.isfinite(sample_gate_model.sample_gate[-1].weight.grad).all()
    assert sample_gate_model.sample_gate[-1].weight.grad.abs().sum() > 0

    residual_model = SMAFEdgeEnergyNet(
        atlas_specs=atlas_specs,
        hidden_dim=16,
        embedding_dim=12,
        dropout=0.1,
        temperature=1.0,
        use_sample_gate=True,
        use_residual_classifier=True,
        residual_classifier_scale=0.5,
    )
    residual_model.load_state_dict(sample_gate_model.state_dict(), strict=False)
    sample_gate_model.eval()
    residual_model.eval()
    with torch.no_grad():
        sample_gate_baseline = sample_gate_model(batch)
        residual_output = residual_model(batch)
    assert torch.allclose(
        sample_gate_baseline["fusion_logits"],
        residual_output["fusion_logits"],
        atol=1e-6,
    )
    assert torch.allclose(
        residual_output["residual_logits"],
        torch.zeros_like(residual_output["residual_logits"]),
        atol=1e-6,
    )

    residual_model.train()
    residual_output = residual_model(batch)
    residual_loss_details = criterion(residual_output, labels)
    residual_loss_details["loss"].backward()
    assert residual_model.residual_classifier[-1].weight.grad is not None
    assert torch.isfinite(residual_model.residual_classifier[-1].weight.grad).all()
    assert residual_model.residual_classifier[-1].weight.grad.abs().sum() > 0

    blend_model = SMAFEdgeEnergyNet(
        atlas_specs=atlas_specs,
        hidden_dim=16,
        embedding_dim=12,
        dropout=0.1,
        temperature=1.0,
        use_sample_gate=True,
        use_dual_energy_blend=True,
        dual_energy_blend_alpha=0.5,
    )
    blend_model.load_state_dict(sample_gate_model.state_dict(), strict=False)
    blend_model.eval()
    with torch.no_grad():
        blend_output = blend_model(batch)
    assert torch.allclose(
        blend_output["atlas_weight"].sum(dim=1),
        torch.ones(4),
        atol=1e-6,
    )
    assert torch.allclose(
        blend_output["base_atlas_weight"],
        blend_output["gated_atlas_weight"],
        atol=1e-6,
    )
    assert torch.allclose(
        blend_output["weighted_logits"],
        blend_output["fusion_logits"],
        atol=1e-6,
    )

    blend_model.train()
    blend_output = blend_model(batch)
    blend_loss_details = criterion(blend_output, labels)
    blend_loss_details["loss"].backward()
    assert blend_model.sample_gate[-1].weight.grad is not None
    assert torch.isfinite(blend_model.sample_gate[-1].weight.grad).all()
    assert blend_model.sample_gate[-1].weight.grad.abs().sum() > 0

    shared_correction_model = SMAFEdgeEnergyNet(
        atlas_specs=atlas_specs,
        hidden_dim=16,
        embedding_dim=12,
        dropout=0.1,
        temperature=1.0,
        use_sample_gate=True,
        use_shared_correction=True,
        shared_correction_scale=0.25,
    )
    shared_correction_model.load_state_dict(
        sample_gate_model.state_dict(),
        strict=False,
    )
    sample_gate_model.eval()
    shared_correction_model.eval()
    with torch.no_grad():
        sample_gate_baseline = sample_gate_model(batch)
        shared_correction_output = shared_correction_model(batch)
    assert torch.allclose(
        sample_gate_baseline["fusion_logits"],
        shared_correction_output["fusion_logits"],
        atol=1e-6,
    )
    assert torch.allclose(
        shared_correction_output["shared_correction_logits"],
        torch.zeros_like(shared_correction_output["shared_correction_logits"]),
        atol=1e-6,
    )

    shared_correction_model.train()
    shared_correction_output = shared_correction_model(batch)
    shared_correction_loss_details = criterion(shared_correction_output, labels)
    shared_correction_loss_details["loss"].backward()
    assert shared_correction_model.shared_correction[-1].weight.grad is not None
    assert torch.isfinite(
        shared_correction_model.shared_correction[-1].weight.grad
    ).all()
    assert shared_correction_model.shared_correction[-1].weight.grad.abs().sum() > 0

    override_model = SMAFEdgeEnergyNet(
        atlas_specs=atlas_specs,
        hidden_dim=16,
        embedding_dim=12,
        dropout=0.1,
        temperature=1.0,
        use_sample_gate=True,
        atlas_overrides={
            "cc200": {
                "hidden_dim": 24,
                "embedding_dim": 12,
                "dropout": 0.05,
            }
        },
    )
    override_output = override_model(batch)
    assert override_output["fusion_logits"].shape == (4, 2)
    assert override_output["branch_logits"].shape == (4, 3, 2)
    assert override_model.embedding_dims["cc200"] == 12
    override_loss_details = criterion(override_output, labels)
    override_loss_details["loss"].backward()
    assert torch.isfinite(override_loss_details["loss"])

    node_summary_model = SMAFEdgeEnergyNet(
        atlas_specs=atlas_specs,
        hidden_dim=16,
        embedding_dim=12,
        dropout=0.1,
        temperature=1.0,
        use_sample_gate=True,
        use_node_summary=True,
        node_summary_hidden_dim=8,
        node_summary_embedding_dim=4,
    )
    node_summary_output = node_summary_model(batch)
    assert node_summary_output["fusion_logits"].shape == (4, 2)
    assert node_summary_output["branch_logits"].shape == (4, 3, 2)
    node_summary_loss_details = criterion(node_summary_output, labels)
    node_summary_loss_details["loss"].backward()
    assert torch.isfinite(node_summary_loss_details["loss"])

    print("Edge energy test passed.")
    print("Fusion logits:", tuple(output["fusion_logits"].shape))
    print("Atlas weights:", tuple(output["atlas_weight"].shape))


if __name__ == "__main__":
    main()
