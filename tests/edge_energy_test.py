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

    branch_residual_model = SMAFEdgeEnergyNet(
        atlas_specs=atlas_specs,
        hidden_dim=16,
        embedding_dim=12,
        dropout=0.1,
        temperature=1.0,
        use_sample_gate=True,
        use_branch_residual_correction=True,
        branch_residual_correction_scale=0.1,
        branch_residual_hidden_dim=8,
    )
    branch_residual_model.load_state_dict(
        sample_gate_model.state_dict(),
        strict=False,
    )
    sample_gate_model.eval()
    branch_residual_model.eval()
    with torch.no_grad():
        sample_gate_baseline = sample_gate_model(batch)
        branch_residual_output = branch_residual_model(batch)
    assert torch.allclose(
        sample_gate_baseline["fusion_logits"],
        branch_residual_output["fusion_logits"],
        atol=1e-6,
    )
    assert torch.allclose(
        branch_residual_output["branch_residual_correction_logits"],
        torch.zeros_like(
            branch_residual_output["branch_residual_correction_logits"]
        ),
        atol=1e-6,
    )

    branch_residual_model.train()
    branch_residual_output = branch_residual_model(batch)
    branch_residual_loss_details = criterion(branch_residual_output, labels)
    branch_residual_loss_details["loss"].backward()
    assert branch_residual_model.branch_residual_correction[-1].weight.grad is not None
    assert torch.isfinite(
        branch_residual_model.branch_residual_correction[-1].weight.grad
    ).all()
    assert (
        branch_residual_model.branch_residual_correction[-1]
        .weight.grad.abs()
        .sum()
        > 0
    )

    zero_consensus_model = SMAFEdgeEnergyNet(
        atlas_specs=atlas_specs,
        hidden_dim=16,
        embedding_dim=12,
        dropout=0.1,
        temperature=1.0,
        use_sample_gate=True,
        use_consensus_gate=True,
        consensus_gate_scale=0.0,
    )
    zero_consensus_model.load_state_dict(sample_gate_model.state_dict(), strict=False)
    sample_gate_model.eval()
    zero_consensus_model.eval()
    with torch.no_grad():
        sample_gate_baseline = sample_gate_model(batch)
        zero_consensus_output = zero_consensus_model(batch)
    assert torch.allclose(
        sample_gate_baseline["atlas_weight"],
        zero_consensus_output["atlas_weight"],
        atol=1e-6,
    )
    assert torch.allclose(
        sample_gate_baseline["fusion_logits"],
        zero_consensus_output["fusion_logits"],
        atol=1e-6,
    )

    consensus_model = SMAFEdgeEnergyNet(
        atlas_specs=atlas_specs,
        hidden_dim=16,
        embedding_dim=12,
        dropout=0.1,
        temperature=1.0,
        use_sample_gate=True,
        use_consensus_gate=True,
        consensus_gate_scale=0.5,
    )
    consensus_output = consensus_model(batch)
    assert consensus_output["consensus_disagreement"].shape == (4, 3)
    assert torch.allclose(
        consensus_output["atlas_weight"].sum(dim=1),
        torch.ones(4),
        atol=1e-6,
    )
    consensus_loss_details = criterion(consensus_output, labels)
    consensus_loss_details["loss"].backward()
    assert torch.isfinite(consensus_loss_details["loss"])

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

    profile_attention_model = SMAFEdgeEnergyNet(
        atlas_specs=atlas_specs,
        hidden_dim=16,
        embedding_dim=12,
        dropout=0.1,
        temperature=1.0,
        use_sample_gate=True,
        use_roi_profile_attention=True,
        roi_profile_dim=8,
        roi_profile_num_heads=2,
        roi_profile_dropout=0.0,
        roi_profile_residual_scale=0.25,
    )
    profile_attention_model.load_state_dict(
        sample_gate_model.state_dict(),
        strict=False,
    )
    sample_gate_model.eval()
    profile_attention_model.eval()
    with torch.no_grad():
        sample_gate_baseline = sample_gate_model(batch)
        profile_attention_output = profile_attention_model(batch)
    assert torch.allclose(
        sample_gate_baseline["fusion_logits"],
        profile_attention_output["fusion_logits"],
        atol=1e-6,
    )
    assert all(
        "roi_profile_embedding"
        in profile_attention_output["branch_details"][atlas_name]
        for atlas_name in atlas_specs
    )
    profile_attention_model.train()
    profile_attention_output = profile_attention_model(batch)
    profile_attention_loss_details = criterion(profile_attention_output, labels)
    profile_attention_loss_details["loss"].backward()
    assert torch.isfinite(profile_attention_loss_details["loss"])
    assert profile_attention_model.roi_profile_adapters["aal"].weight.grad is not None

    edge_residual_model = SMAFEdgeEnergyNet(
        atlas_specs=atlas_specs,
        hidden_dim=16,
        embedding_dim=12,
        dropout=0.1,
        temperature=1.0,
        use_sample_gate=True,
        use_edge_residual=True,
        edge_residual_hidden_dim=8,
        edge_residual_scale=0.25,
    )
    edge_residual_model.load_state_dict(sample_gate_model.state_dict(), strict=False)
    sample_gate_model.eval()
    edge_residual_model.eval()
    with torch.no_grad():
        sample_gate_baseline = sample_gate_model(batch)
        edge_residual_output = edge_residual_model(batch)
    assert torch.allclose(
        sample_gate_baseline["fusion_logits"],
        edge_residual_output["fusion_logits"],
        atol=1e-6,
    )

    edge_residual_model.train()
    edge_residual_output = edge_residual_model(batch)
    edge_residual_loss_details = criterion(edge_residual_output, labels)
    edge_residual_loss_details["loss"].backward()
    assert torch.isfinite(edge_residual_loss_details["loss"])

    edge_dropout_model = SMAFEdgeEnergyNet(
        atlas_specs=atlas_specs,
        hidden_dim=16,
        embedding_dim=12,
        dropout=0.1,
        temperature=1.0,
        use_sample_gate=True,
        edge_dropout=0.1,
    )
    for atlas_name in atlas_specs:
        assert edge_dropout_model.encoders[atlas_name].edge_dropout == 0.1
    edge_dropout_model.train()
    edge_dropout_output = edge_dropout_model(batch)
    edge_dropout_loss_details = criterion(edge_dropout_output, labels)
    edge_dropout_loss_details["loss"].backward()
    assert torch.isfinite(edge_dropout_loss_details["loss"])

    edge_topk_model = SMAFEdgeEnergyNet(
        atlas_specs=atlas_specs,
        hidden_dim=16,
        embedding_dim=12,
        dropout=0.1,
        temperature=1.0,
        use_sample_gate=True,
        edge_topk_ratio=0.5,
    )
    for atlas_name in atlas_specs:
        assert edge_topk_model.encoders[atlas_name].edge_topk_ratio == 0.5
    edge_topk_output = edge_topk_model(batch)
    edge_topk_loss_details = criterion(edge_topk_output, labels)
    edge_topk_loss_details["loss"].backward()
    assert torch.isfinite(edge_topk_loss_details["loss"])

    atlas_dropout_model = SMAFEdgeEnergyNet(
        atlas_specs=atlas_specs,
        hidden_dim=16,
        embedding_dim=12,
        dropout=0.1,
        temperature=1.0,
        use_sample_gate=True,
        atlas_dropout=1.0,
        atlas_dropout_mode="single",
    )
    atlas_dropout_model.train()
    atlas_dropout_output = atlas_dropout_model(batch)
    assert atlas_dropout_output["atlas_keep_mask"].shape == (4, 3)
    assert torch.all(atlas_dropout_output["atlas_keep_mask"].sum(dim=1) == 2)
    assert torch.allclose(
        atlas_dropout_output["atlas_weight"].sum(dim=1),
        torch.ones(4),
        atol=1e-6,
    )
    assert torch.all(
        atlas_dropout_output["atlas_weight"][
            ~atlas_dropout_output["atlas_keep_mask"]
        ]
        < 1e-6
    )
    atlas_dropout_loss_details = criterion(atlas_dropout_output, labels)
    atlas_dropout_loss_details["loss"].backward()
    assert torch.isfinite(atlas_dropout_loss_details["loss"])

    atlas_dropout_model.eval()
    with torch.no_grad():
        atlas_dropout_eval_output = atlas_dropout_model(batch)
    assert atlas_dropout_eval_output["atlas_keep_mask"] is None

    logit_meta_model = SMAFEdgeEnergyNet(
        atlas_specs=atlas_specs,
        hidden_dim=16,
        embedding_dim=12,
        dropout=0.1,
        temperature=1.0,
        use_sample_gate=True,
        use_logit_meta_fusion=True,
        logit_meta_hidden_dim=8,
        logit_meta_dropout=0.0,
    )
    logit_meta_output = logit_meta_model(batch)
    assert logit_meta_output["logit_meta_fusion_logits"].shape == (4, 2)
    assert not torch.allclose(
        logit_meta_output["fusion_logits"],
        logit_meta_output["weighted_logits"],
    )
    logit_meta_loss_details = criterion(logit_meta_output, labels)
    logit_meta_loss_details["loss"].backward()
    assert torch.isfinite(logit_meta_loss_details["loss"])
    assert logit_meta_model.logit_meta_fusion[-1].weight.grad is not None
    assert torch.isfinite(
        logit_meta_model.logit_meta_fusion[-1].weight.grad
    ).all()

    site_model = SMAFEdgeEnergyNet(
        atlas_specs=atlas_specs,
        hidden_dim=16,
        embedding_dim=12,
        dropout=0.1,
        temperature=1.0,
        use_sample_gate=True,
        use_site_embedding=True,
        num_sites=3,
        site_embedding_dim=4,
    )
    site_batch = dict(batch)
    site_batch["site"] = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    site_output = site_model(site_batch)
    assert site_output["fusion_logits"].shape == (4, 2)
    assert site_output["branch_logits"].shape == (4, 3, 2)
    site_loss_details = criterion(site_output, labels)
    site_loss_details["loss"].backward()
    assert site_model.site_embedding.weight.grad is not None
    assert torch.isfinite(site_model.site_embedding.weight.grad).all()

    try:
        site_model(batch)
    except ValueError as exc:
        assert "site ids" in str(exc)
    else:
        raise AssertionError("site model should require site ids in the batch")

    site_adv_model = SMAFEdgeEnergyNet(
        atlas_specs=atlas_specs,
        hidden_dim=16,
        embedding_dim=12,
        dropout=0.1,
        temperature=1.0,
        use_sample_gate=True,
        use_site_adversarial=True,
        num_sites=3,
        site_adversarial_hidden_dim=8,
        site_adversarial_grl_lambda=1.0,
    )
    site_adv_output = site_adv_model(site_batch)
    assert site_adv_output["fusion_logits"].shape == (4, 2)
    assert site_adv_output["site_logits"].shape == (4, 3)
    site_adv_criterion = ProposalLoss(
        lambda_branch=0.2,
        lambda_reg=0.1,
        margin=0.1,
        lambda_site_adversarial=0.05,
    )
    site_adv_loss_details = site_adv_criterion(
        site_adv_output,
        labels,
        site_batch["site"],
    )
    site_adv_loss_details["loss"].backward()
    assert site_adv_model.site_classifier[-1].weight.grad is not None
    assert torch.isfinite(site_adv_model.site_classifier[-1].weight.grad).all()

    try:
        site_adv_model(batch)
    except ValueError as exc:
        assert "site ids" in str(exc)
    else:
        raise AssertionError("site adversarial model should require site ids")

    print("Edge energy test passed.")
    print("Fusion logits:", tuple(output["fusion_logits"].shape))
    print("Atlas weights:", tuple(output["atlas_weight"].shape))


if __name__ == "__main__":
    main()
