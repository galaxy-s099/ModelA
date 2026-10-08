from copy import deepcopy
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.edge_encoder import EdgeBranchEncoder, fc_to_edge_vector
from train import build_loss, build_model, run_repeated_cv


def load_config(filename):
    with (ROOT / "configs" / filename).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def check_matching():
    baseline = load_config("abide_proposal_v6_6.yaml")
    control = load_config("abide_proposal_v17_0_raw_padded.yaml")
    expected = deepcopy(baseline)
    expected["model"].update(
        use_signed_edge_separation=False,
        use_raw_edge_zero_padding=True,
    )
    assert control == expected, "Only the input encoding should change"

    torch.manual_seed(123)
    signed_model = build_model(baseline)
    torch.manual_seed(123)
    raw_model = build_model(control)
    signed_state = signed_model.state_dict()
    raw_state = raw_model.state_dict()
    assert signed_state.keys() == raw_state.keys()
    for name in signed_state:
        assert torch.equal(signed_state[name], raw_state[name]), name
    signed_count = sum(p.numel() for p in signed_model.parameters())
    raw_count = sum(p.numel() for p in raw_model.parameters())
    assert signed_count == raw_count
    for atlas, expected_dim in [("aal", 13340), ("cc200", 39800), ("ho", 12210)]:
        assert raw_model.encoders[atlas].edge_encoder[0].in_features == expected_dim
    print(f"Identical parameter shapes and initial values: {raw_count:,} parameters")


def check_encoding_and_gradients():
    torch.manual_seed(7)
    fc = torch.tensor([
        [[1.0, 0.3, -0.7], [0.3, 1.0, 0.2], [-0.7, 0.2, 1.0]],
        [[1.0, -0.2, 0.8], [-0.2, 1.0, -0.4], [0.8, -0.4, 1.0]],
    ])
    encoder = EdgeBranchEncoder(
        input_dim=6, hidden_dim=8, embedding_dim=5, dropout=0.0,
        use_signed_edge_separation=False, use_raw_edge_zero_padding=True,
    ).eval()
    captured = []
    hook = encoder.edge_encoder[0].register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach().clone())
    )
    output = encoder(fc)
    hook.remove()
    raw = fc_to_edge_vector(fc)
    assert torch.equal(captured[0][:, :3], raw)
    assert torch.count_nonzero(captured[0][:, 3:]) == 0
    assert captured[0].shape == (2, 6)
    output.square().sum().backward()
    gradient = encoder.edge_encoder[0].weight.grad
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient[:, :3]) > 0
    assert torch.count_nonzero(gradient[:, 3:]) == 0

    for invalid in [
        {"input_dim": 6, "use_signed_edge_separation": True},
        {"input_dim": 5, "use_signed_edge_separation": False},
        {"input_dim": 6, "use_signed_edge_separation": False,
         "use_dual_stream_signed_mlp": True},
    ]:
        try:
            EdgeBranchEncoder(**invalid, use_raw_edge_zero_padding=True)
        except ValueError:
            pass
        else:
            raise AssertionError("Incompatible flags must be rejected")
    print("Raw signed values, zero padding, gradients and flag validation passed")


def check_pipeline():
    config = load_config("abide_proposal_v17_0_raw_padded.yaml")
    config["train"].update(
        seeds=[0], n_splits=2, epochs=2, batch_size=4,
        checkpoint_ensemble_start=1, checkpoint_ensemble_interval=1,
    )
    config["model"].update(hidden_dim=16, embedding_dim=12, dropout=0.1)
    nodes = {"aal": 8, "cc200": 10, "ho": 6}
    rng = np.random.default_rng(11)
    sample_count = 12
    with tempfile.TemporaryDirectory(prefix="raw_padded_control_") as directory:
        config["data"]["data_root"] = directory
        np.save(Path(directory) / "labels.npy", np.array([0, 1] * 6))
        for atlas, spec in config["data"]["atlases"].items():
            spec["num_nodes"] = nodes[atlas]
            fc = rng.normal(size=(sample_count, nodes[atlas], nodes[atlas]))
            fc = np.tanh((fc + fc.transpose(0, 2, 1)) / 2).astype(np.float32)
            diagonal = np.arange(nodes[atlas])
            fc[:, diagonal, diagonal] = 1
            np.save(Path(directory) / spec["fc_file"], fc)

        model = build_model(config)
        batch = {atlas: torch.from_numpy(np.load(Path(directory) / spec["fc_file"]))
                 for atlas, spec in config["data"]["atlases"].items()}
        loss = build_loss(config)(model(batch), torch.tensor([0, 1] * 6))["loss"]
        assert torch.isfinite(loss)
        loss.backward()
        for parameter in model.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()

        results, summary, diagnostics = run_repeated_cv(config, return_diagnostics=True)

    assert len(results) == 2
    assert all(np.isfinite(value) for value in summary.values())
    samples = diagnostics["sample_rows"]
    checkpoints = diagnostics["checkpoint_rows"]
    assert len(samples) == sample_count
    assert len(checkpoints) == 2 * sample_count
    assert {row["sample_index"] for row in samples} == set(range(sample_count))
    for row in samples:
        assert row["checkpoint_count"] == 2
        assert row["checkpoint_epochs"] == "1,2"
        assert row["decision_threshold"] == 0.5
        assert np.isclose(sum(row[f"weight_{atlas}"] for atlas in nodes), 1)
        selected = [cp for cp in checkpoints
                    if cp["sample_index"] == row["sample_index"]]
        assert {cp["checkpoint_epoch"] for cp in selected} == {1, 2}
        assert np.isclose(row["prediction_probability"], np.mean(
            [cp["prediction_probability"] for cp in selected]
        ))
    print("Fold-local training, finite gradients and checkpoint probability mean passed")


def main():
    torch.set_num_threads(1)
    check_matching()
    check_encoding_and_gradients()
    check_pipeline()
    print("Raw padded control tests passed.")


if __name__ == "__main__":
    main()
