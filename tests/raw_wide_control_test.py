from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.edge_encoder import fc_to_edge_vector
from train import build_loss, build_model
from utils.metrics import compute_metrics


def load_config(filename):
    with (ROOT / "configs" / filename).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def make_fc(rng, sample_count, num_nodes):
    fc = rng.normal(size=(sample_count, num_nodes, num_nodes))
    fc = np.tanh((fc + fc.transpose(0, 2, 1)) / 2).astype(np.float32)
    diagonal = np.arange(num_nodes)
    fc[:, diagonal, diagonal] = 1
    return fc


def check_budget_and_gradients():
    baseline_config = load_config("abide_proposal_v6_6.yaml")
    config = load_config("abide_proposal_v17_1_raw_wide.yaml")
    expected = deepcopy(baseline_config)
    expected["model"].update(
        use_signed_edge_separation=False,
        use_raw_edge_zero_padding=False,
        atlas_overrides={"aal": {"hidden_dim": 507},
                         "cc200": {"hidden_dim": 510},
                         "ho": {"hidden_dim": 507}},
    )
    assert config == expected, "Only encoding and calculated widths should differ"
    debug = load_config("abide_proposal_v17_1_raw_wide_debug.yaml")
    assert all(debug[key] == config[key] for key in ["data", "model", "loss", "output"])

    torch.manual_seed(123)
    baseline = build_model(baseline_config)
    model = build_model(config)
    baseline_count = sum(p.numel() for p in baseline.parameters())
    model_count = sum(p.numel() for p in model.parameters())
    assert baseline_count == 16931465
    assert model_count == 16926826
    assert abs(model_count / baseline_count - 1) < 0.0003

    rng = np.random.default_rng(73)
    batch = {}
    captured = {}
    hooks = []
    for atlas, spec in config["data"]["atlases"].items():
        n = spec["num_nodes"]
        edges = n * (n - 1) // 2
        encoder = model.encoders[atlas]
        first = encoder.edge_encoder[0]
        hidden = config["model"]["atlas_overrides"][atlas]["hidden_dim"]
        assert hidden == round(256 * (2 * edges + 131) / (edges + 131))
        assert first.in_features == edges
        assert first.out_features == hidden
        assert encoder.edge_encoder[4].out_features == 128
        assert not encoder.use_signed_edge_separation
        assert not encoder.use_raw_edge_zero_padding
        count = sum(p.numel() for p in encoder.parameters())
        assert count == hidden * (edges + 131) + 384
        signed_count = sum(p.numel() for p in baseline.encoders[atlas].parameters())
        candidates = [hidden - 1, hidden, hidden + 1]
        assert hidden == min(candidates, key=lambda h: abs(h * (edges + 131) + 384 - signed_count))
        batch[atlas] = torch.from_numpy(make_fc(rng, 16, n))

        def capture(module, inputs, name=atlas):
            captured[name] = inputs[0].detach().clone()

        hooks.append(first.register_forward_pre_hook(capture))

    assert model.sample_gate[0].in_features == 384
    assert model.sample_gate[0].out_features == 256
    assert model.sample_gate[-1].out_features == 3
    labels = torch.tensor([0, 1] * 8)
    output = model(batch)
    for hook in hooks:
        hook.remove()
    loss = build_loss(config)(output, labels)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    for atlas in batch:
        raw = fc_to_edge_vector(batch[atlas])
        assert torch.equal(captured[atlas], raw)
        assert (raw < 0).any() and (raw > 0).any()
        assert (raw.abs().sum(dim=0) > 0).all()
        gradient = model.encoders[atlas].edge_encoder[0].weight.grad
        assert (gradient.abs().sum(dim=0) > 0).all()
    print(f"Parameter counts: v6.6={baseline_count:,}; v17.1={model_count:,}")
    print("Nearest per-atlas width budgets, raw inputs and all input-column gradients verified")


def check_cli_exports():
    config = load_config("abide_proposal_v17_1_raw_wide.yaml")
    config["train"].update(seeds=[0, 1], n_splits=2, epochs=2, batch_size=4,
                            checkpoint_ensemble_start=1, checkpoint_ensemble_interval=1)
    rng = np.random.default_rng(11)
    sample_count = 12
    with tempfile.TemporaryDirectory(prefix="raw_wide_cli_") as directory:
        path = Path(directory)
        config["data"]["data_root"] = str(path)
        np.save(path / "labels.npy", np.array([0, 1] * 6))
        np.save(path / "sub_ids.npy", np.arange(100, 112))
        np.save(path / "file_ids.npy", np.array([f"test_{i}" for i in range(12)]))
        np.save(path / "site_ids.npy", np.array(["site_a"] * 6 + ["site_b"] * 6))
        for atlas, nodes in {"aal": 8, "cc200": 10, "ho": 6}.items():
            spec = config["data"]["atlases"][atlas]
            spec["num_nodes"] = nodes
            np.save(path / spec["fc_file"], make_fc(rng, sample_count, nodes))
        config_path = path / "abide_proposal_v17_1_raw_wide.yaml"
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        subprocess.run([
            sys.executable, "-c",
            "import runpy,sys,torch; from pathlib import Path; torch.set_num_threads(1); "
            "sys.argv=sys.argv[1:]; sys.path.insert(0,str(Path(sys.argv[0]).parent)); "
            "runpy.run_path(sys.argv[0],run_name='__main__')",
            str(ROOT / "run_abide.py"), "--config", str(config_path),
        ], cwd=path, check=True)
        prefix = "abide_proposal_v17_1_raw_wide"
        frames = {suffix: pd.read_csv(path / f"{prefix}_{suffix}.csv") for suffix in [
            "all_folds", "summary", "sample_diagnostics", "checkpoint_diagnostics", "subject_summary"
        ]}
        folds = frames["all_folds"]
        samples = frames["sample_diagnostics"]
        checkpoints = frames["checkpoint_diagnostics"]
        assert len(folds) == 4
        assert len(samples) == 2 * sample_count
        assert len(checkpoints) == 4 * sample_count
        assert len(frames["subject_summary"]) == sample_count
        assert frames["subject_summary"].observation_count.eq(2).all()
        key = ["seed", "fold", "sample_index"]
        assert not samples.duplicated(["seed", "sample_index"]).any()
        assert not checkpoints.duplicated(key + ["checkpoint_epoch"]).any()
        assert samples.groupby("seed").sample_index.nunique().eq(sample_count).all()
        assert samples.checkpoint_count.eq(2).all()
        assert samples.checkpoint_epochs.eq("1,2").all()
        assert samples.checkpoint_weighting.eq("uniform").all()
        assert samples.decision_threshold.eq(0.5).all()
        assert samples.subject_id.eq(samples.sample_index + 100).all()
        assert set(checkpoints.checkpoint_epoch) == {1, 2}
        assert np.allclose(checkpoints.checkpoint_contribution, 0.5)
        averaged = checkpoints.groupby(key).prediction_probability.mean().rename("recomputed")
        aligned = samples.merge(averaged.reset_index(), on=key, validate="one_to_one")
        assert np.allclose(aligned.prediction_probability, aligned.recomputed, atol=2e-7, rtol=0)
        reconstructed = []
        for _, frame in samples.groupby(["seed", "fold"], sort=True):
            reconstructed.append(compute_metrics(
                frame.label.to_numpy(), frame.prediction_probability.to_numpy(),
                frame.prediction.to_numpy(),
            ))
        metrics = ["ACC", "AUC", "SEN", "SPE", "F1"]
        assert np.allclose(folds[metrics], pd.DataFrame(reconstructed)[metrics])
        for metric in metrics:
            assert np.isclose(frames["summary"].iloc[0][f"{metric}_mean"], folds[metric].mean())
    print("CLI training, five CSV exports, IDs and checkpoint probability means verified")


def main():
    torch.set_num_threads(1)
    check_budget_and_gradients()
    check_cli_exports()
    print("Raw wide control tests passed.")


if __name__ == "__main__":
    main()
