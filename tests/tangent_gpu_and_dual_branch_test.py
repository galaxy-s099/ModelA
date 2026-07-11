"""Verify GPU/CPU Tangent preprocessing and the v11 dual-branch interface."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.tangent_pearson import TangentPearsonTransformer
from data.tangent_fc import GPUTangentPearsonTransformer
from data.abide_dataset import ABIDEMultiAtlasDataset
from models.smaf_edge_energy_net import SMAFEdgeEnergyNet


def make_correlation(seed, nodes):
    random = np.random.default_rng(seed)
    timeseries = random.normal(size=(80, nodes))
    return np.corrcoef(timeseries, rowvar=False).astype(np.float32)


def make_fc(batch_size, num_nodes):
    matrix = torch.randn(batch_size, num_nodes, num_nodes)
    matrix = (matrix + matrix.transpose(-1, -2)) / 2
    matrix = torch.tanh(matrix)
    identity = torch.eye(num_nodes).unsqueeze(0)
    return matrix * (1.0 - identity) + identity


def main():
    train = np.stack([make_correlation(seed, 8) for seed in range(5)])
    test = np.stack([make_correlation(seed + 10, 8) for seed in range(2)])
    cpu = TangentPearsonTransformer(shrinkage=0.05)
    torch_transformer = GPUTangentPearsonTransformer(
        device="cpu",
        shrinkage=0.05,
        batch_size=2,
    )
    cpu_train = cpu.fit_transform(train)
    torch_train = torch_transformer.fit(train).transform(train)
    cpu_test = cpu.transform(test)
    torch_test = torch_transformer.transform(test)
    upper = np.triu_indices(8, k=1)
    assert np.allclose(cpu_train, torch_train[:, upper[0], upper[1]], atol=2e-4)
    assert np.allclose(cpu_test, torch_test[:, upper[0], upper[1]], atol=2e-4)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        np.save(path / "labels.npy", np.array([0, 1], dtype=np.int64))
        raw = np.stack([make_correlation(30, 8), make_correlation(31, 8)])
        tangent = raw.copy()
        tangent[:, 0, 1] = tangent[:, 1, 0] = 2.5
        np.save(path / "X_aal.npy", raw)
        dataset = ABIDEMultiAtlasDataset(
            path,
            {"aal": {"num_nodes": 8, "fc_file": "X_aal.npy"}},
            fc_overrides={"aal": tangent},
        )
        # Tangent values are intentionally not clipped to correlation bounds.
        assert dataset[0]["aal"][0, 1].item() == 2.5

    torch.manual_seed(7)
    atlas_specs = {"aal": {"num_nodes": 8}, "cc200": {"num_nodes": 10}}
    model = SMAFEdgeEnergyNet(
        atlas_specs=atlas_specs,
        hidden_dim=16,
        embedding_dim=12,
        dropout=0.0,
        temperature=1.0,
        use_sample_gate=True,
        use_tangent_branch=True,
    )
    model.eval()
    batch = {}
    for atlas_name, spec in atlas_specs.items():
        batch[atlas_name] = make_fc(4, spec["num_nodes"])
        batch[f"{atlas_name}_tangent"] = make_fc(4, spec["num_nodes"])
    output = model(batch)
    assert output["fusion_logits"].shape == (4, 2)
    assert output["branch_logits"].shape == (4, 2, 2)
    for atlas_name in atlas_specs:
        details = output["branch_details"][atlas_name]
        assert "tangent_embedding" in details
        assert torch.allclose(
            details["graph_embedding"],
            details["raw_graph_embedding"],
            atol=1e-6,
        )
    print("GPU Tangent and dual-branch test passed.")


if __name__ == "__main__":
    main()
