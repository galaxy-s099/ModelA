import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import run_repeated_cv


def make_fc(rng, sample_count, num_nodes):
    matrix = rng.normal(size=(sample_count, num_nodes, num_nodes)).astype(np.float32)
    matrix = (matrix + matrix.transpose(0, 2, 1)) / 2
    matrix = np.tanh(matrix)
    diagonal = np.arange(num_nodes)
    matrix[:, diagonal, diagonal] = 1.0
    return matrix


def main():
    rng = np.random.default_rng(11)
    sample_count = 12
    atlas_specs = {
        "aal": {"num_nodes": 8, "fc_file": "X_aal.npy"},
        "cc200": {"num_nodes": 10, "fc_file": "X_cc200.npy"},
        "ho": {"num_nodes": 6, "fc_file": "X_ho.npy"},
    }

    with tempfile.TemporaryDirectory(prefix="smaf_proposal_") as temp_dir:
        temp_path = Path(temp_dir)
        np.save(temp_path / "labels.npy", np.asarray([0, 1] * 6, dtype=np.int64))
        for atlas_name, spec in atlas_specs.items():
            np.save(
                temp_path / spec["fc_file"],
                make_fc(rng, sample_count, spec["num_nodes"]),
            )

        config = {
            "data": {
                "data_root": str(temp_path),
                "atlases": atlas_specs,
            },
            "train": {
                "seeds": [0],
                "n_splits": 2,
                "epochs": 1,
                "batch_size": 4,
                "lr": 1.0e-3,
                "weight_decay": 1.0e-4,
                "use_best_val": False,
            },
            "model": {
                "hidden_dim": 16,
                "embedding_dim": 12,
                "num_signed_layers": 2,
                "dropout": 0.1,
                "temperature": 1.0,
                "add_negative_self_loops": True,
            },
            "loss": {
                "lambda_branch": 0.2,
                "lambda_reg": 0.1,
                "margin": 0.1,
            },
        }

        results, summary = run_repeated_cv(config)

    assert len(results) == 2
    assert "ACC_mean" in summary
    assert "Weight_aal" in results[0]
    assert "Weight_cc200" in results[0]
    assert "Weight_ho" in results[0]
    print("Pipeline smoke test passed.")


if __name__ == "__main__":
    main()
