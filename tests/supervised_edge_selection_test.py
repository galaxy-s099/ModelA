import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import sklearn  # noqa: F401
except ModuleNotFoundError:
    print("Supervised edge selection test skipped: sklearn is not installed.")
    sys.exit(0)

from data.abide_dataset import ABIDEMultiAtlasDataset
from train import build_supervised_edge_selected_atlas_specs


def main():
    with TemporaryDirectory() as tmpdir:
        data_root = Path(tmpdir)
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        fc = np.zeros((4, 3, 3), dtype=np.float32)

        # Edge 0-1 is label-discriminative; other edges are weak noise.
        fc[:2, 0, 1] = -0.8
        fc[:2, 1, 0] = -0.8
        fc[2:, 0, 1] = 0.8
        fc[2:, 1, 0] = 0.8
        fc[:, 0, 2] = 0.05
        fc[:, 2, 0] = 0.05
        fc[:, 1, 2] = -0.02
        fc[:, 2, 1] = -0.02

        np.save(data_root / "labels.npy", labels)
        np.save(data_root / "X_toy.npy", fc)

        atlas_specs = {
            "toy": {
                "num_nodes": 3,
                "fc_file": "X_toy.npy",
            }
        }
        selected_specs, stats = build_supervised_edge_selected_atlas_specs(
            data_root,
            atlas_specs,
            train_indices=np.arange(4),
            labels=labels,
            selection_config={
                "enabled": True,
                "ratio": 1.0 / 6.0,
            },
        )
        edge_mask = selected_specs["toy"]["edge_mask"]
        assert edge_mask[0, 1] == 1.0
        assert edge_mask[1, 0] == 1.0
        assert edge_mask[0, 2] == 0.0
        assert stats["SelectedEdges_toy"] == 1

        dataset = ABIDEMultiAtlasDataset(data_root, selected_specs)
        sample = dataset[0]["toy"].numpy()
        assert sample[0, 1] == -0.8
        assert sample[0, 2] == 0.0
        assert sample[1, 2] == 0.0

    print("Supervised edge selection test passed.")


if __name__ == "__main__":
    main()
