from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ABIDEMultiAtlasDataset(Dataset):
    """
    Load precomputed FC matrices for multiple atlases.

    By default the model uses each FC row as the ROI node feature. An atlas
    spec may optionally provide node_feature_file for externally extracted
    BOLD/ROI features with shape samples x nodes x feature_dim.
    """

    def __init__(self, data_root, atlas_specs, indices=None):
        self.data_root = Path(data_root)
        self.atlas_specs = OrderedDict(atlas_specs)
        self.labels = np.load(self.data_root / "labels.npy").astype(np.int64)
        self.site_ids = None
        site_path = self.data_root / "site_ids.npy"
        if site_path.exists():
            raw_site_ids = np.load(site_path, allow_pickle=True)
            if len(raw_site_ids) != len(self.labels):
                raise ValueError(
                    f"site_ids sample count {len(raw_site_ids)} does not match "
                    f"label count {len(self.labels)}"
                )
            if np.issubdtype(raw_site_ids.dtype, np.number):
                self.site_ids = raw_site_ids.astype(np.int64)
            else:
                _, encoded_site_ids = np.unique(raw_site_ids, return_inverse=True)
                self.site_ids = encoded_site_ids.astype(np.int64)
        self.fc_matrices = {}
        self.node_features = {}

        for atlas_name, spec in self.atlas_specs.items():
            fc_file = spec.get("fc_file", f"X_{atlas_name}.npy")
            fc = np.load(self.data_root / fc_file, mmap_mode="r")

            num_nodes = int(spec["num_nodes"])
            expected_shape = (num_nodes, num_nodes)
            if fc.ndim != 3 or fc.shape[1:] != expected_shape:
                raise ValueError(
                    f"{atlas_name}: expected FC matrices with trailing shape "
                    f"{expected_shape}, received {fc.shape}"
                )
            if len(fc) != len(self.labels):
                raise ValueError(
                    f"{atlas_name}: FC sample count {len(fc)} does not match "
                    f"label count {len(self.labels)}"
                )

            self.fc_matrices[atlas_name] = fc

            node_feature_file = spec.get("node_feature_file")
            if node_feature_file:
                features = np.load(self.data_root / node_feature_file, mmap_mode="r")
                feature_dim = int(spec["feature_dim"])
                expected_feature_shape = (num_nodes, feature_dim)
                if features.ndim != 3 or features.shape[1:] != expected_feature_shape:
                    raise ValueError(
                        f"{atlas_name}: expected node features with trailing shape "
                        f"{expected_feature_shape}, received {features.shape}"
                    )
                if len(features) != len(self.labels):
                    raise ValueError(
                        f"{atlas_name}: node feature sample count {len(features)} "
                        f"does not match label count {len(self.labels)}"
                    )
                self.node_features[atlas_name] = features

        if indices is None:
            self.indices = np.arange(len(self.labels))
        else:
            self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        real_index = self.indices[index]
        sample = {}
        for atlas_name, fc in self.fc_matrices.items():
            matrix = np.clip(fc[real_index], -1.0, 1.0).astype(np.float32, copy=False)
            sample[atlas_name] = torch.from_numpy(matrix)

        for atlas_name, features in self.node_features.items():
            node_features = np.asarray(features[real_index], dtype=np.float32).copy()
            sample[f"{atlas_name}_features"] = torch.from_numpy(node_features)

        sample["label"] = torch.tensor(self.labels[real_index], dtype=torch.long)
        if self.site_ids is not None:
            sample["site"] = torch.tensor(
                self.site_ids[real_index],
                dtype=torch.long,
            )
        return sample


def load_labels(data_root):
    return np.load(Path(data_root) / "labels.npy").astype(np.int64)


def load_site_count(data_root, site_file="site_ids.npy"):
    site_path = Path(data_root) / site_file
    site_ids = np.load(site_path, allow_pickle=True)
    if np.issubdtype(site_ids.dtype, np.number):
        site_ids = site_ids.astype(np.int64)
        return int(site_ids.max()) + 1
    return int(len(np.unique(site_ids)))
