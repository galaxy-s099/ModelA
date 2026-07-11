"""GPU-accelerated fold-local Tangent Pearson preprocessing for FC matrices."""

import numpy as np
import torch


def _regularize_correlation(matrices, shrinkage):
    matrices = 0.5 * (matrices + matrices.transpose(-1, -2))
    node_count = matrices.shape[-1]
    identity = torch.eye(
        node_count,
        dtype=matrices.dtype,
        device=matrices.device,
    )
    regularized = (1.0 - shrinkage) * matrices + shrinkage * identity
    diagonal = torch.arange(node_count, device=matrices.device)
    regularized[:, diagonal, diagonal] = 1.0
    return regularized


def _matrix_log_spd(matrices, eigenvalue_floor):
    eigenvalues, eigenvectors = torch.linalg.eigh(matrices)
    eigenvalues = eigenvalues.clamp_min(eigenvalue_floor)
    return (eigenvectors * eigenvalues.log().unsqueeze(-2)) @ eigenvectors.transpose(
        -1,
        -2,
    )


def _matrix_exp_symmetric(matrix):
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    return (eigenvectors * eigenvalues.exp().unsqueeze(-2)) @ eigenvectors.transpose(
        -1,
        -2,
    )


def _inverse_square_root_spd(matrix, eigenvalue_floor):
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    inverse_square_root = eigenvalues.clamp_min(eigenvalue_floor).rsqrt()
    return (eigenvectors * inverse_square_root.unsqueeze(-2)) @ eigenvectors.transpose(
        -1,
        -2,
    )


class GPUTangentPearsonTransformer:
    """Log-Euclidean Tangent Pearson transform with batched torch eigensolvers.

    ``fit`` receives only the current outer-training FC matrices.  ``transform``
    can then be applied to train, validation, and outer-test matrices using the
    fitted reference without accessing their labels.
    """

    def __init__(
        self,
        device,
        shrinkage=0.05,
        eigenvalue_floor=1.0e-6,
        batch_size=32,
    ):
        if not 0.0 < float(shrinkage) < 1.0:
            raise ValueError("shrinkage must be in (0, 1)")
        if float(eigenvalue_floor) <= 0:
            raise ValueError("eigenvalue_floor must be positive")
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        self.device = torch.device(device)
        self.shrinkage = float(shrinkage)
        self.eigenvalue_floor = float(eigenvalue_floor)
        self.batch_size = int(batch_size)
        self.inverse_square_root_ = None

    def _to_device(self, matrices):
        # FC memmaps are read-only; make a writable float32 batch before torch.
        batch = np.array(matrices, dtype=np.float32, copy=True)
        return torch.from_numpy(batch).to(self.device, non_blocking=True)

    def fit(self, train_matrices):
        train_matrices = np.asarray(train_matrices)
        if train_matrices.ndim != 3 or train_matrices.shape[-1] != train_matrices.shape[-2]:
            raise ValueError(
                "Expected train FC matrices with shape samples x nodes x nodes"
            )
        node_count = train_matrices.shape[-1]
        log_sum = torch.zeros(
            (node_count, node_count),
            device=self.device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            for start in range(0, len(train_matrices), self.batch_size):
                batch = self._to_device(
                    train_matrices[start : start + self.batch_size]
                )
                log_sum += _matrix_log_spd(
                    _regularize_correlation(batch, self.shrinkage),
                    self.eigenvalue_floor,
                ).sum(dim=0)
            mean_log = log_sum / len(train_matrices)
            reference = _matrix_exp_symmetric(mean_log)
            self.inverse_square_root_ = _inverse_square_root_spd(
                reference,
                self.eigenvalue_floor,
            )
        return self

    def transform(self, matrices):
        if self.inverse_square_root_ is None:
            raise RuntimeError("GPUTangentPearsonTransformer must be fitted first")
        matrices = np.asarray(matrices)
        if matrices.ndim != 3:
            raise ValueError("Expected FC matrices with shape samples x nodes x nodes")
        output = np.empty(matrices.shape, dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(matrices), self.batch_size):
                stop = min(start + self.batch_size, len(matrices))
                batch = self._to_device(matrices[start:stop])
                batch = _regularize_correlation(batch, self.shrinkage)
                whitened = (
                    self.inverse_square_root_.unsqueeze(0)
                    @ batch
                    @ self.inverse_square_root_.unsqueeze(0)
                )
                tangent = _matrix_log_spd(whitened, self.eigenvalue_floor)
                output[start:stop] = tangent.cpu().numpy()
        return output


def build_fold_tangent_matrices(
    data_root,
    atlas_specs,
    fit_indices,
    device,
    tangent_config,
):
    """Create fitted Tangent FC arrays for all samples in one outer fold.

    The returned arrays are CPU NumPy arrays for DataLoader access. Their
    expensive matrix-log operations run on ``device`` and are performed once
    per atlas/fold, not during every training epoch.
    """
    data_root = str(data_root)
    tangent_matrices = {}
    for atlas_name, spec in atlas_specs.items():
        fc_path = f"{data_root}/{spec.get('fc_file', f'X_{atlas_name}.npy')}"
        fc_matrices = np.load(fc_path, mmap_mode="r")
        transformer = GPUTangentPearsonTransformer(
            device=device,
            shrinkage=tangent_config.get("shrinkage", 0.05),
            eigenvalue_floor=tangent_config.get("eigenvalue_floor", 1.0e-6),
            batch_size=tangent_config.get("batch_size", 32),
        )
        transformer.fit(fc_matrices[fit_indices])
        tangent_matrices[atlas_name] = transformer.transform(fc_matrices)
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    return tangent_matrices
