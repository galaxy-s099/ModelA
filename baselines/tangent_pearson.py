"""Fold-local tangent-space features for precomputed Pearson FC matrices.

The original Tangent Pearson workflow computes a Pearson correlation matrix for
each participant and maps those matrices to the tangent space defined by the
training cohort.  This project stores the Pearson matrices directly, so this
module starts from those matrices and never accesses test-fold samples while
estimating the reference matrix.
"""

import numpy as np


def _as_symmetric(matrices):
    matrices = np.asarray(matrices, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[-1] != matrices.shape[-2]:
        raise ValueError(
            "Expected correlation matrices with shape samples x nodes x nodes"
        )
    return 0.5 * (matrices + np.swapaxes(matrices, -1, -2))


def _matrix_log_spd(matrices, eigenvalue_floor):
    eigenvalues, eigenvectors = np.linalg.eigh(matrices)
    eigenvalues = np.clip(eigenvalues, eigenvalue_floor, None)
    return (eigenvectors * np.log(eigenvalues)[..., None, :]) @ np.swapaxes(
        eigenvectors,
        -1,
        -2,
    )


def _matrix_exp_symmetric(matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return (eigenvectors * np.exp(eigenvalues)[None, :]) @ eigenvectors.T


def _inverse_square_root_spd(matrix, eigenvalue_floor):
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, eigenvalue_floor, None)
    return (eigenvectors * np.power(eigenvalues, -0.5)[None, :]) @ eigenvectors.T


class TangentPearsonTransformer:
    """Map FC matrices to a log-Euclidean tangent space.

    A log-Euclidean reference is used instead of an iterative Riemannian mean.
    It is positive definite after shrinkage, deterministic, and feasible for
    repeated cross-validation with the 200-ROI atlas.
    """

    def __init__(self, shrinkage=0.05, eigenvalue_floor=1.0e-6):
        if not 0.0 < shrinkage < 1.0:
            raise ValueError("shrinkage must be in (0, 1)")
        if eigenvalue_floor <= 0:
            raise ValueError("eigenvalue_floor must be positive")
        self.shrinkage = float(shrinkage)
        self.eigenvalue_floor = float(eigenvalue_floor)
        self.reference_ = None
        self.inverse_square_root_ = None
        self.upper_indices_ = None

    def _regularize(self, matrices):
        matrices = _as_symmetric(matrices)
        node_count = matrices.shape[-1]
        identity = np.eye(node_count, dtype=matrices.dtype)
        regularized = (
            (1.0 - self.shrinkage) * matrices + self.shrinkage * identity
        )
        regularized[..., np.arange(node_count), np.arange(node_count)] = 1.0
        return regularized

    def fit(self, train_matrices):
        train_matrices = self._regularize(train_matrices)
        mean_log = _matrix_log_spd(
            train_matrices,
            self.eigenvalue_floor,
        ).mean(axis=0)
        self.reference_ = _matrix_exp_symmetric(mean_log)
        self.inverse_square_root_ = _inverse_square_root_spd(
            self.reference_,
            self.eigenvalue_floor,
        )
        self.upper_indices_ = np.triu_indices(self.reference_.shape[0], k=1)
        return self

    def transform(self, matrices):
        if self.inverse_square_root_ is None:
            raise RuntimeError("TangentPearsonTransformer must be fitted first")
        matrices = self._regularize(matrices)
        whitened = (
            self.inverse_square_root_[None, :, :]
            @ matrices
            @ self.inverse_square_root_[None, :, :]
        )
        tangent_matrices = _matrix_log_spd(
            whitened,
            self.eigenvalue_floor,
        )
        return tangent_matrices[:, self.upper_indices_[0], self.upper_indices_[1]]

    def fit_transform(self, train_matrices):
        return self.fit(train_matrices).transform(train_matrices)
