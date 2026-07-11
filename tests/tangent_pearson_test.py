"""Small no-data test for the fold-local Tangent Pearson transformer."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.tangent_pearson import TangentPearsonTransformer


def make_correlation(seed, nodes):
    random = np.random.default_rng(seed)
    samples = random.normal(size=(80, nodes))
    return np.corrcoef(samples, rowvar=False)


def main():
    train = np.stack([make_correlation(seed, 8) for seed in range(4)])
    test = np.stack([make_correlation(seed + 10, 8) for seed in range(2)])
    transformer = TangentPearsonTransformer(shrinkage=0.05)
    train_features = transformer.fit_transform(train)
    test_features = transformer.transform(test)
    expected_features = 8 * 7 // 2
    assert train_features.shape == (4, expected_features)
    assert test_features.shape == (2, expected_features)
    assert np.isfinite(train_features).all()
    assert np.isfinite(test_features).all()
    print("Tangent Pearson transformer test passed.")


if __name__ == "__main__":
    main()
