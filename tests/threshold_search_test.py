import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from train import search_best_threshold
except ModuleNotFoundError as exc:
    if exc.name == "sklearn":
        print("Threshold search test skipped: sklearn is not installed.")
        sys.exit(0)
    raise


def main():
    labels = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.35, 0.45, 0.55, 0.65])
    threshold, acc = search_best_threshold(labels, probabilities)

    assert 0.46 <= threshold <= 0.55
    assert acc == 1.0

    print("Threshold search test passed.")
    print("threshold:", threshold)
    print("acc:", acc)


if __name__ == "__main__":
    main()
