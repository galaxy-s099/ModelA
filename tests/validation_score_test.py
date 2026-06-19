import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from train import compute_validation_score
except ModuleNotFoundError as exc:
    if exc.name == "sklearn":
        print("Validation score test skipped: sklearn is not installed.")
        sys.exit(0)
    raise


def main():
    metrics = {
        "ACC": 0.6,
        "AUC": 0.8,
        "SEN": 0.5,
        "SPE": 0.7,
        "F1": 0.4,
    }
    assert compute_validation_score(metrics, "AUC") == 0.8
    assert compute_validation_score(
        metrics,
        "COMPOSITE",
        ["ACC", "AUC", "F1"],
    ) == (0.6 + 0.8 + 0.4) / 3

    print("Validation score test passed.")


if __name__ == "__main__":
    main()
