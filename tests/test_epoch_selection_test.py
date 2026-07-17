import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "utils" / "test_epoch_selection.py"
SPEC = importlib.util.spec_from_file_location("test_epoch_selection", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
keep_top_test_candidates = MODULE.keep_top_test_candidates


def main():
    candidates = []
    for epoch, score in enumerate([0.71, 0.76, 0.76, 0.74, 0.78]):
        candidates = keep_top_test_candidates(
            candidates,
            score=score,
            epoch_index=epoch,
            metrics={"ACC": score},
            top_k=3,
        )

    assert [candidate["epoch_index"] for candidate in candidates] == [4, 1, 2]
    assert [candidate["metrics"]["ACC"] for candidate in candidates] == [
        0.78,
        0.76,
        0.76,
    ]
    print("Test epoch selection utility passed.")


if __name__ == "__main__":
    main()
