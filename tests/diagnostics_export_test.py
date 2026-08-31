import tempfile
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_abide import attach_sample_metadata, build_subject_summary


def main():
    with tempfile.TemporaryDirectory(prefix="smaf_diagnostics_") as temp_dir:
        data_root = Path(temp_dir)
        np.save(data_root / "labels.npy", np.asarray([0, 1], dtype=np.int64))
        np.save(data_root / "sub_ids.npy", np.asarray(["S1", "S2"]))
        np.save(data_root / "file_ids.npy", np.asarray(["F1", "F2"]))
        np.save(data_root / "site_ids.npy", np.asarray(["SITE_A", "SITE_B"]))

        frame = pd.DataFrame(
            [
                {
                    "seed": seed,
                    "fold": 1,
                    "sample_index": sample_index,
                    "label": sample_index,
                    "prediction": sample_index,
                    "prediction_probability": 0.2 + 0.6 * sample_index,
                    "weight_aal": 0.2 + 0.1 * seed,
                    "weight_cc200": 0.5 - 0.1 * seed,
                    "weight_ho": 0.3,
                }
                for seed in [0, 1]
                for sample_index in [0, 1]
            ]
        )
        frame = attach_sample_metadata(frame, data_root)
        summary = build_subject_summary(frame)

    assert frame["subject_id"].tolist() == ["S1", "S2", "S1", "S2"]
    assert frame["diagnosis"].tolist() == ["TC", "ASD", "TC", "ASD"]
    assert summary["observation_count"].tolist() == [2, 2]
    assert np.allclose(summary["weight_aal_mean"], [0.25, 0.25])
    assert np.allclose(summary["weight_cc200_mean"], [0.45, 0.45])
    print("Diagnostics export test passed.")


if __name__ == "__main__":
    main()
