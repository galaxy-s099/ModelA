from pathlib import Path
import argparse
import zipfile

import numpy as np
import pandas as pd


REQUIRED_FILES = [
    "file_ids.npy",
    "labels.npy",
    "X_aal.npy",
    "X_cc200.npy",
    "X_ho.npy",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a QC-filtered ABIDE FC dataset from new_processed."
    )
    parser.add_argument(
        "--source",
        default=r"D:\ABIDE_Project\new_processed",
        help="Source folder containing FC npy files and qc_report.csv.",
    )
    parser.add_argument(
        "--output",
        default=r"D:\ABIDE_Project\new_processed_qc10",
        help="Output folder for filtered npy files.",
    )
    parser.add_argument(
        "--max-zero-roi",
        type=int,
        default=10,
        help="Keep subjects with zero_roi_max <= this threshold.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source_dir = Path(args.source)
    output_dir = Path(args.output)
    qc_path = source_dir / "qc_report.csv"
    if not qc_path.exists():
        raise FileNotFoundError(f"Missing QC report: {qc_path}")

    qc = pd.read_csv(qc_path)
    keep_mask = qc["zero_roi_max"].to_numpy() <= args.max_zero_roi
    keep_indices = np.flatnonzero(keep_mask)
    if len(keep_indices) == 0:
        raise RuntimeError("QC filter removed all subjects")

    output_dir.mkdir(parents=True, exist_ok=True)
    filtered_qc = qc.iloc[keep_indices].reset_index(drop=True)
    filtered_qc.to_csv(output_dir / "qc_report.csv", index=False, encoding="utf-8-sig")

    removed_qc = qc.loc[~keep_mask].reset_index(drop=True)
    removed_qc.to_csv(output_dir / "removed_qc_report.csv", index=False, encoding="utf-8-sig")

    for filename in REQUIRED_FILES:
        array = np.load(source_dir / filename, allow_pickle=False)
        np.save(output_dir / filename, array[keep_indices])

    for optional_file in ["site_ids.npy", "sub_ids.npy"]:
        optional_path = source_dir / optional_file
        if optional_path.exists():
            array = np.load(optional_path, allow_pickle=False)
            np.save(output_dir / optional_file, array[keep_indices])

    zip_path = output_dir / f"abide-fc-three-atlas-new-qc{args.max_zero_roi}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename in REQUIRED_FILES:
            zf.write(output_dir / filename, arcname=filename)

    labels = np.load(output_dir / "labels.npy", allow_pickle=False)
    label_values, label_counts = np.unique(labels, return_counts=True)
    label_summary = {
        int(label): int(count)
        for label, count in zip(label_values, label_counts)
    }

    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Threshold: zero_roi_max <= {args.max_zero_roi}")
    print(f"Kept: {len(keep_indices)} / {len(qc)}")
    print(f"Removed: {len(qc) - len(keep_indices)}")
    print(f"Label counts: {label_summary}")
    for atlas in ["aal", "cc200", "ho"]:
        x = np.load(output_dir / f"X_{atlas}.npy", mmap_mode="r")
        print(f"X_{atlas}: shape={x.shape}, min={float(x.min()):.6f}, max={float(x.max()):.6f}")
    print(f"Kaggle zip: {zip_path}")


if __name__ == "__main__":
    main()
