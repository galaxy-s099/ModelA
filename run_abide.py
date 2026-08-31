import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from train import run_repeated_cv


def load_optional_metadata(data_root, filename, sample_count):
    path = Path(data_root) / filename
    if not path.exists():
        return None
    values = np.load(path, allow_pickle=True)
    if len(values) != sample_count:
        raise ValueError(
            f"{filename} sample count {len(values)} does not match "
            f"label count {sample_count}"
        )
    cleaned = []
    for value in values:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        elif isinstance(value, np.generic):
            value = value.item()
        cleaned.append(value)
    return np.asarray(cleaned, dtype=object)


def attach_sample_metadata(frame, data_root):
    if frame.empty:
        return frame

    labels = np.load(Path(data_root) / "labels.npy")
    sample_indices = frame["sample_index"].to_numpy(dtype=np.int64)
    if sample_indices.min() < 0 or sample_indices.max() >= len(labels):
        raise ValueError("Diagnostic sample_index is outside the dataset range")
    if not np.array_equal(
        frame["label"].to_numpy(dtype=np.int64),
        labels[sample_indices].astype(np.int64),
    ):
        raise ValueError("Diagnostic labels do not match labels.npy")

    sub_ids = load_optional_metadata(data_root, "sub_ids.npy", len(labels))
    file_ids = load_optional_metadata(data_root, "file_ids.npy", len(labels))
    site_ids = load_optional_metadata(data_root, "site_ids.npy", len(labels))

    if sub_ids is not None:
        frame["subject_id"] = sub_ids[sample_indices]
    elif file_ids is not None:
        frame["subject_id"] = file_ids[sample_indices]
    else:
        frame["subject_id"] = sample_indices
    if file_ids is not None:
        frame["file_id"] = file_ids[sample_indices]
    if site_ids is not None:
        frame["site_id"] = site_ids[sample_indices]
    frame["diagnosis"] = np.where(frame["label"] == 1, "ASD", "TC")

    preferred_columns = [
        "seed",
        "fold",
        "sample_index",
        "subject_id",
        "file_id",
        "site_id",
        "label",
        "diagnosis",
    ]
    ordered_columns = [
        column for column in preferred_columns if column in frame.columns
    ]
    ordered_columns.extend(
        column for column in frame.columns if column not in ordered_columns
    )
    return frame[ordered_columns]


def build_subject_summary(sample_frame):
    if sample_frame.empty:
        return sample_frame

    grouped = sample_frame.groupby("sample_index", sort=True)
    summary = pd.DataFrame(
        {
            "sample_index": grouped.size().index,
            "observation_count": grouped.size().to_numpy(),
        }
    )
    for column in [
        "subject_id",
        "file_id",
        "site_id",
        "label",
        "diagnosis",
    ]:
        if column in sample_frame.columns:
            summary[column] = grouped[column].first().to_numpy()

    excluded = {
        "seed",
        "fold",
        "sample_index",
        "subject_id",
        "file_id",
        "site_id",
        "label",
        "diagnosis",
        "prediction",
        "checkpoint_count",
    }
    numeric_columns = [
        column
        for column in sample_frame.select_dtypes(include=[np.number]).columns
        if column not in excluded
    ]
    for column in numeric_columns:
        summary[f"{column}_mean"] = grouped[column].mean().to_numpy()
        summary[f"{column}_std"] = grouped[column].std(ddof=0).to_numpy()
    summary["positive_prediction_rate"] = grouped["prediction"].mean().to_numpy()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/abide_proposal.yaml",
        help="Path to an experiment YAML file.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    export_diagnostics = config.get("output", {}).get(
        "export_sample_diagnostics",
        False,
    )
    run_result = run_repeated_cv(
        config,
        return_diagnostics=export_diagnostics,
    )
    if export_diagnostics:
        all_results, summary, diagnostics = run_result
    else:
        all_results, summary = run_result
    experiment_name = config_path.stem
    all_results_frame = pd.DataFrame(all_results)
    all_results_frame.to_csv(
        f"{experiment_name}_all_folds.csv",
        index=False,
    )
    pd.DataFrame([summary]).to_csv(
        f"{experiment_name}_summary.csv",
        index=False,
    )

    print("\nSaved:")
    print(f"{experiment_name}_all_folds.csv")
    print(f"{experiment_name}_summary.csv")
    if export_diagnostics:
        sample_frame = attach_sample_metadata(
            pd.DataFrame(diagnostics["sample_rows"]),
            config["data"]["data_root"],
        )
        checkpoint_frame = attach_sample_metadata(
            pd.DataFrame(diagnostics["checkpoint_rows"]),
            config["data"]["data_root"],
        )
        subject_summary = build_subject_summary(sample_frame)
        sample_path = f"{experiment_name}_sample_diagnostics.csv"
        checkpoint_path = f"{experiment_name}_checkpoint_diagnostics.csv"
        subject_path = f"{experiment_name}_subject_summary.csv"
        sample_frame.to_csv(sample_path, index=False)
        checkpoint_frame.to_csv(checkpoint_path, index=False)
        subject_summary.to_csv(subject_path, index=False)
        print(sample_path)
        print(checkpoint_path)
        print(subject_path)
    if "Test_Rank" in all_results_frame.columns:
        distribution = (
            all_results_frame.groupby("Best_Test_Epoch", as_index=False)
            .agg(
                Count=("Best_Test_Epoch", "size"),
                Mean_ACC=("ACC", "mean"),
                Mean_AUC=("AUC", "mean"),
            )
            .sort_values("Best_Test_Epoch")
        )
        distribution.to_csv(
            f"{experiment_name}_epoch_distribution.csv",
            index=False,
        )
        epoch_bins = all_results_frame.assign(
            Epoch_Bin_Start=(
                ((all_results_frame["Best_Test_Epoch"] - 1) // 10) * 10 + 1
            )
        )
        epoch_bins["Epoch_Bin"] = epoch_bins["Epoch_Bin_Start"].map(
            lambda start: f"{int(start)}-{int(start) + 9}"
        )
        bin_distribution = (
            epoch_bins.groupby("Epoch_Bin", as_index=False, sort=False)
            .agg(
                Count=("Epoch_Bin", "size"),
                Mean_ACC=("ACC", "mean"),
                Mean_AUC=("AUC", "mean"),
            )
        )
        bin_distribution.to_csv(
            f"{experiment_name}_epoch_bin_distribution.csv",
            index=False,
        )
        print(f"{experiment_name}_epoch_distribution.csv")
        print(f"{experiment_name}_epoch_bin_distribution.csv")


if __name__ == "__main__":
    main()
