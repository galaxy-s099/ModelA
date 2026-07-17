import argparse
from pathlib import Path

import pandas as pd
import yaml

from train import run_repeated_cv


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

    all_results, summary = run_repeated_cv(config)
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
