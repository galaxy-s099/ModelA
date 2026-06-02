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
    pd.DataFrame(all_results).to_csv(
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


if __name__ == "__main__":
    main()
