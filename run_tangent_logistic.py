"""Repeated-CV Tangent Pearson + L2 logistic diagnostic baseline.

The script deliberately does not import the PyTorch proposal model.  It tests
whether a fold-local second-order FC representation has useful signal before
adding a tangent branch to SMAF-Net.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baselines.tangent_pearson import TangentPearsonTransformer
from utils.metrics import compute_metrics, summarize_results


def load_atlas_matrices(data_root, atlas_config):
    fc_file = atlas_config["fc_file"]
    matrices = np.load(Path(data_root) / fc_file, mmap_mode="r")
    node_count = int(atlas_config["num_nodes"])
    expected_shape = (node_count, node_count)
    if matrices.ndim != 3 or matrices.shape[1:] != expected_shape:
        raise ValueError(
            f"{fc_file}: expected samples x {expected_shape}, got {matrices.shape}"
        )
    return matrices


def build_classifier(config, seed):
    classifier_config = config["classifier"]
    return make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(
            Cs=classifier_config["C_values"],
            cv=classifier_config["inner_splits"],
            penalty="l2",
            solver="liblinear",
            scoring="roc_auc",
            max_iter=classifier_config["max_iter"],
            class_weight=classifier_config.get("class_weight"),
            random_state=seed,
        ),
    )


def run_fold(atlas_matrices, labels, train_index, test_index, config, seed):
    tangent_config = config["tangent"]
    atlas_probabilities = []
    selected_c = {}
    for atlas_name, matrices in atlas_matrices.items():
        transformer = TangentPearsonTransformer(
            shrinkage=tangent_config["shrinkage"],
            eigenvalue_floor=tangent_config["eigenvalue_floor"],
        )
        train_features = transformer.fit_transform(matrices[train_index])
        test_features = transformer.transform(matrices[test_index])
        classifier = build_classifier(config, seed)
        classifier.fit(train_features, labels[train_index])
        atlas_probabilities.append(classifier.predict_proba(test_features)[:, 1])
        selected_c[atlas_name] = float(
            classifier.named_steps["logisticregressioncv"].C_[0]
        )

    probabilities = np.mean(np.stack(atlas_probabilities, axis=0), axis=0)
    threshold = float(config["train"].get("decision_threshold", 0.5))
    predictions = (probabilities >= threshold).astype(np.int64)
    metrics = compute_metrics(labels[test_index], probabilities, predictions)
    metrics["Decision_Threshold"] = threshold
    for atlas_name, value in selected_c.items():
        metrics[f"{atlas_name}_C"] = value
    return metrics


def run_repeated_cv(config):
    data_root = Path(config["data"]["data_root"])
    labels = np.load(data_root / "labels.npy").astype(np.int64)
    atlas_matrices = {
        name: load_atlas_matrices(data_root, atlas_config)
        for name, atlas_config in config["data"]["atlases"].items()
    }
    if any(len(matrices) != len(labels) for matrices in atlas_matrices.values()):
        raise ValueError("Every atlas must contain the same number of samples")

    results = []
    for seed in config["train"]["seeds"]:
        print(f"\n===== Seed {seed} =====")
        splitter = StratifiedKFold(
            n_splits=config["train"]["n_splits"],
            shuffle=True,
            random_state=seed,
        )
        for fold, (train_index, test_index) in enumerate(
            splitter.split(np.zeros(len(labels)), labels),
            start=1,
        ):
            metrics = run_fold(
                atlas_matrices,
                labels,
                train_index,
                test_index,
                config,
                seed * 100 + fold,
            )
            metrics.update({"Seed": seed, "Fold": fold})
            results.append(metrics)
            print(
                f"Seed {seed} | Fold {fold}: "
                f"ACC={metrics['ACC']:.4f}, AUC={metrics['AUC']:.4f}, "
                f"SEN={metrics['SEN']:.4f}, SPE={metrics['SPE']:.4f}, "
                f"F1={metrics['F1']:.4f}"
            )
    return results, summarize_results(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/abide_tangent_logistic_v10_0.yaml",
        help="Path to the Tangent Pearson experiment YAML file.",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    results, summary = run_repeated_cv(config)
    experiment_name = config_path.stem
    pd.DataFrame(results).to_csv(f"{experiment_name}_all_folds.csv", index=False)
    pd.DataFrame([summary]).to_csv(f"{experiment_name}_summary.csv", index=False)
    print("\n========== Final Result ==========")
    for metric, value in summary.items():
        print(f"{metric}: {value:.4f}")
    print("\nSaved:")
    print(f"{experiment_name}_all_folds.csv")
    print(f"{experiment_name}_summary.csv")


if __name__ == "__main__":
    main()
