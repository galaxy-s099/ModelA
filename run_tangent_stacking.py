"""OOF-stacked fusion for the fold-local Tangent Pearson baseline.

Each outer training fold is split again to create out-of-fold atlas
probabilities. A three-feature L2 logistic meta-classifier learns the fusion
only from those out-of-fold predictions. The outer test fold is untouched
until final inference.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from baselines.tangent_pearson import TangentPearsonTransformer
from run_tangent_logistic import build_classifier, load_atlas_matrices
from utils.metrics import compute_metrics, summarize_results


def fit_predict_atlases(
    atlas_matrices,
    labels,
    fit_index,
    predict_index,
    config,
    seed,
):
    """Fit one fold-local Tangent classifier per atlas and predict one split."""
    tangent_config = config["tangent"]
    atlas_names = list(atlas_matrices)
    probabilities = np.zeros((len(predict_index), len(atlas_names)), dtype=np.float64)
    selected_c = {}

    for atlas_position, atlas_name in enumerate(atlas_names):
        transformer = TangentPearsonTransformer(
            shrinkage=tangent_config["shrinkage"],
            eigenvalue_floor=tangent_config["eigenvalue_floor"],
        )
        matrices = atlas_matrices[atlas_name]
        train_features = transformer.fit_transform(matrices[fit_index])
        predict_features = transformer.transform(matrices[predict_index])
        classifier = build_classifier(config, seed + atlas_position)
        classifier.fit(train_features, labels[fit_index])
        probabilities[:, atlas_position] = classifier.predict_proba(
            predict_features
        )[:, 1]
        selected_c[atlas_name] = float(
            classifier.named_steps["logisticregressioncv"].C_[0]
        )
    return probabilities, selected_c


def build_oof_probabilities(atlas_matrices, labels, outer_train_index, config, seed):
    """Generate base-model probabilities for every outer-training sample.

    No classifier sees the label of the sample it predicts in this function.
    """
    stacking_config = config["stacking"]
    outer_labels = labels[outer_train_index]
    splitter = StratifiedKFold(
        n_splits=stacking_config["inner_splits"],
        shuffle=True,
        random_state=seed,
    )
    oof_probabilities = np.zeros(
        (len(outer_train_index), len(atlas_matrices)),
        dtype=np.float64,
    )
    for inner_fold, (fit_local, predict_local) in enumerate(
        splitter.split(np.zeros(len(outer_train_index)), outer_labels),
        start=1,
    ):
        fit_index = outer_train_index[fit_local]
        predict_index = outer_train_index[predict_local]
        fold_probabilities, _ = fit_predict_atlases(
            atlas_matrices,
            labels,
            fit_index,
            predict_index,
            config,
            seed * 100 + inner_fold,
        )
        oof_probabilities[predict_local] = fold_probabilities
    return oof_probabilities


def build_meta_classifier(config, seed):
    stacking_config = config["stacking"]
    return LogisticRegression(
        penalty="l2",
        C=stacking_config["meta_C"],
        solver="liblinear",
        max_iter=stacking_config["meta_max_iter"],
        class_weight=stacking_config.get("meta_class_weight"),
        random_state=seed,
    )


def run_fold(atlas_matrices, labels, train_index, test_index, config, seed):
    oof_probabilities = build_oof_probabilities(
        atlas_matrices,
        labels,
        train_index,
        config,
        seed,
    )
    meta_classifier = build_meta_classifier(config, seed)
    meta_classifier.fit(oof_probabilities, labels[train_index])

    test_probabilities_by_atlas, selected_c = fit_predict_atlases(
        atlas_matrices,
        labels,
        train_index,
        test_index,
        config,
        seed,
    )
    uniform_probabilities = test_probabilities_by_atlas.mean(axis=1)
    stacked_probabilities = meta_classifier.predict_proba(
        test_probabilities_by_atlas
    )[:, 1]
    threshold = float(config["train"].get("decision_threshold", 0.5))
    stacked_metrics = compute_metrics(
        labels[test_index],
        stacked_probabilities,
        (stacked_probabilities >= threshold).astype(np.int64),
    )
    uniform_metrics = compute_metrics(
        labels[test_index],
        uniform_probabilities,
        (uniform_probabilities >= threshold).astype(np.int64),
    )
    stacked_metrics["Decision_Threshold"] = threshold
    for metric_name, value in uniform_metrics.items():
        stacked_metrics[f"Uniform_{metric_name}"] = value
    for atlas_name, value in selected_c.items():
        stacked_metrics[f"{atlas_name}_C"] = value
    for atlas_name, coefficient in zip(atlas_matrices, meta_classifier.coef_[0]):
        stacked_metrics[f"Meta_Coefficient_{atlas_name}"] = float(coefficient)
    stacked_metrics["Meta_Intercept"] = float(meta_classifier.intercept_[0])
    return stacked_metrics


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
        default="configs/abide_tangent_stacking_v10_1.yaml",
        help="Path to the Tangent Pearson stacking experiment YAML file.",
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
