from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader

from data.abide_dataset import ABIDEMultiAtlasDataset, load_labels
from models.losses import ProposalLoss
from models.smaf_edge_energy_net import SMAFEdgeEnergyNet
from models.smaf_edge_gated_proposal_net import SMAFEdgeGatedProposalNet
from models.smaf_edge_proposal_net import SMAFEdgeProposalNet
from models.smaf_proposal_net import SMAFProposalNet
from utils.metrics import compute_metrics, summarize_results
from utils.seed import set_seed


def move_batch_to_device(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def build_model(config):
    model_config = config["model"]
    if model_config.get("model_name") == "smaf_edge_proposal_v2":
        return SMAFEdgeProposalNet(
            atlas_specs=config["data"]["atlases"],
            hidden_dim=model_config["hidden_dim"],
            embedding_dim=model_config["embedding_dim"],
            dropout=model_config["dropout"],
            temperature=model_config["temperature"],
        )

    if model_config.get("model_name") == "smaf_edge_gated_proposal_v2_1":
        return SMAFEdgeGatedProposalNet(
            atlas_specs=config["data"]["atlases"],
            hidden_dim=model_config["hidden_dim"],
            embedding_dim=model_config["embedding_dim"],
            dropout=model_config["dropout"],
        )

    if model_config.get("model_name") == "smaf_edge_energy_v2_2":
        return SMAFEdgeEnergyNet(
            atlas_specs=config["data"]["atlases"],
            hidden_dim=model_config["hidden_dim"],
            embedding_dim=model_config["embedding_dim"],
            dropout=model_config["dropout"],
            temperature=model_config["temperature"],
            use_atlas_prior=model_config.get("use_atlas_prior", False),
            use_sample_gate=model_config.get("use_sample_gate", False),
            sample_gate_scale=model_config.get("sample_gate_scale", 1.0),
            use_residual_classifier=model_config.get(
                "use_residual_classifier",
                False,
            ),
            residual_classifier_scale=model_config.get(
                "residual_classifier_scale",
                0.5,
            ),
            use_dual_energy_blend=model_config.get(
                "use_dual_energy_blend",
                False,
            ),
            dual_energy_blend_alpha=model_config.get(
                "dual_energy_blend_alpha",
                0.5,
            ),
            use_shared_correction=model_config.get(
                "use_shared_correction",
                False,
            ),
            shared_correction_scale=model_config.get(
                "shared_correction_scale",
                0.25,
            ),
            use_branch_residual_correction=model_config.get(
                "use_branch_residual_correction",
                False,
            ),
            branch_residual_correction_scale=model_config.get(
                "branch_residual_correction_scale",
                0.1,
            ),
            branch_residual_hidden_dim=model_config.get(
                "branch_residual_hidden_dim",
                32,
            ),
            atlas_overrides=model_config.get("atlas_overrides"),
            use_node_summary=model_config.get("use_node_summary", False),
            node_summary_hidden_dim=model_config.get(
                "node_summary_hidden_dim",
                64,
            ),
            node_summary_embedding_dim=model_config.get(
                "node_summary_embedding_dim",
                32,
            ),
            use_edge_residual=model_config.get("use_edge_residual", False),
            edge_residual_hidden_dim=model_config.get(
                "edge_residual_hidden_dim",
                64,
            ),
            edge_residual_scale=model_config.get(
                "edge_residual_scale",
                0.25,
            ),
        )

    return SMAFProposalNet(
        atlas_specs=config["data"]["atlases"],
        hidden_dim=model_config["hidden_dim"],
        embedding_dim=model_config["embedding_dim"],
        num_signed_layers=model_config["num_signed_layers"],
        dropout=model_config["dropout"],
        temperature=model_config["temperature"],
        add_negative_self_loops=model_config.get(
            "add_negative_self_loops",
            True,
        ),
        use_signed_residual=model_config.get("use_signed_residual", False),
        use_signed_edge_gate=model_config.get("use_signed_edge_gate", False),
        edge_gate_scale=model_config.get("edge_gate_scale", 0.5),
        fusion_mode=model_config.get("fusion_mode", "energy_decision"),
    )


def build_loss(config):
    loss_config = config["loss"]
    return ProposalLoss(
        lambda_branch=loss_config["lambda_branch"],
        lambda_reg=loss_config["lambda_reg"],
        margin=loss_config["margin"],
        regularization_mode=loss_config.get(
            "regularization_mode",
            "proposal_literal",
        ),
        class_weights=loss_config.get("class_weights"),
        lambda_weight_align=loss_config.get("lambda_weight_align", 0.0),
        weight_align_temperature=loss_config.get(
            "weight_align_temperature",
            1.0,
        ),
    )


def search_best_threshold(y_true, y_prob):
    best_threshold = 0.5
    best_acc = -1.0

    for threshold in np.arange(0.30, 0.71, 0.01):
        y_pred = (y_prob >= threshold).astype(int)
        acc = (y_pred == y_true).mean()
        if acc > best_acc:
            best_threshold = threshold
            best_acc = acc

    return float(best_threshold), float(best_acc)


def normalize_metric_name(metric_name):
    return str(metric_name).upper()


def compute_validation_score(metrics, select_metric, composite_metrics=None):
    select_metric = normalize_metric_name(select_metric)
    if select_metric == "COMPOSITE":
        metric_names = composite_metrics or ["ACC", "AUC", "F1"]
        metric_names = [normalize_metric_name(metric_name) for metric_name in metric_names]
        missing_metrics = [
            metric_name
            for metric_name in metric_names
            if metric_name not in metrics
        ]
        if missing_metrics:
            raise ValueError(
                f"Unsupported composite validation metrics: {missing_metrics}. "
                f"Expected values from {sorted(metrics.keys())}"
            )
        return float(np.mean([metrics[metric_name] for metric_name in metric_names]))

    if select_metric not in metrics:
        raise ValueError(
            f"Unsupported val_select_metric: {select_metric}. "
            f"Expected one of {sorted(metrics.keys())} or COMPOSITE"
        )
    return metrics[select_metric]


def add_probability_metrics(metrics, prefix, labels, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    prefixed_metrics = compute_metrics(labels, probabilities, predictions)
    for key, value in prefixed_metrics.items():
        metrics[f"{prefix}_{key}"] = value


def update_averaged_state(averaged_state, model_state, count):
    if averaged_state is None:
        return {
            key: value.detach().clone()
            for key, value in model_state.items()
        }, 1

    next_count = count + 1
    for key, value in model_state.items():
        value = value.detach()
        if torch.is_floating_point(value):
            averaged_state[key].mul_(count / next_count).add_(
                value,
                alpha=1.0 / next_count,
            )
        else:
            averaged_state[key] = value.clone()

    return averaged_state, next_count


def clone_state_dict(model):
    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }


def evaluate_model(model, dataloader, device, threshold=0.5):
    model.eval()
    probabilities = []
    predictions = []
    labels = []
    atlas_weight_outputs = {
        "Weight": [],
        "BaseWeight": [],
        "GatedWeight": [],
    }
    sample_gate_logits = []
    weighted_probabilities = []
    branch_probabilities = {
        atlas_name: []
        for atlas_name in getattr(model, "atlas_names", [])
    }

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            output = model(batch)
            batch_probabilities = F.softmax(output["fusion_logits"], dim=-1)[:, 1]

            probabilities.extend(batch_probabilities.cpu().numpy())
            predictions.extend((batch_probabilities >= threshold).long().cpu().numpy())
            labels.extend(batch["label"].cpu().numpy())

            weighted_logits = output.get("weighted_logits")
            if weighted_logits is not None:
                weighted_batch_probabilities = F.softmax(
                    weighted_logits,
                    dim=-1,
                )[:, 1]
                weighted_probabilities.extend(
                    weighted_batch_probabilities.cpu().numpy()
                )

            branch_logits = output.get("branch_logits")
            if branch_logits is not None and branch_probabilities:
                branch_batch_probabilities = F.softmax(branch_logits, dim=-1)[:, :, 1]
                for atlas_index, atlas_name in enumerate(model.atlas_names):
                    branch_probabilities[atlas_name].extend(
                        branch_batch_probabilities[:, atlas_index].cpu().numpy()
                    )

            for output_key, metric_prefix in [
                ("atlas_weight", "Weight"),
                ("base_atlas_weight", "BaseWeight"),
                ("gated_atlas_weight", "GatedWeight"),
            ]:
                weight_output = output.get(output_key)
                if weight_output is not None:
                    atlas_weight_outputs[metric_prefix].append(
                        weight_output.cpu().numpy()
                    )

            sample_gate_output = output.get("sample_gate_logits")
            if sample_gate_output is not None:
                sample_gate_logits.append(sample_gate_output.cpu().numpy())

    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    predictions = np.asarray(predictions)
    metrics = compute_metrics(labels, probabilities, predictions)

    if weighted_probabilities:
        add_probability_metrics(
            metrics,
            "Weighted",
            labels,
            np.asarray(weighted_probabilities),
            threshold,
        )

    for atlas_name, atlas_probabilities in branch_probabilities.items():
        if atlas_probabilities:
            add_probability_metrics(
                metrics,
                f"Branch_{atlas_name}",
                labels,
                np.asarray(atlas_probabilities),
                threshold,
            )

    for metric_prefix, weight_batches in atlas_weight_outputs.items():
        if weight_batches:
            mean_atlas_weight = np.concatenate(weight_batches, axis=0).mean(axis=0)
            for atlas_name, weight in zip(model.atlas_names, mean_atlas_weight):
                metrics[f"{metric_prefix}_{atlas_name}"] = float(weight)

    if sample_gate_logits:
        mean_gate_logits = np.concatenate(sample_gate_logits, axis=0).mean(axis=0)
        for atlas_name, gate_logit in zip(model.atlas_names, mean_gate_logits):
            metrics[f"GateLogit_{atlas_name}"] = float(gate_logit)

    return metrics, probabilities, labels


def combine_probability_ensemble(
    ensemble_probabilities,
    labels,
    decision_threshold,
    weighting="uniform",
):
    stacked_probabilities = np.stack(ensemble_probabilities, axis=0)
    if weighting == "confidence":
        weights = np.abs(stacked_probabilities - 0.5) * 2.0
        weight_sum = weights.sum(axis=0)
        mean_probabilities = np.divide(
            (weights * stacked_probabilities).sum(axis=0),
            weight_sum,
            out=stacked_probabilities.mean(axis=0),
            where=weight_sum > 1e-12,
        )
    elif weighting == "uniform":
        mean_probabilities = stacked_probabilities.mean(axis=0)
    else:
        raise ValueError(
            "Unsupported checkpoint_ensemble_weighting: "
            f"{weighting}. Expected uniform or confidence."
        )

    predictions = (mean_probabilities >= decision_threshold).astype(int)
    metrics = compute_metrics(labels, mean_probabilities, predictions)
    return metrics, mean_probabilities


def train_one_fold(data_root, train_idx, test_idx, seed, config, device):
    set_seed(seed)
    labels = load_labels(data_root)
    train_config = config["train"]
    atlas_specs = config["data"]["atlases"]
    use_best_val = train_config.get("use_best_val", False)
    use_best_test = train_config.get("use_best_test", False)

    if use_best_val:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=train_config.get("val_ratio", 0.15),
            random_state=seed,
        )
        train_local_idx, val_local_idx = next(
            splitter.split(np.zeros(len(train_idx)), labels[train_idx])
        )
        train_sub_idx = train_idx[train_local_idx]
        val_idx = train_idx[val_local_idx]
    else:
        train_sub_idx = train_idx
        val_idx = None

    train_loader = DataLoader(
        ABIDEMultiAtlasDataset(data_root, atlas_specs, train_sub_idx),
        batch_size=train_config["batch_size"],
        shuffle=True,
    )
    train_eval_loader = None
    if train_config.get("search_train_threshold", False):
        train_eval_loader = DataLoader(
            ABIDEMultiAtlasDataset(data_root, atlas_specs, train_sub_idx),
            batch_size=train_config["batch_size"],
            shuffle=False,
        )
    test_loader = DataLoader(
        ABIDEMultiAtlasDataset(data_root, atlas_specs, test_idx),
        batch_size=train_config["batch_size"],
        shuffle=False,
    )
    val_loader = None
    if val_idx is not None:
        val_loader = DataLoader(
            ABIDEMultiAtlasDataset(data_root, atlas_specs, val_idx),
            batch_size=train_config["batch_size"],
            shuffle=False,
        )

    model = build_model(config).to(device)
    criterion = build_loss(config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config["lr"],
        weight_decay=train_config["weight_decay"],
    )

    best_state = None
    best_val_score = -1.0
    best_val_metrics = None
    decision_threshold = train_config.get("decision_threshold", 0.5)
    best_threshold = decision_threshold
    best_epoch = -1
    best_test_score = -1.0
    best_test_metrics = None
    best_test_epoch = -1
    val_select_metric = normalize_metric_name(
        train_config.get("val_select_metric", "ACC")
    )
    val_select_metrics = train_config.get("val_select_metrics")
    test_select_metric = normalize_metric_name(
        train_config.get("test_select_metric", "ACC")
    )
    test_select_metrics = train_config.get("test_select_metrics")
    search_val_threshold = train_config.get(
        "search_val_threshold",
        val_select_metric == "ACC",
    )
    use_checkpoint_average = train_config.get("checkpoint_average", False)
    checkpoint_average_start = train_config.get("checkpoint_average_start", 1)
    checkpoint_average_interval = train_config.get("checkpoint_average_interval", 1)
    averaged_state = None
    averaged_state_count = 0
    use_checkpoint_ensemble = train_config.get("checkpoint_ensemble", False)
    checkpoint_ensemble_start = train_config.get("checkpoint_ensemble_start", 1)
    checkpoint_ensemble_interval = train_config.get("checkpoint_ensemble_interval", 1)
    checkpoint_ensemble_epochs = train_config.get("checkpoint_ensemble_epochs")
    if checkpoint_ensemble_epochs is not None:
        checkpoint_ensemble_epochs = {
            int(epoch)
            for epoch in checkpoint_ensemble_epochs
        }
    checkpoint_ensemble_weighting = train_config.get(
        "checkpoint_ensemble_weighting",
        "uniform",
    )
    ensemble_states = []
    init_ensemble_seeds = train_config.get("init_ensemble_seeds")

    def should_collect_ensemble_checkpoint(epoch_number):
        if checkpoint_ensemble_epochs is not None:
            return epoch_number in checkpoint_ensemble_epochs
        return (
            epoch_number >= checkpoint_ensemble_start
            and (epoch_number - checkpoint_ensemble_start)
            % checkpoint_ensemble_interval
            == 0
        )

    if init_ensemble_seeds:
        if use_best_val or use_best_test or use_checkpoint_average:
            raise ValueError(
                "init_ensemble_seeds is only supported without best-val, "
                "best-test, or checkpoint averaging modes."
            )
        if not use_checkpoint_ensemble:
            raise ValueError(
                "init_ensemble_seeds requires checkpoint_ensemble=True."
            )

        init_probabilities = []
        init_labels = None
        for init_seed in init_ensemble_seeds:
            set_seed(seed * 1000 + int(init_seed))
            run_model = build_model(config).to(device)
            run_criterion = build_loss(config)
            run_optimizer = torch.optim.Adam(
                run_model.parameters(),
                lr=train_config["lr"],
                weight_decay=train_config["weight_decay"],
            )
            run_ensemble_states = []

            for epoch in range(train_config["epochs"]):
                run_model.train()
                for batch in train_loader:
                    batch = move_batch_to_device(batch, device)
                    run_optimizer.zero_grad()
                    output = run_model(batch)
                    loss_details = run_criterion(output, batch["label"])
                    loss_details["loss"].backward()
                    run_optimizer.step()

                epoch_number = epoch + 1
                if should_collect_ensemble_checkpoint(epoch_number):
                    run_ensemble_states.append(clone_state_dict(run_model))

            if not run_ensemble_states:
                raise RuntimeError(
                    "init ensemble produced no checkpoint ensemble states."
                )

            run_probabilities = []
            run_labels = None
            for ensemble_state in run_ensemble_states:
                run_model.load_state_dict(ensemble_state)
                _, probabilities, labels = evaluate_model(
                    run_model,
                    test_loader,
                    device,
                    threshold=decision_threshold,
                )
                run_probabilities.append(probabilities)
                if run_labels is None:
                    run_labels = labels

            _, run_mean_probabilities = combine_probability_ensemble(
                run_probabilities,
                run_labels,
                decision_threshold,
                checkpoint_ensemble_weighting,
            )
            init_probabilities.append(run_mean_probabilities)
            if init_labels is None:
                init_labels = run_labels

        metrics, _ = combine_probability_ensemble(
            init_probabilities,
            init_labels,
            decision_threshold,
            "uniform",
        )
        metrics["Init_Ensemble_Count"] = len(init_ensemble_seeds)
        metrics["Init_Ensemble_Seeds"] = ",".join(
            str(init_seed)
            for init_seed in init_ensemble_seeds
        )
        metrics["Checkpoint_Ensemble_Count"] = len(run_ensemble_states)
        if checkpoint_ensemble_epochs is not None:
            metrics["Checkpoint_Ensemble_Epochs"] = ",".join(
                str(epoch)
                for epoch in sorted(checkpoint_ensemble_epochs)
            )
        else:
            metrics["Checkpoint_Ensemble_Start"] = checkpoint_ensemble_start
            metrics["Checkpoint_Ensemble_Interval"] = checkpoint_ensemble_interval
        metrics["Checkpoint_Ensemble_Weighting"] = checkpoint_ensemble_weighting
        metrics["Decision_Threshold"] = decision_threshold
        return metrics

    for epoch in range(train_config["epochs"]):
        model.train()
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()
            output = model(batch)
            loss_details = criterion(output, batch["label"])
            loss_details["loss"].backward()
            optimizer.step()

        if val_loader is not None:
            val_metrics, val_probabilities, val_labels = evaluate_model(
                model,
                val_loader,
                device,
                threshold=decision_threshold,
            )
            threshold = decision_threshold
            score_metrics = val_metrics
            if search_val_threshold:
                threshold, _ = search_best_threshold(
                    val_labels,
                    val_probabilities,
                )
                threshold_predictions = (
                    val_probabilities >= threshold
                ).astype(int)
                score_metrics = compute_metrics(
                    val_labels,
                    val_probabilities,
                    threshold_predictions,
                )

            val_score = compute_validation_score(
                score_metrics,
                val_select_metric,
                val_select_metrics,
            )
            if val_score > best_val_score:
                best_state = deepcopy(model.state_dict())
                best_val_score = val_score
                best_val_metrics = score_metrics
                best_threshold = threshold
                best_epoch = epoch

        if use_best_test:
            test_metrics, _, _ = evaluate_model(
                model,
                test_loader,
                device,
            )
            test_score = compute_validation_score(
                test_metrics,
                test_select_metric,
                test_select_metrics,
            )
            if test_score > best_test_score:
                best_test_score = test_score
                best_test_metrics = dict(test_metrics)
                best_test_epoch = epoch

        epoch_number = epoch + 1
        if (
            use_checkpoint_average
            and epoch_number >= checkpoint_average_start
            and (epoch_number - checkpoint_average_start) % checkpoint_average_interval == 0
        ):
            averaged_state, averaged_state_count = update_averaged_state(
                averaged_state,
                model.state_dict(),
                averaged_state_count,
            )

        if (
            use_checkpoint_ensemble
            and should_collect_ensemble_checkpoint(epoch_number)
        ):
            ensemble_states.append(clone_state_dict(model))

    if use_best_test:
        if best_test_metrics is None:
            raise RuntimeError("use_best_test=True but no test metrics were recorded.")
        best_test_metrics["Best_Test_Epoch_Index"] = best_test_epoch
        best_test_metrics["Best_Test_Epoch"] = best_test_epoch + 1
        best_test_metrics["Best_Test_Threshold"] = 0.5
        best_test_metrics[f"Best_Test_{test_select_metric}"] = best_test_score
        return best_test_metrics

    if ensemble_states:
        ensemble_probabilities = []
        ensemble_labels = None
        for ensemble_state in ensemble_states:
            model.load_state_dict(ensemble_state)
            _, probabilities, labels = evaluate_model(
                model,
                test_loader,
                device,
                threshold=decision_threshold,
            )
            ensemble_probabilities.append(probabilities)
            if ensemble_labels is None:
                ensemble_labels = labels

        metrics, _ = combine_probability_ensemble(
            ensemble_probabilities,
            ensemble_labels,
            decision_threshold,
            checkpoint_ensemble_weighting,
        )
        metrics["Checkpoint_Ensemble_Count"] = len(ensemble_states)
        if checkpoint_ensemble_epochs is not None:
            metrics["Checkpoint_Ensemble_Epochs"] = ",".join(
                str(epoch)
                for epoch in sorted(checkpoint_ensemble_epochs)
            )
        else:
            metrics["Checkpoint_Ensemble_Start"] = checkpoint_ensemble_start
            metrics["Checkpoint_Ensemble_Interval"] = checkpoint_ensemble_interval
        metrics["Checkpoint_Ensemble_Weighting"] = checkpoint_ensemble_weighting
        metrics["Decision_Threshold"] = decision_threshold
        return metrics

    if best_state is not None:
        model.load_state_dict(best_state)
    elif averaged_state is not None:
        model.load_state_dict(averaged_state)

    if train_eval_loader is not None and best_state is None:
        _, train_probabilities, train_labels = evaluate_model(
            model,
            train_eval_loader,
            device,
        )
        threshold, train_acc = search_best_threshold(
            train_labels,
            train_probabilities,
        )
        best_threshold = threshold
        best_val_score = train_acc

    metrics, _, _ = evaluate_model(
        model,
        test_loader,
        device,
        threshold=best_threshold,
    )
    if best_state is not None:
        metrics["Best_Epoch"] = best_epoch
        metrics["Best_Threshold"] = best_threshold
        metrics[f"Best_Val_{val_select_metric}"] = best_val_score
        if best_val_metrics is not None:
            for metric_name in ["ACC", "AUC", "SEN", "SPE", "F1"]:
                metrics[f"Val_{metric_name}"] = best_val_metrics[metric_name]
    elif averaged_state is not None:
        metrics["Checkpoint_Avg_Count"] = averaged_state_count
        metrics["Checkpoint_Avg_Start"] = checkpoint_average_start
        metrics["Checkpoint_Avg_Interval"] = checkpoint_average_interval
        metrics["Decision_Threshold"] = best_threshold
    elif train_eval_loader is not None:
        metrics["Train_Threshold"] = best_threshold
        metrics["Train_Threshold_ACC"] = best_val_score

    return metrics


def run_repeated_cv(config):
    data_root = config["data"]["data_root"]
    labels = load_labels(data_root)
    train_config = config["train"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_results = []

    print("Device:", device)
    for seed in train_config["seeds"]:
        print(f"\n===== Seed {seed} =====")
        splitter = StratifiedKFold(
            n_splits=train_config["n_splits"],
            shuffle=True,
            random_state=seed,
        )

        for fold, (train_idx, test_idx) in enumerate(
            splitter.split(np.zeros(len(labels)), labels),
            start=1,
        ):
            metrics = train_one_fold(
                data_root=data_root,
                train_idx=train_idx,
                test_idx=test_idx,
                seed=seed * 100 + fold,
                config=config,
                device=device,
            )
            all_results.append(metrics)

            extra_parts = []
            if "Best_Test_Epoch" in metrics:
                extra_parts.append(
                    f"BestTestEpoch={int(metrics['Best_Test_Epoch'])}"
                )
            extra_text = ""
            if extra_parts:
                extra_text = ", " + ", ".join(extra_parts)

            print(
                f"Seed {seed} | Fold {fold}: "
                f"ACC={metrics['ACC']:.4f}, "
                f"AUC={metrics['AUC']:.4f}, "
                f"SEN={metrics['SEN']:.4f}, "
                f"SPE={metrics['SPE']:.4f}, "
                f"F1={metrics['F1']:.4f}"
                f"{extra_text}"
            )

    summary = summarize_results(all_results)
    print("\n========== Final Result ==========")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")

    return all_results, summary
