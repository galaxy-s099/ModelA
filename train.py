from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader

from data.abide_dataset import ABIDEMultiAtlasDataset, load_labels, load_site_count
from data.tangent_fc import build_fold_tangent_matrices
from models.losses import ProposalLoss
from models.smaf_edge_energy_net import SMAFEdgeEnergyNet
from models.smaf_edge_gated_proposal_net import SMAFEdgeGatedProposalNet
from models.smaf_edge_proposal_net import SMAFEdgeProposalNet
from models.smaf_proposal_net import SMAFProposalNet
from utils.metrics import compute_metrics, summarize_results
from utils.seed import set_seed
from utils.test_epoch_selection import keep_top_test_candidates


def move_batch_to_device(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def build_model(config):
    model_config = config["model"]
    num_sites = model_config.get("num_sites")
    if (
        (
            model_config.get("use_site_embedding", False)
            or model_config.get("use_site_adversarial", False)
        )
        and num_sites is None
    ):
        num_sites = load_site_count(
            config["data"]["data_root"],
            model_config.get("site_file", "site_ids.npy"),
        )
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
            reliability_mode=model_config.get("reliability_mode", "energy"),
            use_atlas_prior=model_config.get("use_atlas_prior", False),
            use_sample_gate=model_config.get("use_sample_gate", False),
            sample_gate_scale=model_config.get("sample_gate_scale", 1.0),
            use_uniform_atlas_weights=model_config.get(
                "use_uniform_atlas_weights",
                False,
            ),
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
            use_consensus_gate=model_config.get("use_consensus_gate", False),
            consensus_gate_scale=model_config.get(
                "consensus_gate_scale",
                0.5,
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
            edge_dropout=model_config.get("edge_dropout", 0.0),
            edge_topk_ratio=model_config.get("edge_topk_ratio"),
            edge_projection_rank=model_config.get("edge_projection_rank"),
            use_dual_stream_signed_mlp=model_config.get(
                "use_dual_stream_signed_mlp",
                False,
            ),
            use_signed_edge_separation=model_config.get(
                "use_signed_edge_separation",
                True,
            ),
            use_roi_profile_attention=model_config.get(
                "use_roi_profile_attention",
                False,
            ),
            roi_profile_dim=model_config.get("roi_profile_dim", 64),
            roi_profile_num_heads=model_config.get(
                "roi_profile_num_heads",
                4,
            ),
            roi_profile_dropout=model_config.get("roi_profile_dropout", 0.1),
            roi_profile_residual_scale=model_config.get(
                "roi_profile_residual_scale",
                0.25,
            ),
            atlas_dropout=model_config.get("atlas_dropout", 0.0),
            atlas_dropout_mode=model_config.get("atlas_dropout_mode", "single"),
            use_logit_meta_fusion=model_config.get(
                "use_logit_meta_fusion",
                False,
            ),
            logit_meta_hidden_dim=model_config.get(
                "logit_meta_hidden_dim",
                16,
            ),
            logit_meta_dropout=model_config.get("logit_meta_dropout"),
            use_site_embedding=model_config.get("use_site_embedding", False),
            num_sites=num_sites,
            site_embedding_dim=model_config.get("site_embedding_dim", 8),
            use_site_adversarial=model_config.get(
                "use_site_adversarial",
                False,
            ),
            site_adversarial_hidden_dim=model_config.get(
                "site_adversarial_hidden_dim",
                64,
            ),
            site_adversarial_grl_lambda=model_config.get(
                "site_adversarial_grl_lambda",
                1.0,
            ),
            use_tangent_branch=model_config.get("use_tangent_branch", False),
            use_fisher_z=model_config.get("use_fisher_z", False),
            fisher_z_clip=model_config.get("fisher_z_clip", 0.999999),
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
        lambda_site_adversarial=loss_config.get(
            "lambda_site_adversarial",
            0.0,
        ),
    )


def compute_effect_size_scores(group_a, group_b, eps=1e-6):
    mean_a = group_a.mean(axis=0)
    mean_b = group_b.mean(axis=0)
    var_a = group_a.var(axis=0)
    var_b = group_b.var(axis=0)
    pooled_std = np.sqrt((var_a + var_b) / 2.0)
    return np.abs(mean_b - mean_a) / (pooled_std + eps)


def build_supervised_edge_selected_atlas_specs(
    data_root,
    atlas_specs,
    train_indices,
    labels,
    selection_config,
):
    if not selection_config or not selection_config.get("enabled", False):
        return atlas_specs, {}

    ratio = selection_config.get("ratio", 0.2)
    atlas_ratios = selection_config.get("atlas_ratios", {})
    if not 0.0 < float(ratio) <= 1.0:
        raise ValueError("supervised edge selection ratio must be in (0, 1]")

    train_indices = np.asarray(train_indices)
    train_labels = labels[train_indices]
    negative_mask = train_labels == 0
    positive_mask = train_labels == 1
    if not negative_mask.any() or not positive_mask.any():
        raise ValueError("supervised edge selection needs both classes in train fold")

    selected_specs = deepcopy(atlas_specs)
    selection_stats = {}
    data_root = str(data_root)

    for atlas_name, spec in selected_specs.items():
        atlas_ratio = float(atlas_ratios.get(atlas_name, ratio))
        if not 0.0 < atlas_ratio <= 1.0:
            raise ValueError(
                f"{atlas_name}: supervised edge selection ratio must be in (0, 1]"
            )

        num_nodes = int(spec["num_nodes"])
        edge_index = np.triu_indices(num_nodes, k=1)
        edge_count = len(edge_index[0])
        fc_file = spec.get("fc_file", f"X_{atlas_name}.npy")
        fc = np.load(Path(data_root) / fc_file, mmap_mode="r")
        train_fc = np.clip(fc[train_indices], -1.0, 1.0)
        edge_values = train_fc[:, edge_index[0], edge_index[1]]
        signed_edges = np.concatenate(
            [
                np.clip(edge_values, 0.0, None),
                np.clip(-edge_values, 0.0, None),
            ],
            axis=1,
        )

        scores = compute_effect_size_scores(
            signed_edges[negative_mask],
            signed_edges[positive_mask],
        )
        edge_scores = np.maximum(scores[:edge_count], scores[edge_count:])
        edge_keep_count = max(1, int(round(edge_count * atlas_ratio)))
        if edge_keep_count >= edge_count:
            selected_edge_indices = np.arange(edge_count)
        else:
            selected_edge_indices = np.argpartition(
                edge_scores,
                -edge_keep_count,
            )[-edge_keep_count:]

        edge_mask = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        edge_mask[
            edge_index[0][selected_edge_indices],
            edge_index[1][selected_edge_indices],
        ] = 1.0
        edge_mask = edge_mask + edge_mask.T
        spec["edge_mask"] = edge_mask
        selection_stats[f"SelectedEdges_{atlas_name}"] = int(
            len(selected_edge_indices)
        )
        selection_stats[f"SelectedEdgeRatio_{atlas_name}"] = float(
            len(selected_edge_indices) / edge_count
        )

    return selected_specs, selection_stats


def compute_threshold_score(y_true, y_pred, metric="ACC"):
    metric = normalize_metric_name(metric)
    if metric == "ACC":
        return float((y_pred == y_true).mean())
    if metric in {"BALANCED_ACC", "BAC"}:
        positive_mask = y_true == 1
        negative_mask = y_true == 0
        sensitivity = (
            (y_pred[positive_mask] == 1).mean()
            if positive_mask.any()
            else 0.0
        )
        specificity = (
            (y_pred[negative_mask] == 0).mean()
            if negative_mask.any()
            else 0.0
        )
        return float((sensitivity + specificity) / 2.0)
    raise ValueError(
        f"Unsupported threshold score metric: {metric}. "
        "Expected ACC or BALANCED_ACC."
    )


def search_best_threshold(
    y_true,
    y_prob,
    threshold_min=0.30,
    threshold_max=0.70,
    threshold_step=0.01,
    score_metric="ACC",
    tie_break="first",
    tie_break_target=0.5,
):
    best_threshold = 0.5
    best_score = -1.0
    tie_break = str(tie_break).lower()

    thresholds = np.arange(
        threshold_min,
        threshold_max + threshold_step / 2.0,
        threshold_step,
    )
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        score = compute_threshold_score(y_true, y_pred, score_metric)
        is_better = score > best_score
        if (
            tie_break == "closest_to_target"
            and np.isclose(score, best_score)
            and abs(threshold - tie_break_target)
            < abs(best_threshold - tie_break_target)
        ):
            is_better = True
        if is_better:
            best_threshold = threshold
            best_score = score

    return float(best_threshold), float(best_score)


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


def evaluate_model(
    model,
    dataloader,
    device,
    threshold=0.5,
    return_details=False,
):
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
    sample_indices = []
    weighted_probabilities = []
    branch_probabilities = {
        atlas_name: []
        for atlas_name in getattr(model, "atlas_names", [])
    }
    diagnostic_outputs = {
        key: []
        for key in [
            "branch_logits",
            "branch_logit_mean",
            "branch_logit_margin",
            "raw_energy_score",
            "centered_energy_score",
            "entropy_confidence",
            "reliability_score",
        ]
    }

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            output = model(batch)
            batch_probabilities = F.softmax(output["fusion_logits"], dim=-1)[:, 1]

            probabilities.extend(batch_probabilities.cpu().numpy())
            predictions.extend((batch_probabilities >= threshold).long().cpu().numpy())
            labels.extend(batch["label"].cpu().numpy())
            if return_details:
                if "sample_index" not in batch:
                    raise ValueError(
                        "return_details=True requires sample_index in every batch"
                    )
                sample_indices.extend(batch["sample_index"].cpu().numpy())

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
                if return_details:
                    diagnostic_outputs["branch_logits"].append(
                        branch_logits.cpu().numpy()
                    )
                    if hasattr(model, "compute_reliability_diagnostics"):
                        reliability_diagnostics = (
                            model.compute_reliability_diagnostics(branch_logits)
                        )
                        for output_key, diagnostic_output in (
                            reliability_diagnostics.items()
                        ):
                            diagnostic_outputs[output_key].append(
                                diagnostic_output.cpu().numpy()
                            )
                    reliability_output = output.get("reliability_score")
                    if reliability_output is not None:
                        diagnostic_outputs["reliability_score"].append(
                            reliability_output.cpu().numpy()
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

    if not return_details:
        return metrics, probabilities, labels

    details = {
        "sample_index": np.asarray(sample_indices, dtype=np.int64),
        "label": labels,
        "probability": probabilities,
        "prediction": predictions,
        "decision_threshold": np.full(
            probabilities.shape,
            threshold,
            dtype=np.float64,
        ),
    }
    if weighted_probabilities:
        details["weighted_probability"] = np.asarray(weighted_probabilities)
    if branch_probabilities and all(branch_probabilities.values()):
        details["branch_probability"] = np.column_stack(
            [
                np.asarray(branch_probabilities[atlas_name])
                for atlas_name in model.atlas_names
            ]
        )
    for metric_prefix, weight_batches in atlas_weight_outputs.items():
        if weight_batches:
            details_key = {
                "Weight": "atlas_weight",
                "BaseWeight": "base_atlas_weight",
                "GatedWeight": "gated_atlas_weight",
            }[metric_prefix]
            details[details_key] = np.concatenate(weight_batches, axis=0)
    if sample_gate_logits:
        details["sample_gate_logits"] = np.concatenate(
            sample_gate_logits,
            axis=0,
        )
    for output_key, output_batches in diagnostic_outputs.items():
        if output_batches:
            details[output_key] = np.concatenate(output_batches, axis=0)
    details["atlas_names"] = list(model.atlas_names)
    details["reliability_mode"] = getattr(model, "reliability_mode", "unknown")
    return metrics, probabilities, labels, details


def compute_probability_ensemble_weights(
    ensemble_probabilities,
    weighting="uniform",
    consensus_temperature=0.15,
):
    stacked_probabilities = np.stack(ensemble_probabilities, axis=0)
    if weighting == "uniform":
        raw_weights = np.ones_like(stacked_probabilities)
    elif weighting == "confidence":
        raw_weights = np.abs(stacked_probabilities - 0.5) * 2.0
    elif weighting == "consensus_confidence":
        if consensus_temperature <= 0:
            raise ValueError("checkpoint_consensus_temperature must be positive")

        consensus = stacked_probabilities.mean(axis=0, keepdims=True)
        confidence = np.abs(stacked_probabilities - 0.5) * 2.0
        agreement = np.exp(
            -np.abs(stacked_probabilities - consensus) / consensus_temperature
        )
        raw_weights = confidence * agreement
    else:
        raise ValueError(
            "Unsupported checkpoint_ensemble_weighting: "
            f"{weighting}. Expected uniform, confidence, or consensus_confidence."
        )

    weight_sum = raw_weights.sum(axis=0, keepdims=True)
    uniform_weights = np.full_like(
        raw_weights,
        1.0 / raw_weights.shape[0],
    )
    return np.divide(
        raw_weights,
        weight_sum,
        out=uniform_weights,
        where=weight_sum > 1e-12,
    )


def aggregate_checkpoint_diagnostics(
    checkpoint_details,
    checkpoint_weights,
    mean_probabilities,
    decision_threshold,
):
    if not checkpoint_details:
        raise ValueError("checkpoint_details cannot be empty")

    reference = checkpoint_details[0]
    sample_indices = reference["sample_index"]
    labels = reference["label"]
    for details in checkpoint_details[1:]:
        if not np.array_equal(details["sample_index"], sample_indices):
            raise ValueError("Checkpoint diagnostics have different sample order")
        if not np.array_equal(details["label"], labels):
            raise ValueError("Checkpoint diagnostics have different labels")
        if details["atlas_names"] != reference["atlas_names"]:
            raise ValueError("Checkpoint diagnostics have different atlas order")
        if details.get("reliability_mode") != reference.get("reliability_mode"):
            raise ValueError("Checkpoint diagnostics have different reliability modes")

    aggregated = {
        "sample_index": sample_indices.copy(),
        "label": labels.copy(),
        "probability": np.asarray(mean_probabilities),
        "prediction": (
            np.asarray(mean_probabilities) >= decision_threshold
        ).astype(np.int64),
        "decision_threshold": np.full(
            len(sample_indices),
            decision_threshold,
            dtype=np.float64,
        ),
        "atlas_names": list(reference["atlas_names"]),
    }
    aggregate_keys = [
        "weighted_probability",
        "branch_probability",
        "atlas_weight",
        "base_atlas_weight",
        "gated_atlas_weight",
        "sample_gate_logits",
        "branch_logits",
        "branch_logit_mean",
        "branch_logit_margin",
        "raw_energy_score",
        "centered_energy_score",
        "entropy_confidence",
        "reliability_score",
    ]
    for key in aggregate_keys:
        if not all(key in details for details in checkpoint_details):
            continue
        stacked_values = np.stack(
            [details[key] for details in checkpoint_details],
            axis=0,
        )
        value_weights = checkpoint_weights
        while value_weights.ndim < stacked_values.ndim:
            value_weights = value_weights[..., np.newaxis]
        aggregated[key] = np.sum(value_weights * stacked_values, axis=0)

    aggregated["reliability_mode"] = reference.get(
        "reliability_mode",
        "unknown",
    )

    return aggregated


def sample_diagnostics_to_rows(details):
    atlas_names = details["atlas_names"]
    sample_count = len(details["sample_index"])
    rows = []
    for sample_offset in range(sample_count):
        row = {
            "sample_index": int(details["sample_index"][sample_offset]),
            "label": int(details["label"][sample_offset]),
            "prediction_probability": float(
                details["probability"][sample_offset]
            ),
            "prediction": int(details["prediction"][sample_offset]),
            "decision_threshold": float(
                details["decision_threshold"][sample_offset]
            ),
            "reliability_mode": details.get("reliability_mode", "unknown"),
        }
        scalar_keys = {
            "weighted_probability": "weighted_probability",
        }
        for details_key, column_name in scalar_keys.items():
            if details_key in details:
                row[column_name] = float(
                    details[details_key][sample_offset]
                )

        vector_keys = {
            "branch_probability": "branch_probability",
            "atlas_weight": "weight",
            "base_atlas_weight": "base_weight",
            "gated_atlas_weight": "gated_weight",
            "sample_gate_logits": "gate_logit",
            "branch_logit_mean": "branch_logit_mean",
            "branch_logit_margin": "branch_logit_margin",
            "raw_energy_score": "raw_energy_score",
            "centered_energy_score": "centered_energy_score",
            "entropy_confidence": "entropy_confidence",
            "reliability_score": "reliability_score",
        }
        for details_key, column_prefix in vector_keys.items():
            if details_key not in details:
                continue
            for atlas_index, atlas_name in enumerate(atlas_names):
                row[f"{column_prefix}_{atlas_name}"] = float(
                    details[details_key][sample_offset, atlas_index]
                )
        if "branch_logits" in details:
            for atlas_index, atlas_name in enumerate(atlas_names):
                row[f"branch_logit_class0_{atlas_name}"] = float(
                    details["branch_logits"][sample_offset, atlas_index, 0]
                )
                row[f"branch_logit_class1_{atlas_name}"] = float(
                    details["branch_logits"][sample_offset, atlas_index, 1]
                )
        rows.append(row)
    return rows


def combine_probability_ensemble(
    ensemble_probabilities,
    labels,
    decision_threshold,
    weighting="uniform",
    consensus_temperature=0.15,
    return_weights=False,
):
    stacked_probabilities = np.stack(ensemble_probabilities, axis=0)
    normalized_weights = compute_probability_ensemble_weights(
        ensemble_probabilities,
        weighting,
        consensus_temperature,
    )
    mean_probabilities = np.sum(
        normalized_weights * stacked_probabilities,
        axis=0,
    )

    predictions = (mean_probabilities >= decision_threshold).astype(int)
    metrics = compute_metrics(labels, mean_probabilities, predictions)
    if return_weights:
        return metrics, mean_probabilities, normalized_weights
    return metrics, mean_probabilities


def select_inner_cv_checkpoint_epochs(
    data_root,
    outer_train_idx,
    seed,
    config,
    device,
    labels,
):
    """Select final-training checkpoints using inner CV only.

    The outer test fold is deliberately not constructed or evaluated here.
    """
    train_config = config["train"]
    selection_config = train_config.get("inner_cv_checkpoint_selection", {})
    if not selection_config.get("enabled", False):
        return None, {}

    if config.get("feature_selection", {}).get("supervised_edges", {}).get(
        "enabled",
        False,
    ):
        raise ValueError(
            "inner CV checkpoint selection does not support supervised edge "
            "selection yet"
        )
    if config["model"].get("use_tangent_branch", False) or config["model"].get(
        "use_tangent_fc_as_input",
        False,
    ):
        raise ValueError(
            "inner CV checkpoint selection does not support fold-local Tangent FC"
        )

    inner_splits = int(selection_config.get("n_splits", 3))
    top_k_epochs = int(selection_config.get("top_k_epochs", 6))
    epochs = int(train_config["epochs"])
    if inner_splits < 2:
        raise ValueError("inner CV n_splits must be at least 2")
    if not 0 < top_k_epochs <= epochs:
        raise ValueError("inner CV top_k_epochs must be in [1, epochs]")

    select_metric = normalize_metric_name(
        selection_config.get("select_metric", "ACC")
    )
    select_metrics = selection_config.get("select_metrics")
    atlas_specs = config["data"]["atlases"]
    splitter = StratifiedKFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=seed,
    )
    epoch_scores = [[] for _ in range(epochs)]

    print(
        f"Selecting {top_k_epochs} checkpoints with inner {inner_splits}-fold "
        f"CV by {select_metric}..."
    )
    for inner_fold, (inner_train_local, inner_val_local) in enumerate(
        splitter.split(np.zeros(len(outer_train_idx)), labels[outer_train_idx]),
        start=1,
    ):
        inner_train_idx = outer_train_idx[inner_train_local]
        inner_val_idx = outer_train_idx[inner_val_local]
        set_seed(seed * 1000 + inner_fold)

        train_loader = DataLoader(
            ABIDEMultiAtlasDataset(data_root, atlas_specs, inner_train_idx),
            batch_size=train_config["batch_size"],
            shuffle=True,
            pin_memory=device == "cuda",
        )
        val_loader = DataLoader(
            ABIDEMultiAtlasDataset(data_root, atlas_specs, inner_val_idx),
            batch_size=train_config["batch_size"],
            shuffle=False,
            pin_memory=device == "cuda",
        )
        model = build_model(config).to(device)
        criterion = build_loss(config)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_config["lr"],
            weight_decay=train_config["weight_decay"],
        )

        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                batch = move_batch_to_device(batch, device)
                optimizer.zero_grad()
                output = model(batch)
                loss_details = criterion(
                    output,
                    batch["label"],
                    batch.get("site"),
                )
                loss_details["loss"].backward()
                optimizer.step()

            val_metrics, _, _ = evaluate_model(model, val_loader, device)
            epoch_scores[epoch].append(
                compute_validation_score(
                    val_metrics,
                    select_metric,
                    select_metrics,
                )
            )

    mean_scores = np.asarray([np.mean(scores) for scores in epoch_scores])
    ranking = sorted(
        range(epochs),
        key=lambda epoch_index: (-mean_scores[epoch_index], epoch_index),
    )
    selected_epochs = [epoch_index + 1 for epoch_index in ranking[:top_k_epochs]]
    stats = {
        "Inner_CV_Folds": inner_splits,
        "Inner_CV_Selected_Epochs": ",".join(map(str, sorted(selected_epochs))),
        "Inner_CV_Selection_Metric": select_metric,
        "Inner_CV_Mean_Selection_Score": float(
            mean_scores[ranking[:top_k_epochs]].mean()
        ),
    }
    print(f"Inner CV selected epochs: {stats['Inner_CV_Selected_Epochs']}")
    return set(selected_epochs), stats


def train_one_fold(
    data_root,
    train_idx,
    test_idx,
    seed,
    config,
    device,
    return_diagnostics=False,
):
    set_seed(seed)
    labels = load_labels(data_root)
    train_config = config["train"]
    atlas_specs = config["data"]["atlases"]
    use_best_val = train_config.get("use_best_val", False)
    use_best_test = train_config.get("use_best_test", False)
    test_top_k_epochs = int(train_config.get("test_top_k_epochs", 0))
    if test_top_k_epochs < 0:
        raise ValueError("test_top_k_epochs must be non-negative")
    if use_best_test and test_top_k_epochs > 0:
        raise ValueError(
            "use_best_test and test_top_k_epochs cannot both be enabled"
        )
    if return_diagnostics and (use_best_test or test_top_k_epochs > 0):
        raise ValueError(
            "Sample diagnostics are not supported by test-best selection modes"
        )
    if return_diagnostics and train_config.get("init_ensemble_seeds"):
        raise ValueError(
            "Sample diagnostics are not supported by initialization ensembles"
        )

    inner_cv_epochs, inner_cv_stats = select_inner_cv_checkpoint_epochs(
        data_root=data_root,
        outer_train_idx=train_idx,
        seed=seed,
        config=config,
        device=device,
        labels=labels,
    )
    # Inner model selection must not change the final outer-fold initialization.
    set_seed(seed)

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

    selection_config = config.get("feature_selection", {}).get(
        "supervised_edges",
    )
    fold_atlas_specs, selection_stats = build_supervised_edge_selected_atlas_specs(
        data_root,
        atlas_specs,
        train_sub_idx,
        labels,
        selection_config,
    )

    use_tangent_branch = config["model"].get("use_tangent_branch", False)
    use_tangent_fc_as_input = config["model"].get(
        "use_tangent_fc_as_input",
        False,
    )
    if use_tangent_branch and use_tangent_fc_as_input:
        raise ValueError(
            "use_tangent_branch and use_tangent_fc_as_input cannot both be true"
        )
    tangent_matrices = None
    if use_tangent_branch or use_tangent_fc_as_input:
        print(f"Building fold-local Tangent Pearson matrices on {device}...")
        tangent_matrices = build_fold_tangent_matrices(
            data_root=data_root,
            atlas_specs=fold_atlas_specs,
            fit_indices=train_sub_idx,
            device=device,
            tangent_config=config.get("tangent", {}),
        )

    def attach_selection_stats(metrics):
        if selection_stats:
            metrics.update(selection_stats)
        if inner_cv_stats:
            metrics.update(inner_cv_stats)
        return metrics

    def build_dataset(indices):
        return ABIDEMultiAtlasDataset(
            data_root,
            fold_atlas_specs,
            indices,
            tangent_matrices=(tangent_matrices if use_tangent_branch else None),
            fc_overrides=(tangent_matrices if use_tangent_fc_as_input else None),
        )

    train_loader = DataLoader(
        build_dataset(train_sub_idx),
        batch_size=train_config["batch_size"],
        shuffle=True,
        pin_memory=device == "cuda",
    )
    train_eval_loader = None
    if train_config.get("search_train_threshold", False):
        train_eval_loader = DataLoader(
            build_dataset(train_sub_idx),
            batch_size=train_config["batch_size"],
            shuffle=False,
            pin_memory=device == "cuda",
        )
    test_loader = DataLoader(
        build_dataset(test_idx),
        batch_size=train_config["batch_size"],
        shuffle=False,
        pin_memory=device == "cuda",
    )
    val_loader = None
    if val_idx is not None:
        val_loader = DataLoader(
            build_dataset(val_idx),
            batch_size=train_config["batch_size"],
            shuffle=False,
            pin_memory=device == "cuda",
        )

    fold_config = deepcopy(config)
    fold_config["data"]["atlases"] = fold_atlas_specs
    model = build_model(fold_config).to(device)
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
    top_test_candidates = []
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
    threshold_search_min = train_config.get("threshold_search_min", 0.30)
    threshold_search_max = train_config.get("threshold_search_max", 0.70)
    threshold_search_step = train_config.get("threshold_search_step", 0.01)
    threshold_score_metric = train_config.get("threshold_score_metric", "ACC")
    threshold_tie_break = train_config.get("threshold_tie_break", "first")
    threshold_tie_break_target = train_config.get(
        "threshold_tie_break_target",
        decision_threshold,
    )
    use_checkpoint_average = train_config.get("checkpoint_average", False)
    checkpoint_average_start = train_config.get("checkpoint_average_start", 1)
    checkpoint_average_interval = train_config.get("checkpoint_average_interval", 1)
    averaged_state = None
    averaged_state_count = 0
    use_checkpoint_ensemble = (
        train_config.get("checkpoint_ensemble", False)
        or inner_cv_epochs is not None
    )
    checkpoint_ensemble_start = train_config.get("checkpoint_ensemble_start", 1)
    checkpoint_ensemble_interval = train_config.get("checkpoint_ensemble_interval", 1)
    checkpoint_ensemble_epochs = train_config.get("checkpoint_ensemble_epochs")
    if checkpoint_ensemble_epochs is not None:
        checkpoint_ensemble_epochs = {
            int(epoch)
            for epoch in checkpoint_ensemble_epochs
        }
    if inner_cv_epochs is not None:
        if checkpoint_ensemble_epochs is not None:
            raise ValueError(
                "inner CV checkpoint selection cannot be combined with "
                "checkpoint_ensemble_epochs"
            )
        checkpoint_ensemble_epochs = inner_cv_epochs
    checkpoint_ensemble_weighting = train_config.get(
        "checkpoint_ensemble_weighting",
        "uniform",
    )
    checkpoint_consensus_temperature = train_config.get(
        "checkpoint_consensus_temperature",
        0.15,
    )
    ensemble_states = []
    ensemble_epochs = []
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
                    loss_details = run_criterion(
                        output,
                        batch["label"],
                        batch.get("site"),
                    )
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
                checkpoint_consensus_temperature,
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
        return attach_selection_stats(metrics)

    for epoch in range(train_config["epochs"]):
        model.train()
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()
            output = model(batch)
            loss_details = criterion(output, batch["label"], batch.get("site"))
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
                    threshold_min=threshold_search_min,
                    threshold_max=threshold_search_max,
                    threshold_step=threshold_search_step,
                    score_metric=threshold_score_metric,
                    tie_break=threshold_tie_break,
                    tie_break_target=threshold_tie_break_target,
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

        if use_best_test or test_top_k_epochs > 0:
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
            if use_best_test and test_score > best_test_score:
                best_test_score = test_score
                best_test_metrics = dict(test_metrics)
                best_test_epoch = epoch
            if test_top_k_epochs > 0:
                top_test_candidates = keep_top_test_candidates(
                    top_test_candidates,
                    score=test_score,
                    epoch_index=epoch,
                    metrics=test_metrics,
                    top_k=test_top_k_epochs,
                )

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
            ensemble_epochs.append(epoch_number)

    if use_best_test:
        if best_test_metrics is None:
            raise RuntimeError("use_best_test=True but no test metrics were recorded.")
        best_test_metrics["Best_Test_Epoch_Index"] = best_test_epoch
        best_test_metrics["Best_Test_Epoch"] = best_test_epoch + 1
        best_test_metrics["Best_Test_Threshold"] = 0.5
        best_test_metrics[f"Best_Test_{test_select_metric}"] = best_test_score
        return attach_selection_stats(best_test_metrics)

    if test_top_k_epochs > 0:
        if len(top_test_candidates) != test_top_k_epochs:
            raise RuntimeError(
                "test_top_k_epochs exceeds the number of evaluated epochs"
            )
        ranked_metrics = []
        for test_rank, candidate in enumerate(top_test_candidates, start=1):
            metrics = dict(candidate["metrics"])
            metrics["Test_Rank"] = test_rank
            metrics["Best_Test_Epoch_Index"] = candidate["epoch_index"]
            metrics["Best_Test_Epoch"] = candidate["epoch_index"] + 1
            metrics["Best_Test_Threshold"] = 0.5
            metrics["Test_Select_Metric"] = test_select_metric
            metrics["Test_Select_Score"] = candidate["score"]
            ranked_metrics.append(attach_selection_stats(metrics))
        return ranked_metrics

    if ensemble_states:
        ensemble_threshold = decision_threshold
        train_threshold_score = None
        if train_eval_loader is not None:
            train_ensemble_probabilities = []
            train_ensemble_labels = None
            for ensemble_state in ensemble_states:
                model.load_state_dict(ensemble_state)
                _, probabilities, labels = evaluate_model(
                    model,
                    train_eval_loader,
                    device,
                    threshold=decision_threshold,
                )
                train_ensemble_probabilities.append(probabilities)
                if train_ensemble_labels is None:
                    train_ensemble_labels = labels

            _, train_mean_probabilities = combine_probability_ensemble(
                train_ensemble_probabilities,
                train_ensemble_labels,
                decision_threshold,
                checkpoint_ensemble_weighting,
                checkpoint_consensus_temperature,
            )
            ensemble_threshold, train_threshold_score = search_best_threshold(
                train_ensemble_labels,
                train_mean_probabilities,
                threshold_min=threshold_search_min,
                threshold_max=threshold_search_max,
                threshold_step=threshold_search_step,
                score_metric=threshold_score_metric,
                tie_break=threshold_tie_break,
                tie_break_target=threshold_tie_break_target,
            )

        ensemble_probabilities = []
        ensemble_details = []
        ensemble_labels = None
        for ensemble_state in ensemble_states:
            model.load_state_dict(ensemble_state)
            evaluation = evaluate_model(
                model,
                test_loader,
                device,
                threshold=decision_threshold,
                return_details=return_diagnostics,
            )
            if return_diagnostics:
                _, probabilities, labels, details = evaluation
                ensemble_details.append(details)
            else:
                _, probabilities, labels = evaluation
            ensemble_probabilities.append(probabilities)
            if ensemble_labels is None:
                ensemble_labels = labels

        ensemble_result = combine_probability_ensemble(
            ensemble_probabilities,
            ensemble_labels,
            ensemble_threshold,
            checkpoint_ensemble_weighting,
            checkpoint_consensus_temperature,
            return_weights=return_diagnostics,
        )
        if return_diagnostics:
            metrics, mean_probabilities, checkpoint_weights = ensemble_result
        else:
            metrics, mean_probabilities = ensemble_result
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
        metrics["Decision_Threshold"] = ensemble_threshold
        if train_threshold_score is not None:
            metrics["Train_Threshold"] = ensemble_threshold
            metrics["Train_Threshold_Score"] = train_threshold_score
            metrics["Train_Threshold_Score_Metric"] = normalize_metric_name(
                threshold_score_metric
            )
            if normalize_metric_name(threshold_score_metric) == "ACC":
                metrics["Train_Threshold_ACC"] = train_threshold_score
        metrics = attach_selection_stats(metrics)
        if not return_diagnostics:
            return metrics

        final_details = aggregate_checkpoint_diagnostics(
            ensemble_details,
            checkpoint_weights,
            mean_probabilities,
            ensemble_threshold,
        )
        sample_rows = sample_diagnostics_to_rows(final_details)
        checkpoint_epoch_text = ",".join(map(str, ensemble_epochs))
        for row in sample_rows:
            row["checkpoint_count"] = len(ensemble_states)
            row["checkpoint_epochs"] = checkpoint_epoch_text
            row["checkpoint_weighting"] = checkpoint_ensemble_weighting

        checkpoint_rows = []
        for checkpoint_index, (epoch_number, details) in enumerate(
            zip(ensemble_epochs, ensemble_details)
        ):
            rows = sample_diagnostics_to_rows(details)
            for sample_offset, row in enumerate(rows):
                row["checkpoint_epoch"] = epoch_number
                row["checkpoint_contribution"] = float(
                    checkpoint_weights[checkpoint_index, sample_offset]
                )
                checkpoint_rows.append(row)

        return metrics, {
            "sample_rows": sample_rows,
            "checkpoint_rows": checkpoint_rows,
        }

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
            threshold_min=threshold_search_min,
            threshold_max=threshold_search_max,
            threshold_step=threshold_search_step,
            score_metric=threshold_score_metric,
            tie_break=threshold_tie_break,
            tie_break_target=threshold_tie_break_target,
        )
        best_threshold = threshold
        best_val_score = train_acc

    evaluation = evaluate_model(
        model,
        test_loader,
        device,
        threshold=best_threshold,
        return_details=return_diagnostics,
    )
    if return_diagnostics:
        metrics, _, _, final_details = evaluation
    else:
        metrics, _, _ = evaluation
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
        metrics["Train_Threshold_Score"] = best_val_score
        metrics["Train_Threshold_Score_Metric"] = normalize_metric_name(
            threshold_score_metric
        )
        if normalize_metric_name(threshold_score_metric) == "ACC":
            metrics["Train_Threshold_ACC"] = best_val_score

    metrics = attach_selection_stats(metrics)
    if return_diagnostics:
        return metrics, {
            "sample_rows": sample_diagnostics_to_rows(final_details),
            "checkpoint_rows": [],
        }
    return metrics


def run_repeated_cv(config, return_diagnostics=False):
    data_root = config["data"]["data_root"]
    labels = load_labels(data_root)
    train_config = config["train"]
    test_top_k_epochs = int(train_config.get("test_top_k_epochs", 0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_results = []
    diagnostics = {
        "sample_rows": [],
        "checkpoint_rows": [],
    }

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
            fold_result = train_one_fold(
                data_root=data_root,
                train_idx=train_idx,
                test_idx=test_idx,
                seed=seed * 100 + fold,
                config=config,
                device=device,
                return_diagnostics=return_diagnostics,
            )
            if return_diagnostics:
                fold_metrics, fold_diagnostics = fold_result
                for diagnostics_key in ["sample_rows", "checkpoint_rows"]:
                    for row in fold_diagnostics[diagnostics_key]:
                        row["seed"] = seed
                        row["fold"] = fold
                    diagnostics[diagnostics_key].extend(
                        fold_diagnostics[diagnostics_key]
                    )
            else:
                fold_metrics = fold_result
            if not isinstance(fold_metrics, list):
                fold_metrics = [fold_metrics]
            for metrics in fold_metrics:
                all_results.append(metrics)

                extra_parts = []
                if "Test_Rank" in metrics:
                    extra_parts.append(f"TestRank={int(metrics['Test_Rank'])}")
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
    if test_top_k_epochs > 0:
        summary["Test_Top_K_Per_Fold"] = test_top_k_epochs
        summary["Test_Top_K_Observation_Count"] = len(all_results)
    print("\n========== Final Result ==========")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")

    if return_diagnostics:
        return all_results, summary, diagnostics
    return all_results, summary
