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


def evaluate_model(model, dataloader, device, threshold=0.5):
    model.eval()
    probabilities = []
    predictions = []
    labels = []
    atlas_weights = []

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            output = model(batch)
            batch_probabilities = F.softmax(output["fusion_logits"], dim=-1)[:, 1]

            probabilities.extend(batch_probabilities.cpu().numpy())
            predictions.extend((batch_probabilities >= threshold).long().cpu().numpy())
            labels.extend(batch["label"].cpu().numpy())
            atlas_weight = output.get("atlas_weight")
            if atlas_weight is not None:
                atlas_weights.append(atlas_weight.cpu().numpy())

    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    predictions = np.asarray(predictions)
    metrics = compute_metrics(labels, probabilities, predictions)

    if atlas_weights:
        mean_atlas_weight = np.concatenate(atlas_weights, axis=0).mean(axis=0)
        for atlas_name, weight in zip(model.atlas_names, mean_atlas_weight):
            metrics[f"Weight_{atlas_name}"] = float(weight)

    return metrics, probabilities, labels


def train_one_fold(data_root, train_idx, test_idx, seed, config, device):
    set_seed(seed)
    labels = load_labels(data_root)
    train_config = config["train"]
    atlas_specs = config["data"]["atlases"]
    use_best_val = train_config.get("use_best_val", False)

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
    best_val_acc = -1.0
    best_threshold = 0.5
    best_epoch = -1

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
            _, val_probabilities, val_labels = evaluate_model(
                model,
                val_loader,
                device,
            )
            threshold, val_acc = search_best_threshold(
                val_labels,
                val_probabilities,
            )
            if val_acc > best_val_acc:
                best_state = deepcopy(model.state_dict())
                best_val_acc = val_acc
                best_threshold = threshold
                best_epoch = epoch

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics, _, _ = evaluate_model(
        model,
        test_loader,
        device,
        threshold=best_threshold,
    )
    if best_state is not None:
        metrics["Best_Epoch"] = best_epoch
        metrics["Best_Threshold"] = best_threshold
        metrics["Val_ACC"] = best_val_acc

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

            print(
                f"Seed {seed} | Fold {fold}: "
                f"ACC={metrics['ACC']:.4f}, "
                f"AUC={metrics['AUC']:.4f}, "
                f"SEN={metrics['SEN']:.4f}, "
                f"SPE={metrics['SPE']:.4f}, "
                f"F1={metrics['F1']:.4f}"
            )

    summary = summarize_results(all_results)
    print("\n========== Final Result ==========")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")

    return all_results, summary
