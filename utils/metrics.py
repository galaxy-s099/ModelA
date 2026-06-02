import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_prob, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "ACC": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "SEN": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "SPE": tn / (tn + fp + 1e-8),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }


def summarize_results(results):
    metric_keys = ["ACC", "AUC", "SEN", "SPE", "F1"]
    summary = {}

    for key in metric_keys:
        values = np.asarray([result[key] for result in results], dtype=np.float64)
        summary[f"{key}_mean"] = values.mean()
        summary[f"{key}_std"] = values.std()

    return summary
