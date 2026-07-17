"""Utilities for diagnostic Test-epoch ranking experiments only."""


def keep_top_test_candidates(candidates, score, epoch_index, metrics, top_k):
    """Return the highest-scoring candidates, breaking ties by earlier epoch."""
    candidate = {
        "score": float(score),
        "epoch_index": int(epoch_index),
        "metrics": dict(metrics),
    }
    ranked = [*candidates, candidate]
    ranked.sort(key=lambda item: (-item["score"], item["epoch_index"]))
    return ranked[:top_k]
