"""ECE / Brier / reliability-diagram helpers.

Deliberately ported to match Cost-Aware-Multi-Agent-LLM-Router's
v2/scripts/stats_utils.py exactly (same bin-edge convention, same weighted-gap
ECE) -- that's what makes the cross-project calibration comparison legitimate
rather than two different formulas that happen to look similar.
"""

import numpy as np


def expected_calibration_error(probs: list[float], labels: list[int], n_bins: int = 5) -> float:
    """ECE: weighted average gap between predicted confidence and observed accuracy per bin."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    if n == 0:
        return 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        if not mask.any():
            continue
        bin_conf = probs[mask].mean()
        bin_acc = labels[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def brier_score(probs: list[float], labels: list[int]) -> float:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if len(probs) == 0:
        return 0.0
    return float(np.mean((probs - labels) ** 2))


def reliability_bins(probs: list[float], labels: list[int], n_bins: int = 5):
    """Returns (bin_centers, bin_confidence, bin_accuracy, bin_counts) for a reliability diagram."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, confs, accs, counts = [], [], [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        centers.append((lo + hi) / 2)
        if mask.any():
            confs.append(float(probs[mask].mean()))
            accs.append(float(labels[mask].mean()))
        else:
            confs.append(float("nan"))
            accs.append(float("nan"))
        counts.append(int(mask.sum()))
    return centers, confs, accs, counts
