"""Token-level calibration study: raw confidence vs. temperature-scaled confidence.

Confidence definition (Guo et al. 2017, "On Calibration of Modern Neural
Networks"): for each held-out token position, take the model's max softmax
probability as "confidence" and whether the argmax prediction matches the
actual next token as "correctness". This produces the same (probability,
binary label) shape as Cost-Aware-Multi-Agent-LLM-Router's calibrator ECE/Brier,
which is what makes the cross-project comparison meaningful rather than
comparing two differently-defined numbers.

val.bin is split down the middle: the first half fits the temperature scalar,
the second half (never touched during fitting) is what ECE/Brier are measured
on -- a genuine generalization test, not a same-distribution sanity check.

Usage:
    python -m src.eval.calibration --checkpoint checkpoints/base.pt \
        --val-bin data/tokenized/val.bin --out results/calibration_summary.json \
        --report results/calibration_report.md --plot results/calibration_curve.png
"""

import argparse
import json
import os
import sys

sys.stdout.reconfigure(line_buffering=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.eval.calibration_stats import brier_score, expected_calibration_error, reliability_bins
from src.eval.evaluate import load_model

ROUTER_ECE = 0.1671
ROUTER_BRIER = 0.1521


def collect_logits(model, ids, block_size, device, max_tokens, batch_size=8):
    """Cache full logits for a bounded number of tokens -- enough to fit one
    scalar (T) reliably without holding the whole vocab-sized tensor for the
    entire fit split in memory (16384-vocab logits over hundreds of thousands
    of tokens would be tens of GB)."""
    data = np.array(ids[: max_tokens + 1], dtype=np.int64)
    n_blocks = (len(data) - 1) // block_size
    all_logits, all_targets = [], []
    with torch.no_grad():
        for start in range(0, n_blocks, batch_size):
            end = min(start + batch_size, n_blocks)
            xs = [data[i * block_size : i * block_size + block_size] for i in range(start, end)]
            ys = [data[i * block_size + 1 : i * block_size + 1 + block_size] for i in range(start, end)]
            x = torch.from_numpy(np.stack(xs)).to(device)
            y = torch.from_numpy(np.stack(ys)).to(device)
            logits, _, _ = model(x)
            all_logits.append(logits.reshape(-1, logits.size(-1)))
            all_targets.append(y.reshape(-1))
    return torch.cat(all_logits), torch.cat(all_targets)


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor, max_iter: int = 50) -> float:
    temperature = torch.nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=0.05, max_iter=max_iter)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll(logits / temperature, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return temperature.item()


def collect_confidence_correctness(model, ids, block_size, device, temperature=1.0, batch_size=8):
    """Streams over the full split batch-by-batch (never materializes all
    logits at once) collecting (max softmax prob, is-argmax-correct) pairs."""
    data = np.array(ids, dtype=np.int64)
    n_blocks = (len(data) - 1) // block_size
    probs, labels = [], []
    with torch.no_grad():
        for start in range(0, n_blocks, batch_size):
            end = min(start + batch_size, n_blocks)
            xs = [data[i * block_size : i * block_size + block_size] for i in range(start, end)]
            ys = [data[i * block_size + 1 : i * block_size + 1 + block_size] for i in range(start, end)]
            x = torch.from_numpy(np.stack(xs)).to(device)
            y = torch.from_numpy(np.stack(ys)).to(device)
            logits, _, _ = model(x)
            softmax = torch.softmax(logits / temperature, dim=-1)
            max_prob, argmax = softmax.max(dim=-1)
            correct = (argmax == y).float()
            probs.extend(max_prob.reshape(-1).tolist())
            labels.extend(correct.reshape(-1).tolist())
    return probs, labels


def plot_reliability(raw, scaled, path, n_bins=5):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    for label, (probs, labels), marker in [("raw (T=1)", raw, "o"), ("temperature-scaled", scaled, "s")]:
        _, confs, accs, _ = reliability_bins(probs, labels, n_bins=n_bins)
        xs = [c for c, a in zip(confs, accs) if not (np.isnan(c) or np.isnan(a))]
        ys = [a for c, a in zip(confs, accs) if not (np.isnan(c) or np.isnan(a))]
        ax.plot(xs, ys, marker=marker, label=label)
    ax.set_xlabel("confidence (mean predicted probability, per bin)")
    ax.set_ylabel("accuracy (fraction of top-1 predictions correct, per bin)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Token-level reliability diagram")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/base.pt")
    parser.add_argument("--val-bin", default="data/tokenized/val.bin")
    parser.add_argument("--fit-tokens", type=int, default=20480, help="tokens used to fit T (subsample of the fit half)")
    parser.add_argument("--out", default="results/calibration_summary.json")
    parser.add_argument("--report", default="results/calibration_report.md")
    parser.add_argument("--plot", default="results/calibration_curve.png")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"loading model from {args.checkpoint} ...")
    model, ckpt = load_model(args.checkpoint, args.device)
    block_size = model.cfg.block_size

    ids = np.memmap(args.val_bin, dtype=np.uint16, mode="r").tolist()
    mid = len(ids) // 2
    fit_ids, eval_ids = ids[:mid], ids[mid:]
    print(f"val.bin split: {len(fit_ids):,} tokens to fit T, {len(eval_ids):,} tokens held out for ECE/Brier")

    print(f"fitting temperature on {args.fit_tokens:,} tokens from the fit half ...")
    logits, targets = collect_logits(model, fit_ids, block_size, args.device, max_tokens=args.fit_tokens)
    temperature = fit_temperature(logits, targets)
    print(f"fitted temperature T = {temperature:.4f}")

    print("computing raw (T=1) confidence/correctness on the held-out half ...")
    raw_probs, raw_labels = collect_confidence_correctness(model, eval_ids, block_size, args.device, temperature=1.0)
    print("computing temperature-scaled confidence/correctness on the held-out half ...")
    scaled_probs, scaled_labels = collect_confidence_correctness(
        model, eval_ids, block_size, args.device, temperature=temperature
    )

    raw_ece = expected_calibration_error(raw_probs, raw_labels, n_bins=5)
    raw_brier = brier_score(raw_probs, raw_labels)
    scaled_ece = expected_calibration_error(scaled_probs, scaled_labels, n_bins=5)
    scaled_brier = brier_score(scaled_probs, scaled_labels)
    top1_acc = float(np.mean(raw_labels))

    result = {
        "n_eval_tokens": len(raw_labels),
        "n_fit_tokens": len(targets),
        "temperature": temperature,
        "top1_accuracy": top1_acc,
        "raw": {"ece": raw_ece, "brier": raw_brier},
        "temperature_scaled": {"ece": scaled_ece, "brier": scaled_brier},
        "router_comparison": {"router_ece": ROUTER_ECE, "router_brier": ROUTER_BRIER},
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))

    plot_reliability((raw_probs, raw_labels), (scaled_probs, scaled_labels), args.plot)

    report = f"""# Token-Level Calibration Report

- Held-out tokens (fit T / measure ECE-Brier split of val.bin): {len(fit_ids):,} / {len(eval_ids):,}
- Tokens used to fit temperature T: {len(targets):,} (subsample of the fit half; one scalar parameter
  doesn't need the full half to fit reliably)
- Fitted temperature: T = {temperature:.4f}
- Top-1 accuracy (argmax == actual next token) on the held-out half: {top1_acc:.4f}

| | ECE (5 bins) | Brier score |
|---|---|---|
| raw (T=1) | {raw_ece:.4f} | {raw_brier:.4f} |
| temperature-scaled | {scaled_ece:.4f} | {scaled_brier:.4f} |

See `calibration_curve.png` for the reliability diagram (raw vs. temperature-scaled).

**Methodology note:** confidence = the model's max softmax probability per
token position; label = whether that top-1 prediction matches the actual next
token. This is the standard Guo et al. 2017 definition, and structurally
identical to the router's `(predicted_probability, is_correct)` pairs -- same
ECE/Brier formulas (`src/eval/calibration_stats.py`, ported from the router's
`stats_utils.py`), applied to a different task.

**Split note:** unlike the router's own calibration report, T is fit and ECE/Brier
are measured on two disjoint halves of the held-out set -- a genuine
generalization test for the temperature fix, not a same-distribution sanity
check.

## Cross-project comparison

| | ECE | Brier |
|---|---|---|
| router calibrator (task-level: is the routed response correct) | {ROUTER_ECE:.4f} | {ROUTER_BRIER:.4f} |
| this model, raw (token-level: is the top-1 next-token prediction correct) | {raw_ece:.4f} | {raw_brier:.4f} |
| this model, temperature-scaled | {scaled_ece:.4f} | {scaled_brier:.4f} |

**Caveat, stated plainly:** these are not apples-to-apples measurements of the
same thing. The router's calibrator predicts *task-level* correctness of a
routed LLM response (a judged, semantic notion of "right answer"), fit with a
trained logistic-regression calibrator on hand-engineered features. This
model's numbers are *token-level* next-token prediction confidence, straight
out of the softmax, with no calibrator fit at all beyond the single
temperature scalar. Different task, different label distribution, different
downstream stakes. What *is* comparable is the calibration behavior itself --
whether raw model confidence over- or under-shoots actual accuracy, and
whether a standard post-hoc fix (temperature scaling here, a trained
calibrator there) narrows that gap.
"""
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w") as f:
        f.write(report)
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
