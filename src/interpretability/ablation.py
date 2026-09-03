"""Causal ablation: zero one head's value contribution at a time, measure the
held-out loss delta. This is what turns "this head has a high prefix-matching
score" into an actual finding -- a head that matters when zeroed is causally
doing something, independent of whether its attention pattern looked
interesting on its own.

Uses a bounded subset of val.bin (not the full ~497K tokens) so a full sweep
over every head in every layer stays CPU-feasible: base.pt is 8x12=96 heads,
interp_small.pt is 4x4=16, each requiring its own forward pass over the
subset.

Usage:
    python -m src.interpretability.ablation --checkpoint checkpoints/base.pt \
        --val-bin data/tokenized/val.bin --n-tokens 24576 \
        --out results/ablation_base.json --plot results/ablation_base.png
"""

import argparse
import json
import os

import numpy as np
import torch

torch.set_num_threads(4)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.eval.evaluate import load_model


def compute_loss(model, data: np.ndarray, block_size: int, device: str, ablate=None, batch_size: int = 4):
    n_blocks = (len(data) - 1) // block_size
    total_nll, total_tokens = 0.0, 0
    with torch.no_grad():
        for start in range(0, n_blocks, batch_size):
            end = min(start + batch_size, n_blocks)
            xs = [data[i * block_size : i * block_size + block_size] for i in range(start, end)]
            ys = [data[i * block_size + 1 : i * block_size + 1 + block_size] for i in range(start, end)]
            x = torch.from_numpy(np.stack(xs)).to(device)
            y = torch.from_numpy(np.stack(ys)).to(device)
            _, loss, _ = model(x, y, ablate=ablate)
            ntok = x.numel()
            total_nll += loss.item() * ntok
            total_tokens += ntok
    return total_nll / total_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/base.pt")
    parser.add_argument("--val-bin", default="data/tokenized/val.bin")
    parser.add_argument("--n-tokens", type=int, default=24576, help="bounded held-out subset for the full sweep")
    parser.add_argument("--out", default="results/ablation.json")
    parser.add_argument("--plot", default="results/ablation.png")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model, ckpt = load_model(args.checkpoint, args.device)
    block_size = model.cfg.block_size
    n_layer, n_head = model.cfg.n_layer, model.cfg.n_head

    ids = np.memmap(args.val_bin, dtype=np.uint16, mode="r")[: args.n_tokens + 1]
    data = np.array(ids, dtype=np.int64)

    print(f"baseline held-out loss over {len(data):,} tokens ...")
    baseline_loss = compute_loss(model, data, block_size, args.device, ablate=None)
    print(f"baseline loss: {baseline_loss:.4f}")

    deltas = np.zeros((n_layer, n_head))
    for layer in range(n_layer):
        for head in range(n_head):
            loss = compute_loss(model, data, block_size, args.device, ablate=(layer, head))
            deltas[layer, head] = loss - baseline_loss
            print(f"  layer {layer} head {head}: loss {loss:.4f} (delta {deltas[layer, head]:+.4f})")

    ranked = sorted(
        [{"layer": l, "head": h, "delta_loss": float(deltas[l, h])} for l in range(n_layer) for h in range(n_head)],
        key=lambda d: -d["delta_loss"],
    )
    result = {
        "checkpoint": args.checkpoint,
        "n_tokens": len(data),
        "baseline_loss": baseline_loss,
        "n_layer": n_layer,
        "n_head": n_head,
        "delta_loss": deltas.tolist(),
        "ranked_most_important": ranked,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("most important heads (largest loss increase when ablated):")
    print(json.dumps(ranked[:5], indent=2))

    fig, ax = plt.subplots(figsize=(1.2 * n_head + 2, 1.2 * n_layer + 1))
    vmax = max(abs(deltas.min()), abs(deltas.max()), 1e-6)
    im = ax.imshow(deltas, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_xticks(range(n_head))
    ax.set_yticks(range(n_layer))
    ax.set_title(f"Ablation loss delta (higher = more important)\n{os.path.basename(args.checkpoint)}")
    fig.colorbar(im, ax=ax, label="held-out loss delta (nats)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
    fig.savefig(args.plot, dpi=150)
    print(f"wrote {args.out}, {args.plot}")


if __name__ == "__main__":
    main()
