"""Residual-stream norm growth across depth.

Uses external forward hooks (no transformer.py changes needed for this one --
register_forward_hook on each Block captures its output without touching the
model's forward logic at all) to record the mean L2 norm of the residual
stream after each block, over a batch of held-out text.

Usage:
    python -m src.interpretability.residual_norms --checkpoint checkpoints/base.pt \
        --val-bin data/tokenized/val.bin --out results/residual_norms_base.json
"""

import argparse
import json
import os

import numpy as np
import torch

torch.set_num_threads(4)

from src.eval.evaluate import load_model


def measure_norms(model, x: torch.Tensor) -> list[float]:
    norms = []

    def hook(module, inputs, output):
        block_out, _ = output
        norms.append(block_out.norm(dim=-1).mean().item())

    handles = [block.register_forward_hook(hook) for block in model.blocks]
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    return norms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/base.pt")
    parser.add_argument("--val-bin", default="data/tokenized/val.bin")
    parser.add_argument("--n-tokens", type=int, default=8192)
    parser.add_argument("--out", default="results/residual_norms.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model, ckpt = load_model(args.checkpoint, args.device)
    block_size = model.cfg.block_size

    ids = np.memmap(args.val_bin, dtype=np.uint16, mode="r")[: args.n_tokens]
    data = np.array(ids, dtype=np.int64)
    n_blocks = len(data) // block_size
    xs = np.stack([data[i * block_size : (i + 1) * block_size] for i in range(n_blocks)])
    x = torch.from_numpy(xs).to(args.device)

    norms = measure_norms(model, x)
    result = {"checkpoint": args.checkpoint, "n_layer": model.cfg.n_layer, "norms_by_layer": norms}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
