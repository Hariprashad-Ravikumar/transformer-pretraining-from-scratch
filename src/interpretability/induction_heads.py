"""Induction-head probe via synthetic repeated-token sequences.

Methodology: Olsson et al., "In-context Learning and Induction Heads"
(Anthropic, 2022). Feed sequences of random tokens repeated twice
([A B C ... Z] [A B C ... Z]) and measure, for each head, how much attention
weight the second occurrence of a token puts on the position that followed
that token's *first* occurrence -- the "prefix matching + copying" signature
of an induction head. A head with a high score has learned "the token after
the last time I saw this token is probably what comes next," independent of
which specific tokens those are (that's what makes it a general, reusable
circuit rather than a memorized bigram).

Usage:
    python -m src.interpretability.induction_heads --checkpoint checkpoints/base.pt \
        --out results/induction_heads_base.json --plot results/induction_heads_base.png
"""

import argparse
import json
import os

import numpy as np
import torch

# Capped low deliberately -- this runs on a memory-constrained laptop (M1, 8GB)
# and shouldn't peg every core while other work is happening on the machine.
torch.set_num_threads(4)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.eval.evaluate import load_model


def make_repeated_sequences(vocab_size: int, seq_len: int, n_seqs: int, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    # reserve a small low-id range often used for special tokens; sample from the rest
    lo = 8
    seqs = []
    for _ in range(n_seqs):
        half = rng.integers(lo, vocab_size, size=seq_len)
        seqs.append(np.concatenate([half, half]))
    return torch.from_numpy(np.stack(seqs)).long()


def induction_scores(model, seqs: torch.Tensor, seq_len: int) -> np.ndarray:
    """Returns an (n_layer, n_head) array of mean prefix-matching attention
    weight, averaged over the second half of each sequence and over sequences."""
    with torch.no_grad():
        _, _, attn_per_layer = model(seqs, capture_attn=True)
    n_layer = len(attn_per_layer)
    n_head = attn_per_layer[0].shape[1]
    scores = np.zeros((n_layer, n_head))
    # Query position (seq_len + j) is the second occurrence of the token that
    # first appeared at position j. An induction head attends from there to
    # position j+1 -- the token that followed the *first* occurrence -- since
    # that's the token it should copy forward as the prediction for what
    # comes after this (second) occurrence. Valid for j in [0, seq_len - 2]
    # (j+1 must stay inside the first half).
    query_positions = list(range(seq_len, 2 * seq_len - 1))
    target_positions = list(range(1, seq_len))
    for layer_idx, attn in enumerate(attn_per_layer):
        # attn: (B, n_head, T, T)
        picked = attn[:, :, query_positions, :][:, :, :, target_positions]
        # picked[b, h, j, j] is the weight from query (seq_len+1+j) to target (j)
        diag = torch.diagonal(picked, dim1=-2, dim2=-1)  # (B, n_head, min(len(q),len(t)))
        scores[layer_idx] = diag.mean(dim=(0, 2)).numpy()
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/base.pt")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--n-seqs", type=int, default=16)
    parser.add_argument("--out", default="results/induction_heads.json")
    parser.add_argument("--plot", default="results/induction_heads.png")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model, ckpt = load_model(args.checkpoint, args.device)
    seqs = make_repeated_sequences(model.cfg.vocab_size, args.seq_len, args.n_seqs)
    assert 2 * args.seq_len <= model.cfg.block_size, "sequence too long for block_size"

    scores = induction_scores(model, seqs, args.seq_len)
    n_layer, n_head = scores.shape

    ranked = sorted(
        [{"layer": l, "head": h, "score": float(scores[l, h])} for l in range(n_layer) for h in range(n_head)],
        key=lambda d: -d["score"],
    )
    result = {"checkpoint": args.checkpoint, "n_layer": n_layer, "n_head": n_head, "scores": scores.tolist(), "ranked": ranked}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(ranked[:5], indent=2))

    fig, ax = plt.subplots(figsize=(1.2 * n_head + 2, 1.2 * n_layer + 1))
    im = ax.imshow(scores, cmap="viridis", vmin=0, vmax=max(scores.max(), 1e-6))
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_xticks(range(n_head))
    ax.set_yticks(range(n_layer))
    ax.set_title(f"Induction-head prefix-matching score\n{os.path.basename(args.checkpoint)}")
    fig.colorbar(im, ax=ax, label="prefix-matching score")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
    fig.savefig(args.plot, dpi=150)
    print(f"wrote {args.out}, {args.plot}")


if __name__ == "__main__":
    main()
