"""Held-out evaluation for our from-scratch model: loss, perplexity, bits-per-byte.

Also reconstructs the held-out set's original text (byte-level BPE decode is
lossless) and writes it to disk so baseline_pythia.py can evaluate a different
tokenizer's model on the exact same held-out content.

Usage:
    python -m src.eval.evaluate --checkpoint checkpoints/base.pt \
        --val-bin data/tokenized/val.bin --tokenizer data/tokenizer \
        --out results/eval_summary.json --holdout-text data/eval_holdout.txt
"""

import argparse
import json
import math
import os

import numpy as np
import torch
from tokenizers import ByteLevelBPETokenizer

from src.model.transformer import DecoderTransformer, ModelConfig


def load_model(checkpoint_path: str, device: str) -> tuple[DecoderTransformer, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = ModelConfig(**ckpt["config"]["model"])
    model = DecoderTransformer(model_cfg)
    # training used torch.compile, which wraps the model and prefixes all
    # state_dict keys with "_orig_mod." -- strip it before loading into the
    # uncompiled eval-time model.
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, ckpt


def reconstruct_holdout_text(val_bin: str, tokenizer_dir: str) -> tuple[str, list[int]]:
    tok = ByteLevelBPETokenizer(
        os.path.join(tokenizer_dir, "vocab.json"),
        os.path.join(tokenizer_dir, "merges.txt"),
    )
    eot_id = tok.token_to_id("<|endoftext|>")
    ids = np.memmap(val_bin, dtype=np.uint16, mode="r").tolist()

    # roundtrip sanity check: decode(encode(text)) must reproduce text exactly,
    # since BPB depends on the byte count of the reconstructed text being correct.
    sample = "The transformer architecture uses self-attention, 123 & symbols too."
    assert tok.decode(tok.encode(sample).ids) == sample, "tokenizer decode roundtrip failed"

    segments, current = [], []
    for tid in ids:
        if tid == eot_id:
            if current:
                segments.append(current)
            current = []
        else:
            current.append(tid)
    if current:
        segments.append(current)

    texts = [tok.decode(seg) for seg in segments]
    return "\n".join(texts), ids


def compute_loss(model: DecoderTransformer, ids: list[int], block_size: int, device: str, batch_size: int = 8):
    data = np.array(ids, dtype=np.int64)
    n_blocks = (len(data) - 1) // block_size
    total_nll, total_tokens = 0.0, 0
    with torch.no_grad():
        for start in range(0, n_blocks, batch_size):
            end = min(start + batch_size, n_blocks)
            xs = [data[i * block_size : i * block_size + block_size] for i in range(start, end)]
            ys = [data[i * block_size + 1 : i * block_size + 1 + block_size] for i in range(start, end)]
            x = torch.from_numpy(np.stack(xs)).to(device)
            y = torch.from_numpy(np.stack(ys)).to(device)
            _, loss, _ = model(x, y)
            ntok = x.numel()
            total_nll += loss.item() * ntok
            total_tokens += ntok
    return total_nll / total_tokens, total_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/base.pt")
    parser.add_argument("--val-bin", default="data/tokenized/val.bin")
    parser.add_argument("--tokenizer", default="data/tokenizer")
    parser.add_argument("--out", default="results/eval_summary.json")
    parser.add_argument("--holdout-text", default="data/eval_holdout.txt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("reconstructing held-out text from val.bin ...")
    text, ids = reconstruct_holdout_text(args.val_bin, args.tokenizer)
    os.makedirs(os.path.dirname(args.holdout_text) or ".", exist_ok=True)
    with open(args.holdout_text, "w") as f:
        f.write(text)
    total_bytes = len(text.encode("utf-8"))
    print(f"held-out set: {len(ids):,} tokens, {total_bytes:,} bytes, wrote {args.holdout_text}")

    print(f"loading model from {args.checkpoint} ...")
    model, ckpt = load_model(args.checkpoint, args.device)
    block_size = model.cfg.block_size

    print("computing held-out loss ...")
    loss, n_eval_tokens = compute_loss(model, ids, block_size, args.device)
    perplexity = math.exp(loss)
    bpb = (loss * n_eval_tokens) / math.log(2) / total_bytes

    result = {
        "checkpoint_step": ckpt["step"],
        "held_out_tokens": len(ids),
        "held_out_bytes": total_bytes,
        "eval_tokens": n_eval_tokens,
        "loss_nats": loss,
        "perplexity": perplexity,
        "bits_per_byte": bpb,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
