"""Evaluate a Pythia checkpoint on the exact same held-out text used for our
model (written by evaluate.py to data/eval_holdout.txt), so bits-per-byte is
comparable across the two different tokenizers.

Note: Pythia was pretrained on the Pile. It's very unlikely these specific
FineWeb-Edu rows leaked into its training data, but that isn't independently
verified here -- treat this as a reasonable size-matched reference point, not
an airtight controlled comparison.

Usage:
    python -m src.eval.baseline_pythia --model EleutherAI/pythia-70m \
        --holdout-text data/eval_holdout.txt --out results/baseline_pythia.json
"""

import argparse
import json
import math
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-70m")
    parser.add_argument("--holdout-text", default="data/eval_holdout.txt")
    parser.add_argument("--out", default="results/baseline_pythia.json")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.holdout_text) as f:
        text = f.read()
    total_bytes = len(text.encode("utf-8"))

    print(f"loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)
    model.eval()

    ids = tokenizer(text)["input_ids"]
    print(f"held-out set re-tokenized with {args.model}'s tokenizer: {len(ids):,} tokens")

    block_size = args.block_size
    n_blocks = (len(ids) - 1) // block_size
    total_nll, total_tokens = 0.0, 0
    with torch.no_grad():
        for start in range(0, n_blocks, args.batch_size):
            end = min(start + args.batch_size, n_blocks)
            xs = [ids[i * block_size : i * block_size + block_size] for i in range(start, end)]
            ys = [ids[i * block_size + 1 : i * block_size + 1 + block_size] for i in range(start, end)]
            x = torch.tensor(xs, dtype=torch.long, device=args.device)
            y = torch.tensor(ys, dtype=torch.long, device=args.device)
            logits = model(x).logits
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            ntok = x.numel()
            total_nll += loss.item() * ntok
            total_tokens += ntok

    loss = total_nll / total_tokens
    perplexity = math.exp(loss)
    bpb = (loss * total_tokens) / math.log(2) / total_bytes

    result = {
        "model": args.model,
        "held_out_bytes": total_bytes,
        "eval_tokens": total_tokens,
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
