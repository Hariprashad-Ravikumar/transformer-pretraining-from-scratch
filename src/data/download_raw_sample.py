"""Download a raw-text sample from the corpus, used only to train the tokenizer.
Kept separate from the full tokenize-and-shard pass in prepare_data.py so the
tokenizer isn't re-downloading the whole target token budget just to bootstrap.

Usage:
    python -m src.data.download_raw_sample \
        --dataset HuggingFaceFW/fineweb-edu --subset sample-10BT \
        --out data/raw --target-chars 300_000_000
"""

import argparse
import os

from datasets import load_dataset
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--subset", default="sample-10BT")
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-chars", type=int, default=300_000_000)
    parser.add_argument("--shard-chars", type=int, default=50_000_000)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ds = load_dataset(args.dataset, name=args.subset, split="train", streaming=True)

    n_chars = 0
    shard_idx = 0
    shard_chars = 0
    f = open(os.path.join(args.out, f"shard_{shard_idx:03d}.txt"), "w")
    pbar = tqdm(total=args.target_chars, unit="char")
    for row in ds:
        text = row.get("text", "")
        if not text:
            continue
        f.write(text)
        f.write("\n")
        n_chars += len(text)
        shard_chars += len(text)
        pbar.update(len(text))
        if shard_chars >= args.shard_chars:
            f.close()
            shard_idx += 1
            shard_chars = 0
            f = open(os.path.join(args.out, f"shard_{shard_idx:03d}.txt"), "w")
        if n_chars >= args.target_chars:
            break
    pbar.close()
    f.close()
    print(f"wrote {n_chars:,} chars across {shard_idx + 1} shard(s) to {args.out}")


if __name__ == "__main__":
    main()
