"""Download a corpus subset, tokenize it with the trained tokenizer, and shard
to flat uint16 .bin files for memory-mapped reading during training.

Usage:
    python -m src.data.prepare_data \
        --dataset HuggingFaceFW/fineweb-edu --subset sample-10BT \
        --tokenizer data/tokenizer --out data/tokenized --target-tokens 1_000_000_000
"""

import argparse
import os

import numpy as np
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--subset", default="sample-10BT")
    parser.add_argument("--tokenizer", required=True, help="directory with vocab.json/merges.txt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-tokens", type=int, default=1_000_000_000)
    parser.add_argument("--val-fraction", type=float, default=0.0005)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tok = ByteLevelBPETokenizer(
        os.path.join(args.tokenizer, "vocab.json"),
        os.path.join(args.tokenizer, "merges.txt"),
    )
    eot_id = tok.token_to_id("<|endoftext|>")

    ds = load_dataset(args.dataset, name=args.subset, split="train", streaming=True)

    train_path = os.path.join(args.out, "train.bin")
    val_path = os.path.join(args.out, "val.bin")
    train_f = open(train_path, "wb")
    val_f = open(val_path, "wb")

    n_tokens = 0
    pbar = tqdm(total=args.target_tokens, unit="tok")
    for i, row in enumerate(ds):
        text = row.get("text", "")
        if not text:
            continue
        ids = tok.encode(text).ids + [eot_id]
        arr = np.array(ids, dtype=np.uint16)
        dest = val_f if (i % int(1 / args.val_fraction) == 0) else train_f
        dest.write(arr.tobytes())
        n_tokens += len(ids)
        pbar.update(len(ids))
        if n_tokens >= args.target_tokens:
            break
    pbar.close()
    train_f.close()
    val_f.close()
    print(f"wrote {n_tokens:,} tokens total to {args.out} (train.bin + val.bin)")


if __name__ == "__main__":
    main()
