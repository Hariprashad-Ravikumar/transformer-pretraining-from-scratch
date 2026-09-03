"""Train a byte-level BPE tokenizer from scratch on the training corpus.

Usage:
    python -m src.tokenizer.train_tokenizer --input data/raw/*.txt --vocab-size 16384 --out data/tokenizer
"""

import argparse
import glob

from tokenizers import ByteLevelBPETokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, help="glob pattern(s) for training text files")
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--out", required=True, help="output directory for tokenizer files")
    args = parser.parse_args()

    files = []
    for pattern in args.input:
        files.extend(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no files matched {args.input}")

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=files,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=["<|endoftext|>"],
    )
    tokenizer.save_model(args.out)
    print(f"trained byte-level BPE, vocab_size={args.vocab_size}, saved to {args.out}")


if __name__ == "__main__":
    main()
