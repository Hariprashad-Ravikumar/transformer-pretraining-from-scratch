"""Convert our vocab.json/merges.txt tokenizer into the unified tokenizer.json
format HF's PreTrainedTokenizerFast (and therefore AutoTokenizer) expects.

Usage:
    python -m src.tokenizer.export_hf_tokenizer --in data/tokenizer --out hf_export
"""

import argparse
import os

from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tok = ByteLevelBPETokenizer(
        os.path.join(args.in_dir, "vocab.json"),
        os.path.join(args.in_dir, "merges.txt"),
    )

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok._tokenizer,  # the underlying tokenizers.Tokenizer
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        pad_token="<|endoftext|>",
    )
    hf_tokenizer.save_pretrained(args.out)

    # sanity check: encode/decode roundtrip through the HF-facing tokenizer
    text = "The transformer architecture uses self-attention."
    ids = hf_tokenizer.encode(text)
    decoded = hf_tokenizer.decode(ids)
    print(f"encoded: {ids}")
    print(f"decoded: {decoded!r}")
    assert decoded.strip() == text, "HF tokenizer roundtrip mismatch"
    print(f"saved HF-compatible tokenizer to {args.out}")


if __name__ == "__main__":
    main()
