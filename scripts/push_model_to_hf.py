"""Convert checkpoints/base.pt into the hf_export/ HF-compatible format and
push model + tokenizer to the Hub.

Verifies the converted model produces identical logits to the original
training-time model (src/model/transformer.py) before ever calling
push_to_hub -- a silently-wrong conversion pushed publicly is much worse
than one caught locally.

Usage:
    python -m scripts.push_model_to_hf --repo-id hari-8/transformer-pretraining-from-scratch
"""

import argparse
import os
import shutil
import sys

import torch
import yaml

from src.model.transformer import DecoderTransformer, ModelConfig

sys.path.insert(0, "hf_export")
from modeling_decoder_transformer import DecoderTransformerConfig, DecoderTransformerForCausalLM  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/base.pt")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--tokenizer-dir", default="data/tokenizer")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--local-dir", default="/tmp/hf_model_push")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--push", action="store_true", help="actually push; omit to only build+verify locally")
    args = parser.parse_args()

    with open(args.config) as f:
        train_cfg = yaml.safe_load(f)
    model_cfg_dict = train_cfg["model"]

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}

    print("building reference (training-arch) model for verification ...")
    ref_model = DecoderTransformer(ModelConfig(**model_cfg_dict))
    ref_model.load_state_dict(state_dict)
    ref_model.eval()

    print("building HF-wrapper model ...")
    hf_config = DecoderTransformerConfig(
        vocab_size=model_cfg_dict["vocab_size"],
        n_layer=model_cfg_dict["n_layer"],
        n_head=model_cfg_dict["n_head"],
        n_embd=model_cfg_dict["n_embd"],
        block_size=model_cfg_dict["block_size"],
        dropout=model_cfg_dict["dropout"],
        bias=model_cfg_dict["bias"],
    )
    hf_config.auto_map = {
        "AutoConfig": "modeling_decoder_transformer.DecoderTransformerConfig",
        "AutoModelForCausalLM": "modeling_decoder_transformer.DecoderTransformerForCausalLM",
    }
    hf_model = DecoderTransformerForCausalLM(hf_config)
    # rope_cos/rope_sin are non-persistent (not saved) in the training-time
    # checkpoint but persistent=True in the HF wrapper (see HANDOFF.md's
    # documented bug #5) -- already correctly populated by __init__'s
    # build_rope_cache call, so strict=False here is expected, not a bug.
    missing, unexpected = hf_model.model.load_state_dict(state_dict, strict=False)
    assert set(missing) <= {"rope_cos", "rope_sin"}, f"unexpected missing keys: {missing}"
    assert not unexpected, f"unexpected keys in checkpoint: {unexpected}"
    hf_model.eval()

    print("verifying converted model matches the training-time model exactly ...")
    torch.manual_seed(0)
    x = torch.randint(0, model_cfg_dict["vocab_size"], (2, 64))
    with torch.no_grad():
        ref_logits, _, _ = ref_model(x)
        hf_logits = hf_model(x).logits
    assert torch.allclose(ref_logits, hf_logits, atol=1e-4), "conversion mismatch -- do not push"
    print("PASS: converted model logits match the training model exactly")

    shutil.rmtree(args.local_dir, ignore_errors=True)
    os.makedirs(args.local_dir, exist_ok=True)
    hf_model.save_pretrained(args.local_dir)
    hf_config.save_pretrained(args.local_dir)
    shutil.copy("hf_export/modeling_decoder_transformer.py", args.local_dir)

    print("exporting tokenizer to HF format ...")
    from tokenizers import ByteLevelBPETokenizer
    from transformers import PreTrainedTokenizerFast

    tok = ByteLevelBPETokenizer(
        os.path.join(args.tokenizer_dir, "vocab.json"),
        os.path.join(args.tokenizer_dir, "merges.txt"),
    )
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok._tokenizer,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        pad_token="<|endoftext|>",
    )
    hf_tokenizer.save_pretrained(args.local_dir)

    print(f"local export ready at {args.local_dir}")
    for f in sorted(os.listdir(args.local_dir)):
        print(f"  {f}")

    if not args.push:
        print("--push not set, stopping here (local build+verify only)")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=args.local_dir, repo_id=args.repo_id, repo_type="model")
    print(f"pushed to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
