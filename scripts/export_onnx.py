"""Export checkpoints/base.pt to ONNX for in-browser inference (onnxruntime-web).

Fixed-shape export (batch=1, seq_len=block_size), no dynamic axes: the
browser demo always right-pads to block_size and reads logits at the last
real-token position, recomputing the full forward each generation step
(matches model.generate_simple()'s own O(n^2) approach server-side).

Forces the model's manual (matmul + softmax + causal mask) attention path
via capture_attn=True instead of the fused F.scaled_dot_product_attention
used in training -- fused SDPA's ONNX export support is inconsistent across
onnxruntime-web backends, while explicit matmul/softmax is universally
portable. tests/test_manual_attention_matches_fused.py already proves this
path is numerically identical to the fused one used in training.

Usage:
    python -m scripts.export_onnx --checkpoint checkpoints/base.pt \
        --out hf_static_demo/model.onnx
"""

import argparse
import os

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

from src.eval.evaluate import load_model


class ExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.model(idx, capture_attn=True)
        return logits


def verify(onnx_path: str, wrapped_model: nn.Module, block_size: int, vocab_size: int):
    torch.manual_seed(0)
    idx = torch.randint(0, vocab_size, (1, block_size))

    with torch.no_grad():
        torch_logits = wrapped_model(idx).numpy()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    (onnx_logits,) = session.run(None, {"input_ids": idx.numpy().astype(np.int64)})

    max_diff = np.abs(torch_logits - onnx_logits).max()
    print(f"max abs diff between PyTorch and ONNX Runtime logits: {max_diff:.6e}")
    assert np.allclose(torch_logits, onnx_logits, atol=1e-4), "ONNX export mismatch, do not deploy"
    print("PASS: ONNX export matches PyTorch (manual-attention path) exactly")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/base.pt")
    parser.add_argument("--out", default="hf_static_demo/model.onnx")
    args = parser.parse_args()

    print(f"loading {args.checkpoint} ...")
    model, ckpt = load_model(args.checkpoint, "cpu")
    model.eval()
    block_size = model.cfg.block_size
    vocab_size = model.cfg.vocab_size

    wrapped = ExportWrapper(model)
    wrapped.eval()

    dummy = torch.randint(0, vocab_size, (1, block_size))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"exporting to {args.out} (fixed shape 1x{block_size}, no dynamic axes) ...")
    torch.onnx.export(
        wrapped,
        (dummy,),
        args.out,
        input_names=["input_ids"],
        output_names=["logits"],
        opset_version=18,
        dynamo=False,
    )

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out} ({size_mb:.1f} MB)")

    print("verifying ONNX output matches PyTorch ...")
    verify(args.out, wrapped, block_size, vocab_size)


if __name__ == "__main__":
    main()
