"""Correctness + gradient checks for the Triton RMSNorm kernel.

Triton has no CPU/MPS backend -- these tests only run where CUDA is
available (the GPU VM), and are skipped everywhere else (including your
Mac) rather than failing or hanging.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton requires CUDA")


def _reference_rmsnorm(x, weight, eps=1e-5):
    norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return norm * weight


def test_forward_matches_reference():
    from src.triton_kernels.rmsnorm import triton_rmsnorm

    torch.manual_seed(0)
    x = torch.randn(4, 16, 256, device="cuda", dtype=torch.float32)
    weight = torch.randn(256, device="cuda", dtype=torch.float32)
    out_triton = triton_rmsnorm(x, weight)
    out_ref = _reference_rmsnorm(x, weight)
    assert torch.allclose(out_triton, out_ref, atol=1e-4, rtol=1e-4)


def test_gradients_match_reference():
    from src.triton_kernels.rmsnorm import triton_rmsnorm

    torch.manual_seed(0)
    x_ref = torch.randn(4, 16, 256, device="cuda", dtype=torch.float32, requires_grad=True)
    w_ref = torch.randn(256, device="cuda", dtype=torch.float32, requires_grad=True)
    x_tri = x_ref.detach().clone().requires_grad_(True)
    w_tri = w_ref.detach().clone().requires_grad_(True)

    out_ref = _reference_rmsnorm(x_ref, w_ref)
    out_tri = triton_rmsnorm(x_tri, w_tri)

    go = torch.randn_like(out_ref)
    out_ref.backward(go)
    out_tri.backward(go)

    assert torch.allclose(x_ref.grad, x_tri.grad, atol=1e-3, rtol=1e-3)
    assert torch.allclose(w_ref.grad, w_tri.grad, atol=1e-3, rtol=1e-3)


def test_matches_reference_within_full_model():
    from src.model.transformer import DecoderTransformer, ModelConfig

    torch.manual_seed(0)
    cfg_plain = ModelConfig(vocab_size=100, n_layer=2, n_head=2, n_embd=32, block_size=16, use_triton_rmsnorm=False)
    cfg_triton = ModelConfig(vocab_size=100, n_layer=2, n_head=2, n_embd=32, block_size=16, use_triton_rmsnorm=True)

    model_plain = DecoderTransformer(cfg_plain).cuda()
    model_triton = DecoderTransformer(cfg_triton).cuda()
    model_triton.load_state_dict(model_plain.state_dict())

    x = torch.randint(0, 100, (2, 10), device="cuda")
    with torch.no_grad():
        logits_plain, _, _ = model_plain(x)
        logits_triton, _, _ = model_triton(x)
    assert torch.allclose(logits_plain, logits_triton, atol=1e-3, rtol=1e-3)
