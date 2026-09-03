"""Fused RMSNorm forward + backward, written directly in Triton.

RMSNorm (no mean-subtraction, unlike LayerNorm): for a row x (last dim of
size N), rstd = 1/sqrt(mean(x^2) + eps), y = x * rstd * weight.

Backward derivation (rstd depends on every x_j in the row, not just x_i, so
the elementwise-looking forward has a full row-wide gradient):
    dL/dw_i = sum_over_rows(go_i * x_i * rstd)
    dL/dx_k = rstd * w_k * go_k - (rstd^3 * x_k / N) * sum_i(x_i * w_i * go_i)
where go is the upstream gradient (dL/dy). Matches the standard RMSNorm
backward used in LLaMA/T5-style implementations.

Only runs on CUDA (Triton has no CPU/MPS backend) -- this file, and its test,
only execute on the GPU VM, never on a Mac.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fwd_kernel(
    x_ptr, w_ptr, y_ptr, rstd_ptr,
    stride_row, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x_row_ptr = x_ptr + row * stride_row
    x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    mean_sq = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(mean_sq + eps)
    tl.store(rstd_ptr + row, rstd)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    y_row_ptr = y_ptr + row * stride_row
    tl.store(y_row_ptr + cols, y.to(x_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _rmsnorm_bwd_kernel(
    x_ptr, w_ptr, go_ptr, rstd_ptr, dx_ptr, dw_partial_ptr,
    stride_row, N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    go = tl.load(go_ptr + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)

    row_sum = tl.sum(x * w * go, axis=0)
    dx = rstd * w * go - (rstd * rstd * rstd) * x * row_sum / N
    tl.store(dx_ptr + row * stride_row + cols, dx.to(x_ptr.dtype.element_ty), mask=mask)

    dw_row = x * rstd * go
    tl.atomic_add(dw_partial_ptr + cols, dw_row, mask=mask)


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


class _TritonRMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.reshape(-1, N).contiguous()
        n_rows = x2d.shape[0]
        y = torch.empty_like(x2d)
        rstd = torch.empty(n_rows, device=x.device, dtype=torch.float32)
        BLOCK_SIZE = _next_pow2(N)
        _rmsnorm_fwd_kernel[(n_rows,)](
            x2d, weight, y, rstd, x2d.stride(0), N, eps, BLOCK_SIZE=BLOCK_SIZE
        )
        ctx.save_for_backward(x2d, weight, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.N = N
        return y.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x2d, weight, rstd = ctx.saved_tensors
        N = ctx.N
        go2d = grad_output.reshape(-1, N).contiguous()
        n_rows = x2d.shape[0]
        dx = torch.empty_like(x2d)
        dw_partial = torch.zeros(N, device=x2d.device, dtype=torch.float32)
        _rmsnorm_bwd_kernel[(n_rows,)](
            x2d, weight, go2d, rstd, dx, dw_partial, x2d.stride(0), N, BLOCK_SIZE=ctx.BLOCK_SIZE
        )
        return dx.reshape(grad_output.shape), dw_partial.to(weight.dtype), None


def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return _TritonRMSNormFn.apply(x, weight, eps)
