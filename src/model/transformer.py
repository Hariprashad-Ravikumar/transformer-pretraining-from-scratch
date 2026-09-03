"""Decoder-only transformer, implemented from scratch (no AutoModel/AutoConfig).

Architecture: pre-norm, RMSNorm, rotary position embeddings, causal self-attention
via F.scaled_dot_product_attention, SwiGLU MLP, weight-tied embedding/output head.
Every module below is authored here; the only borrowed primitives are
torch.nn.Linear/Embedding and F.scaled_dot_product_attention (a fused kernel, not
an architecture choice).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 1024
    dropout: float = 0.0
    bias: bool = False
    use_triton_rmsnorm: bool = False


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, use_triton: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.use_triton = use_triton

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_triton and x.is_cuda:
            from src.triton_kernels.rmsnorm import triton_rmsnorm

            return triton_rmsnorm(x, self.weight, self.eps)
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def build_rope_cache(seq_len: int, head_dim: int, base: int = 10000, device=None):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, n_head, T, head_dim)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[: x.size(-2)].unsqueeze(0).unsqueeze(0)
    sin = sin[: x.size(-2)].unsqueeze(0).unsqueeze(0)
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = cfg.dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        return_attn: bool = False,
        ablate_head: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        attn_weights = None
        if return_attn or ablate_head is not None:
            # manual (non-fused) attention -- only path that exposes per-head
            # weights and lets a head's value contribution be zeroed before
            # recombination. Not used in training/normal eval, where the
            # fused F.scaled_dot_product_attention kernel is faster.
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
            scores = scores.masked_fill(causal_mask, float("-inf"))
            attn_weights = torch.softmax(scores, dim=-1)
            if ablate_head is not None:
                v = v.clone()
                v[:, ablate_head, :, :] = 0.0
            y = attn_weights @ v
            if not return_attn:
                attn_weights = None
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y), attn_weights


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = int(8 * cfg.n_embd / 3)
        hidden = ((hidden + 63) // 64) * 64  # round to multiple of 64
        self.w1 = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.w2 = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.w3 = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd, use_triton=cfg.use_triton_rmsnorm)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.n_embd, use_triton=cfg.use_triton_rmsnorm)
        self.mlp = SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        return_attn: bool = False,
        ablate_head: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_out, attn_weights = self.attn(self.ln1(x), cos, sin, return_attn, ablate_head)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, attn_weights


class DecoderTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.n_embd, use_triton=cfg.use_triton_rmsnorm)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

        head_dim = cfg.n_embd // cfg.n_head
        cos, sin = build_rope_cache(cfg.block_size, head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # scaled init for residual projections, per GPT-2 paper
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w3.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        capture_attn: bool = False,
        ablate: tuple[int, int] | None = None,
    ):
        """ablate: (layer_idx, head_idx) to zero that head's value contribution
        in that layer only. capture_attn: return each layer's attention
        weights as a list (index i = layer i), else None."""
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} exceeds block_size {self.cfg.block_size}"
        x = self.drop(self.tok_emb(idx))
        cos, sin = self.rope_cos.to(x.device), self.rope_sin.to(x.device)
        attn_weights_per_layer = [] if capture_attn else None
        for layer_idx, block in enumerate(self.blocks):
            ablate_head = ablate[1] if ablate is not None and ablate[0] == layer_idx else None
            x, attn_weights = block(x, cos, sin, return_attn=capture_attn, ablate_head=ablate_head)
            if capture_attn:
                attn_weights_per_layer.append(attn_weights)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss, attn_weights_per_layer

    def configure_optimizers(self, weight_decay: float, lr: float, betas: tuple[float, float]):
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=torch.cuda.is_available())
