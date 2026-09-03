"""Hugging Face-compatible wrapper for the from-scratch decoder transformer.

This file is self-contained (no imports from the rest of this repo) because
Hugging Face's `trust_remote_code=True` loading only pulls in files that live
in the model's own HF repo, not the GitHub repo. The actual architecture is
identical to src/model/transformer.py, duplicated here so this file can be
uploaded on its own alongside config.json and the weights.

Registered via config.json's "auto_map" so it loads with:
    AutoModelForCausalLM.from_pretrained("<repo>", trust_remote_code=True)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput


class DecoderTransformerConfig(PretrainedConfig):
    model_type = "decoder_transformer_scratch"

    def __init__(
        self,
        vocab_size: int = 16384,
        n_layer: int = 8,
        n_head: int = 12,
        n_embd: int = 768,
        block_size: int = 1024,
        dropout: float = 0.0,
        bias: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        self.dropout = dropout
        self.bias = bias
        kwargs.setdefault("tie_word_embeddings", True)
        super().__init__(**kwargs)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def build_rope_cache(seq_len: int, head_dim: int, base: int = 10000, device=None):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[: x.size(-2)].unsqueeze(0).unsqueeze(0)
    sin = sin[: x.size(-2)].unsqueeze(0).unsqueeze(0)
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: DecoderTransformerConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: DecoderTransformerConfig):
        super().__init__()
        hidden = int(8 * cfg.n_embd / 3)
        hidden = ((hidden + 63) // 64) * 64
        self.w1 = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.w2 = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.w3 = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class Block(nn.Module):
    def __init__(self, cfg: DecoderTransformerConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


class DecoderTransformerCore(nn.Module):
    """Identical structure/state_dict keys to src/model/transformer.py's
    DecoderTransformer, so a training checkpoint loads in directly."""

    def __init__(self, cfg: DecoderTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        head_dim = cfg.n_embd // cfg.n_head
        cos, sin = build_rope_cache(cfg.block_size, head_dim)
        # persistent=True here (unlike the training-time version of this model)
        # because HF's from_pretrained constructs modules under a meta-device
        # fast-init path: non-persistent buffers never get re-filled after
        # loading, silently leaving them as uninitialized garbage.
        self.register_buffer("rope_cos", cos, persistent=True)
        self.register_buffer("rope_sin", sin, persistent=True)

        self.apply(self._init_weights)
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

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} exceeds block_size {self.cfg.block_size}"
        x = self.drop(self.tok_emb(idx))
        cos, sin = self.rope_cos.to(x.device), self.rope_sin.to(x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss


class DecoderTransformerForCausalLM(PreTrainedModel):
    """The HF-facing wrapper. Delegates all real computation to `self.model`
    (DecoderTransformerCore), which keeps this file's forward pass identical
    to what was actually trained rather than a reimplementation that could
    drift from it."""

    config_class = DecoderTransformerConfig
    # tells HF's save/load machinery that model.head.weight is intentionally
    # the same tensor as the input embeddings, not an accidental duplicate,
    # so it's saved once and re-tied automatically on load
    _tied_weights_keys = {"model.head.weight": "model.tok_emb.weight"}

    def __init__(self, config: DecoderTransformerConfig):
        super().__init__(config)
        self.model = DecoderTransformerCore(config)
        self.post_init()

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None, **kwargs):
        logits, loss = self.model(input_ids, labels)
        return CausalLMOutput(loss=loss, logits=logits)

    def get_input_embeddings(self):
        return self.model.tok_emb

    def set_input_embeddings(self, value):
        self.model.tok_emb = value

    def get_output_embeddings(self):
        return self.model.head

    def set_output_embeddings(self, new_embeddings):
        self.model.head = new_embeddings

    @torch.no_grad()
    def generate_simple(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
    ):
        """Minimal sampling decode loop (temperature + top-k), recomputing the
        full sequence each step. This custom architecture doesn't implement
        HF's KV-cache generation hooks, so this is O(n^2) in sequence length
        instead of O(n) - fine for short demo completions, not for serving
        at scale, which is exactly the kind of tradeoff worth being explicit
        about rather than pretending .generate() here is production-grade."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.config.block_size :]
            logits, _ = self.model(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids
