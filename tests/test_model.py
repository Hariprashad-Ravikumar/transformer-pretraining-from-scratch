import torch

from src.model.transformer import DecoderTransformer, ModelConfig


def test_forward_shapes_and_loss():
    cfg = ModelConfig(vocab_size=100, n_layer=2, n_head=2, n_embd=32, block_size=16)
    model = DecoderTransformer(cfg)
    x = torch.randint(0, 100, (4, 16))
    y = torch.randint(0, 100, (4, 16))
    logits, loss, attn = model(x, y)
    assert logits.shape == (4, 16, 100)
    assert loss.item() > 0
    assert attn is None


def test_weight_tying():
    cfg = ModelConfig(vocab_size=100, n_layer=1, n_head=2, n_embd=32, block_size=16)
    model = DecoderTransformer(cfg)
    assert model.head.weight.data_ptr() == model.tok_emb.weight.data_ptr()


def test_causal_masking_no_future_leakage():
    cfg = ModelConfig(vocab_size=100, n_layer=2, n_head=2, n_embd=32, block_size=16)
    model = DecoderTransformer(cfg)
    model.eval()
    x = torch.randint(0, 100, (1, 8))
    with torch.no_grad():
        logits_full, _, _ = model(x)
    x_truncated = x[:, :4]
    with torch.no_grad():
        logits_trunc, _, _ = model(x_truncated)
    assert torch.allclose(logits_full[:, :4], logits_trunc, atol=1e-4)


def test_manual_attention_matches_fused():
    cfg = ModelConfig(vocab_size=100, n_layer=2, n_head=2, n_embd=32, block_size=16)
    model = DecoderTransformer(cfg)
    model.eval()
    x = torch.randint(0, 100, (2, 10))
    with torch.no_grad():
        logits_fused, _, attn_none = model(x)
        logits_manual, _, attn_weights = model(x, capture_attn=True)
    assert attn_none is None
    assert len(attn_weights) == cfg.n_layer
    for w in attn_weights:
        assert w.shape == (2, cfg.n_head, 10, 10)
    assert torch.allclose(logits_fused, logits_manual, atol=1e-4)


def test_ablate_head_zeroes_its_contribution():
    cfg = ModelConfig(vocab_size=100, n_layer=2, n_head=2, n_embd=32, block_size=16)
    model = DecoderTransformer(cfg)
    model.eval()
    x = torch.randint(0, 100, (2, 10))
    with torch.no_grad():
        logits_base, _, _ = model(x)
        logits_ablated, _, _ = model(x, ablate=(0, 0))
    assert not torch.allclose(logits_base, logits_ablated)
