import torch

from src.model.transformer import DecoderTransformer, ModelConfig


def test_forward_shapes_and_loss():
    cfg = ModelConfig(vocab_size=100, n_layer=2, n_head=2, n_embd=32, block_size=16)
    model = DecoderTransformer(cfg)
    x = torch.randint(0, 100, (4, 16))
    y = torch.randint(0, 100, (4, 16))
    logits, loss = model(x, y)
    assert logits.shape == (4, 16, 100)
    assert loss.item() > 0


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
        logits_full, _ = model(x)
    x_truncated = x[:, :4]
    with torch.no_grad():
        logits_trunc, _ = model(x_truncated)
    assert torch.allclose(logits_full[:, :4], logits_trunc, atol=1e-4)
