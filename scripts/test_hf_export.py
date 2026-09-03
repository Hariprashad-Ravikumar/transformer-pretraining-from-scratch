"""Sanity check: does the HF wrapper actually round-trip through
save_pretrained -> from_pretrained(trust_remote_code=True) and produce the
same output as calling the model directly? Uses random weights - this only
tests the plumbing, not model quality.
"""

import shutil
import sys

import torch

sys.path.insert(0, "hf_export")
from modeling_decoder_transformer import DecoderTransformerConfig, DecoderTransformerForCausalLM

OUT = "/tmp/hf_export_test"
shutil.rmtree(OUT, ignore_errors=True)

config = DecoderTransformerConfig(vocab_size=100, n_layer=2, n_head=2, n_embd=32, block_size=64)
config.auto_map = {
    "AutoConfig": "modeling_decoder_transformer.DecoderTransformerConfig",
    "AutoModelForCausalLM": "modeling_decoder_transformer.DecoderTransformerForCausalLM",
}
model = DecoderTransformerForCausalLM(config)
model.eval()

x = torch.randint(0, 100, (1, 16))
with torch.no_grad():
    out_before = model(x).logits

model.save_pretrained(OUT)
config.save_pretrained(OUT)
shutil.copy("hf_export/modeling_decoder_transformer.py", OUT)

# Reload exactly the way a real user on Hugging Face would, from scratch,
# using only the files on disk plus trust_remote_code.
from transformers import AutoModelForCausalLM

reloaded = AutoModelForCausalLM.from_pretrained(OUT, trust_remote_code=True)
reloaded.eval()
with torch.no_grad():
    out_after = reloaded(x).logits

assert torch.allclose(out_before, out_after, atol=1e-6), "round-trip mismatch!"
print("PASS: save_pretrained -> AutoModelForCausalLM.from_pretrained(trust_remote_code=True) round-trips correctly")
print(f"logits shape: {out_after.shape}")
