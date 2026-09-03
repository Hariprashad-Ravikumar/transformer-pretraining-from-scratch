---
license: mit
language:
- en
library_name: transformers
tags:
- decoder-only
- from-scratch
- pytorch
- causal-lm
pipeline_tag: text-generation
---

# transformer-pretraining-from-scratch

A ~57M-non-embedding-parameter decoder-only transformer, **pretrained from scratch in
PyTorch** — every `nn.Module` (RMSNorm, rotary position embeddings, causal self-attention,
SwiGLU MLP, weight-tied embedding/output head) is authored in the training repo, not
`AutoModel`, not a fine-tune of an existing checkpoint. Trained on ~1B tokens of
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) with a custom
byte-level BPE tokenizer (also trained from scratch, not GPT-2's).

Full training code, tokenizer, evaluation, calibration study, interpretability pass, and
writeups: [GitHub repo](https://github.com/Hariprashad-Ravikumar/transformer-pretraining-from-scratch).

## Why this exists

A portfolio project built to close a specific gap: demonstrating hands-on PyTorch/transformer
experience with a real trained-and-evaluated model, not just framework usage. It is a
companion to a separate, already-shipped production project (a cost-aware LLM router), and
is explicitly **not** trying to be state of the art, instruction-tuned, or chat-capable.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "hari-8/transformer-pretraining-from-scratch"
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(repo)

ids = tokenizer("The history of the Roman Empire", return_tensors="pt").input_ids
out = model.generate_simple(ids, max_new_tokens=40, temperature=0.8, top_k=40)
print(tokenizer.decode(out[0].tolist()))
```

`trust_remote_code=True` is required — the architecture (`modeling_decoder_transformer.py`)
ships in this repo rather than being a stock `transformers` model class.
`model.generate_simple(...)` is a minimal temperature/top-k sampling loop, not HF's
KV-cached `.generate()` — it recomputes the full sequence each step (fine for short demo
completions, not optimized for long generations or serving at scale).

## Architecture

| | |
|---|---|
| Layers | 8 |
| Attention heads | 12 |
| Embedding dim | 768 |
| Context length | 1024 |
| Vocab size | 16,384 (custom byte-level BPE) |
| Non-embedding params | ~56.6M |
| Position encoding | RoPE |
| Normalization | RMSNorm |
| MLP | SwiGLU |
| Attention | causal, `F.scaled_dot_product_attention` |
| Embedding/output head | weight-tied |

## Training

- ~1B tokens, [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) sample
- Single NVIDIA L4 GPU, bf16, `torch.compile`, gradient accumulation
- 7,630 steps, cosine LR schedule with warmup
- Final training loss: 3.16

## Evaluation

Held-out loss/perplexity/bits-per-byte (BPB), measured on ~497K held-out tokens never seen
during training. BPB (not raw perplexity) is the metric that's actually comparable across
models with different tokenizers — it normalizes by the byte length of the original text
rather than by token count.

| | loss (nats) | perplexity | BPB |
|---|---|---|---|
| **this model** | 3.19 | 24.2 | **1.057** |
| Pythia-70M (reference) | 3.68 | 39.7 | 1.135 |

This model scores lower BPB than Pythia-70M on this held-out set — likely explained by
domain match (trained exclusively on FineWeb-Edu, a narrower and more predictable
distribution than Pythia's Pile-trained generalist scope), not general capability; the two
are close in parameter count. Full methodology: `src/eval/` in the GitHub repo.

## Calibration

Token-level calibration (confidence = max softmax probability, label = whether that top-1
prediction matches the actual next token — the standard Guo et al. 2017 definition), on
~248K held-out tokens disjoint from the temperature-fitting split.

| | ECE (5 bins) | Brier |
|---|---|---|
| raw (T=1) | 0.0071 | 0.157 |
| temperature-scaled (T=1.014) | 0.0047 | 0.157 |

The model's raw confidence is already close to perfectly calibrated — the fitted
temperature is barely different from 1. Full write-up: `results/calibration_report.md` in
the GitHub repo.

## Interpretability

Induction-head probing (Olsson et al. 2022 methodology) found two clear induction heads
concentrated in layers 6-7 (prefix-matching scores 0.79 and 0.74, far above every other
head). A full causal ablation sweep (every head, every layer) found these aren't
necessarily the most causally important heads on natural text — a genuine finding about
the limits of attention-pattern-only probing. Full write-up:
`results/interpretability_report.md` in the GitHub repo.

## Limitations

- Not instruction-tuned — this is a raw next-token-prediction language model, not a chat
  assistant. Expect continuation-style completions, not question-answering or dialogue.
- Small (~57M params) and trained on ~1B tokens — coherent short-range text, not reliable
  long-form factual or reasoning capability.
- `generate_simple`'s O(n²) decoding (no KV cache) makes long generations slow.
- Custom architecture requires `trust_remote_code=True` to load.

## Citation

If this is useful, please link back to the
[GitHub repo](https://github.com/Hariprashad-Ravikumar/transformer-pretraining-from-scratch)
rather than citing formally — this is a portfolio/learning project, not a paper.
