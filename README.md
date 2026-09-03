# Transformer Pretraining From Scratch

A decoder-only transformer (~57M non-embedding parameters), pretrained from scratch
in PyTorch on a public web-text corpus, with a calibration study on token-level
confidence and a custom fused Triton kernel on the hot path.

**Why this exists:** its companion project, the
[Cost-Aware LLM Router](https://github.com/Hariprashad-Ravikumar/Cost-Aware-Multi-Agent-LLM-Router),
calls a pretrained transformer via a high-level API (`SentenceTransformer.encode()`) and
asks *when to trust a model's confidence*. This project builds the model and asks
*where that confidence comes from in the first place* — no `AutoModel`, no fine-tune,
every `nn.Module` here is authored in this repo.

## Status

Actively in progress. See commit history for what's actually done vs. planned below.

## What this is

- **Model**: pre-norm decoder-only transformer, RMSNorm, rotary position embeddings,
  SwiGLU MLP, weight-tied embedding/output head — `src/model/transformer.py`.
- **Tokenizer**: custom byte-level BPE trained on the training corpus (not a reused
  pretrained tokenizer) — `src/tokenizer/`.
- **Training**: bf16 mixed precision, `torch.compile`, gradient accumulation,
  DDP for multi-GPU, checkpoint/resume built to survive Spot preemption —
  `src/train/train.py`.
- **Evaluation**: held-out validation loss/perplexity plus bits-per-byte (BPB) against
  a named baseline — BPB rather than raw perplexity because our tokenizer's vocabulary
  isn't the baseline's, so per-token perplexity isn't directly comparable.
- **Calibration study**: reliability diagrams, ECE, Brier score on token-level
  confidence, same methodology as the router project.
- **Triton kernel**: one fused op on the hot path, correctness-tested against the
  PyTorch reference, benchmarked before/after on identical hardware.

## What this deliberately is not

Not state of the art, no instruction tuning, no chat model, no RLHF, no demo UI.
See `PROJECT_PLAN.md` (private planning doc, not in this repo) for the full scope
and non-goals.

## Repro

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Verify the pipeline end to end on CPU with fake data (no GPU spend)
python scripts/make_tiny_data.py
python -m src.train.train --config configs/tiny.yaml

# 2. Real run: train tokenizer, prepare data, then train
python -m src.tokenizer.train_tokenizer --input "data/raw/*.txt" --vocab-size 16384 --out data/tokenizer
python -m src.data.prepare_data --tokenizer data/tokenizer --out data/tokenized --target-tokens 1000000000
python -m src.train.train --config configs/base.yaml

# Multi-GPU
torchrun --standalone --nproc_per_node=4 -m src.train.train --config configs/base.yaml
```

## Results

Filled in once measured — perplexity/BPB vs. baseline, scaling efficiency at 1/2/4
GPUs (tokens/sec, MFU), Triton kernel speedup, calibration ECE/Brier. Not estimated.

## Limitations

Written once training and evaluation are complete, in the author's own words, not
generated. This section matters more than it looks — see `PROJECT_PLAN.md`.
