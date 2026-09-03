# Transformer Pretraining From Scratch

A decoder-only transformer (~57M non-embedding parameters), pretrained from scratch
in PyTorch on a public web-text corpus, with a calibration study on token-level
confidence and a custom fused Triton kernel on the hot path.

**Why this exists:** its companion project, the
[Cost-Aware LLM Router](https://github.com/Hariprashad-Ravikumar/Cost-Aware-Multi-Agent-LLM-Router),
calls a pretrained transformer via a high-level API (`SentenceTransformer.encode()`) and
asks *when to trust a model's confidence*. This project builds the model and asks
*where that confidence comes from in the first place*. No `AutoModel`, no fine-tune:
every `nn.Module` here is authored in this repo.

## Status

Actively in progress. See commit history for what's actually done vs. planned below.

## What this is

- Model: pre-norm decoder-only transformer, RMSNorm, rotary position embeddings,
  SwiGLU MLP, weight-tied embedding/output head. `src/model/transformer.py`.
- Tokenizer: custom byte-level BPE trained on the training corpus, not a reused
  pretrained tokenizer. `src/tokenizer/`.
- Training: bf16 mixed precision, `torch.compile`, gradient accumulation,
  DDP for multi-GPU, checkpoint/resume built to survive Spot preemption.
  `src/train/train.py`.
- Evaluation: held-out validation loss/perplexity plus bits-per-byte (BPB) against
  a named baseline. BPB rather than raw perplexity because our tokenizer's vocabulary
  isn't the baseline's, so per-token perplexity isn't directly comparable.
- Calibration study: reliability diagrams, ECE, Brier score on token-level
  confidence, same methodology as the router project.
- Triton kernel: one fused op on the hot path, correctness-tested against the
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

**Pretraining** (`configs/base.yaml`, 57M non-embedding params, ~1B tokens on a
FineWeb-Edu sample, single L4 GPU): 7,630 steps, ~67,400 tok/s, final training
loss 3.16, no divergence.

**Held-out evaluation** (`src/eval/evaluate.py`, 497K held-out tokens):

| | loss (nats) | perplexity | bits-per-byte |
|---|---|---|---|
| this model | 3.19 | 24.2 | **1.057** |
| Pythia-70M baseline | 3.68 | 39.7 | 1.135 |

BPB (not raw perplexity) is the comparable metric here, since the two models
use different tokenizers. See `src/eval/baseline_pythia.py` for the byte-count
methodology. Our model scores lower BPB than Pythia-70M on this held-out set.
The likely explanation is domain match, not general capability: this model was
trained exclusively on FineWeb-Edu (educational web text), a narrower and more
predictable distribution than Pythia's Pile-trained generalist scope, and both
models are close in parameter count. Caveat: Pythia's training data isn't
independently verified to exclude these exact FineWeb-Edu rows, though a leak is
very unlikely.

**Token-level calibration study** (`src/eval/calibration.py`, methodology and
ECE/Brier formulas matched to the
[router repo](https://github.com/Hariprashad-Ravikumar/Cost-Aware-Multi-Agent-LLM-Router)'s
calibrator report for cross-project comparability; ~249K held-out tokens fit a
single temperature scalar, and ~249K disjoint tokens measure ECE/Brier, a genuine
generalization test rather than a same-distribution sanity check):

| | ECE (5 bins) | Brier score |
|---|---|---|
| this model, raw (T=1) | **0.0071** | 0.1572 |
| this model, temperature-scaled (T=1.014) | 0.0047 | 0.1571 |
| router calibrator (task-level correctness, for reference) | 0.1671 | 0.1521 |

The model's raw token-level confidence is already close to perfectly
calibrated: the fitted temperature (1.014) is barely different from 1, and
temperature scaling only marginally improves an already-low ECE. This is a
real, honest finding, not a failure to find something more dramatic. Models
trained end-to-end with softmax cross-entropy on next-token prediction are
well documented to calibrate more naturally than classifiers trained on
one-hot labels, because the training objective directly targets the true
conditional distribution rather than a decision boundary. The router's
0.1671 ECE isn't a fair "worse" comparison; it's a different task
entirely (task-level response correctness, judged, via a separately-trained
logistic-regression calibrator on hand-engineered features), included here
for reference, not as a competing number. Full write-up and the reliability
diagram: `results/calibration_report.md` / `results/calibration_curve.png`.

**Interpretability pass** (`src/interpretability/`, methodology from Olsson et al.
2022, "In-context Learning and Induction Heads"): induction-head probing (synthetic
repeated-token sequences, prefix-matching attention score), a full causal ablation
sweep (every head in every layer, held-out loss delta when zeroed), and residual-stream
norm growth, run on both `base.pt` and a smaller companion checkpoint
(`configs/interp_small.yaml`, 4 layers/256 dim, 3,000 steps on the same real corpus).

`base.pt` has two clear induction heads (layer 6 head 8, score 0.79; layer 6 head 4,
score 0.74, a scale more than 10x anything elsewhere in the model), concentrated in
layers 6-7. `interp_small.pt`'s strongest score is 0.03, an order of magnitude weaker,
consistent with induction heads emerging as a fairly sharp phase change during training
rather than gradually (caveat: this model differs in both scale *and* training length
from `base.pt` at once, so it isn't a clean scale-only comparison).

The causal ablation sweep is what makes this more than a picture: the heads with the
highest induction score are *not* reliably the most important by ablation. On `base.pt`,
the two strongest induction heads rank only 8th and 11th of 12 within their own layer by
held-out loss delta when zeroed; the heads that matter most (layer 5 head 4, layer 0 head
2) show no elevated induction score at all. Likely explanation: the synthetic probe
targets exact token repeats, a narrow pattern rarely seen verbatim in natural held-out
text, so a head specialized for it may matter less for real-text loss than a head doing
more general work the probe doesn't target. Full write-up:
`results/interpretability_report.md`.

**Triton kernel** (`src/triton_kernels/rmsnorm.py`, hand-written forward + backward,
correctness/gradient-tested against the PyTorch reference; 3/3 tests pass on the L4):

| | tok/s | MFU |
|---|---|---|
| PyTorch RMSNorm (baseline) | 65,484 | 22.47% |
| Triton RMSNorm | 61,172 | 21.00% |

The Triton kernel is **~6.6% slower**, not faster, and that's reported honestly rather
than hidden. Reason: `torch.compile` already lowers the plain-PyTorch RMSNorm into a
fused, autotuned Triton kernel automatically as part of compiling the surrounding graph.
The hand-written kernel is wrapped in a `torch.autograd.Function`, which forces a
`torch.compile` graph-break around it, adding dispatch overhead the compiler's own
fusion doesn't pay. It's a real, useful finding about when custom kernels are (and
aren't) worth writing, not a result to paper over. Full write-up:
`results/triton_benchmark.md`.

**DDP / multi-GPU scaling**: deferred. `train.py` is DDP-ready (`torchrun` +
`DistributedDataParallel`, already used for the interpretability/calibration phases'
single-GPU runs without any DDP-path changes needed), but a global GCP GPU-quota
increase (1 to 2) was denied, blocking any multi-GPU run regardless of the regional
quota (3). See `HANDOFF.md` for the quota gotcha and what's next.

## Try it

The trained model and tokenizer are live on the Hugging Face Hub:
[hari-8/transformer-pretraining-from-scratch](https://huggingface.co/hari-8/transformer-pretraining-from-scratch).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "hari-8/transformer-pretraining-from-scratch"
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(repo)

ids = tokenizer("The history of the Roman Empire", return_tensors="pt").input_ids
out = model.generate_simple(ids, max_new_tokens=40, temperature=0.8, top_k=40)
print(tokenizer.decode(out[0].tolist()))
```

`trust_remote_code=True` is required: `modeling_decoder_transformer.py` (the
from-scratch architecture) ships in the model repo itself, not as a stock
`AutoModel` class.

**Live demo**: [hari-8/transformer-pretraining-demo](https://huggingface.co/spaces/hari-8/transformer-pretraining-demo)
runs the model entirely in the browser via ONNX Runtime Web (`hf_static_demo/`,
`scripts/export_onnx.py`), sidestepping Hugging Face's requirement of a PRO
subscription to host Gradio/Docker Spaces on free `cpu-basic` hardware, since
a Static Space has no server compute at all. The Gradio version
(`hf_space/`) is also fully built and verified, kept dormant rather than
deployed, in case a PRO subscription is ever worth it later.

## Limitations

Written once training and evaluation are complete, in the author's own words, not
generated. This section matters more than it looks. See `PROJECT_PLAN.md`.
