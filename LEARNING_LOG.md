# Learning Log: Pretraining a Transformer From Scratch on GCP

This is a walkthrough of everything built and run so far, written so it can be
read end to end and reproduced by hand, not just a changelog. Each section
explains *why* a choice was made, not just what command was run, since "I called
`.encode()` on a pretrained model" and "I trained this" are different claims and
the difference only survives an interview if you can explain the mechanism.

Companion piece: [`PROJECT_PLAN.md`](../Cost-Aware-Multi-Agent-LLM-Router/PROJECT_PLAN.md)
in the router repo explains *why* this project exists at all.

---

## 1. The GCP setup

### 1.1 Billing, APIs, budget

Before touching compute, three things had to be true: billing enabled, the right
APIs turned on, and a budget alert live so a runaway job can't silently burn
money.

```bash
gcloud billing projects describe cost-aware-multi-agent-llm   # confirm billing is on
gcloud services enable billingbudgets.googleapis.com --project=cost-aware-multi-agent-llm
gcloud services enable compute.googleapis.com --project=cost-aware-multi-agent-llm

gcloud billing budgets create \
  --billing-account=015E9B-E3687C-B1B39E \
  --display-name="transformer-pretrain-50usd" \
  --budget-amount=50USD \
  --threshold-rule=percent=0.5 \
  --threshold-rule=percent=0.9 \
  --threshold-rule=percent=1.0 \
  --filter-projects=projects/cost-aware-multi-agent-llm
```

**Why a budget alert before a VM exists, not after:** a Spot GPU VM left running
by accident is the single easiest way to blow through cloud credit on a project
like this. The alert doesn't stop spending. It emails at 50/90/100% of the
$50 threshold so a mistake gets noticed in minutes, not the next time the
billing page happens to get opened.

### 1.2 GPU quota, the part that actually went wrong

New GCP projects start with **zero GPU quota**. There isn't one quota to raise.
There are at least two, and they're independent:

1. A **per-region, per-GPU-type quota**, e.g. `NVIDIA_L4_GPUS` in `us-central1`.
2. A **global, all-regions, all-GPU-types quota**, `GPUS_ALL_REGIONS`. This is
   an account-wide cap that sits *underneath* every regional quota: raising the
   regional one does nothing if this one is still 0.

This project's account already had regional L4 quota of 1 (probably a default
grant), which made the first VM-creation attempt fail on the *global* quota
instead, with an error that doesn't obviously explain the two-tier system:

```
ERROR: Quota 'GPUS_ALL_REGIONS' exceeded. Limit: 0.0 globally.
```

Filing the increase used the `gcloud alpha quotas` command group (needed
`gcloud components install alpha` first, since it's not in the default install).
Finding the exact quota ID mattered: the human-readable name in the console
("GPUs (all regions)") isn't the ID the CLI wants.

```bash
gcloud alpha quotas info list --service=compute.googleapis.com \
  --project=cost-aware-multi-agent-llm --format="value(name)" | grep -i L4
# -> .../quotaInfos/PREEMPTIBLE-NVIDIA-L4-GPUS-per-project-region  (this is the id, dashes not underscores)

gcloud alpha quotas preferences create \
  --service=compute.googleapis.com \
  --quota-id=GPUS-ALL-REGIONS-per-project \
  --preferred-value=4 \
  --project=cost-aware-multi-agent-llm \
  --justification="..." --email=you@example.com
```

**The interesting finding:** requesting `--preferred-value=4` for the global quota
was **denied instantly**. Requesting `--preferred-value=1` on the same quota a
few minutes later was **auto-approved instantly**. GCP's automated quota approval
appears to have an implicit ceiling on fresh accounts: asking for less can
succeed where asking for more fails outright, with no queue or wait either way.
If you hit a denial, don't assume you have to wait a day; try a smaller number
first. (Multi-GPU work later in this project will need to raise this again,
now from an account that has *some* history, which should go easier.)

Check what actually got granted (not what you asked for) with:

```bash
gcloud alpha quotas preferences list --project=cost-aware-multi-agent-llm \
  --format="table(quotaId,quotaConfig.grantedValue,quotaConfig.preferredValue)"
```

Or in the console: **IAM & Admin -> Quotas & System Limits -> Increase Requests**
tab. This shows Denied/Partially approved/Approved per request, which the CLI's
`reconciling: true` field doesn't surface clearly.

### 1.3 The VM itself

```bash
gcloud compute instances create pretrain-l4-spot \
  --zone=us-central1-a \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=150GB --boot-disk-type=pd-ssd
```

Three choices worth understanding, not just copying:

- **`--provisioning-model=SPOT`**: Spot VMs cost roughly 60-90% less than
  on-demand but can be reclaimed by GCP at any time with about 30 seconds notice.
  This is *why* the training loop has checkpoint/resume. It's not a nice-to-have,
  it's the thing that makes Spot pricing usable at all for anything that runs
  longer than a few minutes.
- **`--instance-termination-action=STOP`** (not delete): on preemption, the VM
  stops but the boot disk survives. `gcloud compute instances start` brings it
  back with everything still on disk, no re-cloning, no re-installing.
- **`--image-family=pytorch-2-9-cu129-...`**: a Google-maintained "Deep Learning
  VM" image with CUDA drivers and PyTorch already installed. Installing NVIDIA
  drivers from scratch is its own multi-step yak-shave; picking the right
  pre-built image skipped a solid chunk of Wednesday-night setup time.

**Cost hygiene**: stop the VM (`gcloud compute instances stop pretrain-l4-spot`)
between work sessions. Spot billing is per-second while running; a stopped VM
only costs (tiny) disk storage, not compute.

---

## 2. The repo layout

```
transformer-pretraining-from-scratch/
├── src/
│   ├── model/transformer.py      # the actual nn.Module, no AutoModel
│   ├── tokenizer/train_tokenizer.py
│   ├── data/
│   │   ├── download_raw_sample.py   # small sample, for training the tokenizer
│   │   ├── prepare_data.py          # full corpus, tokenized + sharded to .bin
│   │   └── dataset.py               # memory-mapped reader for training
│   └── train/train.py            # the training loop
├── configs/
│   ├── tiny.yaml        # CPU-testable, synthetic data, seconds to run
│   ├── gpu_smoke.yaml    # real GPU, real torch.compile path, still synthetic data
│   └── base.yaml         # the real ~57M param / ~1B token run
├── tests/test_model.py
└── scripts/make_tiny_data.py
```

The split between `tiny.yaml` and `gpu_smoke.yaml` and `base.yaml` is deliberate:
each one isolates a different class of bug before spending real time or money.
`tiny.yaml` catches Python-level bugs (shapes, missing directories, checkpoint
logic) on a laptop in under a second. `gpu_smoke.yaml` catches CUDA/`torch.compile`
issues on real hardware but still with fake data, so a bad training run can't
waste an hour before you find out something's broken. Only `base.yaml` spends
real GPU-hours on real data. This is the same idea as a staging environment
before production: cheap, fast checks first, expensive ones last, each one
only run once the cheaper one already passed.

---

## 3. The model, `src/model/transformer.py`

This is a decoder-only transformer (the same family as GPT), authored as raw
`nn.Module` subclasses. No `AutoModel.from_pretrained`, no `AutoConfig`. Every
layer below is written out so the "why" of each piece can actually be defended.

### 3.1 RMSNorm instead of LayerNorm

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight
```

Original GPT-2 uses LayerNorm, which centers *and* rescales activations
(subtracts the mean, divides by the std, learns a scale and a bias). RMSNorm
only rescales: it skips the mean-centering step and the bias term entirely.
The empirical finding (from the LLaMA line of models) is that the
mean-centering step doesn't actually help stability much; dropping it saves
compute and parameters for basically free. This is a small, defensible,
modern deviation from the "vanilla GPT-2" architecture, the kind of detail
worth knowing you *chose*, not just inherited.

### 3.2 Rotary position embeddings (RoPE) instead of learned position embeddings

GPT-2 gives every position 0..N a learned embedding vector, added to the token
embedding. RoPE instead **rotates** the query and key vectors by an angle that
depends on their position, before the attention dot product:

```python
def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)
```

The mechanical intuition: pair up each head's dimensions as 2D coordinates and
rotate each pair by an angle proportional to the token's position times a
frequency. Different dimension-pairs rotate at different frequencies (like the
hands of a clock moving at different speeds), the same trick as the sinusoidal
encodings in the original "Attention is All You Need" paper. But instead of
*adding* a position signal to the embedding, RoPE bakes position into *how
query and key vectors align with each other* in the attention dot product. The
practical payoff: relative position (how far apart two tokens are) falls out
of the math naturally, and the model generalizes better to sequence lengths
longer than anything it was trained on. This is what LLaMA, Mistral, and most
2024+ open models use instead of learned position embeddings.

### 3.3 Causal self-attention via a fused kernel

```python
y = F.scaled_dot_product_attention(
    q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
)
```

`is_causal=True` is what makes this a *decoder*: token `i` can only attend to
tokens `0..i`, never to the future. Under the hood, PyTorch dispatches this
call to a fused kernel (FlashAttention on supported GPUs) that never
materializes the full N×N attention matrix in memory. Computing attention the
naive way (`Q @ K.T`, mask, softmax, `@ V` as four separate ops) uses O(N²)
memory and is markedly slower. Using this call is a legitimate architecture
decision (it's *how* attention gets computed), not a shortcut around writing
the model. It's the standard way every serious training codebase does it.

### 3.4 SwiGLU instead of a plain ReLU MLP

```python
class SwiGLU(nn.Module):
    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))
```

GPT-2's MLP block is `Linear -> GELU -> Linear`. SwiGLU instead computes two
parallel linear projections and multiplies one (passed through SiLU) against
the other elementwise, before a final projection back down. It's a "gated"
activation: one branch acts as a learned gate controlling how much of the
other branch passes through. Empirically this gives a small but consistent
quality improvement over GELU/ReLU MLPs for the same parameter budget, and is
what LLaMA/PaLM/Mistral all use.

### 3.5 Weight tying

```python
self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
self.head.weight = self.tok_emb.weight
```

The input token-embedding matrix (`vocab_size x n_embd`, converting a token id
into a vector) and the output head (`n_embd x vocab_size`, converting a vector
back into logits over the vocabulary) are given the *same* weight tensor. The
intuition: both matrices are really doing the same job, a lookup table
between token identities and their vector representation, so sharing it saves
a meaningful fraction of total parameters (the embedding matrix is often the
single largest parameter tensor in a small model) with little to no quality
cost. This is standard since the original GPT-2 paper.

### 3.6 Verifying it's actually correct: causal masking test

Code that "runs without erroring" is not the same as code that's *correct*.
The test that actually matters here checks that a token's output never changes
based on tokens that come after it: the causal mask genuinely prevents
future information from leaking backward.

```python
def test_causal_masking_no_future_leakage():
    model.eval()
    x = torch.randint(0, 100, (1, 8))
    logits_full, _ = model(x)
    logits_trunc, _ = model(x[:, :4])
    assert torch.allclose(logits_full[:, :4], logits_trunc, atol=1e-4)
```

If you truncate the input to its first 4 tokens, the model's output logits for
those first 4 positions must be *identical* whether or not tokens 5-8 exist,
because a real decoder can't see the future. If this test failed, it would mean
the causal mask is broken, a bug that a shape-only test (`logits.shape == (...)`)
would never catch, since the output would still have the right shape, just
the wrong values.

---

## 4. The tokenizer: why train your own, and what it costs you

### 4.1 Training it

```python
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=files,
    vocab_size=16384,
    min_frequency=2,
    special_tokens=["<|endoftext|>"],
)
tokenizer.save_model(out_dir)
```

**Byte-level BPE** (Byte-Pair Encoding) starts from raw bytes (so it can
represent *any* text, no "unknown token" problem) and iteratively merges the
most frequent adjacent pair of tokens into a new token, repeating until it
hits the target vocabulary size. GPT-2 and GPT-3 both use this exact scheme.
Training it here (on 300M characters of the actual training corpus) rather
than downloading GPT-2's pretrained tokenizer means the vocabulary is fit to
*this* corpus's actual word/subword frequency distribution, not someone else's.

### 4.2 The tradeoff this creates for evaluation

A trained tokenizer's vocabulary is specific to the text it was trained on.
That means **perplexity numbers from this model are not directly comparable
to a baseline model that uses a different tokenizer**: perplexity is a
per-token metric, and "per token" means something different when the two
models don't agree on what a token is. The fix used here is to report
**bits-per-byte (BPB)** as the primary comparison metric instead of raw
perplexity. BPB normalizes by the number of *bytes* of underlying text rather
than the number of tokens, which makes it comparable across different
tokenizers. (Perplexity is still reported alongside it, just labeled as
"not directly comparable to the baseline.")

### 4.3 A bug, found by actually running it

```python
tokenizer.save_model(out_dir)   # -> Exception: No such file or directory
```

`ByteLevelBPETokenizer.save_model()` does not create its output directory. It
assumes the directory already exists and fails otherwise. The exact same
bug independently showed up in the checkpoint-saving code
(`torch.save(..., path)` also assumes the parent directory exists). Both were
one-line fixes (`os.makedirs(dir, exist_ok=True)` before the save call), but
neither would have been caught by reading the code carefully. They only
surfaced by actually running the scripts against a fresh checkout. This is the
same lesson the router project's case study already learned the hard way with
Cloud Run memory limits: some bugs are invisible until code runs in the real
environment, not the one you developed it in.

---

## 5. The data pipeline

### 5.1 Two separate scripts, on purpose

- `download_raw_sample.py`: pulls ~300MB of raw text, saved as plain `.txt`
  shards. Used *only* to train the tokenizer.
- `prepare_data.py`: streams the full ~1B-token target from the same dataset,
  tokenizes each document with the now-trained tokenizer, and writes the token
  ids straight to a binary file.

Splitting these matters because training the tokenizer doesn't need anywhere
near the full corpus (300M characters is already enough to see the real
frequency distribution of subwords), so there's no reason to pay the
download+tokenize cost for the full 1B tokens twice.

### 5.2 Streaming, not downloading-then-processing

```python
ds = load_dataset(args.dataset, name=args.subset, split="train", streaming=True)
for row in ds:
    ids = tok.encode(row["text"]).ids + [eot_id]
    dest.write(np.array(ids, dtype=np.uint16).tobytes())
```

`streaming=True` means Hugging Face's `datasets` library never downloads the
full dataset to disk up front. It yields documents one at a time as they're
pulled over the network, tokenize-and-discard, tokenize-and-discard. For a
dataset that's many times larger than the ~1B tokens actually needed
(FineWeb-Edu's full sample is 10B+ tokens), this avoids downloading and storing
data that's never used.

### 5.3 uint16, not int64 or int32

```python
arr = np.array(ids, dtype=np.uint16)
```

Token ids only need to represent values 0 to `vocab_size - 1` (here, 0-16383).
`uint16` covers 0-65535, comfortably enough, at a quarter the size of the
`int64` PyTorch would otherwise default to. For a billion-token dataset that's
the difference between an 8GB file and a 2GB file, genuinely material when
data has to be read repeatedly off disk during training.

### 5.4 Reading it back: memory-mapped, not loaded into RAM

```python
class MemmapTokenDataset:
    def __init__(self, bin_path: str, block_size: int):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")

    def get_batch(self, batch_size, device):
        ix = np.random.randint(0, len(self), size=batch_size)
        x = np.stack([self.data[i:i+self.block_size].astype(np.int64) for i in ix])
        ...
```

`np.memmap` opens the file without reading it into memory: the OS pages in
only the byte ranges actually touched, on demand. For a training run that
picks random windows into a multi-GB file millions of times, this means the
process's memory footprint stays tiny (just the batch itself) regardless of
how large the total dataset file is.

---

## 6. The training loop, `src/train/train.py`

### 6.1 The pieces, and why each one is there

**bf16 mixed precision** (`torch.autocast(dtype=torch.bfloat16)`): runs most of
the forward/backward pass in 16-bit floats instead of 32-bit, roughly doubling
throughput and halving memory on GPUs that support it (the L4 does). `bf16`
specifically (as opposed to `fp16`) keeps the same exponent range as `fp32`, so
it doesn't need the loss-scaling tricks `fp16` requires to avoid underflow:
simpler code, one less thing to get subtly wrong.

**`torch.compile`**: JIT-compiles the model's forward pass into fused GPU
kernels the first time it runs, trading a slower first step for faster
subsequent ones. Left off in `tiny.yaml` (CPU compilation is slow and pointless
for a 20-step debug run) and on in `gpu_smoke.yaml`/`base.yaml`.

**Gradient accumulation**:

```python
for micro_step in range(grad_accum):
    x, y = train_ds.get_batch(micro_bs, device)
    loss = model(x, y)[1] / grad_accum
    loss.backward()
optimizer.step()
```

Runs several small "micro-batches" and sums their gradients before taking one
optimizer step, simulating a larger batch size than would fit in GPU memory at
once. `base.yaml` uses `micro_batch_size: 32` x `grad_accum_steps: 4` for an
effective batch of 128 sequences per optimizer step.

**Cosine learning-rate schedule with warmup**:

```python
def get_lr(step, cfg):
    if step < warmup: return lr * (step + 1) / warmup
    decay_ratio = (step - warmup) / (max_steps - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)
```

Warmup (linearly ramping the learning rate up from ~0 over the first couple
hundred steps) avoids the large, unstable gradient updates that can happen at
initialization when the optimizer's moment estimates haven't stabilized yet.
The cosine decay afterward smoothly reduces the learning rate to `min_lr` by
the end of training, standard practice, used by GPT-3, LLaMA, and effectively
every modern pretraining run.

**Weight decay applied selectively**:

```python
decay, no_decay = [], []
for n, p in self.named_parameters():
    (decay if p.dim() >= 2 else no_decay).append(p)
```

Weight decay (L2-style regularization pulling weights toward zero) is applied
to 2D+ weight matrices (the actual linear layers) but *not* to 1D parameters,
biases and RMSNorm's scale weights. Regularizing a normalization layer's scale
doesn't make sense the same way it does for a weight matrix, and in practice
excluding 1D params from decay is standard (again, GPT-2/LLaMA convention).

### 6.2 Checkpoint/resume: the part that makes Spot VMs viable

```python
def save_checkpoint(path, model, optimizer, step, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw_model = model.module if isinstance(model, DDP) else model
    torch.save({"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "config": cfg}, path)

def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["step"]
```

Two details worth noticing:

- **Both model *and* optimizer state are saved.** AdamW keeps a running
  per-parameter estimate of gradient mean and variance (its "moments"). If you
  only saved the model weights and resumed with a freshly initialized
  optimizer, training would technically continue but the loss curve would show
  a visible bump: the optimizer effectively forgets its momentum. Saving both
  makes resume genuinely seamless, not just "doesn't crash."
- **`model.module if isinstance(model, DDP) else model`**: when training is
  wrapped in `DistributedDataParallel` for multi-GPU, the actual model lives
  one level down at `model.module`. Saving `model.state_dict()` directly (DDP
  wrapper and all) would embed `module.` prefixes into every parameter name,
  breaking a later single-GPU load. Unwrapping here means the same checkpoint
  file loads cleanly whether it's resumed on 1 GPU or 4.

This was tested directly, not just written and assumed correct: train the tiny
config to completion, then run the identical command again. The second run's
log shows exactly what should happen:

```
resumed from checkpoints/tiny.pt at step 20
training complete, final checkpoint saved
```

It picked up at step 20 (where the prior run left off) and correctly
recognized `max_steps: 20` was already satisfied, rather than either
re-training from scratch or crashing.

---

## 7. The verification ladder, and what each rung actually caught

| Step | What it checks | What it can't catch |
|---|---|---|
| `pytest tests/` | Model math is correct (shapes, weight tying, causal masking) | Anything about training dynamics or I/O |
| `tiny.yaml` on CPU | The full pipeline runs end to end: data → model → optimizer → checkpoint → resume | GPU-specific bugs (`torch.compile`, CUDA memory, autocast dtype support) |
| `gpu_smoke.yaml` on the L4 | `torch.compile` + bf16 actually work on real hardware, at real (if tiny) scale | Whether the *real* corpus/tokenizer/config combination behaves, still synthetic random-token data |
| `base.yaml`, real data | The actual training run | This is the real thing, nothing left to catch cheaply after this |

Each rung is deliberately cheaper than the next, and each one was actually run,
not just written. The two directory-creation bugs above were caught at the
`tiny.yaml`/tokenizer-training rungs specifically *because* those steps were
run for real rather than assumed to work.

---

## 8. The real training run, and what evaluation found

The real `base.yaml` run completed: 7,630 steps on the full ~1B-token FineWeb-Edu
sample on one L4, ~67,400 tok/s throughout, final training loss 3.16. No NaNs,
no divergence, no intervention needed. The loss curve was noisy step-to-step
(mini-batch variance at `micro_batch_size: 32`) but trended down smoothly across
the run, consistent with the warmup/cosine-decay schedule in `get_lr()`.

**Evaluation** (`src/eval/evaluate.py`, `src/eval/baseline_pythia.py`) ran on
497K held-out tokens the model never trained on:

- Held-out loss (3.19 nats) came in essentially identical to the last few
  logged training-loss values (3.1-3.3), the expected result for a model that
  saw each token roughly once (Chinchilla-ish ~18 tok/param ratio, not enough
  passes over the data to meaningfully overfit at this scale).
- Bits-per-byte, not perplexity, is the comparable metric across models with
  different tokenizers. Perplexity is defined per-token, and our custom BPE's
  tokens aren't the same unit as Pythia's. BPB normalizes by the byte length of
  the *original text*, which is tokenizer-invariant. The methodology: decode
  `val.bin` back to its original text (byte-level BPE decode is lossless, so
  this reconstructs the exact held-out text with no information loss), feed
  that identical text through each model's own tokenizer, and divide total nats
  by `ln(2) x text byte length` for both.
- Result: this model scored 1.057 BPB vs. Pythia-70M's 1.135 BPB on the same
  held-out text. The honest read isn't "beats a real research baseline." It's
  that this model trained exclusively on FineWeb-Edu, a narrower, more
  predictable educational-text distribution, while Pythia-70M trained on the
  much broader Pile. Domain match, not general capability, is the more likely
  explanation, and the two are close in parameter count. Worth stating plainly
  in any writeup rather than letting the number imply more than it shows.

Full numbers are in `README.md`'s Results section and
`results/eval_summary.json` / `results/baseline_pythia.json`.

## 9. Calibration study, and a genuinely surprising result

`src/eval/calibration.py` measures token-level calibration: for each held-out
token, confidence = the model's max softmax probability, label = whether that
top-1 prediction matches the actual next token (Guo et al. 2017's definition,
structurally identical to the router project's `(predicted_probability,
is_correct)` pairs, so the same ECE/Brier code applies to both;
`src/eval/calibration_stats.py` ports the router's `stats_utils.py` formulas
verbatim rather than reimplementing them slightly differently).

**Split methodology, stronger than the router's own precedent**: `val.bin` was
split down the middle. The first half (subsampled to 20,480 tokens, since one
scalar parameter doesn't need hundreds of thousands of examples to fit) fit a
temperature scalar via LBFGS on cached logits; the second, disjoint half (never
touched during fitting) is what ECE/Brier are measured on. The router's own
report explicitly flags its validation split as "a sanity check, not the real
generalization test." This setup avoids that caveat entirely.

**Result**: raw ECE = 0.0071, Brier = 0.1572, on ~249K held-out tokens. Fitted
temperature T = 1.014, essentially 1, meaning temperature scaling barely
moved anything (scaled ECE = 0.0047, a small further improvement, not a fix
for a real problem). The reliability diagram (`results/calibration_curve.png`)
sits almost exactly on the diagonal at every confidence bin.

This is the kind of result worth resisting the urge to spin up: it isn't a
weak or unfinished study, it's an honest negative-ish finding. The model
turned out to already be well-calibrated, so there wasn't much for temperature
scaling to fix. The likely reason: next-token prediction trained end-to-end
with softmax cross-entropy on a large corpus directly targets the true
conditional distribution over the vocabulary, unlike a classifier trained on
one-hot labels (the setting Guo et al. 2017's headline "modern neural networks
are overconfident" result comes from) which only has to get the *argmax*
right, and can freely overshoot confidence elsewhere without being penalized
at training time. A from-scratch, from-first-principles calibration
measurement landing near-diagonal is itself evidence the training objective
did what it's supposed to do. That's the honest way to frame it, not
"nothing interesting happened here."

**Cross-project comparison, with the confound stated up front**: the router's
task-level calibrator (is a routed LLM response *correct*, a judged semantic
notion, predicted via a trained logistic regression on hand-engineered
features) scores 0.1671 ECE / 0.1521 Brier, worse-looking numbers, but not a
fair "our model calibrates better" claim. Different task, different label
distribution, different downstream stakes, and a very different amount of
engineering behind the confidence estimate. What's comparable is the
calibration *behavior*: whether raw confidence over/under-shoots accuracy,
and whether a standard post-hoc fix narrows the gap, not the raw numbers
side by side. Full write-up: `results/calibration_report.md`.

## 10. Interpretability: instrumenting the model without breaking training

`F.scaled_dot_product_attention` (the fused kernel `CausalSelfAttention` uses for speed)
never exposes attention weights, and gives no way to zero one head's contribution before
the output projection, both needed for interpretability, neither needed for training or
eval. Rather than write a separate, duplicated forward pass for probing (which risks
drifting out of sync with the real model), `CausalSelfAttention`/`Block`/
`DecoderTransformer.forward` got optional `return_attn`/`ablate` kwargs that switch to a
manual (non-fused) attention computation *only* when set. The fast fused path is
untouched for every existing caller. This did change `forward`'s return signature from
`(logits, loss)` to `(logits, loss, attn_weights)` everywhere (attn_weights is `None`
unless requested), so every existing caller (`train.py`, `evaluate.py`, `calibration.py`,
`tests/test_model.py`) needed a one-line update to unpack three values instead of two.
A new test (`test_manual_attention_matches_fused`) checks the manual path reproduces the
fused path's output exactly when no ablation is applied. This is the safety net that
would have caught a bug in the manual reimplementation before it silently corrupted every
downstream interpretability number.

**Induction-head probe** (`src/interpretability/induction_heads.py`, Olsson et al. 2022
methodology): feed sequences of random tokens repeated twice, measure how much attention
weight the second occurrence of a token puts on the position that followed that token's
*first* occurrence. Sanity-checked on a random-init model first: scores came back flat
and near chance level (~0.043 vs. an expected ~0.06 for seq_len=16), confirming the metric
itself isn't spuriously elevated for an untrained model before trusting it on real
checkpoints.

**Result**: `base.pt` has two unmistakable induction heads (layer 6 head 8, score 0.79;
layer 6 head 4, score 0.74, both far above every other head), concentrated specifically
in layers 6-7 of 8. `interp_small.pt`'s strongest score is 0.03, an order of magnitude
weaker, consistent with induction heads being a fairly sharp phase-change phenomenon
during training that a shorter, smaller run has only begun to develop. Caveat worth
repeating: `interp_small.pt` differs from `base.pt` in both architecture *and* training
length simultaneously (a property of the config design decided before this phase, not
something fixable after the fact), so this doesn't cleanly isolate scale from training
length as the cause.

**Causal ablation** (`src/interpretability/ablation.py`): full sweep, every head in every
layer on both checkpoints (96 + 16 ablations), held-out loss delta when each is zeroed
(bounded to a 16,384-token subset of `val.bin`, since a full sweep over the full
497K-token set wasn't compute-sensible, and one forward pass per ablation calibrated at
~1s per pass on this scale, confirmed by timing a small run before committing to the full
sweep).

This is the finding that turns the induction-head numbers into an actual causal claim
rather than a picture: **the highest-scoring induction heads are not reliably the most
important by ablation.** On `base.pt`, layer 6 heads 8 and 4 (the two clear induction
heads) rank only 8th and 11th of 12 heads in their own layer by held-out loss delta when
ablated. The heads that matter *most* causally (layer 5 head 4, delta +0.074; layer 0 head
2, delta +0.063) show no elevated induction score at all. On `interp_small.pt` the two
measures agree better (the top induction-score head is also the 2nd-most-important by
ablation) but still not perfectly. The likely reason: the synthetic probe targets *exact*
token repeats, a narrow pattern rare in natural held-out text. A head highly specialized
for that pattern may matter less for real-text loss than a head doing more diffuse work
the synthetic probe was never designed to detect. This is a genuine limitation of a
purely attention-pattern-based probe, worth stating as a finding rather than smoothing
over to make the story cleaner.

**Residual-stream norm growth** (`src/interpretability/residual_norms.py`, external
`register_forward_hook`s on each `Block`, no `transformer.py` changes needed for this
one): monotonic growth with depth in both checkpoints (`base.pt`: 12.7 → 88.6 nats-scale
norm across 8 layers, roughly 7x; `interp_small.pt`: 1.2 → 3.6 across 4 layers, roughly
3x). This is the expected signature of a pre-norm transformer, where each block adds to
the residual stream without renormalizing it (RMSNorm only rescales what's *read* from
the stream at the start of a block, not the stream itself), so growth compounds with
depth.

Full write-up, all six JSON result files, and six plots: `results/interpretability_report.md`.

## 12. Triton RMSNorm: a real kernel that turned out to be slower

`F.scaled_dot_product_attention` was the only fused kernel already in the model; this
phase added a hand-written one for RMSNorm, the mean-square reduction plus rescale that
runs on every token, every layer. Implemented forward and backward directly in Triton
(`src/triton_kernels/rmsnorm.py`), wrapped in a `torch.autograd.Function` so it's a
drop-in replacement for the existing `RMSNorm` module (`ModelConfig.use_triton_rmsnorm`,
default `False`, so no existing checkpoint or config changes behavior). The backward
derivation matters here specifically because RMSNorm's `rstd` term depends on *every*
element in the row (via the mean-square), not just the one being differentiated, so it's
not a simple elementwise gradient; see the docstring in that file for the full derivation
(`dL/dx_k = rstd*w_k*go_k - (rstd^3 * x_k / N) * sum_i(x_i*w_i*go_i)`).

Correctness (`tests/test_triton_kernels.py`, CUDA-only, skipped on this Mac, verified on
the VM): forward matches the PyTorch reference, gradients w.r.t. both `x` and `weight`
match via a direct comparison against PyTorch autograd, and a full model with the kernel
swapped in produces identical logits to the same model without it, given identical
weights. All three passed on the first real run; the gradient derivation above held up.

**Benchmark result, and why it matters more than a clean win would have**: at
`base.yaml`'s real scale (56.6M non-embedding params, 200 steps, `torch.compile` on),
the Triton kernel measured **~6.6% slower** than plain PyTorch RMSNorm (61,172 tok/s /
21.00% MFU vs. 65,484 tok/s / 22.47% MFU). This is the actual, honestly-reported result,
not a discarded failed attempt. The reason is instructive: `torch.compile`'s Inductor
backend already autotunes and fuses the plain-PyTorch RMSNorm into its own Triton kernel
as part of compiling the surrounding graph. The hand-written kernel isn't competing
against "slow uncompiled PyTorch," it's competing against the compiler's own generated
kernel. Worse, wrapping the custom kernel in a `torch.autograd.Function` forces
`torch.compile` to graph-break around it, so every call pays eager-mode dispatch overhead
right where the rest of the model flows through as one compiled region. The lesson: a
correct, hand-written kernel doesn't automatically beat a modern compiler's fused
baseline, and the *integration* cost (breaking the compiler's fusion boundary) can matter
more than the kernel's own execution time. Full write-up: `results/triton_benchmark.md`.

## 13. DDP scaling: blocked by quota, not by the code

`src/train/train.py` was already DDP-ready before this phase (`setup_ddp()`,
`DistributedDataParallel`, gradient-sync gating via `require_backward_grad_sync`).
Running it under `torchrun --nproc_per_node=N` needs no code changes. What's actually
blocking a multi-GPU run is GCP quota: the **global** `GPUS-ALL-REGIONS-per-project`
quota is granted 1, independently of the **regional** `PREEMPTIBLE-NVIDIA-L4-GPUS`
quota (granted 3), two separate caps, the global one sitting underneath the regional
one (documented in `HANDOFF.md`'s original quota investigation). A small increase
request (1 to 2, following the "ask small" lesson that worked for the original 0 to 1
grant) was submitted and denied outright this time. Worth noting that "ask small" isn't
a guaranteed pattern, just a better-odds one. Deferred rather than blocking the rest of
this phase; the moment quota allows it, the 2/4-GPU throughput and MFU comparison is
ready to run with no further engineering.

## 14. Hugging Face push: the conversion bug the verification step exists to catch

`HF_TOKEN` was set (via `~/.zshrc`, never pasted into a session directly per the original
plan). `scripts/push_model_to_hf.py` converts `checkpoints/base.pt` into the
`hf_export/` format, and deliberately verifies before ever calling `push_to_hub`: it
builds both the training-arch model and the HF wrapper from the *same* checkpoint state
dict, runs an identical input through both, and asserts the logits match exactly. This
caught a real issue on the first run: `load_state_dict(strict=True)` (the default)
failed on `rope_cos`/`rope_sin`, because those buffers are `persistent=False` in the
training-time model (not saved in the checkpoint) but `persistent=True` in the HF wrapper
(HANDOFF's already-documented bug #5, about `from_pretrained`'s meta-device fast-init path
never filling in non-persistent buffers). Fix: `strict=False` for that specific load, with
an explicit assertion that the *only* missing keys are those two rope buffers, narrow
enough that a genuinely wrong/incomplete state dict would still fail loudly rather than
silently loading with strict=False papering over a real problem.

After that fix, the full pipeline verified clean: converted-model logits matched the
training model exactly, the full `save_pretrained` → `AutoModelForCausalLM.from_pretrained
(trust_remote_code=True)` round-trip from disk loaded correctly, and a real generation
sample from the trained weights ("The history of the Roman Empire...") came back
grammatically coherent and topically on-target. This is the clearest evidence yet that
training actually produced a working language model, not just a checkpoint with a good
loss number.

Pushed to
[hari-8/transformer-pretraining-from-scratch](https://huggingface.co/hari-8/transformer-pretraining-from-scratch)
(model + tokenizer, public). The Gradio Space was not deployed: creating it hit a `402
Payment Required`. Hugging Face now requires a PRO subscription to host Gradio/Docker
Spaces on free `cpu-basic` hardware, which wasn't true when `hf_space/` was originally
built and tested against placeholder weights. Decided to skip paying for a recurring
subscription for a portfolio demo rather than silently working around it. `hf_space/` is
fully built and its `MODEL_REPO` default now points at the real pushed model, ready to
deploy immediately if that decision changes later.

## 15. What's next (not done yet as of this log)

- DDP scaling to 2/4 GPUs, once quota allows (see above, the code is ready, this is
  purely a GCP quota wait, not an engineering task).
- Gradio Space deployment, if HF PRO is ever worth it for this project.
