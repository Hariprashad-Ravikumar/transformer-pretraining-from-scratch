# Learning Log: Pretraining a Transformer From Scratch on GCP

This is a walkthrough of everything built and run so far, written so it can be
read end to end and reproduced by hand — not just a changelog. Each section
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
like this. The alert doesn't stop spending — it emails at 50/90/100% of the
$50 threshold so a mistake gets noticed in minutes, not the next time the
billing page happens to get opened.

### 1.2 GPU quota — the part that actually went wrong

New GCP projects start with **zero GPU quota**. There isn't one quota to raise —
there are at least two, and they're independent:

1. A **per-region, per-GPU-type quota** — e.g. `NVIDIA_L4_GPUS` in `us-central1`.
2. A **global, all-regions, all-GPU-types quota** — `GPUS_ALL_REGIONS`. This is
   an account-wide cap that sits *underneath* every regional quota — raising the
   regional one does nothing if this one is still 0.

This project's account already had regional L4 quota of 1 (probably a default
grant), which made the first VM-creation attempt fail on the *global* quota
instead, with an error that doesn't obviously explain the two-tier system:

```
ERROR: Quota 'GPUS_ALL_REGIONS' exceeded. Limit: 0.0 globally.
```

Filing the increase used the `gcloud alpha quotas` command group (needed
`gcloud components install alpha` first, since it's not in the default install).
Finding the exact quota ID mattered — the human-readable name in the console
("GPUs (all regions)") isn't the ID the CLI wants:

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
appears to have an implicit ceiling on fresh accounts — asking for less can
succeed where asking for more fails outright, with no queue or wait either way.
If you hit a denial, don't assume you have to wait a day; try a smaller number
first. (Multi-GPU work later in this project will need to raise this again,
now from a account that has *some* history, which should go easier.)

Check what actually got granted (not what you asked for) with:

```bash
gcloud alpha quotas preferences list --project=cost-aware-multi-agent-llm \
  --format="table(quotaId,quotaConfig.grantedValue,quotaConfig.preferredValue)"
```

Or in the console: **IAM & Admin → Quotas & System Limits → Increase Requests**
tab — this shows Denied/Partially approved/Approved per request, which the CLI's
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
  on-demand but can be reclaimed by GCP at any time with ~30 seconds notice.
  This is *why* the training loop has checkpoint/resume — it's not a nice-to-have,
  it's the thing that makes Spot pricing usable at all for anything that runs
  longer than a few minutes.
- **`--instance-termination-action=STOP`** (not delete): on preemption, the VM
  stops but the boot disk survives. `gcloud compute instances start` brings it
  back with everything still on disk — no re-cloning, no re-installing.
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
before production — cheap, fast checks first, expensive ones last, each one
only run once the cheaper one already passed.

---

## 3. The model — `src/model/transformer.py`

This is a decoder-only transformer (the same family as GPT), authored as raw
`nn.Module` subclasses. No `AutoModel.from_pretrained`, no `AutoConfig` — every
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
only rescales — it skips the mean-centering step and the bias term entirely.
The empirical finding (from the LLaMA line of models) is that the
mean-centering step doesn't actually help stability much; dropping it saves
compute and parameters for basically free. This is a small, defensible,
modern deviation from the "vanilla GPT-2" architecture — the kind of detail
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
encodings in the original "Attention is All You Need" paper — but instead of
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
materializes the full N×N attention matrix in memory — computing attention the
naive way (`Q @ K.T`, mask, softmax, `@ V` as four separate ops) uses O(N²)
memory and is markedly slower. Using this call is a legitimate architecture
decision (it's *how* attention gets computed), not a shortcut around writing
the model — it's the standard way every serious training codebase does it.

### 3.4 SwiGLU instead of a plain ReLU MLP

```python
class SwiGLU(nn.Module):
    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))
```

GPT-2's MLP block is `Linear -> GELU -> Linear`. SwiGLU instead computes two
parallel linear projections and multiplies one (passed through SiLU) against
the other elementwise, before a final projection back down. It's a "gated"
activation — one branch acts as a learned gate controlling how much of the
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
intuition: both matrices are really doing the same job — a lookup table
between token identities and their vector representation — so sharing it saves
a meaningful fraction of total parameters (the embedding matrix is often the
single largest parameter tensor in a small model) with little to no quality
cost. This is standard since the original GPT-2 paper.

### 3.6 Verifying it's actually correct: causal masking test

Code that "runs without erroring" is not the same as code that's *correct*.
The test that actually matters here checks that a token's output never changes
based on tokens that come after it — i.e., the causal mask genuinely prevents
future information from leaking backward:

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
the causal mask is broken — a bug that a shape-only test (`logits.shape == (...)`)
would never catch, since the output would still have the right shape, just
the wrong values.

---

## 4. The tokenizer — why train your own, and what it costs you

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
to a baseline model that uses a different tokenizer** — perplexity is a
per-token metric, and "per token" means something different when the two
models don't agree on what a token is. The fix used here is to report
**bits-per-byte (BPB)** as the primary comparison metric instead of raw
perplexity: BPB normalizes by the number of *bytes* of underlying text rather
than the number of tokens, which makes it comparable across different
tokenizers. (Perplexity is still reported alongside it, just labeled as
"not directly comparable to the baseline.")

### 4.3 A bug, found by actually running it

```python
tokenizer.save_model(out_dir)   # -> Exception: No such file or directory
```

`ByteLevelBPETokenizer.save_model()` does not create its output directory —
it assumes the directory already exists and fails otherwise. The exact same
bug independently showed up in the checkpoint-saving code
(`torch.save(..., path)` also assumes the parent directory exists). Both were
one-line fixes (`os.makedirs(dir, exist_ok=True)` before the save call), but
neither would have been caught by reading the code carefully — they only
surfaced by actually running the scripts against a fresh checkout. This is the
same lesson the router project's case study already learned the hard way with
Cloud Run memory limits: some bugs are invisible until code runs in the real
environment, not the one you developed it in.

---

## 5. The data pipeline

### 5.1 Two separate scripts, on purpose

- `download_raw_sample.py` — pulls ~300MB of raw text, saved as plain `.txt`
  shards. Used *only* to train the tokenizer.
- `prepare_data.py` — streams the full ~1B-token target from the same dataset,
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
full dataset to disk up front — it yields documents one at a time as they're
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
the difference between an 8GB file and a 2GB file — genuinely material when
data has to be read repeatedly off disk during training.

### 5.4 Reading it back — memory-mapped, not loaded into RAM

```python
class MemmapTokenDataset:
    def __init__(self, bin_path: str, block_size: int):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")

    def get_batch(self, batch_size, device):
        ix = np.random.randint(0, len(self), size=batch_size)
        x = np.stack([self.data[i:i+self.block_size].astype(np.int64) for i in ix])
        ...
```

`np.memmap` opens the file without reading it into memory — the OS pages in
only the byte ranges actually touched, on demand. For a training run that
picks random windows into a multi-GB file millions of times, this means the
process's memory footprint stays tiny (just the batch itself) regardless of
how large the total dataset file is.

---

## 6. The training loop — `src/train/train.py`

### 6.1 The pieces, and why each one is there

**bf16 mixed precision** (`torch.autocast(dtype=torch.bfloat16)`): runs most of
the forward/backward pass in 16-bit floats instead of 32-bit, roughly doubling
throughput and halving memory on GPUs that support it (the L4 does). `bf16`
specifically (as opposed to `fp16`) keeps the same exponent range as `fp32`, so
it doesn't need the loss-scaling tricks `fp16` requires to avoid underflow —
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
once. `base.yaml` uses `micro_batch_size: 32` × `grad_accum_steps: 4` for an
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
the end of training — standard practice, used by GPT-3, LLaMA, and effectively
every modern pretraining run.

**Weight decay applied selectively**:

```python
decay, no_decay = [], []
for n, p in self.named_parameters():
    (decay if p.dim() >= 2 else no_decay).append(p)
```

Weight decay (L2-style regularization pulling weights toward zero) is applied
to 2D+ weight matrices (the actual linear layers) but *not* to 1D parameters —
biases and RMSNorm's scale weights. Regularizing a normalization layer's scale
doesn't make sense the same way it does for a weight matrix, and in practice
excluding 1D params from decay is standard (again, GPT-2/LLaMA convention).

### 6.2 Checkpoint/resume — the part that makes Spot VMs viable

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
  a visible bump — the optimizer effectively forgets its momentum. Saving both
  makes resume genuinely seamless, not just "doesn't crash."
- **`model.module if isinstance(model, DDP) else model`**: when training is
  wrapped in `DistributedDataParallel` for multi-GPU, the actual model lives
  one level down at `model.module`. Saving `model.state_dict()` directly (DDP
  wrapper and all) would embed `module.` prefixes into every parameter name,
  breaking a later single-GPU load. Unwrapping here means the same checkpoint
  file loads cleanly whether it's resumed on 1 GPU or 4.

This was tested directly, not just written and assumed correct: train the tiny
config to completion, then run the identical command again. The second run's
log shows exactly what should happen —

```
resumed from checkpoints/tiny.pt at step 20
training complete, final checkpoint saved
```

— it picked up at step 20 (where the prior run left off) and correctly
recognized `max_steps: 20` was already satisfied, rather than either
re-training from scratch or crashing.

---

## 7. The verification ladder, and what each rung actually caught

| Step | What it checks | What it can't catch |
|---|---|---|
| `pytest tests/` | Model math is correct (shapes, weight tying, causal masking) | Anything about training dynamics or I/O |
| `tiny.yaml` on CPU | The full pipeline runs end to end: data → model → optimizer → checkpoint → resume | GPU-specific bugs (`torch.compile`, CUDA memory, autocast dtype support) |
| `gpu_smoke.yaml` on the L4 | `torch.compile` + bf16 actually work on real hardware, at real (if tiny) scale | Whether the *real* corpus/tokenizer/config combination behaves — still synthetic random-token data |
| `base.yaml`, real data | The actual training run | — this is the real thing, nothing left to catch cheaply after this |

Each rung is deliberately cheaper than the next, and each one was actually run,
not just written — the two directory-creation bugs above were caught at the
`tiny.yaml`/tokenizer-training rungs specifically *because* those steps were
run for real rather than assumed to work.

---

## 8. What's next (not done yet as of this log)

- Launch the real `base.yaml` training run on the full ~1B-token corpus.
- Evaluate: held-out loss, perplexity, and bits-per-byte against a named
  baseline (baseline model not yet chosen — needs to be picked once the actual
  corpus domain is confirmed, so BPB comparison is apples-to-apples).
- Scale to 2 and 4 GPUs with `torchrun`, measure throughput (tokens/sec) and
  MFU (model FLOPs utilization — measured achieved FLOPs divided by the GPU's
  theoretical peak, the standard way to report "how efficiently is this
  actually using the hardware").
- Triton kernel: one fused op on the hot path, correctness-tested against the
  plain PyTorch version, benchmarked before/after.
- Calibration study (ECE, Brier score, reliability diagrams) on token-level
  confidence — same methodology already used in the router project, applied
  here to a model trained from scratch instead of a pretrained one called via
  API.
