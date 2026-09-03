# Build It Yourself: A Beginner's Walkthrough

This document exists so you could rebuild this project from scratch, without
AI help, just by reading it. It assumes no prior deep learning background.
Every code snippet below is real code from this repo — nothing simplified away
or faked. Where a concept is genuinely subtle, there's a small worked example
by hand before the code, so the code isn't the first time you see the idea.

This covers five things, in the order they actually happened:

1. Teaching the computer to read text (the tokenizer)
2. Writing the neural network itself (the model)
3. Getting real training text (the data pipeline)
4. Actually training it (the training loop)
5. Two real bugs, as a debugging case study

---

## 1. Teaching the computer to read text — the tokenizer

### The problem

A neural network only understands numbers. It has no idea what a "word" is.
So step one, before any learning can happen, is converting text into a
sequence of numbers. The obvious idea — "just give every word a number" —
breaks immediately: English has hundreds of thousands of words, new words get
invented constantly, and typos/misspellings would each need their own number
forever. You need something that can represent *any* possible text, including
text it's never seen, using a small, fixed set of numbers.

### The idea: Byte-Pair Encoding (BPE), by hand first

Here's the actual algorithm, done on a tiny made-up example so you can trace
it yourself.

Start with a tiny "corpus" (training text): `"low low lower lowest"`.

**Step 0 — start at the character level.** Every word is just its individual
characters, plus a special "end of word" marker (call it `_`):

```
l o w _       (appears 2 times: "low low")
l o w e r _   (appears 1 time)
l o w e s t _ (appears 1 time)
```

**Step 1 — count every adjacent pair of symbols, find the most common one.**
Looking across all the words: `l`+`o` appears 4 times (once in each of the 4
word occurrences), `o`+`w` appears 4 times too. Say we pick `l`+`o` (ties are
broken by whichever the algorithm sees first). **Merge every occurrence** of
`l` followed by `o` into a single new symbol `lo`:

```
lo w _
lo w e r _
lo w e s t _
```

**Step 2 — repeat.** Now `lo`+`w` is the most common pair (appears in every
word). Merge it into `low`:

```
low _
low e r _
low e s t _
```

**Step 3 — repeat again.** Now `low`+`_` is most common (2 occurrences, from
"low low"). Merge into `low_`:

```
low_
low e r _
low e s t _
```

Keep repeating this — count pairs, merge the most frequent — for as many
rounds as you want (a real tokenizer does this until it reaches a target
vocabulary size, e.g. 16,384 symbols). Notice what's happening: **common whole
words become single symbols** (`low_` is now one token), while **rare or
unseen words stay broken into smaller pieces** that the model has still seen
before. `"lowly"` — a word never in the training text — would still tokenize
fine, as some combination of already-known pieces like `low` + `l` + `y`.
That's the whole point: no word is ever "unknown," because you can always fall
back to individual characters/bytes as the smallest possible unit.

This project uses the **byte-level** version specifically: the starting
symbols are raw bytes (0-255), not characters. This means literally any
text — any language, emoji, weird punctuation — can always be represented,
because every possible character is made of bytes.

### The real code

```python
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=files,               # your raw .txt files
    vocab_size=16384,          # stop merging once you have this many symbols
    min_frequency=2,           # ignore pairs that occur only once
    special_tokens=["<|endoftext|>"],  # a marker token, not a real byte sequence
)
tokenizer.save_model(out_dir)  # writes vocab.json + merges.txt
```

`vocab.json` ends up being the final list of ~16,384 symbols (some single
bytes, most short byte-sequences learned by merging). `merges.txt` is the
ordered list of merge rules — literally the steps from the worked example
above, just thousands of them, learned automatically from your actual corpus
instead of a toy 4-word example.

**Why train your own instead of downloading one:** a tokenizer trained on your
specific corpus has merges that reflect *that* corpus's actual vocabulary and
style. Downloading GPT-2's tokenizer would work fine too, but then you
haven't actually built the "read text" step yourself — you'd be reusing
someone else's work for a step that's genuinely learnable in an afternoon.

### Using it once trained

```python
tok = ByteLevelBPETokenizer("data/tokenizer/vocab.json", "data/tokenizer/merges.txt")
ids = tok.encode("The quick brown fox").ids   # -> [258, 288, 282, 284, ...] a list of integers
text = tok.decode(ids)                         # -> "The quick brown fox", exactly recovered
```

That's it — that's the entire "reading" step. From here on, the model never
sees text again, only lists of integers.

---

## 2. Writing the neural network itself

### The one-sentence job description

A decoder-only transformer's job is: **given a sequence of tokens so far,
predict a probability distribution over what token comes next.** That's
genuinely the whole task. Everything below is machinery in service of that one
prediction.

### Step 1: turn token numbers into vectors (the embedding)

A token id like `258` is just an arbitrary integer — it doesn't encode any
meaning by itself. The first thing the model does is look that integer up in a
big table and get back a vector (a list of, say, 768 numbers) instead:

```python
self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)  # a (16384 x 768) lookup table
x = self.tok_emb(idx)   # idx: token ids, shape (batch, sequence_length)
                         # x:   shape (batch, sequence_length, 768)
```

This table starts out as random numbers. Training is what teaches the model
to place similar tokens near each other in this 768-number space — that's the
"meaning" the model builds up, entirely through the training process, never
hand-specified.

### Step 2: attention — the mechanism that lets tokens "look at" each other

This is the actual novel idea behind transformers. Before a token can predict
what comes next, it needs to gather relevant context from everything that
came before it in the sequence. Attention is the mechanism for that.

**The intuition, in plain terms:** each token produces three vectors from
itself — a **query** ("what am I looking for?"), a **key** ("what do I
contain, that others might want?"), and a **value** ("what information do I
actually offer, if picked?"). To decide how much attention token A should pay
to an earlier token B, you compare A's query against B's key (literally a dot
product — a similarity score). Do that between A and every earlier token,
turn the scores into probabilities (softmax — they sum to 1), and then take a
weighted average of everyone's *value* vectors, weighted by those
probabilities. High-relevance tokens contribute more; irrelevant ones
contribute almost nothing.

```python
class CausalSelfAttention(nn.Module):
    def forward(self, x, cos, sin):
        B, T, C = x.shape                    # batch, sequence length, embedding size
        q, k, v = self.qkv(x).split(C, dim=2)  # one big matrix multiply, split 3 ways
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)   # see "position" note below
        k = apply_rope(k, cos, sin)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # under the hood, conceptually, this line is doing:
        #   scores = (q @ k.transpose(-2, -1)) / sqrt(head_dim)
        #   scores = mask_out_future_positions(scores)   # <- the "causal" part
        #   probs  = softmax(scores, dim=-1)
        #   y      = probs @ v
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))
```

**`is_causal=True` is the single most important line in the whole model.**
It's what makes this a valid *language model* rather than cheating: token
number 5 is mathematically prevented from seeing tokens 6, 7, 8... (the
future). If it could see the future, "predicting the next word" would be
trivial and useless — the model would just learn to copy the answer. The mask
enforces that every prediction is made using only information available at
that point in the sequence, which is the actual task.

**"Multi-head":** instead of doing this once with a big vector, the model
splits the vector into several smaller chunks (`n_head` of them) and does
attention independently on each chunk, then recombines. Each head is free to
specialize — one might learn to track "what's the subject of this sentence,"
another "what rhymes," etc. Nobody tells it to specialize this way; it falls
out of training.

**How the model knows position (RoPE):** attention as described above doesn't
know order at all — "the cat sat" and "sat cat the" would look identical to
it, since it's comparing every pair of tokens regardless of how far apart they
are. Rotary Position Embeddings fix this by rotating the query and key vectors
by an angle proportional to their position before comparing them — nearby
tokens end up rotated similarly (so they still align well), far-apart tokens
end up rotated very differently (so their similarity score naturally drops).
Position information gets baked into the comparison itself, not tacked on as
an extra input.

### Step 3: the "thinking" part (the MLP)

After attention gathers context from other tokens, each token individually
gets pushed through a small two-layer network that can do arbitrary
computation on what it just gathered:

```python
class SwiGLU(nn.Module):
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
        # w1, w2: two different projections of x, up to a bigger hidden size
        # silu(w1(x)): one branch, squashed through an activation function
        # * w2(x):     multiplied elementwise by the other branch (a learned "gate")
        # w3(...):     projected back down to the original size
```

Attention moves information *between* tokens (token 5 gathers from tokens
1-4). The MLP processes information *within* each token, independently. A
transformer block alternates these two: gather from others, then think about
what you gathered, repeated `n_layer` times (8 times in this project).

### Step 4: stacking it all together

```python
class Block(nn.Module):
    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)   # "+x" = residual connection
        x = x + self.mlp(self.ln2(x))
        return x
```

Two details that matter:

- **`self.ln1(x)` before attention, not after** — this is called "pre-norm."
  `ln1`/`ln2` are normalization layers (RMSNorm here) that rescale the
  numbers to keep them in a healthy range before each sub-layer processes
  them. Doing this *before* rather than after each sub-layer is what makes it
  practical to stack many layers without training becoming unstable.
- **`x = x + ...` (the residual connection)** — the output isn't just "the
  result of attention," it's "the input, plus whatever attention added to it."
  This means information has a direct, unobstructed path all the way from the
  first layer to the last — each layer only has to learn a small *adjustment*
  to add, not carry the entire signal through by itself. This is the single
  biggest reason very deep networks are trainable at all.

Stack `n_layer` of these blocks, and at the very end, one more normalization
and a final linear layer that turns the last token's 768 numbers back into
16,384 scores — one per vocabulary token — which softmax turns into "here's
how likely each possible next token is."

### How training actually pushes the weights

```python
logits = self.head(x)   # shape (batch, sequence_length, vocab_size) - raw scores
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
```

`targets` here is just the input sequence shifted over by one position — the
model is trained to predict, at every position, whatever token actually comes
next in the real text. `cross_entropy` measures how wrong the model's
probability distribution was compared to the real answer (low loss = the
model gave high probability to the correct next token). That single number,
`loss`, is what everything in the next section exists to minimize.

---

## 3. Getting real training text

### Why not just download the whole dataset

Public datasets like FineWeb-Edu are enormous (billions of documents,
hundreds of gigabytes). Downloading all of it just to use a small slice would
waste huge amounts of time and disk space for no benefit. The fix is
**streaming**: pull one document at a time over the network, process it
immediately, and never store the parts you don't need.

```python
from datasets import load_dataset

ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                   split="train", streaming=True)

for row in ds:                          # pulls one document at a time, on demand
    text = row["text"]
    ids = tokenizer.encode(text).ids + [eot_id]   # tokenize it right away
    # ... write ids to disk, discard the text ...
    if total_tokens_written >= target:
        break                            # stop as soon as we have enough
```

`eot_id` is the "end of text" special token from step 1 — it gets inserted
between documents so the model can learn "this is where one document ends and
an unrelated one begins," not accidentally treat two unrelated articles as one
continuous story.

### How much text, and why that specific number

This project targets **1 billion tokens**. That number isn't arbitrary — it
comes from a rule of thumb called the "Chinchilla ratio" (~20 tokens of
training data per model parameter, from a well-known scaling-laws research
paper). This model has ~57 million parameters, so `57M × 20 ≈ 1.1B tokens` is
roughly the sweet spot: enough data to actually use the model's capacity well,
without wildly overshooting into a token budget that would take far longer to
train through for limited additional benefit at this size.

### Storing it efficiently

```python
import numpy as np
arr = np.array(ids, dtype=np.uint16)   # not int64!
train_bin_file.write(arr.tobytes())
```

Token ids are just small integers (0 to 16,383 here). `uint16` (unsigned
16-bit integer) can hold values up to 65,535 — plenty — while using only 2
bytes per token instead of 8 bytes for the default 64-bit integer type. For a
billion tokens, that's the difference between a 2GB file and an 8GB file.
Small detail, real practical impact when you're reading this file over and
over during training.

### Reading it back without loading it all into memory

```python
self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")   # doesn't load the file!
# later, to grab a random training example:
chunk = self.data[i : i + block_size].astype(np.int64)
```

`np.memmap` treats the file on disk as if it were an array in memory, but the
operating system only actually reads the specific bytes you access, on
demand, and can discard them again once you're done. This means a 2GB file
can be "opened" instantly and read from randomly, without ever needing 2GB of
RAM to hold it — the file is the storage, RAM only holds whatever tiny slice
is currently being used.

---

## 4. Actually training it

### The core loop, in plain English before code

Training a neural network is: **look at some data, measure how wrong the
model currently is (the loss), figure out which direction to nudge every one
of the model's 57 million numbers to make it slightly less wrong, take a
small step in that direction, and repeat this hundreds of thousands of
times.** "Which direction to nudge" is computed automatically by PyTorch via
**backpropagation** — you never compute it by hand, but it's worth knowing
what `.backward()` is actually doing: calculus (the chain rule), applied
automatically to every parameter in the model, computing exactly how much
increasing or decreasing each individual number would change the final loss.

```python
optimizer.zero_grad(set_to_none=True)   # clear old gradients from last step
x, y = train_ds.get_batch(batch_size, device)   # x: input tokens, y: correct next tokens
logits, loss = model(x, y)              # forward pass: make predictions, measure error
loss.backward()                         # backward pass: compute the nudge direction for every parameter
optimizer.step()                        # actually apply the nudge to every parameter
```

Those four lines, repeated 7,630 times (once per "step" in this project), are
the entire mechanism. Everything else in the training script is refinements
on top of this core loop.

### Refinement 1: batches, not one example at a time

```python
x, y = train_ds.get_batch(batch_size, device)   # e.g. 32 sequences at once, not 1
```

Processing many sequences at once (a "batch") rather than one-at-a-time is
both much faster on a GPU (which is built for doing the same operation on
lots of data in parallel) and gives a more stable, less noisy estimate of
which direction to nudge the weights — one random sequence might give a
misleading signal, but the average over 32 is more trustworthy.

### Refinement 2: gradient accumulation (simulating a bigger batch than fits)

```python
for micro_step in range(grad_accum_steps):     # e.g. 4 micro-steps
    x, y = train_ds.get_batch(micro_batch_size, device)
    loss = model(x, y)[1] / grad_accum_steps
    loss.backward()          # gradients ADD UP across these calls, they aren't reset
optimizer.step()             # one real update, using the combined gradient from all 4
```

GPU memory is limited — you can't always fit as large a batch as you'd like.
The trick: run several smaller "micro-batches," and instead of updating the
weights after each one, let PyTorch accumulate (sum) the gradients across all
of them, then take one optimizer step using the *combined* gradient. The
result behaves like one big batch of `micro_batch_size × grad_accum_steps`,
without ever needing that much memory at once.

### Refinement 3: the learning rate schedule

The "learning rate" controls how big each nudge is. Too big, and training can
become unstable or even diverge (loss shoots up instead of down). Too small,
and training crawls.

```python
def get_lr(step, cfg):
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps       # ramp up linearly from ~0
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # smooth curve from 1 down to 0
    return min_lr + coeff * (lr - min_lr)
```

**Warmup** (the first ~150 steps here): start with a tiny learning rate and
ramp it up. Early in training, the model's weights are random and the
gradient signal can be erratic — jumping straight to a large learning rate
risks a bad, unstable update right at the start.

**Cosine decay** (the rest of training): after warmup, smoothly reduce the
learning rate following a cosine curve, ending near zero by the last step.
Big steps early (when there's a lot of "obviously wrong" to fix quickly),
progressively smaller steps later (fine-tuning once the model is already
decent) — this consistently trains better than a constant learning rate the
whole way through. This exact shape is used by GPT-3, LLaMA, and effectively
every serious pretraining run.

### Refinement 4: checkpointing (save your progress)

```python
def save_checkpoint(path, model, optimizer, step, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)   # (bug fix, see part 5)
    torch.save({
        "model": model.state_dict(),         # every weight in the network
        "optimizer": optimizer.state_dict(), # the optimizer's own internal memory
        "step": step,
    }, path)
```

Training runs for hours. If the computer restarts partway through (this
project deliberately uses a cheap "Spot" VM that Google can reclaim at any
time to save money), you don't want to lose all that progress. Every 500
steps, the entire state gets saved to disk — not just the model's weights,
but the optimizer's state too (it's been tracking a running average of recent
gradients per-parameter, and losing that would cause a visible hiccup in the
loss curve right after resuming). Resuming is just loading this file back and
continuing the loop from `step` instead of 0.

---

## 5. Two real bugs — a debugging case study

Both of these are worth understanding in detail, because the *process* of
diagnosing them is the actually transferable skill — the specific bugs
themselves are less important than how they were found.

### Bug 1: "the training looks frozen, but the GPU is at 100%"

**Symptom:** after launching the real training run, nothing appeared in the
log file for over 9 minutes. No error. No crash. Just silence.

**The tempting wrong conclusion:** "it's hung, kill it and start over."

**Why that would have been wrong:** checking `nvidia-smi` (a command that
shows live GPU activity) the whole time showed the GPU at 100% utilization,
drawing real power, with real memory allocated. That's strong evidence the
GPU was actually doing genuine computation — a truly hung/frozen process
would typically show 0% GPU utilization, since nothing would be issuing new
work to it.

**The actual cause:** Python's `print()` doesn't necessarily write to the file
immediately. When output goes to your terminal directly, Python flushes
(writes out) each line right away. But when output is redirected to a file —
exactly what happens when you run `python3 script.py > train.log` in the
background — Python switches to a different buffering mode and holds output
in memory, writing it out only occasionally (roughly every few KB of text).
The training was working correctly the entire time; the *evidence* of it
working just hadn't been flushed to the file yet.

**The fix:**

```python
import sys
sys.stdout.reconfigure(line_buffering=True)
```

One line, placed near the top of the script. This forces Python to flush
output after every line, regardless of whether it's going to a terminal or a
file. Added once, it fixes this permanently for every future run.

**The general lesson:** "no output" and "not working" are not the same claim.
When you see silence, check for *independent* evidence of activity (here,
`nvidia-smi`) before assuming the worst and restarting something that might
have been fine all along.

### Bug 2: training was configured to do 2.6x more work than intended

**Symptom:** nothing crashed, nothing looked wrong — the loss was dropping
normally. The bug wasn't a crash, it was a **silent mismatch between two
numbers that were supposed to agree.**

**How it was caught:** by explicitly re-deriving the numbers instead of
trusting that a config file's `max_steps: 20000` was correct just because it
had been written down. The math:

```
tokens per step = micro_batch_size × grad_accum_steps × block_size
                 = 32 × 4 × 1024
                 = 131,072 tokens/step

total tokens if max_steps = 20,000:
131,072 × 20,000 = 2,621,440,000  (2.62 billion)
```

But the actual data-prep step earlier had deliberately targeted **1 billion**
tokens (the Chinchilla-ratio number derived in part 3). `2.62B ≠ 1B` — the
config was internally inconsistent with the project's own stated plan, off by
more than 2.6x.

**Why this specific mismatch matters, beyond just "it'd run too long":** the
cosine learning-rate schedule (part 4) is shaped by `max_steps` — it's
designed to decay smoothly down to `min_lr` right as training finishes. If
`max_steps` is wrong, the learning rate would still be relatively high when
the actual 1-billion-token dataset runs out (since the dataset itself is
finite — after 1B tokens' worth of random samples, you're now re-reading data
in a way the schedule didn't account for), meaning the "smoothly wind down and
converge" behavior the schedule exists to provide wouldn't line up with when
training data actually needs to stop.

**The fix:**

```yaml
# 1,000,000,000 / 131,072 ≈ 7630 steps
max_steps: 7630        # was 20000
```

**The general lesson:** a config value that "looks reasonable" and doesn't
crash anything is not the same as a config value that's *correct*. Whenever
two numbers are supposed to be derived from each other (here: token budget
and step count), the right move is to actually recompute one from the other
and check they agree, rather than trust that whoever wrote the file down did
that arithmetic correctly. This is exactly the same category of bug as a
tier-pricing inversion or a unit mismatch — the code runs fine either way, so
nothing forces you to notice unless you deliberately check.
