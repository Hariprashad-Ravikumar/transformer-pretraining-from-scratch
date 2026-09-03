# Interpretability Report

## Induction heads (synthetic repeated-token probe, Olsson et al. 2022 methodology)

**base.pt** (8 layers, 12 heads, 7,630 training steps): 6 heads with a
clearly elevated prefix-matching score, concentrated in layers 6-7:

  1. layer 6 head 8: 0.7881
  2. layer 6 head 4: 0.7438
  3. layer 7 head 10: 0.4505
  4. layer 7 head 0: 0.4473
  5. layer 6 head 11: 0.2492

**interp_small.pt** (4 layers, 4 heads, 3,000 training steps): much weaker signal overall
(scores an order of magnitude lower than base.pt's strongest heads):

  1. layer 3 head 3: 0.0286
  2. layer 1 head 2: 0.0243
  3. layer 1 head 1: 0.0090
  4. layer 0 head 2: 0.0065
  5. layer 0 head 3: 0.0047

`base.pt` and `interp_small.pt` differ in both model scale *and* training length at once
(this was a decision baked into the config design before this phase started, not an
artifact of the analysis), so this isn't a clean scale-only ablation. What it does show:
with a smaller model and roughly 2.5x fewer training steps, induction heads have only
barely started to emerge, consistent with the literature's framing of induction heads as a
fairly sharp phase change during training rather than a smooth, continuous improvement. It
doesn't isolate whether scale or training length is the driver.

## Causal ablation (zero one head, measure held-out loss delta, full sweep, every head)

**base.pt** baseline held-out loss: 3.2247 nats. Most important heads
by ablation:

  1. layer 5 head 4: 0.0745
  2. layer 0 head 2: 0.0631
  3. layer 4 head 2: 0.0428
  4. layer 7 head 8: 0.0410
  5. layer 1 head 3: 0.0329

**interp_small.pt** baseline held-out loss: 4.1177 nats. Most
important heads by ablation:

  1. layer 2 head 3: 0.2176
  2. layer 3 head 3: 0.1952
  3. layer 3 head 1: 0.1474
  4. layer 0 head 2: 0.1428
  5. layer 0 head 1: 0.1172

The heads with the highest induction (prefix-matching) score are *not* reliably the heads
with the largest ablation loss delta, which is what makes this a causal test rather than
just a picture. In `base.pt`, the two strongest induction heads (layer 6 head 8, score
0.79; layer 6 head 4, score 0.74) rank only 8th and 11th out of 12 heads in their own layer
by causal importance on held-out natural text. The heads that matter most when ablated
(layer 5 head 4, layer 0 head 2) show no elevated induction score at all. In
`interp_small.pt` the two measures agree better (the top induction-score head, layer 3 head
3, is also the 2nd-most-important by ablation), but the agreement isn't clean there either.

The likely explanation: the induction probe measures behavior on synthetic *exact* token
repeats, a narrow and somewhat artificial pattern. Natural held-out text has far fewer
verbatim repeats, so a head that's highly specialized for that exact synthetic pattern may
contribute comparatively little to loss on real text, while heads doing more general
work (that the synthetic probe doesn't target at all) turn out to matter more when
ablated. This is a real, reportable finding about the limits of a purely
attention-pattern-based probe, not a failure of the method.

## Residual-stream norm growth

**base.pt**: 12.7, 17.8, 22.1, 26.6, 33.4, 42.0, 56.0, 88.6 (layers 0-7)

**interp_small.pt**: 1.20, 1.78, 2.20, 3.62 (layers 0-3)

Both grow monotonically with depth, as expected for a pre-norm transformer: each block adds
to the residual stream without renormalizing it, so norm growth compounds. RMSNorm only
rescales what's *read* from the stream at the start of each block, not the stream itself.
`base.pt` grows roughly 7x from layer 0 to layer 7; `interp_small.pt` grows roughly 3x over
its 4 layers, consistent with more layers compounding more growth, though again confounded
with the scale/training-length difference noted above.
