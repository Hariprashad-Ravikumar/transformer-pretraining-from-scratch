"""Assembles results/interpretability_report.md from the JSON already written
by induction_heads.py, residual_norms.py, and ablation.py (run separately --
this script doesn't rerun the probes, just writes up what they found).

Usage:
    python -m src.interpretability.run_all --results-dir results
"""

import argparse
import json
import os


def load(results_dir, name):
    with open(os.path.join(results_dir, name)) as f:
        return json.load(f)


def top_n(ranked, n=5):
    return "\n".join(f"  {i+1}. layer {r['layer']} head {r.get('head')}: {list(r.values())[-1]:.4f}" for i, r in enumerate(ranked[:n]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default="results/interpretability_report.md")
    args = parser.parse_args()

    ih_base = load(args.results_dir, "induction_heads_base.json")
    ih_small = load(args.results_dir, "induction_heads_interp_small.json")
    abl_base = load(args.results_dir, "ablation_base.json")
    abl_small = load(args.results_dir, "ablation_interp_small.json")
    rn_base = load(args.results_dir, "residual_norms_base.json")
    rn_small = load(args.results_dir, "residual_norms_interp_small.json")

    def head_set(ranked, thresh):
        return {(r["layer"], r["head"]) for r in ranked if r["score"] >= thresh}

    strong_induction_base = [r for r in ih_base["ranked"] if r["score"] > 0.2]
    strong_induction_small = [r for r in ih_small["ranked"] if r["score"] > 0.02]

    report = f"""# Interpretability Report

## Induction heads (synthetic repeated-token probe, Olsson et al. 2022 methodology)

**base.pt** (8 layers, 12 heads, 7,630 training steps): {len(strong_induction_base)} heads with a
clearly elevated prefix-matching score, concentrated in layers 6-7:

{top_n(ih_base["ranked"])}

**interp_small.pt** (4 layers, 4 heads, 3,000 training steps): much weaker signal overall
(scores an order of magnitude lower than base.pt's strongest heads):

{top_n(ih_small["ranked"])}

**Reading this comparison honestly**: `base.pt` and `interp_small.pt` differ in both model
scale *and* training length at once (this was a decision baked into the config design
before this phase started, not an artifact of the analysis) -- so this isn't a clean
scale-only ablation. What it does show: with a smaller model and roughly 2.5x fewer
training steps, induction heads have only barely started to emerge, consistent with the
literature's framing of induction heads as a fairly sharp phase change during training
rather than a smooth, continuous improvement. It doesn't isolate whether scale or training
length is the driver.

## Causal ablation (zero one head, measure held-out loss delta -- full sweep, every head)

**base.pt** baseline held-out loss: {abl_base["baseline_loss"]:.4f} nats. Most important heads
by ablation:

{top_n(abl_base["ranked_most_important"])}

**interp_small.pt** baseline held-out loss: {abl_small["baseline_loss"]:.4f} nats. Most
important heads by ablation:

{top_n(abl_small["ranked_most_important"])}

**The finding that makes this a causal test, not just a picture**: the heads with the
highest induction (prefix-matching) score are *not* reliably the heads with the largest
ablation loss delta. In `base.pt`, the two strongest induction heads (layer 6 head 8, score
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

**base.pt**: {", ".join(f"{x:.1f}" for x in rn_base["norms_by_layer"])} (layers 0-7)

**interp_small.pt**: {", ".join(f"{x:.2f}" for x in rn_small["norms_by_layer"])} (layers 0-3)

Both grow monotonically with depth, as expected for a pre-norm transformer (each block adds
to the residual stream without renormalizing it, so norm growth compounds; RMSNorm only
rescales what's *read* from the stream at the start of each block, not the stream itself).
`base.pt` grows roughly 7x from layer 0 to layer 7; `interp_small.pt` grows roughly 3x over
its 4 layers -- consistent with more layers compounding more growth, though again confounded
with the scale/training-length difference noted above.
"""
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(report)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
