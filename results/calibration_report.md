# Token-Level Calibration Report

- Held-out tokens (fit T / measure ECE-Brier split of val.bin): 248,748 / 248,749
- Tokens used to fit temperature T: 20,480 (subsample of the fit half; one scalar parameter
  doesn't need the full half to fit reliably)
- Fitted temperature: T = 1.0144
- Top-1 accuracy (argmax == actual next token) on the held-out half: 0.3958

| | ECE (5 bins) | Brier score |
|---|---|---|
| raw (T=1) | 0.0071 | 0.1572 |
| temperature-scaled | 0.0047 | 0.1571 |

See `calibration_curve.png` for the reliability diagram (raw vs. temperature-scaled).

**Methodology note:** confidence = the model's max softmax probability per
token position; label = whether that top-1 prediction matches the actual next
token. This is the standard Guo et al. 2017 definition, and structurally
identical to the router's `(predicted_probability, is_correct)` pairs -- same
ECE/Brier formulas (`src/eval/calibration_stats.py`, ported from the router's
`stats_utils.py`), applied to a different task.

**Split note:** unlike the router's own calibration report, T is fit and ECE/Brier
are measured on two disjoint halves of the held-out set -- a genuine
generalization test for the temperature fix, not a same-distribution sanity
check.

## Cross-project comparison

| | ECE | Brier |
|---|---|---|
| router calibrator (task-level: is the routed response correct) | 0.1671 | 0.1521 |
| this model, raw (token-level: is the top-1 next-token prediction correct) | 0.0071 | 0.1572 |
| this model, temperature-scaled | 0.0047 | 0.1571 |

**Caveat, stated plainly:** these are not apples-to-apples measurements of the
same thing. The router's calibrator predicts *task-level* correctness of a
routed LLM response (a judged, semantic notion of "right answer"), fit with a
trained logistic-regression calibrator on hand-engineered features. This
model's numbers are *token-level* next-token prediction confidence, straight
out of the softmax, with no calibrator fit at all beyond the single
temperature scalar. Different task, different label distribution, different
downstream stakes. What *is* comparable is the calibration behavior itself --
whether raw model confidence over- or under-shoots actual accuracy, and
whether a standard post-hoc fix (temperature scaling here, a trained
calibrator there) narrows that gap.
