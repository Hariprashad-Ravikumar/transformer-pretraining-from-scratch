# Triton RMSNorm Kernel: Correctness and Benchmark

## Correctness (VM, CUDA required)

`tests/test_triton_kernels.py` (3/3 passed on the L4):

- `test_forward_matches_reference`: Triton forward output matches the plain-PyTorch
  RMSNorm reference within `atol=1e-4`.
- `test_gradients_match_reference`: gradients w.r.t. both `x` and `weight` match the
  PyTorch-autograd reference within `atol=1e-3`, confirming the hand-derived backward
  pass (see `src/triton_kernels/rmsnorm.py`'s docstring for the derivation) is correct,
  not just the forward.
- `test_matches_reference_within_full_model`: a full `DecoderTransformer` with
  `use_triton_rmsnorm=True` produces matching logits to the same model with it `False`,
  given identical weights, confirming the kernel is correctly wired into the real model,
  not just correct in isolation.

## Benchmark (before/after, real model scale)

`configs/triton_bench_off.yaml` vs `configs/triton_bench_on.yaml` are identical in every
respect (base.yaml's real dimensions: 8 layers, 12 heads, 768 dim, ~56.6M non-embedding
params, real FineWeb-Edu corpus) except `model.use_triton_rmsnorm`. 200 steps each,
`torch.compile` on, mean over steps 20-180 (post-warmup):

| | tok/s | MFU |
|---|---|---|
| PyTorch RMSNorm (baseline) | 65,484 | 22.47% |
| Triton RMSNorm | 61,172 | 21.00% |

The Triton kernel is ~6.6% slower, not faster. This is the honest result, not the result a
"we optimized it" narrative would want, and it's a real, explicable finding rather than a
failed experiment to bury:

- `torch.compile`'s Inductor backend already lowers the plain-PyTorch RMSNorm (a handful
  of elementwise ops plus a reduction) into a fused Triton kernel automatically, as part
  of compiling the surrounding graph. A hand-written kernel doesn't get to start from
  "uncompiled PyTorch" as its baseline here; it's competing against Inductor's own
  autotuned fusion.
- The custom kernel is wrapped in a `torch.autograd.Function`, which forces `torch.compile`
  (Dynamo) to graph-break around it. Every other op fuses into one compiled region, and
  the custom RMSNorm becomes an isolated eager-mode call sandwiched between compiled
  segments, adding dispatch overhead exactly where the graph would otherwise flow through
  uninterrupted.
- The kernel itself is also not autotuned (`BLOCK_SIZE` from a fixed next-power-of-2
  rule, no autotuning search over block size/warps/stages), unlike Inductor's generated
  kernel, which the compiler autotunes as part of compilation.

Not every hand-written kernel beats a modern compiler baseline, and integration cost
(breaking `torch.compile`'s fusion boundary) can outweigh a kernel's own execution time.
That's a real, useful lesson about when custom kernels are and aren't worth writing, and a
more honest result than a fabricated "we made it 20% faster" would be. The path
to an actual win here would be autotuning the kernel and/or registering it as a proper
custom op Dynamo can trace through instead of graph-breaking around, both out of scope for
this pass.

## DDP / multi-GPU scaling: deferred

Requested a small increase to the global GPU quota (`GPUS-ALL-REGIONS-per-project`,
1 -> 2) to unblock a 2-GPU comparison run. It was denied outright, unlike the earlier
0 -> 1 grant that made the single-GPU setup possible in the first place. The regional
preemptible-L4 quota is 3, but the global cap (currently 1) sits underneath it and blocks
any multi-GPU run regardless. Deferred rather than blocking the rest of this phase:
`train.py` is already DDP-wired (`setup_ddp()`, `DDP(...)`, gradient-sync gating) and
ready to run a scaling comparison whenever quota allows.
