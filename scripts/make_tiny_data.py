"""Generate random token shards for the CPU smoke test only. Not real data."""

import os

import numpy as np

os.makedirs("data/tiny", exist_ok=True)
rng = np.random.default_rng(0)
for name, n in [("train", 20_000), ("val", 2_000)]:
    arr = rng.integers(0, 512, size=n, dtype=np.uint16)
    arr.tofile(f"data/tiny/{name}.bin")
print("wrote data/tiny/{train,val}.bin")
