"""Memory-mapped token dataset. Avoids loading the full shard into RAM."""

import numpy as np
import torch


class MemmapTokenDataset:
    def __init__(self, bin_path: str, block_size: int):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def get_batch(self, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        ix = np.random.randint(0, len(self), size=batch_size)
        x = np.stack([self.data[i : i + self.block_size].astype(np.int64) for i in ix])
        y = np.stack([self.data[i + 1 : i + 1 + self.block_size].astype(np.int64) for i in ix])
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        if device == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y
