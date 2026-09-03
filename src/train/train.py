"""Training loop: bf16 mixed precision, torch.compile, gradient accumulation,
DDP-ready, checkpoint/resume built to survive Spot preemption.

Single GPU:
    python -m src.train.train --config configs/base.yaml

Multi-GPU (DDP):
    torchrun --standalone --nproc_per_node=4 -m src.train.train --config configs/base.yaml
"""

import argparse
import math
import os
import time

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from src.data.dataset import MemmapTokenDataset
from src.model.transformer import DecoderTransformer, ModelConfig


def setup_ddp():
    if "RANK" not in os.environ:
        return False, 0, 1, "cuda" if torch.cuda.is_available() else "cpu"
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return True, rank, world_size, f"cuda:{local_rank}"


def get_lr(step: int, cfg: dict) -> float:
    warmup = cfg["warmup_steps"]
    max_steps = cfg["max_steps"]
    min_lr = cfg["min_lr"]
    lr = cfg["lr"]
    if step < warmup:
        return lr * (step + 1) / warmup
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup) / max(1, max_steps - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


def save_checkpoint(path, model, optimizer, step, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw_model = model.module if isinstance(model, DDP) else model
    torch.save(
        {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": cfg,
        },
        path,
    )


def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["step"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    is_ddp, rank, world_size, device = setup_ddp()
    is_master = rank == 0
    use_cuda = "cuda" in device
    torch.manual_seed(cfg.get("seed", 1337) + rank)

    model_cfg = ModelConfig(**cfg["model"])
    model = DecoderTransformer(model_cfg).to(device)
    if is_master:
        print(f"model params (non-embedding): {model.num_params():,}")

    if cfg.get("compile", True) and use_cuda:
        model = torch.compile(model)

    if is_ddp:
        model = DDP(model, device_ids=[int(device.split(":")[1])])

    raw_for_opt = model.module if isinstance(model, DDP) else model
    optimizer = raw_for_opt.configure_optimizers(
        weight_decay=cfg["weight_decay"], lr=cfg["lr"], betas=tuple(cfg["betas"])
    )

    train_ds = MemmapTokenDataset(cfg["train_bin"], model_cfg.block_size)
    val_ds = MemmapTokenDataset(cfg["val_bin"], model_cfg.block_size)

    start_step = 0
    ckpt_path = cfg["checkpoint_path"]
    if os.path.exists(ckpt_path):
        start_step = load_checkpoint(ckpt_path, model, optimizer, device)
        if is_master:
            print(f"resumed from {ckpt_path} at step {start_step}")

    grad_accum = cfg["grad_accum_steps"]
    micro_bs = cfg["micro_batch_size"]
    dtype = torch.bfloat16 if use_cuda else torch.float32

    model.train()
    t0 = time.time()
    for step in range(start_step, cfg["max_steps"]):
        lr = get_lr(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(grad_accum):
            x, y = train_ds.get_batch(micro_bs, device)
            if is_ddp:
                model.require_backward_grad_sync = micro_step == grad_accum - 1
            with torch.autocast(device_type="cuda" if use_cuda else "cpu", dtype=dtype):
                _, loss = model(x, y)
                loss = loss / grad_accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
        optimizer.step()

        if is_master and step % cfg["log_interval"] == 0:
            dt = time.time() - t0
            toks_per_sec = (
                micro_bs * grad_accum * model_cfg.block_size * world_size * cfg["log_interval"] / dt
                if step > start_step
                else 0
            )
            print(f"step {step} | loss {loss.item() * grad_accum:.4f} | lr {lr:.2e} | tok/s {toks_per_sec:.0f}")
            t0 = time.time()

        if is_master and step > 0 and step % cfg["checkpoint_interval"] == 0:
            save_checkpoint(ckpt_path, model, optimizer, step, cfg)
            print(f"checkpoint saved at step {step}")

    if is_master:
        save_checkpoint(ckpt_path, model, optimizer, cfg["max_steps"], cfg)
        print("training complete, final checkpoint saved")

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
