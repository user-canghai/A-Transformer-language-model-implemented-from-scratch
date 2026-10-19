"""Transformer 语言模型的训练驱动。

实现讲义第 5 节所述的训练循环：memmap 数据、带余弦学习率衰减与 warmup 的
AdamW、梯度裁剪、周期性验证、checkpoint，以及实验日志（CSV，可选 Weights & Biases）。
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import time

import numpy as np
import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .data import get_batch, load_dataset
from .model import TransformerLM
from .nn_utils import clip_gradients, cross_entropy
from .optimizer import AdamW, lr_cosine_schedule


@dataclasses.dataclass
class TrainConfig:
    """单次训练运行的全部超参数。"""

    train_data: str = "data/tinystories_train.npy"
    valid_data: str = "data/tinystories_valid.npy"

    vocab_size: int = 10000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1344
    rope_theta: float = 10000.0
    use_rope: bool = True
    norm: str = "pre"
    use_rmsnorm: bool = True
    ffn: str = "swiglu"

    batch_size: int = 32
    num_steps: int = 5000
    lr_max: float = 1e-3
    lr_min: float = 1e-4
    warmup_iters: int = 200
    cosine_cycle_iters: int = 5000
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0

    device: str = "cpu"
    seed: int = 42
    log_interval: int = 10
    eval_interval: int = 100
    eval_batches: int = 20
    checkpoint_interval: int = 1000
    out_dir: str = "runs/base"
    compile_model: bool = False
    wandb_project: str = ""
    resume: str = ""


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def resolve_device(device: str) -> torch.device:
    """解析设备字符串；需要时导入 Ascend 后端。"""
    if device.startswith("npu"):
        import torch_npu  # noqa: F401  （注册 npu 后端）
    return torch.device(device)


@torch.no_grad()
def evaluate(model: torch.nn.Module, valid_data, cfg: TrainConfig, device: torch.device) -> float:
    """在固定数量的 batch 上返回平均每 token 的验证损失。"""
    model.eval()
    total_loss = 0.0
    for _ in range(cfg.eval_batches):
        inputs, targets = get_batch(valid_data, cfg.batch_size, cfg.context_length, device)
        logits = model(inputs)
        total_loss += cross_entropy(logits, targets).item()
    model.train()
    return total_loss / cfg.eval_batches


def train(cfg: TrainConfig) -> dict:
    """运行一次训练任务，返回最终指标。"""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = resolve_device(cfg.device)

    train_data = load_dataset(cfg.train_data, dtype="uint16", mmap=True)
    valid_data = load_dataset(cfg.valid_data, dtype="uint16", mmap=True)

    model = TransformerLM(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        rope_theta=cfg.rope_theta,
        use_rope=cfg.use_rope,
        norm=cfg.norm,
        use_rmsnorm=cfg.use_rmsnorm,
        ffn=cfg.ffn,
    ).to(device)
    if cfg.compile_model:
        model = torch.compile(model)  # type: ignore[assignment]

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr_max,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )

    start_step = 0
    if cfg.resume:
        start_step = load_checkpoint(cfg.resume, model, optimizer)
        print(f"已从 {cfg.resume} 恢复，起始 step 为 {start_step}")

    n_params = count_parameters(model)
    print(f"模型参数量：{n_params:,}（约 {n_params / 1e6:.1f}M）")
    print(f"设备：{device}，训练 token 数：{len(train_data):,}，验证 token 数：{len(valid_data):,}")

    wandb_run = None
    if cfg.wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=cfg.wandb_project, dir=cfg.out_dir, config=dataclasses.asdict(cfg), resume="allow"
        )

    log_path = os.path.join(cfg.out_dir, "log.csv")
    new_log = not os.path.exists(log_path) or start_step == 0
    log_file = open(log_path, "a" if not new_log else "w", newline="")  # noqa: SIM115 - 训练期间保持打开
    writer = csv.writer(log_file)
    if new_log:
        writer.writerow(["step", "wall_time_s", "lr", "train_loss", "valid_loss", "tokens_seen", "tokens_per_s"])

    model.train()
    start_time = time.time()
    running_loss = 0.0
    tokens_per_step = cfg.batch_size * cfg.context_length
    best_valid = float("inf")
    final_valid = float("nan")

    for step in range(start_step, cfg.num_steps):
        lr = lr_cosine_schedule(step, cfg.lr_max, cfg.lr_min, cfg.warmup_iters, cfg.cosine_cycle_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr

        inputs, targets = get_batch(train_data, cfg.batch_size, cfg.context_length, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        clip_gradients(model.parameters(), cfg.grad_clip)
        optimizer.step()
        running_loss += loss.item()

        if (step + 1) % cfg.log_interval == 0:
            elapsed = time.time() - start_time
            tokens_seen = step * tokens_per_step
            writer.writerow(
                [
                    step,
                    round(elapsed, 3),
                    f"{lr:.6e}",
                    round(running_loss / cfg.log_interval, 5),
                    "",
                    tokens_seen,
                    round(tokens_seen / elapsed, 1) if elapsed > 0 else 0,
                ]
            )
            log_file.flush()
            print(
                f"step {step:6d} | lr {lr:.3e} | 训练损失 {running_loss / cfg.log_interval:.4f} | "
                f"{tokens_seen / max(elapsed, 1e-9):,.0f} tok/s"
            )
            if wandb_run is not None:
                wandb_run.log({"step": step, "lr": lr, "train_loss": running_loss / cfg.log_interval})
            running_loss = 0.0

        if cfg.eval_interval > 0 and (step + 1) % cfg.eval_interval == 0:
            valid_loss = evaluate(model, valid_data, cfg, device)
            best_valid = min(best_valid, valid_loss)
            final_valid = valid_loss
            elapsed = time.time() - start_time
            writer.writerow(
                [
                    step,
                    round(elapsed, 3),
                    f"{lr:.6e}",
                    "",
                    round(valid_loss, 5),
                    (step + 1) * tokens_per_step,
                    "",
                ]
            )
            log_file.flush()
            print(f"step {step:6d} | 验证损失 {valid_loss:.4f} | 困惑度 {math.exp(valid_loss):.2f}")
            if wandb_run is not None:
                wandb_run.log({"step": step, "valid_loss": valid_loss, "perplexity": math.exp(valid_loss)})

        if cfg.checkpoint_interval > 0 and (step + 1) % cfg.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, step + 1, os.path.join(cfg.out_dir, f"checkpoint_{step + 1}.pt"))

    elapsed = time.time() - start_time
    final_checkpoint = os.path.join(cfg.out_dir, "checkpoint_final.pt")
    save_checkpoint(model, optimizer, cfg.num_steps, final_checkpoint)
    log_file.close()

    metrics = {
        "final_valid_loss": final_valid,
        "final_valid_perplexity": math.exp(final_valid) if not math.isnan(final_valid) else None,
        "best_valid_loss": best_valid if best_valid != float("inf") else None,
        "num_parameters": n_params,
        "num_steps": cfg.num_steps,
        "tokens_processed": cfg.num_steps * tokens_per_step,
        "wall_time_s": round(elapsed, 2),
        "tokens_per_s": round(cfg.num_steps * tokens_per_step / elapsed, 1) if elapsed else None,
    }
    with open(os.path.join(cfg.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)
    print(json.dumps(metrics, indent=2))
    if wandb_run is not None:
        wandb_run.finish()
    return metrics


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    cfg = TrainConfig()
    for field in dataclasses.fields(cfg):
        name = "--" + field.name.replace("_", "-")
        default = getattr(cfg, field.name)
        if isinstance(default, bool):
            parser.add_argument(name, action=argparse.BooleanOptionalAction, default=default)
        elif isinstance(default, int):
            parser.add_argument(name, type=int, default=default)
        elif isinstance(default, float):
            parser.add_argument(name, type=float, default=default)
        else:
            parser.add_argument(name, type=str, default=default)
    args = parser.parse_args()
    return TrainConfig(**{f.name: getattr(args, f.name) for f in dataclasses.fields(cfg)})


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
