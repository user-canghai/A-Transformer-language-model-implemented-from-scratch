"""学习率调参实验：不同学习率下的（衰减步长）SGD。"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import json
import math
import os
from collections.abc import Callable

import torch


class SGD(torch.optim.Optimizer):
    """衰减步长 SGD：lr_t = lr / sqrt(t + 1)。"""

    def __init__(self, params, lr: float = 1e-3) -> None:
        if lr < 0:
            raise ValueError(f"非法学习率：{lr}")
        super().__init__(params, {"lr": lr})

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                t = state.get("t", 0)
                # 参数原地更新：theta <- theta - lr / sqrt(t+1) * grad
                param.data -= lr / math.sqrt(t + 1) * param.grad.data
                state["t"] = t + 1
        return loss


def run(lr: float, steps: int = 10) -> list[float]:
    """在 5*randn(10,10) 权重上最小化 mean(w^2)，返回每一步的 loss。"""
    torch.manual_seed(0)
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    optimizer = SGD([weights], lr=lr)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = (weights**2).mean()
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    return losses


def main() -> None:
    results = {}
    for lr in [1e-1, 1.0, 1e1, 1e2, 1e3]:
        losses = run(lr, steps=100)
        results[str(lr)] = losses
        print(f"lr={lr:>8} 第 1 步={losses[0]:.4f} 第 10 步={losses[9]:.4f} 最后一步={losses[-1]:.4f}")
    out_dir = os.path.join("experiments", "logs")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "toy_sgd.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("已保存", os.path.join(out_dir, "toy_sgd.json"))


if __name__ == "__main__":
    main()
