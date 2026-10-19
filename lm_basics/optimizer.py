"""从零实现的 AdamW 优化器与学习率调度。"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import torch


class AdamW(torch.optim.Optimizer):
    """解耦权重衰减的 AdamW（Loshchilov & Hutter, 2019）。"""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if lr < 0:
            raise ValueError(f"无效的学习率：{lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"无效的 beta 参数：{betas}")
        if eps < 0:
            raise ValueError(f"无效的 epsilon 值：{eps}")
        if weight_decay < 0:
            raise ValueError(f"无效的 weight_decay 值：{weight_decay}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> float | None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                m, v = state["m"], state["v"]
                state["step"] += 1
                t = state["step"]
                # 解耦权重衰减：theta <- theta - lr * lambda * theta。
                p.data.mul_(1.0 - lr * weight_decay)
                # 一阶与二阶矩估计。
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                # 偏差校正后的步长（与 PyTorch 的 AdamW 等价）。
                step_size = lr * math.sqrt(1.0 - beta2**t) / (1.0 - beta1**t)
                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-step_size)
        return loss


def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """带线性 warmup 的余弦退火。"""
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_learning_rate + cosine * (max_learning_rate - min_learning_rate)
    return min_learning_rate
