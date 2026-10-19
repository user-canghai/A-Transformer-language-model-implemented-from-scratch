"""从零实现的数值稳定神经网络基础算子。

这些运算要求手工实现、不使用 torch.nn.functional，因此下面的每个函数
都用普通张量算术实现。
"""

from __future__ import annotations

import math

import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """沿 dim 维计算 softmax，先减去最大值以保证数值稳定。"""
    shifted = x - x.max(dim=dim, keepdim=True).values
    exp = shifted.exp()
    return exp / exp.sum(dim=dim, keepdim=True)


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU / Swish 激活函数：x * sigmoid(x)。"""
    return x * torch.sigmoid(x)


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """带可选布尔掩码的缩放点积注意力。

    参数：
        Q：形状为 (..., queries, d_k) 的 query。
        K：形状为 (..., keys, d_k) 的 key。
        V：形状为 (..., keys, d_v) 的 value。
        mask：形状为 (..., queries, keys) 的布尔张量；True 表示该 query
            可以关注对应的 key，False 表示将其屏蔽。

    返回：
        形状为 (..., queries, d_v) 的输出。
    """
    d_k = Q.shape[-1]
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attention_weights = softmax(scores, dim=-1)
    return attention_weights @ V


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """logits 与整数 target 之间的平均交叉熵损失。

    参数：
        logits：形状为 (..., vocab_size) 的未归一化预测。
        targets：形状为 (...) 的整数类别索引。

    返回：
        对所有前导维取平均后的标量损失张量。
    """
    shifted = logits - logits.max(dim=-1, keepdim=True).values
    log_normalizer = shifted.exp().sum(dim=-1).log()
    target_logits = shifted.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return (log_normalizer - target_logits).mean()


def clip_gradients(parameters, max_l2_norm: float) -> None:
    """原地裁剪梯度的全局 l2 范数。"""
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    total_norm = torch.sqrt(sum((g.to(torch.float32) ** 2).sum() for g in grads))
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for g in grads:
            g.mul_(clip_coef.to(g.dtype))
