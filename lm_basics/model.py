"""从零实现的 Transformer 语言模型组件。

下面每个模块都只用普通张量运算编写；只使用 Parameter、Module 容器和初始化辅助函数这些 PyTorch 基础构件。
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .nn_utils import scaled_dot_product_attention, silu, softmax

__all__ = [
    "Embedding",
    "Linear",
    "MultiHeadSelfAttention",
    "RMSNorm",
    "RotaryPositionalEmbedding",
    "SwiGLU",
    "TransformerBlock",
    "TransformerLM",
    "scaled_dot_product_attention",
    "silu",
    "softmax",
]


class Linear(nn.Module):
    """无偏置项的线性变换 y = x W^T。"""

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.transpose(-2, -1)


class Embedding(nn.Module):
    """将 token id 映射为稠密向量的查找表。"""

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """带可学习增益的均方根层归一化（RMSNorm）。"""

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms * self.weight.to(torch.float32)).to(in_dtype)


def _default_d_ff(d_model: int) -> int:
    """将 8/3 * d_model 向上取整到 64 的倍数。"""
    return 64 * math.ceil((8 * d_model / 3) / 64)


class SwiGLU(nn.Module):
    """使用 SwiGLU 激活的逐位置前馈网络。"""

    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else _default_d_ff(d_model)
        self.w1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, self.d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class SiLUFeedForward(nn.Module):
    """不带门控的 SiLU 前馈网络（SwiGLU 的消融基线）。"""

    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None) -> None:
        super().__init__()
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        self.w1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)))


class RotaryPositionalEmbedding(nn.Module):
    """作用于 query/key 张量的旋转位置编码（RoPE）。"""

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) -> None:
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError(f"RoPE 要求 head 维度为偶数，实际得到 d_k={d_k}")
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        inv_freq = theta ** (-torch.arange(0, d_k, 2, dtype=torch.float32, device=device) / d_k)
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        angles = torch.outer(positions, inv_freq)
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions].to(dtype=x.dtype, device=x.device)
        sin = self.sin[token_positions].to(dtype=x.dtype, device=x.device)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        out = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)
        return out.flatten(-2)


class MultiHeadSelfAttention(nn.Module):
    """带可选 RoPE 的因果多头自注意力。"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float = 10000.0,
        use_rope: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} 必须能被 num_heads={num_heads} 整除")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.use_rope = use_rope
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        if use_rope:
            if max_seq_len is None:
                raise ValueError("use_rope=True 时必须提供 max_seq_len")
            self.rope = RotaryPositionalEmbedding(theta, self.d_head, max_seq_len, device=device)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.unflatten(-1, (self.num_heads, self.d_head)).transpose(-3, -2)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        # 把位置张量的形状对齐到 (..., seq)，使其能在 head 维上广播。
        if self.use_rope and token_positions is not None and token_positions.ndim == x.ndim - 1:
            token_positions = token_positions.unsqueeze(-2)
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        if self.use_rope:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        out = scaled_dot_product_attention(q, k, v, causal_mask)
        out = out.transpose(-3, -2).flatten(-2)
        return self.output_proj(out)


class TransformerBlock(nn.Module):
    """pre-norm（默认）或 post-norm 的 Transformer 块。"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int | None = None,
        theta: float = 10000.0,
        use_rope: bool = True,
        norm: str = "pre",
        use_rmsnorm: bool = True,
        ffn: str = "swiglu",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if norm not in ("pre", "post"):
            raise ValueError(f"norm 只能是 'pre' 或 'post'，实际得到 {norm}")
        self.norm = norm
        self.attn = MultiHeadSelfAttention(
            d_model,
            num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            use_rope=use_rope,
            device=device,
            dtype=dtype,
        )
        if use_rmsnorm:
            self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
            self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        else:
            self.ln1 = nn.Identity()
            self.ln2 = nn.Identity()
        if ffn == "swiglu":
            self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        elif ffn == "silu":
            self.ffn = SiLUFeedForward(d_model, d_ff, device=device, dtype=dtype)
        else:
            raise ValueError(f"ffn 只能是 'swiglu' 或 'silu'，实际得到 {ffn}")

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        if self.norm == "pre":
            x = x + self.attn(self.ln1(x), token_positions)
            x = x + self.ffn(self.ln2(x))
        else:
            x = self.ln1(x + self.attn(x, token_positions))
            x = self.ln2(x + self.ffn(x))
        return x


class TransformerLM(nn.Module):
    """decoder-only 的 Transformer 语言模型。"""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        use_rope: bool = True,
        norm: str = "pre",
        use_rmsnorm: bool = True,
        ffn: str = "swiglu",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    use_rope=use_rope,
                    norm=norm,
                    use_rmsnorm=use_rmsnorm,
                    ffn=ffn,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype) if use_rmsnorm else nn.Identity()
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_final(x))
