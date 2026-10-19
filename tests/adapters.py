from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    给定 Linear 层的权重，计算一批输入的线性变换。

    参数:
        in_dim (int): 输入维度的大小
        out_dim (int): 输出维度的大小
        weights (Float[Tensor, "d_out d_in"]): 使用的线性权重
        in_features (Float[Tensor, "... d_in"]): 要应用该函数的输出张量

    返回:
        Float[Tensor, "... d_out"]: 你的 Linear 模块的变换输出。
    """

    from lm_basics.model import Linear

    linear = Linear(d_in, d_out)
    linear.weight.data = weights
    return linear(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    给定 Embedding 层的权重，获取一批 token id 的嵌入向量。

    参数:
        vocab_size (int): 词表中嵌入向量的数量
        d_model (int): 嵌入维度的大小
        weights (Float[Tensor, "vocab_size d_model"]): 要从中获取的嵌入向量
        token_ids (Int[Tensor, "..."]): 要从 Embedding 层获取的 token id 集合

    返回:
        Float[Tensor, "... d_model"]: 你的 Embedding 层返回的一批嵌入向量。
    """

    from lm_basics.model import Embedding

    embedding = Embedding(vocab_size, d_model)
    embedding.weight.data = weights
    return embedding(token_ids)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """给定 SwiGLU 网络的权重，返回使用这些权重运行你的
    实现所得到的输出。

    参数:
        d_model (int): 前馈网络输入和输出的维度。
        d_ff (int): 你的 SwiGLU 内部上投影的维度。
        w1_weight (Float[Tensor, "d_ff d_model"]): W1 的存储权重
        w2_weight (Float[Tensor, "d_model d_ff"]): W2 的存储权重
        w3_weight (Float[Tensor, "d_ff d_model"]): W3 的存储权重
        in_features (Float[Tensor, "... d_model"]): 前馈层的输入嵌入。

    返回:
        Float[Tensor, "... d_model"]: 与输入嵌入形状相同的输出嵌入。
    """
    # 示例：
    # 如果你的 state dict 键名匹配，可以使用 `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # 你也可以手动赋值权重
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    from lm_basics.model import SwiGLU

    swiglu = SwiGLU(d_model, d_ff)
    swiglu.w1.weight.data = w1_weight
    swiglu.w2.weight.data = w2_weight
    swiglu.w3.weight.data = w3_weight
    return swiglu(in_features)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    给定键（K）、查询（Q）和值（V）张量，返回缩放点积注意力
    实现的输出。

    参数:
        Q (Float[Tensor, " ... queries d_k"]): 查询张量
        K (Float[Tensor, " ... keys d_k"]): 键张量
        V (Float[Tensor, " ... keys d_v"]): 值张量
        mask (Bool[Tensor, " ... queries keys"] | None): 掩码张量
    返回:
        Float[Tensor, " ... queries d_v"]: SDPA 的输出
    """
    from lm_basics.nn_utils import scaled_dot_product_attention

    return scaled_dot_product_attention(Q, K, V, mask)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    给定一个朴素、非批处理的多头注意力实现的键、查询和值投影权重，
    返回优化后的批处理实现的输出。该实现应当在一次矩阵乘法中
    处理所有头的键、查询和值投影。
    本函数不应使用 RoPE。
    参见 Vaswani 等人 2017 年论文的第 3.2.2 节。

    参数:
        d_model (int): 前馈网络输入和输出的维度。
        num_heads (int): 多头注意力中使用的头数。
        max_seq_len (int): 如果你的实现会预缓存，则为预缓存的最大序列长度。
        q_proj_weight (Float[Tensor, "d_model d_model"]): Q 投影的权重
        k_proj_weight (Float[Tensor, "d_model d_model"]): K 投影的权重
        v_proj_weight (Float[Tensor, "d_model d_model"]): V 投影的权重
        o_proj_weight (Float[Tensor, "d_model d_model"]): 输出投影的权重
        in_features (Float[Tensor, "... sequence_length d_model"]): 用于运行你的实现的张量。

    返回:
        Float[Tensor, " ... sequence_length d_model"]: 使用给定 QKV 投影权重和输入特征运行你优化的、
        批处理多头注意力实现所得到的输出张量。
    """
    from lm_basics.model import MultiHeadSelfAttention

    attention = MultiHeadSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=in_features.shape[-2],
        use_rope=False,
    )
    attention.q_proj.weight.data = q_proj_weight
    attention.k_proj.weight.data = k_proj_weight
    attention.v_proj.weight.data = v_proj_weight
    attention.output_proj.weight.data = o_proj_weight
    return attention(in_features)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    给定一个朴素、非批处理的多头注意力实现的键、查询和值投影权重，
    返回优化后的批处理实现的输出。该实现应当在一次矩阵乘法中
    处理所有头的键、查询和值投影。
    此版本的 MHA 应包含 RoPE。
    在这种情况下，RoPE 嵌入维度必须是每个头的嵌入维度（d_model // num_heads）。
    参见 Vaswani 等人 2017 年论文的第 3.2.2 节。

    参数:
        d_model (int): 前馈网络输入和输出的维度。
        num_heads (int): 多头注意力中使用的头数。
        max_seq_len (int): 如果你的实现会预缓存，则为预缓存的最大序列长度。
        theta (float): RoPE 参数。
        q_proj_weight (Float[Tensor, "d_model d_model"]): Q 投影的权重
        k_proj_weight (Float[Tensor, "d_model d_model"]): K 投影的权重
        v_proj_weight (Float[Tensor, "d_model d_model"]): V 投影的权重
        o_proj_weight (Float[Tensor, "d_model d_model"]): 输出投影的权重
        in_features (Float[Tensor, "... sequence_length d_model"]): 用于运行你的实现的张量。
        token_positions (Int[Tensor, " ... sequence_length"] | None): 可选的张量，包含各个 token 的位置

    返回:
        Float[Tensor, " ... sequence_length d_model"]: 使用给定 QKV 投影权重和输入特征运行你优化的、
        批处理多头注意力实现所得到的输出张量。
    """
    from lm_basics.model import MultiHeadSelfAttention

    attention = MultiHeadSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        theta=theta,
        use_rope=True,
    )
    attention.q_proj.weight.data = q_proj_weight
    attention.k_proj.weight.data = k_proj_weight
    attention.v_proj.weight.data = v_proj_weight
    attention.output_proj.weight.data = o_proj_weight
    return attention(in_features, token_positions)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    对给定输入张量运行 RoPE。

    参数:
        d_k (int): 查询或键张量的嵌入维度大小。
        theta (float): RoPE 参数。
        max_seq_len (int): 如果你的实现会预缓存，则为预缓存的最大序列长度。
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): 要运行 RoPE 的输入张量。
        token_positions (Int[Tensor, "... sequence_length"]): 形状为 (batch_size, sequence_length) 的张量，包含 token 位置
    返回:
        Float[Tensor, " ... sequence_length d_k"]: 应用 RoPE 后的输入张量。
    """
    from lm_basics.model import RotaryPositionalEmbedding

    rope = RotaryPositionalEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len)
    return rope(in_query_or_key, token_positions)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    给定 pre-norm Transformer 块的权重和输入特征，返回在输入特征上
    运行 Transformer 块所得到的输出。

    本函数应使用 RoPE。
    根据你的实现方式，你可能只需将相关参数传给 TransformerBlock 构造函数，
    也可能需要初始化自己的 RoPE 类并将其传入。

    参数:
        d_model (int): Transformer 块输入的维度。
        num_heads (int): 多头注意力中使用的头数。`d_model` 必须能被 `num_heads` 整除。
        d_ff (int): 前馈网络内层的维度。
        max_seq_len (int): 如果你的实现会预缓存，则为预缓存的最大序列长度。
        theta (float): RoPE 参数。
        weights (dict[str, Tensor]):
            我们的参考实现的 state dict。
            该字典的键包括：
            - `attn.q_proj.weight`
                所有 `num_heads` 个注意力头的查询投影。
                形状为 (d_model, d_model)。
                各行的排列顺序对应于形状为 (num_heads, d_k) 的矩阵，
                因此 `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`。
            - `attn.k_proj.weight`
                所有 `num_heads` 个注意力头的键投影。
                形状为 (d_model, d_model)。
                各行的排列顺序对应于形状为 (num_heads, d_k) 的矩阵，
                因此 `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`。
            - `attn.v_proj.weight`
                所有 `num_heads` 个注意力头的值投影。
                形状为 (d_model, d_model)。
                各行的排列顺序对应于形状为 (num_heads, d_v) 的矩阵，
                因此 `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`。
            - `attn.output_proj.weight`
                多头自注意力输出投影的权重。
                形状为 (d_model, d_model)。
            - `ln1.weight`
                Transformer 块中应用的第一个 RMSNorm
                的仿射变换权重。
                形状为 (d_model,)。
            - `ffn.w1.weight`
                FFN 中第一个线性变换的权重。
                形状为 (d_ff, d_model)。
            - `ffn.w2.weight`
                FFN 中第二个线性变换的权重。
                形状为 (d_model, d_ff)。
            - `ffn.w3.weight`
                FFN 中第三个线性变换的权重。
                形状为 (d_ff, d_model)。
            - `ln2.weight`
                Transformer 块中应用的第二个 RMSNorm
                的仿射变换权重。
                形状为 (d_model,)。
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            用于运行你的实现的张量。

    返回:
        Float[Tensor, "batch sequence_length d_model"] 在输入特征上使用 RoPE
        运行 Transformer 块所得到的输出张量。
    """
    from lm_basics.model import TransformerBlock

    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        theta=theta,
    )
    block.load_state_dict(weights)
    return block(in_features)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """给定 Transformer 语言模型的权重和输入索引，
    返回在输入索引上运行一次前向传播所得到的输出。

    本函数应使用 RoPE。

    参数:
        vocab_size (int): 需要预测的输出词表中唯一项的数量。
        context_length (int): 一次处理的最大 token 数。
        d_model (int): 模型嵌入和子层输出的维度。
        num_layers (int): 使用的 Transformer 层数。
        num_heads (int): 多头注意力中使用的头数。`d_model` 必须能被 `num_heads` 整除。
        d_ff (int): 前馈网络内层的维度（第 3.3 节）。
        rope_theta (float): RoPE 的 $\\Theta$ 参数。
        weights (dict[str, Tensor]):
            我们的参考实现的 state dict。{num_layers} 指的是
            `0` 到 `num_layers - 1` 之间的整数（层索引）。
            该字典的键包括：
            - `token_embeddings.weight`
                Token 嵌入矩阵。形状为 (vocab_size, d_model)。
            - `layers.{num_layers}.attn.q_proj.weight`
                所有 `num_heads` 个注意力头的查询投影。
                形状为 (num_heads * (d_model / num_heads), d_model)。
                各行的排列顺序对应于形状为 (num_heads, d_k) 的矩阵，
                因此 `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`。
            - `layers.{num_layers}.attn.k_proj.weight`
                所有 `num_heads` 个注意力头的键投影。
                形状为 (num_heads * (d_model / num_heads), d_model)。
                各行的排列顺序对应于形状为 (num_heads, d_k) 的矩阵，
                因此 `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`。
            - `layers.{num_layers}.attn.v_proj.weight`
                所有 `num_heads` 个注意力头的值投影。
                形状为 (num_heads * (d_model / num_heads), d_model)。
                各行的排列顺序对应于形状为 (num_heads, d_v) 的矩阵，
                因此 `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`。
            - `layers.{num_layers}.attn.output_proj.weight`
                多头自注意力输出投影的权重。
                形状为 ((d_model / num_heads) * num_heads, d_model)。
            - `layers.{num_layers}.ln1.weight`
                Transformer 块中应用的第一个 RMSNorm
                的仿射变换权重。
                形状为 (d_model,)。
            - `layers.{num_layers}.ffn.w1.weight`
                FFN 中第一个线性变换的权重。
                形状为 (d_ff, d_model)。
            - `layers.{num_layers}.ffn.w2.weight`
                FFN 中第二个线性变换的权重。
                形状为 (d_model, d_ff)。
            - `layers.{num_layers}.ffn.w3.weight`
                FFN 中第三个线性变换的权重。
                形状为 (d_ff, d_model)。
            - `layers.{num_layers}.ln2.weight`
                Transformer 块中应用的第二个 RMSNorm
                的仿射变换权重。
                形状为 (d_model,)。
            - `ln_final.weight`
                应用于最终 Transformer 块输出的 RMSNorm
                的仿射变换权重。
                形状为 (d_model, )。
            - `lm_head.weight`
                语言模型输出嵌入的权重。
                形状为 (vocab_size, d_model)。
        in_indices (Int[Tensor, "batch_size sequence_length"]) 要在语言模型上运行的输入索引张量。形状为 (batch_size, sequence_length)，
            其中 `sequence_length` 至多为 `context_length`。

    返回:
        Float[Tensor, "batch_size sequence_length vocab_size"]: 包含每个 token 预测的
        未归一化下一词分布的张量。
    """
    from lm_basics.model import TransformerLM

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    )
    model.load_state_dict(weights)
    return model(in_indices)


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """给定 RMSNorm 仿射变换的权重，
    返回在输入特征上运行 RMSNorm 所得到的输出。

    参数:
        d_model (int): RMSNorm 输入的维度。
        eps: (float): 为数值稳定性而加到分母上的值。
        weights (Float[Tensor, "d_model"]): RMSNorm 权重。
        in_features (Float[Tensor, "... d_model"]): 要运行 RMSNorm 的输入特征。可以有任意数量的前导
            维度。

    返回:
        Float[Tensor,"... d_model"]: 与 `in_features` 形状相同的张量，包含对 `in_features`
        运行 RMSNorm 的输出。
    """
    from lm_basics.model import RMSNorm

    rmsnorm = RMSNorm(d_model, eps=eps)
    rmsnorm.weight.data = weights
    return rmsnorm(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """给定输入张量，返回对每个元素应用 SiLU
    所得到的输出。

    参数:
        in_features(Float[Tensor, "..."]): 要运行 SiLU 的输入特征。形状任意。

    返回:
        Float[Tensor,"..."]: 与 `in_features` 形状相同的张量，包含对每个元素
        应用 SiLU 的输出。
    """
    from lm_basics.nn_utils import silu

    return silu(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    给定一个数据集（一维整数 numpy 数组）以及期望的 batch size 和
    上下文长度，从数据集中采样语言建模输入序列及其对应的
    标签。

    参数:
        dataset (np.array): 数据集中整数 token ID 的一维 numpy 数组。
        batch_size (int): 期望采样的 batch size。
        context_length (int): 每个采样样本的期望上下文长度。
        device (str): PyTorch 设备字符串（例如 'cpu' 或 'cuda:0'），指示将采样的
            输入序列和标签放置到哪个设备上。

    返回:
        形状为 (batch_size, context_length) 的 torch.LongTensor 元组。元组的第一个元素
        是采样的输入序列，第二个元素是对应的语言建模标签。
    """
    from lm_basics.data import get_batch

    return get_batch(dataset, batch_size, context_length, device)


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    给定输入张量，返回对输入的指定 `dim` 应用 softmax
    所得到的输出。

    参数:
        in_features (Float[Tensor, "..."]): 要应用 softmax 的输入特征。形状任意。
        dim (int): `in_features` 中要应用 softmax 的维度。

    返回:
        Float[Tensor, "..."]: 与 `in_features` 形状相同的张量，包含沿指定 `dim`
        归一化的 softmax 输出。
    """
    from lm_basics.nn_utils import softmax

    return softmax(in_features, dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """给定输入张量和目标张量，计算各样本上的平均交叉熵
    损失。

    参数:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] 是
            第 i 个样本第 j 类的未归一化 logit。
        targets (Int[Tensor, "batch_size"]): 形状为 (batch_size,) 的张量，包含正确类别的索引。
            每个值必须介于 0 和 `num_classes - 1` 之间。

    返回:
        Float[Tensor, ""]: 各样本上的平均交叉熵损失。
    """
    from lm_basics.nn_utils import cross_entropy

    return cross_entropy(inputs, targets)


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """给定一组参数，将其组合梯度裁剪到 l2 范数至多为 max_l2_norm。

    参数:
        parameters (Iterable[torch.nn.Parameter]): 可训练参数的集合。
        max_l2_norm (float): 一个正值，表示最大 l2 范数。

    参数的梯度（parameter.grad）应当被原地修改。
    """
    from lm_basics.nn_utils import clip_gradients

    clip_gradients(parameters, max_l2_norm)


def get_adamw_cls() -> Any:
    """
    返回一个实现 AdamW 的 torch.optim.Optimizer。
    """
    from lm_basics.optimizer import AdamW

    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    给定余弦学习率衰减调度（带线性预热）的参数和一个迭代轮数，
    返回指定调度下给定迭代轮次的学习率。

    参数:
        it (int): 要获取学习率的迭代轮数。
        max_learning_rate (float): alpha_max，余弦学习率调度
            （带预热）的最大学习率。
        min_learning_rate (float): alpha_min，余弦学习率调度
            （带预热）的最小/最终学习率。
        warmup_iters (int): T_w，线性预热学习率的
            迭代轮数。
        cosine_cycle_iters (int): T_c，余弦退火的迭代轮数。

    返回:
        指定调度下给定迭代轮次的学习率。
    """
    from lm_basics.optimizer import lr_cosine_schedule

    return lr_cosine_schedule(
        it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters
    )


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    给定模型、优化器和迭代轮数，将它们序列化到磁盘。

    参数:
        model (torch.nn.Module): 序列化该模型的状态。
        optimizer (torch.optim.Optimizer): 序列化该优化器的状态。
        iteration (int): 序列化该值，它表示我们已经完成的训练
            迭代轮数。
        out (str | os.PathLike | BinaryIO | IO[bytes]): 用于序列化模型、优化器和迭代轮数的路径或类文件对象。
    """
    from lm_basics.checkpoint import save_checkpoint

    save_checkpoint(model, optimizer, iteration, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    给定一个序列化的检查点（路径或类文件对象），将序列化的状态
    恢复到给定的模型和优化器中。
    返回我们之前序列化在
    检查点中的迭代轮数。

    参数:
        src (str | os.PathLike | BinaryIO | IO[bytes]): 序列化检查点的路径或类文件对象。
        model (torch.nn.Module): 恢复该模型的状态。
        optimizer (torch.optim.Optimizer): 恢复该优化器的状态。
    返回:
        int: 之前序列化的迭代轮数。
    """
    from lm_basics.checkpoint import load_checkpoint

    return load_checkpoint(src, model, optimizer)


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """给定词表、合并列表和特殊 token 列表，
    返回一个使用所提供词表、合并和特殊 token 的 BPE 分词器。

    参数:
        vocab (dict[int, bytes]): 分词器词表，是从 int（词表中的 token ID）
            到 bytes（token 字节）的映射
        merges (list[tuple[bytes, bytes]]): BPE 合并。每个列表项是一个字节元组 (<token1>, <token2>)，
            表示 <token1> 与 <token2> 发生了合并。
            合并按创建顺序排列。
        special_tokens (list[str] | None): 分词器的字符串特殊 token 列表。这些字符串永远不会
            被拆分成多个 token，并且始终作为单个 token 保留。

    返回:
        一个使用所提供词表、合并和特殊 token 的 BPE 分词器。
    """
    from lm_basics.tokenizer import Tokenizer

    return Tokenizer(vocab, merges, special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """给定输入语料的路径，训练一个 BPE 分词器，
    并输出其词表和合并列表。

    参数:
        input_path (str | os.PathLike): BPE 分词器训练数据的路径。
        vocab_size (int): 分词器词表中的总项数（包括特殊 token）。
        special_tokens (list[str]): 要添加到分词器词表中的字符串特殊 token 列表。
            这些字符串永远不会被拆分成多个 token，并且始终
            作为单个 token 保留。如果这些特殊 token 出现在 `input_path` 中，
            它们会被当作普通字符串处理。

    返回:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                训练好的分词器词表，是从 int（词表中的 token ID）
                到 bytes（token 字节）的映射
            merges:
                BPE 合并。每个列表项是一个字节元组 (<token1>, <token2>)，
                表示 <token1> 与 <token2> 发生了合并。
                合并按创建顺序排列。
    """
    from lm_basics.train_bpe import train_bpe

    return train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        **kwargs,
    )
