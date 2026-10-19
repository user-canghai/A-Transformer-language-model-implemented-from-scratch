"""语言模型训练用的数据加载工具。"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """采样一批 (输入, 下一个 token 目标) 序列。

    起始下标从 [0, len(dataset) - context_length) 中均匀抽取，
    从而保证每个样本都是 token 流中合法的连续窗口。
    """
    num_windows = len(dataset) - context_length
    if num_windows <= 0:
        raise ValueError(f"长度为 {len(dataset)} 的 dataset 对 context_length={context_length} 来说太短")
    starts = np.random.randint(0, num_windows, size=(batch_size,))
    offsets = np.arange(context_length)
    indices = starts[:, None] + offsets[None, :]
    inputs = np.asarray(dataset[indices], dtype=np.int64)
    targets = np.asarray(dataset[indices + 1], dtype=np.int64)
    return (
        torch.from_numpy(inputs).to(device=device, non_blocking=True),
        torch.from_numpy(targets).to(device=device, non_blocking=True),
    )


def load_dataset(
    path: str,
    dtype: str = "uint16",
    mmap: bool = True,
) -> npt.NDArray:
    """加载序列化的 token 数组，可选以只读 memmap 方式映射。"""
    np_dtype = np.dtype(dtype)
    if path.endswith(".npy"):
        return np.load(path, mmap_mode="r") if mmap else np.load(path)
    return np.memmap(path, dtype=np_dtype, mode="r") if mmap else np.fromfile(path, dtype=np_dtype)
