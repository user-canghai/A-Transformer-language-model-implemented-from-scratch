import math
from collections import Counter

import numpy as np
import pytest

from .adapters import run_get_batch


def test_get_batch():
    dataset = np.arange(0, 100)
    context_length = 7
    batch_size = 32
    device = "cpu"

    # 合理性检查，确保随机采样确实是随机的。
    starting_indices = Counter()
    num_iters = 1000
    for _ in range(num_iters):
        x, y = run_get_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )

        # 确保形状正确
        assert x.shape == (batch_size, context_length)
        assert y.shape == (batch_size, context_length)

        # 确保 y 总是比 x 偏移 1
        np.testing.assert_allclose((x + 1).detach().numpy(), y.detach().numpy())

        starting_indices.update(x[:, 0].tolist())

    # 确保我们永远不会采样到无效的起始索引
    num_possible_starting_indices = len(dataset) - context_length
    assert max(starting_indices) == num_possible_starting_indices - 1
    assert min(starting_indices) == 0
    # 每个起始索引预期出现的次数
    expected_count = (num_iters * batch_size) / num_possible_starting_indices
    standard_deviation = math.sqrt(
        (num_iters * batch_size) * (1 / num_possible_starting_indices) * (1 - (1 / num_possible_starting_indices))
    )
    # 预期结果范围（mu +/- 5sigma）。对于给定的索引，
    # 这应该有 99.99994% 的时间会落在该范围内。
    # 因此，在有 93 个可能起始索引的情况下，
    # 整个测试应该有 99.9944202% 的概率通过
    occurrences_lower_bound = expected_count - 5 * standard_deviation
    occurrences_upper_bound = expected_count + 5 * standard_deviation

    for starting_index, count in starting_indices.items():
        if count < occurrences_lower_bound:
            raise ValueError(
                f"起始索引 {starting_index} 出现了 {count} 次，但预期至少 {occurrences_lower_bound} 次"
            )
        if count > occurrences_upper_bound:
            raise ValueError(
                f"起始索引 {starting_index} 出现了 {count} 次，但预期至多 {occurrences_upper_bound} 次"
            )

    with pytest.raises((RuntimeError, AssertionError)) as excinfo:
        # 我们假设 cuda:99 是一个无效的设备序号。
        # 在这里加上这个测试只是为了确保 device 标志
        # 被正确处理。
        run_get_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device="cuda:99",
        )
        assert "CUDA error" in str(excinfo.value) or "Torch not compiled with CUDA enabled" in str(excinfo.value)
