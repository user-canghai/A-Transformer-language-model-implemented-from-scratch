import os
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor


class DEFAULT:
    pass


def _canonicalize_array[A: (np.ndarray, Tensor)](arr: A) -> np.ndarray:
    if isinstance(arr, Tensor):
        arr = arr.detach().cpu().numpy()
    return arr


class NumpySnapshot[A: (np.ndarray, Tensor)]:
    """使用 .npz 格式的 NumPy 数组快照测试工具。"""

    def __init__(
        self,
        snapshot_dir: str = "tests/_snapshots",
        default_force_update: bool = False,
        always_match_exact: bool = False,
        default_test_name: str | None = None,
    ):
        self.snapshot_dir = Path(snapshot_dir)
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.default_force_update = default_force_update
        self.always_match_exact = always_match_exact
        self.default_test_name = default_test_name

    def _get_snapshot_path(self, test_name: str) -> Path:
        """获取快照文件的路径。"""
        return self.snapshot_dir / f"{test_name}.npz"

    def assert_match(
        self,
        actual: A | dict[str, A],
        rtol: float = 1e-4,
        atol: float = 1e-2,
        test_name: str | type[DEFAULT] = DEFAULT,
        force_update: bool | type[DEFAULT] = DEFAULT,
    ):
        """
        断言实际数组与快照匹配。

        参数:
            actual: 单个 NumPy 数组或命名数组的字典
            test_name: 测试名称（用于快照文件）
            update: 如果为 True，则更新快照而不是进行比较
        """
        if force_update is DEFAULT:
            force_update = self.default_force_update
        if self.always_match_exact:
            rtol = atol = 0
        if test_name is DEFAULT:
            assert self.default_test_name is not None, "必须提供测试名称或将其设为默认值"
            test_name = self.default_test_name

        snapshot_path = self._get_snapshot_path(test_name)

        # 将单个数组转换为字典，以便统一处理
        arrays_dict = actual if isinstance(actual, dict) else {"array": actual}
        arrays_dict = {k: _canonicalize_array(v) for k, v in arrays_dict.items()}

        # 加载快照
        expected_arrays = dict(np.load(snapshot_path))

        # 验证所有预期的数组都存在
        missing_keys = set(arrays_dict.keys()) - set(expected_arrays.keys())
        if missing_keys:
            raise AssertionError(f"快照中未找到 {test_name} 的键 {missing_keys}")

        # 验证所有实际数组都是预期的
        extra_keys = set(expected_arrays.keys()) - set(arrays_dict.keys())
        if extra_keys:
            raise AssertionError(f"{test_name} 的快照包含多余的键 {extra_keys}")

        # 比较所有数组
        for key in arrays_dict:
            np.testing.assert_allclose(
                _canonicalize_array(arrays_dict[key]),
                expected_arrays[key],
                rtol=rtol,
                atol=atol,
                err_msg=f"数组 '{key}' 与 {test_name} 的快照不匹配",
            )


class Snapshot[A: (np.ndarray, Tensor)]:
    def __init__(
        self,
        snapshot_dir: str = "tests/_snapshots",
        default_force_update: bool = False,
        default_test_name: str | None = None,
    ):
        """
        用于任意数据类型的快照，保存为 pickle 文件。
        """
        self.snapshot_dir = Path(snapshot_dir)
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.default_force_update = default_force_update
        self.default_test_name = default_test_name

    def _get_snapshot_path(self, test_name: str) -> Path:
        return self.snapshot_dir / f"{test_name}.pkl"

    def assert_match(
        self,
        actual: A | dict[str, A],
        test_name: str | type[DEFAULT] = DEFAULT,
        force_update: bool | type[DEFAULT] = DEFAULT,
    ):
        """
        断言实际数据与快照匹配。
        参数:
            actual: 单个对象或命名对象的字典
            test_name: 测试名称（用于快照文件）
            force_update: 如果为 True，则更新快照而不是进行比较
        """

        if force_update is DEFAULT:
            force_update = self.default_force_update
        if test_name is DEFAULT:
            assert self.default_test_name is not None, "必须提供测试名称或将其设为默认值"
            test_name = self.default_test_name

        snapshot_path = self._get_snapshot_path(test_name)

        # 加载快照
        with open(snapshot_path, "rb") as f:
            expected_data = pickle.load(f)

        if isinstance(actual, dict):
            for key in actual:
                if key not in expected_data:
                    raise AssertionError(f"快照中未找到 {test_name} 的键 '{key}'")
                assert actual[key] == expected_data[key], (
                    f"键 '{key}' 的数据与 {test_name} 的快照不匹配"
                )
        else:
            assert actual == expected_data, f"数据与 {test_name} 的快照不匹配"


@pytest.fixture
def snapshot(request):
    """
    提供快照测试功能的 fixture。

    用法:
        def test_my_function(snapshot):
            result = my_function()
            snapshot.assert_match(result, "my_test_name")
    """
    force_update = False

    # 使用默认设置创建快照处理器
    snapshot_handler = Snapshot(default_force_update=force_update, default_test_name=request.node.name)

    return snapshot_handler


# 可在所有测试中使用的 fixture
@pytest.fixture
def numpy_snapshot(request):
    """
    提供 numpy 快照测试功能的 fixture。

    用法:
        def test_my_function(numpy_snapshot):
            result = my_function()
            numpy_snapshot.assert_match(result, "my_test_name")
    """
    force_update = False

    match_exact = request.config.getoption("--snapshot-exact", default=False)

    # 使用默认设置创建快照处理器
    snapshot = NumpySnapshot(
        default_force_update=force_update, always_match_exact=match_exact, default_test_name=request.node.name
    )

    return snapshot


@pytest.fixture
def ts_state_dict(request):
    import json

    from .common import FIXTURES_PATH

    state_dict = torch.load(FIXTURES_PATH / "ts_tests" / "model.pt", map_location="cpu")
    config = json.load(open(FIXTURES_PATH / "ts_tests" / "model_config.json"))
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    return state_dict, config


# 用于模型 fixture 的模型参数


@pytest.fixture
def n_layers():
    return 3


@pytest.fixture
def vocab_size():
    return 10_000


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def n_queries():
    return 12


@pytest.fixture
def n_keys():
    return 16


@pytest.fixture
def n_heads():
    return 4


@pytest.fixture
def d_head():
    return 16


@pytest.fixture
def d_model(n_heads, d_head):
    return n_heads * d_head


@pytest.fixture
def d_ff():
    return 128


@pytest.fixture
def q(batch_size, n_queries, d_model):
    torch.manual_seed(1)
    return torch.randn(batch_size, n_queries, d_model)


@pytest.fixture
def k(batch_size, n_keys, d_model):
    torch.manual_seed(2)
    return torch.randn(batch_size, n_keys, d_model)


@pytest.fixture
def v(batch_size, n_keys, d_model):
    torch.manual_seed(3)
    return torch.randn(batch_size, n_keys, d_model)


@pytest.fixture
def in_embeddings(batch_size, n_queries, d_model):
    torch.manual_seed(4)
    return torch.randn(batch_size, n_queries, d_model)


@pytest.fixture
def mask(batch_size, n_queries, n_keys):
    torch.manual_seed(5)
    return torch.randn(batch_size, n_queries, n_keys) > 0.5


@pytest.fixture
def in_indices(batch_size, n_queries):
    torch.manual_seed(6)
    return torch.randint(0, 10_000, (batch_size, n_queries))


@pytest.fixture
def theta():
    return 10000.0


@pytest.fixture
def pos_ids(n_queries):
    return torch.arange(0, n_queries)


# # 用法示例：
# def test_single_array(numpy_snapshot):
#     # 一个生成 numpy 数组的示例函数
#     def my_function():
#         return np.array([[1.0, 2.0], [3.0, 4.0001]])

#     result = my_function()

#     # 只需提供结果 - 测试名称会被自动推断
#     numpy_snapshot.assert_match(result)


# def test_multiple_arrays(numpy_snapshot):
#     # 生成多个数组的函数
#     def my_function():
#         return {
#             "weights": np.array([0.1, 0.2, 0.3]),
#             "biases": np.array([0.01, 0.02]),
#             "gradients": np.array([[0.001, 0.002], [0.003, 0.004]])
#         }

#     results = my_function()

#     # 使用显式名称和自定义容差进行测试
#     # custom_snapshot = NumpySnapshot()
#     numpy_snapshot.assert_match(
#         results,
#         "my_special_test",
#         rtol=1e-4,
#         atol=1e-5,
#     )

# def test_state_dict(ts_state_dict):
#     print(ts_state_dict)
