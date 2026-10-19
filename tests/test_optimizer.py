import numpy
import torch

from .adapters import get_adamw_cls, run_get_lr_cosine_schedule


def _optimize(opt_class) -> torch.Tensor:
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 2, bias=False)
    opt = opt_class(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    # 使用 1000 个优化步骤进行测试
    for _ in range(1000):
        opt.zero_grad()
        x = torch.rand(model.in_features)
        y_hat = model(x)
        y = torch.tensor([x[0] + x[1], -x[2]])
        loss = ((y - y_hat) ** 2).sum()
        loss.backward()
        opt.step()
    return model.weight.detach()


def test_adamw(numpy_snapshot):
    """
    我们的参考实现产生的结果与 PyTorch AdamW 略有不同，因为应用权重衰减
    有几种不同的方式，它们在原理上等价，但由于浮点运算行为，
    在实践中会有差异。因此，我们测试所提供实现的结果是否与
    _我们的参考实现_ 的预期结果或 PyTorch AdamW 的结果之一匹配。
    """
    # expected_weights = torch.load(FIXTURES_PATH / "adamw_expected_params.pt")
    pytorch_weights = _optimize(torch.optim.AdamW)
    actual_weights = _optimize(get_adamw_cls())

    # 如果权重与 PyTorch 匹配，可能需要提前返回，因为那也应当是有效的
    matches_pytorch = torch.allclose(actual_weights, pytorch_weights, atol=1e-4)
    if matches_pytorch:
        return

    numpy_snapshot.assert_match(
        actual_weights,
        atol=1e-4,
    )


def test_get_lr_cosine_schedule():
    max_learning_rate = 1
    min_learning_rate = 1 * 0.1
    warmup_iters = 7
    cosine_cycle_iters = 21

    expected_lrs = [
        0,
        0.14285714285714285,
        0.2857142857142857,
        0.42857142857142855,
        0.5714285714285714,
        0.7142857142857143,
        0.8571428571428571,
        1.0,
        0.9887175604818206,
        0.9554359905560885,
        0.9018241671106134,
        0.8305704108364301,
        0.7452476826029011,
        0.6501344202803414,
        0.55,
        0.44986557971965857,
        0.3547523173970989,
        0.26942958916356996,
        0.19817583288938662,
        0.14456400944391146,
        0.11128243951817937,
        0.1,
        0.1,
        0.1,
        0.1,
    ]
    actual_lrs = [
        run_get_lr_cosine_schedule(
            it=it,
            max_learning_rate=max_learning_rate,
            min_learning_rate=min_learning_rate,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=cosine_cycle_iters,
        )
        for it in range(25)
    ]
    numpy.testing.assert_allclose(numpy.array(actual_lrs), numpy.array(expected_lrs))
