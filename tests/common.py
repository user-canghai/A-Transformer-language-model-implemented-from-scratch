from __future__ import annotations

import pathlib
from functools import lru_cache

FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent) / "fixtures"


@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    返回每个可能的字节（0 到 255 的整数）到可打印 Unicode 字符串字符表示
    之间的映射。该函数取自 GPT-2 代码。

    例如，`chr(0)` 是 `\x00`，这是一个不可打印字符：

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    因此，该函数返回一个字典 `d`，其中 `d[0]` 返回 `Ā`。
    视觉上可打印的字节保留其原始字符串表示 [1]。
    例如，`chr(33)` 返回 `!`，因此相应地 `d[33]` 返回 `!`。
    特别注意，空格字符 `chr(32)` 会变成 `d[32]`，它
    返回 'Ġ'。

    对于不可打印字符，该函数会取表示该字符 Unicode 码点的整数
    （由 Python 的 `ord` 函数返回），并将其偏移 256。例如，`ord(" ")` 返回 `32`，
    因此空格字符 ' ' 被偏移到 `256 + 32`。由于 `chr(256 + 32)` 返回 `Ġ`，我们用它作为
    空格的字符串表示。

    该函数可以简化 BPE 实现，并使在合并结果序列化到文件后
    手动检查它们稍微容易一些。
    """
    # 这 188 个整数可以直接使用，因为它们不是空白字符或控制字符。
    # 参见 https://www.ssec.wisc.edu/~tomw/java/unicode.html。
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # 现在获取另外 68 个需要偏移的整数的表示
    # 每个都会被映射为 chr(256 + n)，其中 n 在循环中从 0...67 递增
    # 获取剩余 68 个整数的可打印表示。
    n = 0
    for b in range(2**8):
        if b not in bs:
            # 如果该整数不在我们的视觉可表示字符列表中，
            # 则将其映射到下一个合适的字符（偏移 256）
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d
