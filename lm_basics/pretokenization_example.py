import itertools
import os
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    将文件切分成可独立计数的多个部分。
    如果边界最终发生重叠，返回的块数可能少于请求的数量。
    """
    assert isinstance(split_special_token, bytes), "必须将特殊 token 表示为字节串"

    # 获取文件的总字节大小
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # 对块边界位置的初始猜测，均匀间隔分布
    # 块从前一个索引开始，不包含最后一个索引
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # 每次预读 4k 字节

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # 从猜测的边界位置开始
        while True:
            mini_chunk = file.read(mini_chunk_size)  # 读取一个小块

            # 如果到达文件末尾，该边界应位于文件末尾
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # 在小块中查找特殊 token
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # 确保所有边界唯一，但数量可能少于 desired_num_chunks
    return sorted(set(chunk_boundaries))


# 用法
with open(..., "rb") as f:
    num_processes = 4
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # 以下是一个串行实现，但你可以通过把每个起始/结束位置对发送给
    # 一组进程来实现并行化。
    for start, end in itertools.pairwise(boundaries):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # 对你的块运行预分词，并保存每个预 token 的计数
