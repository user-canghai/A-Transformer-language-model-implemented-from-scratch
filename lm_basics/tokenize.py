"""把文本语料分词为 uint16 NumPy 数组（讲义 2.7d 节）。

支持按特殊 token 对齐的字节区间并行编码，从而能快速分词大型语料
（TinyStories、OpenWebText）。

用法：
    python -m lm_basics.tokenize --input data/TinyStoriesV2-GPT4-train.txt \
        --vocab data/tinystories/vocab.json --merges data/tinystories/merges.txt \
        --special-tokens '["<|endoftext|>"]' --output data/tiny/train.npy --num-processes 32
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os

import numpy as np

from .tokenizer import Tokenizer

_WORKER_TOKENIZER: Tokenizer | None = None
_WORKER_SPECIAL_TOKENS: list[str] = []


def _init_worker(vocab_path: str, merges_path: str, special_tokens: list[str]) -> None:
    global _WORKER_TOKENIZER, _WORKER_SPECIAL_TOKENS
    _WORKER_TOKENIZER = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
    _WORKER_SPECIAL_TOKENS = special_tokens


def _encode_range(job: tuple[str, int, int, str]) -> np.ndarray:
    path, start, end, dtype = job
    assert _WORKER_TOKENIZER is not None
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8")
    return np.asarray(_WORKER_TOKENIZER.encode(text), dtype=dtype)


def _find_chunk_boundaries(path: str, num_chunks: int, special_tokens: list[str]) -> list[int]:
    """把文件切分为若干字节区间，每个区间的起点紧跟在特殊 token 之后。"""
    file_size = os.path.getsize(path)
    markers = [token.encode("utf-8") for token in special_tokens] or [b"\n"]
    boundaries = [0]
    window = 1 << 20
    with open(path, "rb") as f:
        for index in range(1, num_chunks):
            target = file_size * index // num_chunks
            boundary = None
            search_at = target
            while search_at < file_size:
                f.seek(search_at)
                data = f.read(window)
                if not data:
                    break
                best = None
                for marker in markers:
                    position = data.find(marker)
                    if position >= 0 and (best is None or position < best[0]):
                        best = (position, len(marker))
                if best is not None:
                    boundary = search_at + best[0] + best[1]
                    break
                search_at += len(data)
            boundaries.append(boundary if boundary is not None else file_size)
    boundaries.append(file_size)
    cleaned = [boundaries[0]]
    for boundary in boundaries[1:]:
        if boundary > cleaned[-1]:
            cleaned.append(boundary)
    return cleaned


def _write_npy(arrays: list[np.ndarray], output_path: str, dtype: str) -> int:
    total = sum(len(array) for array in arrays)
    output = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.dtype(dtype), shape=(total,))
    offset = 0
    for array in arrays:
        output[offset : offset + len(array)] = array
        offset += len(array)
    output.flush()
    del output
    return total


def tokenize_file(
    input_path: str,
    output_path: str,
    vocab_path: str,
    merges_path: str,
    special_tokens: list[str],
    dtype: str = "uint16",
    num_processes: int | None = None,
    quiet: bool = False,
) -> int:
    """对 input_path 做流式或多进程编码，并把 token id 保存为 .npy。"""
    num_processes = num_processes or 1
    file_size = os.path.getsize(input_path)
    if num_processes > 1 and file_size > 1 << 26:
        boundaries = _find_chunk_boundaries(input_path, num_processes, special_tokens)
        jobs = [(input_path, start, end, dtype) for start, end in itertools.pairwise(boundaries)]
        if not quiet:
            print(f"正在用 {num_processes} 个进程分词 {input_path}（共 {len(jobs)} 个分块）")
        with mp.Pool(
            num_processes,
            initializer=_init_worker,
            initargs=(vocab_path, merges_path, special_tokens),
        ) as pool:
            arrays = pool.map(_encode_range, jobs)
        token_count = _write_npy(arrays, output_path, dtype)
        if not quiet:
            print(f"已保存 {token_count:,} 个 token 到 {output_path}")
        return token_count

    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
    tmp_path = output_path + ".bin"
    token_count = 0
    buffer: list[int] = []
    with open(input_path, encoding="utf-8") as fin, open(tmp_path, "wb") as fout:
        for token_id in tokenizer.encode_iterable(fin):
            buffer.append(token_id)
            if len(buffer) >= 1_000_000:
                np.asarray(buffer, dtype=dtype).tofile(fout)
                token_count += len(buffer)
                buffer = []
                if not quiet:
                    print(f"\r已处理 {token_count:,} 个 token", end="", flush=True)
        if buffer:
            np.asarray(buffer, dtype=dtype).tofile(fout)
            token_count += len(buffer)
    output = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.dtype(dtype), shape=(token_count,))
    chunk = 50_000_000
    with open(tmp_path, "rb") as fin:
        written = 0
        while written < token_count:
            n = min(chunk, token_count - written)
            output[written : written + n] = np.fromfile(fin, dtype=dtype, count=n)
            written += n
    output.flush()
    del output
    os.remove(tmp_path)
    if not quiet:
        print(f"\r已保存 {token_count:,} 个 token 到 {output_path}")
    return token_count


def main() -> None:
    parser = argparse.ArgumentParser(description="把语料分词为 uint16 .npy 数组。")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--merges", required=True)
    parser.add_argument("--special-tokens", default='["<|endoftext|>"]')
    parser.add_argument("--dtype", default="uint16")
    parser.add_argument("--num-processes", type=int, default=1)
    args = parser.parse_args()
    tokenize_file(
        args.input,
        args.output,
        args.vocab,
        args.merges,
        json.loads(args.special_tokens),
        dtype=args.dtype,
        num_processes=args.num_processes,
    )


if __name__ == "__main__":
    main()
