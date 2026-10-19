"""字节级 BPE tokenizer 训练（讲义 2.4 与 2.5 节）。

从零实现该算法：GPT-2 正则 pre-token 切分（可选地按特殊 token 对齐的字节区间并行）、
用 256 个字节值初始化的词表，以及增量式 pair 计数缓存，使每次 merge 只需处理
包含被合并 pair 的 pre-token。
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import multiprocessing as mp
import os
import sys
import time
from collections import Counter, defaultdict

import regex as re

from .tokenizer import save_vocab_merges

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _pretoken_counts(text: str, special_tokens: list[str]) -> Counter:
    """统计一个文本块内各 pre-token 的出现次数，并按特殊 token 切分。"""
    counts: Counter = Counter()
    if special_tokens:
        splitter = "|".join(re.escape(token) for token in sorted(special_tokens, key=len, reverse=True))
        segments = re.split(splitter, text)
    else:
        segments = [text]
    for segment in segments:
        if not segment:
            continue
        for match in re.finditer(PAT, segment):
            counts[tuple(match.group(0).encode("utf-8"))] += 1
    return counts


def _worker(job: tuple[str, int, int, list[str]]) -> Counter:
    path, start, end, special_tokens = job
    with open(path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)
    return _pretoken_counts(chunk.decode("utf-8"), special_tokens)


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
    # 去掉空区间与重复边界，同时保持原有顺序。
    cleaned = [boundaries[0]]
    for boundary in boundaries[1:]:
        if boundary > cleaned[-1]:
            cleaned.append(boundary)
    return cleaned


class _ReversedPair:
    """堆包装器：让字典序更大的 pair 先弹出。"""

    __slots__ = ("pair", "pair_bytes")

    def __init__(self, pair: tuple[int, int], pair_bytes: tuple[bytes, bytes]) -> None:
        self.pair = pair
        self.pair_bytes = pair_bytes

    def __lt__(self, other: _ReversedPair) -> bool:
        return self.pair_bytes > other.pair_bytes

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _ReversedPair) and self.pair_bytes == other.pair_bytes


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int | None = None,
    profile: bool = False,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """训练字节级 BPE tokenizer，返回 (vocab, merges)。"""
    start_time = time.time()
    input_path = os.fspath(input_path)
    special_tokens = list(special_tokens or [])
    num_merges = vocab_size - 256 - len(special_tokens)
    if num_merges < 0:
        raise ValueError(f"vocab_size={vocab_size} 小于 256 个字节 token 加 {len(special_tokens)} 个特殊 token 的下限")

    file_size = os.path.getsize(input_path)
    if num_processes is None:
        num_processes = min(os.cpu_count() or 1, 32)
    if num_processes > 1 and file_size > 1 << 20:
        boundaries = _find_chunk_boundaries(input_path, num_processes, special_tokens)
        ranges = list(itertools.pairwise(boundaries))
        with mp.Pool(num_processes) as pool:
            partial_counts = pool.map(_worker, [(input_path, s, e, special_tokens) for s, e in ranges])
        word_counts: Counter = Counter()
        for counts in partial_counts:
            word_counts.update(counts)
    else:
        with open(input_path, encoding="utf-8") as f:
            word_counts = _pretoken_counts(f.read(), special_tokens)

    pretokenize_time = time.time() - start_time

    vocab: dict[int, bytes] = {byte: bytes([byte]) for byte in range(256)}
    next_id = 256

    seqs: list[list[int]] = [list(word) for word in word_counts]
    freqs: list[int] = list(word_counts.values())
    pair_counts: dict[tuple[int, int], int] = {}
    pair_words: dict[tuple[int, int], set[int]] = defaultdict(set)
    for word_index, seq in enumerate(seqs):
        frequency = freqs[word_index]
        for position in range(len(seq) - 1):
            pair = (seq[position], seq[position + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + frequency
            pair_words[pair].add(word_index)

    heap: list[tuple[int, _ReversedPair]] = [
        (-count, _ReversedPair(pair, (vocab[pair[0]], vocab[pair[1]]))) for pair, count in pair_counts.items()
    ]
    heapq.heapify(heap)

    def add_pair(pair: tuple[int, int], word_index: int, frequency: int) -> None:
        new_count = pair_counts.get(pair, 0) + frequency
        pair_counts[pair] = new_count
        pair_words[pair].add(word_index)
        heapq.heappush(heap, (-new_count, _ReversedPair(pair, (vocab[pair[0]], vocab[pair[1]]))))

    merges: list[tuple[bytes, bytes]] = []
    for _ in range(num_merges):
        pair = None
        while heap:
            negative_count, wrapped = heapq.heappop(heap)
            candidate = wrapped.pair
            current_count = pair_counts.get(candidate, 0)
            if current_count <= 0:
                continue
            if current_count != -negative_count:
                # 过期条目：用最新计数重新入堆，保证总能找到出现频率最高的 pair。
                heapq.heappush(heap, (-current_count, wrapped))
                continue
            pair = candidate
            break
        if pair is None:
            break
        first, second = pair
        new_id = next_id
        next_id += 1
        vocab[new_id] = vocab[first] + vocab[second]
        merges.append((vocab[first], vocab[second]))

        affected = pair_words.pop(pair, set())
        for word_index in affected:
            seq = seqs[word_index]
            if len(seq) < 2:
                # 单元素 pre-token 不可能包含该 pair，直接跳过。
                continue
            if pair_counts.get(pair, 0) == 0:
                # 该 pair 的所有出现都已合并完毕。
                break
            if not any(seq[i] == first and seq[i + 1] == second for i in range(len(seq) - 1)):
                continue
            frequency = freqs[word_index]
            merged: list[int] = []
            position = 0
            length = len(seq)
            while position < length:
                if position < length - 1 and seq[position] == first and seq[position + 1] == second:
                    previous = merged[-1] if merged else None
                    following = seq[position + 2] if position + 2 < length else None
                    pair_counts[(first, second)] -= frequency
                    if pair_counts[(first, second)] <= 0:
                        del pair_counts[(first, second)]
                        pair_words.pop((first, second), None)
                    if previous is not None:
                        decrement = pair_counts.get((previous, first), 0) - frequency
                        if decrement > 0:
                            pair_counts[(previous, first)] = decrement
                        else:
                            pair_counts.pop((previous, first), None)
                            pair_words.pop((previous, first), None)
                        add_pair((previous, new_id), word_index, frequency)
                    if following is not None:
                        decrement = pair_counts.get((second, following), 0) - frequency
                        if decrement > 0:
                            pair_counts[(second, following)] = decrement
                        else:
                            pair_counts.pop((second, following), None)
                            pair_words.pop((second, following), None)
                        add_pair((new_id, following), word_index, frequency)
                    merged.append(new_id)
                    position += 2
                else:
                    merged.append(seq[position])
                    position += 1
            seqs[word_index] = merged

    special_offset = next_id
    for offset, token in enumerate(special_tokens):
        vocab[special_offset + offset] = token.encode("utf-8")

    if profile:
        import resource

        merge_time = time.time() - start_time - pretokenize_time
        peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        longest = max(vocab.values(), key=len)
        print(
            f"[profile] 预分词={pretokenize_time:.1f}s 合并={merge_time:.1f}s "
            f"总计={time.time() - start_time:.1f}s 峰值内存={peak_mb:.0f}MB "
            f"唯一 pre-token 数={len(word_counts)} 最长 token={longest!r}",
            file=sys.stderr,
        )
    return vocab, merges


def main() -> None:
    parser = argparse.ArgumentParser(description="训练一个字节级 BPE tokenizer。")
    parser.add_argument("--input", required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--special-tokens", default='["<|endoftext|>"]')
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-processes", type=int, default=None)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    import json

    special_tokens = json.loads(args.special_tokens)
    os.makedirs(args.output_dir, exist_ok=True)
    start = time.time()
    vocab, merges = train_bpe(args.input, args.vocab_size, special_tokens, args.num_processes, profile=args.profile)
    elapsed = time.time() - start
    save_vocab_merges(
        vocab,
        merges,
        os.path.join(args.output_dir, "vocab.json"),
        os.path.join(args.output_dir, "merges.txt"),
    )
    longest = max(vocab.values(), key=len)
    print(f"BPE 训练完成：vocab={len(vocab)}，merges={len(merges)}，耗时 {elapsed:.1f}s")
    print(f"最长 token：{longest!r}（{len(longest)} 字节）")
    print(f"特殊 token：{special_tokens}")


if __name__ == "__main__":
    main()
