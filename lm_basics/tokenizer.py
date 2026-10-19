"""字节级 BPE tokenizer：编码与解码（讲义 2.6 节）。"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator

import regex as re

# GPT-2 风格的 pre-token 切分正则（来自 tiktoken PR #234），tokenizer 训练与推理共用。
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def bytes_to_unicode() -> dict[int, str]:
    """返回 GPT-2 的字节 -> 可打印 unicode 映射（用于序列化）。"""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


_BYTE_ENCODER = bytes_to_unicode()
_BYTE_DECODER = {char: byte for byte, char in _BYTE_ENCODER.items()}


def _encode_bytes(data: bytes) -> str:
    return "".join(_BYTE_ENCODER[byte] for byte in data)


def _decode_bytes(text: str) -> bytes:
    return bytes(_BYTE_DECODER[char] for char in text)


def save_vocab_merges(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
) -> None:
    """把词表与 merge 列表序列化到磁盘（GPT-2 风格文本格式）。"""
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({str(idx): _encode_bytes(token) for idx, token in vocab.items()}, f, ensure_ascii=False)
    with open(merges_path, "w", encoding="utf-8") as f:
        f.writelines(f"{_encode_bytes(first)} {_encode_bytes(second)}\n" for first, second in merges)


def load_vocab_merges(
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """读取由 save_vocab_merges 写出的词表与 merge 列表。"""
    with open(vocab_path, encoding="utf-8") as f:
        raw_vocab = json.load(f)
    vocab = {int(idx): _decode_bytes(token) for idx, token in raw_vocab.items()}
    merges: list[tuple[bytes, bytes]] = []
    with open(merges_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            first, second = line.split(" ")
            merges.append((_decode_bytes(first), _decode_bytes(second)))
    return vocab, merges


class Tokenizer:
    """字节级 BPE tokenizer，可选支持特殊 token。"""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens or [])
        self._bytes_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}
        next_id = max(self.vocab) + 1 if self.vocab else 0
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self._bytes_to_id:
                self.vocab[next_id] = token_bytes
                self._bytes_to_id[token_bytes] = next_id
                next_id += 1
        self._merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self._cache: dict[bytes, tuple[int, ...]] = {}
        self._special_set = set(self.special_tokens)
        self._pattern = re.compile(PAT)
        if self.special_tokens:
            alternation = "|".join(re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True))
            self._special_pattern = re.compile(f"({alternation})")
            self._max_special_len = max(len(token) for token in self.special_tokens)
        else:
            self._special_pattern = None
            self._max_special_len = 0

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        vocab, merges = load_vocab_merges(vocab_filepath, merges_filepath)
        return cls(vocab, merges, special_tokens)

    def _bpe(self, token: bytes) -> tuple[int, ...]:
        cached = self._cache.get(token)
        if cached is not None:
            return cached
        parts = [bytes([byte]) for byte in token]
        while len(parts) > 1:
            best_rank: int | None = None
            best_index = -1
            for index in range(len(parts) - 1):
                rank = self._merge_ranks.get((parts[index], parts[index + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_index = index
            if best_index < 0:
                break
            parts[best_index : best_index + 2] = [parts[best_index] + parts[best_index + 1]]
        ids = tuple(self._bytes_to_id[part] for part in parts)
        if len(self._cache) < 2_000_000:
            self._cache[token] = ids
        return ids

    def _encode_segment(self, segment: str) -> list[int]:
        """对一个不含特殊 token 的文本片段做 BPE 编码。"""
        ids: list[int] = []
        for match in self._pattern.finditer(segment):
            ids.extend(self._bpe(match.group(0).encode("utf-8")))
        return ids

    def encode(self, text: str) -> list[int]:
        if self._special_pattern is None:
            return self._encode_segment(text)
        ids: list[int] = []
        position = 0
        for match in self._special_pattern.finditer(text):
            if match.start() > position:
                ids.extend(self._encode_segment(text[position : match.start()]))
            ids.append(self._bytes_to_id[match.group(0).encode("utf-8")])
            position = match.end()
        if position < len(text):
            ids.extend(self._encode_segment(text[position:]))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """惰性地编码一个字符串可迭代对象（例如文件句柄）。

        缓冲区只保留末尾的 pre-token（以及可能的部分特殊 token），因此内存占用
        与输入规模无关，而输出与一次性编码整段文本完全一致。
        """
        buffer = ""
        for chunk in iterable:
            buffer += chunk
            safe_end = len(buffer)
            # 扣住可能是特殊 token 开头的后缀，暂不处理。
            for size in range(min(self._max_special_len - 1, len(buffer)), 0, -1):
                suffix = buffer[-size:]
                if any(token.startswith(suffix) for token in self.special_tokens):
                    safe_end = len(buffer) - size
                    break
            # 到最后一个完整特殊 token 为止的内容边界已经完全确定。
            last_special_end = 0
            if self._special_pattern is not None and safe_end > 0:
                for match in self._special_pattern.finditer(buffer, 0, safe_end):
                    last_special_end = match.end()
            if last_special_end > 0:
                yield from self.encode(buffer[:last_special_end])
                buffer = buffer[last_special_end:]
                continue
            segment = buffer[:safe_end]
            matches = list(self._pattern.finditer(segment))
            if matches and matches[-1].end() == len(segment):
                cut = matches[-1].start()
            else:
                cut = len(segment)
            yield from self._encode_segment(segment[:cut])
            buffer = segment[cut:] + buffer[safe_end:]
        if buffer:
            yield from self.encode(buffer)

    def decode(self, ids: list[int]) -> str:
        token_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return token_bytes.decode("utf-8", errors="replace")
