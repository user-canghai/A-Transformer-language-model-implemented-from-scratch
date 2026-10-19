"""分词器实验。

计算压缩率、跨分词器行为、编码吞吐量以及 uint16 词表大小论证，
结果写入 JSON 文件。
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import json
import os
import random
import time

from lm_basics.tokenizer import Tokenizer

SPECIALS = ["<|endoftext|>"]
DATA = "data"


def sample_documents(path: str, count: int = 10, seed: int = 0) -> list[str]:
    """从文本文件中按 <|endoftext|> 切分并随机抽取完整文档。"""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    docs = [d for d in text.split("<|endoftext|>") if d.strip()]
    rng = random.Random(seed)
    return rng.sample(docs, min(count, len(docs)))


def measure_throughput(tokenizer: Tokenizer, path: str, chunks: int = 20) -> float:
    """编码文件的一段前缀，返回每秒处理的字节数。"""
    with open(path, "rb") as f:
        data = f.read(2_000_000 * chunks)
    text = data.decode("utf-8", errors="ignore")
    start = time.time()
    tokenizer.encode(text)
    elapsed = time.time() - start
    return len(data) / elapsed


def compression_ratio(tokenizer: Tokenizer, docs: list[str]) -> dict:
    """计算若干文档的压缩率（字节/词元）。"""
    n_bytes = sum(len(doc.encode("utf-8")) for doc in docs)
    n_tokens = sum(len(tokenizer.encode(doc)) for doc in docs)
    return {"bytes": n_bytes, "tokens": n_tokens, "bytes_per_token": n_bytes / n_tokens}


def main() -> None:
    results: dict = {}
    ts_tok = Tokenizer.from_files(
        os.path.join(DATA, "tinystories_tokenizer", "vocab.json"),
        os.path.join(DATA, "tinystories_tokenizer", "merges.txt"),
        SPECIALS,
    )
    owt_tok_path = os.path.join(DATA, "owt_tokenizer", "vocab.json")
    owt_tok = None
    if os.path.exists(owt_tok_path):
        owt_tok = Tokenizer.from_files(
            owt_tok_path,
            os.path.join(DATA, "owt_tokenizer", "merges.txt"),
            SPECIALS,
        )

    ts_docs = sample_documents(os.path.join(DATA, "TinyStoriesV2-GPT4-train.txt"))
    owt_docs = sample_documents(os.path.join(DATA, "owt_train.txt"))

    # (a) 各自分词器在各自语料上的压缩率
    results["tinystories_tokenizer_on_tinystories"] = compression_ratio(ts_tok, ts_docs)
    if owt_tok is not None:
        results["owt_tokenizer_on_owt"] = compression_ratio(owt_tok, owt_docs)
    # (b) 跨语料使用分词器
    results["tinystories_tokenizer_on_owt"] = compression_ratio(ts_tok, owt_docs)
    if owt_tok is not None:
        results["owt_tokenizer_on_tinystories"] = compression_ratio(owt_tok, ts_docs)

    # (c) 编码吞吐量与 Pile（825 GB）耗时估计
    ts_throughput = measure_throughput(ts_tok, os.path.join(DATA, "TinyStoriesV2-GPT4-train.txt"))
    results["tinystories_bytes_per_second"] = ts_throughput
    results["pile_825gb_hours_tinystories_tokenizer"] = 825e9 / ts_throughput / 3600
    if owt_tok is not None:
        owt_throughput = measure_throughput(owt_tok, os.path.join(DATA, "owt_train.txt"))
        results["owt_bytes_per_second"] = owt_throughput
        results["pile_825gb_hours_owt_tokenizer"] = 825e9 / owt_throughput / 3600

    # (d) 词表大小与 uint16 适用性
    results["vocab_sizes"] = {"tinystories": len(ts_tok.vocab)}
    if owt_tok is not None:
        results["vocab_sizes"]["owt"] = len(owt_tok.vocab)
    results["uint16_ok"] = all(size <= 65536 for size in results["vocab_sizes"].values())

    out_dir = os.path.join("experiments", "logs")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "tokenizer_experiments.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
