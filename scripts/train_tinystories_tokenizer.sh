#!/usr/bin/env bash
# 训练 TinyStories 字节级 BPE 分词器（词表 10,000）。
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python -m lm_basics.train_bpe \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --vocab-size 10000 \
  --special-tokens '["<|endoftext|>"]' \
  --output-dir data/tinystories_tokenizer \
  --num-processes 32 --profile
