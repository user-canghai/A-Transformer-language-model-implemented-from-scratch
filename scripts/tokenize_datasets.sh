#!/usr/bin/env bash
# 把 TinyStories / OpenWebText 语料分词为 uint16 NumPy 数组。
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/tinystories data/owt
.venv/bin/python -m lm_basics.tokenize \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --vocab data/tinystories_tokenizer/vocab.json \
  --merges data/tinystories_tokenizer/merges.txt \
  --special-tokens '["<|endoftext|>"]' \
  --output data/tinystories/train.npy --num-processes 32
.venv/bin/python -m lm_basics.tokenize \
  --input data/TinyStoriesV2-GPT4-valid.txt \
  --vocab data/tinystories_tokenizer/vocab.json \
  --merges data/tinystories_tokenizer/merges.txt \
  --special-tokens '["<|endoftext|>"]' \
  --output data/tinystories/valid.npy --num-processes 8
if [ -f data/owt_train_700mb.txt ]; then
  .venv/bin/python -m lm_basics.tokenize \
    --input data/owt_train_700mb.txt \
    --vocab data/owt_tokenizer/vocab.json \
    --merges data/owt_tokenizer/merges.txt \
    --special-tokens '["<|endoftext|>"]' \
    --output data/owt/train.npy --num-processes 32
  .venv/bin/python -m lm_basics.tokenize \
    --input data/owt_valid_30mb.txt \
    --vocab data/owt_tokenizer/vocab.json \
    --merges data/owt_tokenizer/merges.txt \
    --special-tokens '["<|endoftext|>"]' \
    --output data/owt/valid.npy --num-processes 8
fi
