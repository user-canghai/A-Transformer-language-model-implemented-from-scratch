#!/usr/bin/env bash
# 训练 OpenWebText 字节级 BPE 分词器（词表 32,000）。
#
# 说明：交付实验中，我们在 OpenWebText 的前 2 GB（按文档边界对齐）上训练。
# 完整 11.9 GB 语料下 merge 阶段的耗时超出本会话预算（设计允许最多 12 小时）；
# 若要复现全量训练，把 --input 换成 data/owt_train.txt 即可。
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python -m lm_basics.train_bpe \
  --input data/owt_train_2gb.txt \
  --vocab-size 32000 \
  --special-tokens '["<|endoftext|>"]' \
  --output-dir data/owt_tokenizer \
  --num-processes 32 --profile
