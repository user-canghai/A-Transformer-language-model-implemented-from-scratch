#!/usr/bin/env bash
# 从训练好的 TinyStories checkpoint 采样文本。
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${1:-runs/tinystories_base}"
.venv/bin/python -m lm_basics.generate \
  --checkpoint "${RUN}/checkpoint_final.pt" --config "${RUN}/config.json" \
  --vocab data/tinystories_tokenizer/vocab.json \
  --merges data/tinystories_tokenizer/merges.txt \
  --prompt "Once upon a time" --max-new-tokens 256 --temperature 0.8 --top-p 0.95
