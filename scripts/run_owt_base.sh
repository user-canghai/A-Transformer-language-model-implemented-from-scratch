#!/usr/bin/env bash
# OpenWebText 语言模型：与 TinyStories 相同的结构与 token 预算。
set -euo pipefail
cd "$(dirname "$0")/.."
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
.venv-npu/bin/python -m lm_basics.train \
  --train-data data/owt/train.npy --valid-data data/owt/valid.npy \
  --vocab-size 32000 --context-length 256 \
  --d-model 512 --num-layers 4 --num-heads 16 --d-ff 1344 \
  --batch-size 32 --num-steps 5000 \
  --lr-max 1e-3 --lr-min 1e-4 --warmup-iters 200 --cosine-cycle-iters 5000 \
  --weight-decay 0.01 --beta2 0.95 --grad-clip 1.0 \
  --device npu --eval-interval 100 --checkpoint-interval 1000 \
  --out-dir runs/owt_base
