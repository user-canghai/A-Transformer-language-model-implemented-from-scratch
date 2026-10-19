#!/bin/bash
# TinyStories 基线模型训练（低资源配置）。
set -u
cd "$(dirname "$0")/.."
source /usr/local/Ascend/ascend-toolkit/set_env.sh
.venv-npu/bin/python -m lm_basics.train \
  --train-data data/tinystories/train.npy --valid-data data/tinystories/valid.npy \
  --vocab-size 10000 --context-length 256 \
  --d-model 512 --num-layers 4 --num-heads 16 --d-ff 1344 \
  --batch-size 32 --num-steps 5000 \
  --lr-max 2e-3 --lr-min 1e-4 --warmup-iters 200 --cosine-cycle-iters 5000 \
  --weight-decay 0.01 --beta2 0.95 --grad-clip 1.0 \
  --device npu:1 --log-interval 50 --eval-interval 100 --checkpoint-interval 1000 \
  --out-dir runs/tinystories_base
echo "=== 基线训练完成 ==="
