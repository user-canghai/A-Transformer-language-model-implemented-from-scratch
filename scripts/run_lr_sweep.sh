#!/bin/bash
# TinyStories 语言模型的学习率扫描。
set -u
cd "$(dirname "$0")/.."
source /usr/local/Ascend/ascend-toolkit/set_env.sh
for lr in 3e-4 1e-3 2e-3 3e-3 6e-3 1e-2 3e-2; do
  echo "=== 学习率扫描：lr=$lr ==="
  .venv-npu/bin/python -m lm_basics.train \
    --train-data data/tinystories/train.npy --valid-data data/tinystories/valid.npy \
    --vocab-size 10000 --context-length 256 --d-model 512 --num-layers 4 --num-heads 16 --d-ff 1344 \
    --batch-size 32 --num-steps 1000 \
    --lr-max $lr --lr-min 1e-5 --warmup-iters 100 --cosine-cycle-iters 1000 \
    --weight-decay 0.01 --beta2 0.95 --grad-clip 1.0 \
    --device npu --log-interval 50 --eval-interval 100 --eval-batches 20 --checkpoint-interval 0 \
    --out-dir runs/lr_$lr 2>&1 | tail -6
done
echo "=== 学习率扫描完成 ==="
