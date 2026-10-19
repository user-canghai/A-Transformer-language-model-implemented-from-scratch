#!/bin/bash
# 批量大小扫描：固定 token 预算 32*1200*256 ≈ 9.83M tokens。
set -u
cd "$(dirname "$0")/.."
source /usr/local/Ascend/ascend-toolkit/set_env.sh
LR=2e-3
for bs in 8 32 64 128; do
  steps=$(( (32 * 1200) / bs ))
  echo "=== 批量大小扫描：bs=$bs steps=$steps ==="
  .venv-npu/bin/python -m lm_basics.train \
    --train-data data/tinystories/train.npy --valid-data data/tinystories/valid.npy \
    --vocab-size 10000 --context-length 256 --d-model 512 --num-layers 4 --num-heads 16 --d-ff 1344 \
    --batch-size $bs --num-steps $steps \
    --lr-max $LR --lr-min 1e-5 --warmup-iters 100 --cosine-cycle-iters $steps \
    --weight-decay 0.01 --beta2 0.95 --grad-clip 1.0 \
    --device npu:0 --log-interval 50 --eval-interval 100 --eval-batches 20 --checkpoint-interval 0 \
    --out-dir runs/batch_$bs 2>&1 | tail -3
done
echo "=== 批量大小扫描完成 ==="
