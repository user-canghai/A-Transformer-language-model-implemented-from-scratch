#!/bin/bash
# 架构消融：去 RMSNorm、post-norm、NoPE、SwiGLU vs SiLU。
set -u
cd "$(dirname "$0")/.."
source /usr/local/Ascend/ascend-toolkit/set_env.sh
COMMON="--train-data data/tinystories/train.npy --valid-data data/tinystories/valid.npy \
  --vocab-size 10000 --context-length 256 --d-model 512 --num-layers 4 --num-heads 16 --d-ff 1344 \
  --batch-size 32 --num-steps 1200 --lr-min 1e-5 --warmup-iters 100 \
  --cosine-cycle-iters 1200 --weight-decay 0.01 --beta2 0.95 --grad-clip 1.0 \
  --device npu:1 --log-interval 50 --eval-interval 100 --eval-batches 20 --checkpoint-interval 0"
run () {
  name=$1; shift
  echo "=== 消融实验：$name ==="
  .venv-npu/bin/python -m lm_basics.train $COMMON --out-dir runs/ablation_$name "$@" 2>&1 | tail -2
}
run baseline --lr-max 2e-3
run no_rmsnorm_lr2e-3 --lr-max 2e-3 --no-use-rmsnorm
run no_rmsnorm_lr3e-4 --lr-max 3e-4 --no-use-rmsnorm
run post_norm --lr-max 2e-3 --norm post
run nope --lr-max 2e-3 --no-use-rope
run silu_ffn --lr-max 2e-3 --ffn silu --d-ff 2048
echo "=== 消融实验完成 ==="
