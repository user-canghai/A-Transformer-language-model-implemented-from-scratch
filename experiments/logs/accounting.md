# Transformer / AdamW 资源核算

## gpt2_small
- d_ff=2048，参数量=162,148,608（0.162B），fp32 显存=0.65 GB
- 前向 FLOPs（context 1024）：2.916e+11（0.3 TFLOPs）
- 占比：attention_projections=19.9%，attention_matrix_products=13.3%，feed_forward=39.8%，lm_head=27.1%

## gpt2_medium
- d_ff=2752，参数量=406,539,264（0.407B），fp32 显存=1.63 GB
- 前向 FLOPs（context 1024）：8.302e+11（0.8 TFLOPs）
- 占比：attention_projections=24.8%，attention_matrix_products=12.4%，feed_forward=50.1%，lm_head=12.7%

## gpt2_large
- d_ff=3392，参数量=833,591,040（0.834B），fp32 显存=3.33 GB
- 前向 FLOPs（context 1024）：1.769e+12（1.8 TFLOPs）
- 占比：attention_projections=27.3%，attention_matrix_products=10.9%，feed_forward=54.3%，lm_head=7.4%

## gpt2_xl
- d_ff=4288，参数量=1,640,452,800（1.640B），fp32 显存=6.56 GB
- 前向 FLOPs（context 1024）：3.517e+12（3.5 TFLOPs）
- 占比：attention_projections=28.6%，attention_matrix_products=9.2%，feed_forward=57.5%，lm_head=4.7%

## GPT-2 XL（context 16,384）
- 前向 FLOPs：1.336e+14（133.6 TFLOPs）
- 占比：attention_projections=12.1%，attention_matrix_products=61.7%，feed_forward=24.2%，lm_head=2.0%

## AdamW 显存（GPT-2 XL，context 1024，float32）
- 公式：4 * [ L*(B*n*(11*d + 4*d_ff) + 2*B*h*n^2) + B*n*d (embeddings) + B*n*d (final norm) + 2*B*n*V (logits + loss) ] + 16*P
- 固定开销（参数+梯度+优化器状态）：26.25 GB
- 每个 batch 元素：17.32 GB
- 80 GB 内最大 batch size：3

## MFU 训练时间估计
- tokens=4.194e+11，训练 FLOPs=4.128e+21
- 按 495 TFLOP/s 的 50% MFU：4633.4 小时（193.1 天）
