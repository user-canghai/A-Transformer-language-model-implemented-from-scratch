"""Transformer 语言模型与 AdamW 的资源核算（Transformer 与 AdamW）。"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import json
import os

# 四种 GPT-2 规模（d_ff 由 d_model 推导）
MODELS = {
    "gpt2_small": {"vocab_size": 50257, "context_length": 1024, "num_layers": 12, "d_model": 768, "num_heads": 12},
    "gpt2_medium": {"vocab_size": 50257, "context_length": 1024, "num_layers": 24, "d_model": 1024, "num_heads": 16},
    "gpt2_large": {"vocab_size": 50257, "context_length": 1024, "num_layers": 36, "d_model": 1280, "num_heads": 20},
    "gpt2_xl": {"vocab_size": 50257, "context_length": 1024, "num_layers": 48, "d_model": 1600, "num_heads": 25},
}


def d_ff_of(d_model: int) -> int:
    """取 8/3 * d_model 最近的 64 的倍数。"""

    return 64 * round((8 * d_model / 3) / 64)


def parameter_count(vocab_size: int, num_layers: int, d_model: int, d_ff: int) -> dict:
    """按本文档的 Transformer 结构（无 bias、绑定/不绑定 LM head）统计参数量。"""
    embedding = vocab_size * d_model
    per_layer = 4 * d_model * d_model + 3 * d_model * d_ff + 2 * d_model
    lm_head = vocab_size * d_model
    final_norm = d_model
    total = embedding + num_layers * per_layer + final_norm + lm_head
    return {
        "token_embedding": embedding,
        "per_layer": per_layer,
        "blocks": num_layers * per_layer,
        "final_norm": final_norm,
        "lm_head": lm_head,
        "total": total,
        "fp32_memory_gb": total * 4 / 1e9,
    }


def forward_flops(vocab_size: int, context_length: int, num_layers: int, d_model: int, d_ff: int) -> dict:
    """一次前向（完整 context_length 个 token）的矩阵乘 FLOPs。"""
    n = context_length
    d = d_model
    qkv = num_layers * n * 3 * 2 * d * d
    attn_scores = num_layers * 2 * n * n * d
    attn_weighted_sum = num_layers * 2 * n * n * d
    out_proj = num_layers * n * 2 * d * d
    ffn = num_layers * n * 3 * 2 * d * d_ff
    lm_head = n * 2 * d * vocab_size
    total = qkv + attn_scores + attn_weighted_sum + out_proj + ffn + lm_head
    return {
        "qkv_projections": qkv,
        "attention_scores": attn_scores,
        "attention_weighted_sum": attn_weighted_sum,
        "output_projection": out_proj,
        "feed_forward": ffn,
        "lm_head": lm_head,
        "total": total,
        "total_tflops": total / 1e12,
        "proportions": {
            "attention_projections": (qkv + out_proj) / total,
            "attention_matrix_products": (attn_scores + attn_weighted_sum) / total,
            "feed_forward": ffn / total,
            "lm_head": lm_head / total,
        },
    }


def adamw_memory(d_model: int, d_ff: int, num_layers: int, num_heads: int, vocab_size: int) -> dict:
    """以 batch size B 和 context n 为变量的激活显存公式（字节）。"""
    return {
        "formula": (
            "4 * [ L*(B*n*(11*d + 4*d_ff) + 2*B*h*n^2) + B*n*d (embeddings) "
            "+ B*n*d (final norm) + 2*B*n*V (logits + loss) ] + 16*P"
        ),
        "per_batch_bytes": 4
        * (
            num_layers * (1024 * (11 * d_model + 4 * d_ff) + 2 * num_heads * 1024**2)
            + 1024 * d_model
            + 1024 * d_model
            + 2 * 1024 * vocab_size
        ),
        "fixed_bytes": 16 * parameter_count(vocab_size, num_layers, d_model, d_ff)["total"],
        "notes": (
            "B=batch size，n=context 长度，d=d_model，d_ff=前馈网络宽度，"
            "h=头数，V=词表大小，P=参数量。假设 float32 且所有列出的激活都保留到反向传播；"
            "实际实现可以通过重计算换取更小的显存占用。"
        ),
    }


def main() -> None:
    results: dict = {}
    for name, cfg in MODELS.items():
        d_ff = d_ff_of(cfg["d_model"])
        params = parameter_count(cfg["vocab_size"], cfg["num_layers"], cfg["d_model"], d_ff)
        flops = forward_flops(cfg["vocab_size"], cfg["context_length"], cfg["num_layers"], cfg["d_model"], d_ff)
        results[name] = {"config": {**cfg, "d_ff": d_ff}, "parameters": params, "forward_flops": flops}

    # (e) GPT-2 XL 在 16,384 上下文下的前向 FLOPs
    xl = MODELS["gpt2_xl"]
    d_ff = d_ff_of(xl["d_model"])
    results["gpt2_xl_ctx16384"] = {
        "forward_flops": forward_flops(xl["vocab_size"], 16384, xl["num_layers"], xl["d_model"], d_ff)
    }

    # (b) GPT-2 XL（context 1024）的 AdamW 显存与 80 GB 下最大 batch
    mem = adamw_memory(xl["d_model"], d_ff, xl["num_layers"], xl["num_heads"], xl["vocab_size"])
    total_80gb = 80e9
    mem["max_batch_size_80gb"] = int((total_80gb - mem["fixed_bytes"]) // mem["per_batch_bytes"])
    mem["peak_bytes_at_batch"] = {
        str(batch): mem["fixed_bytes"] + batch * mem["per_batch_bytes"] for batch in (1, 2, 4, 8, 16)
    }
    results["gpt2_xl_adamw_memory"] = mem

    # (d) 用 MFU 估计 GPT-2 XL 训练 400K 步的耗时
    tokens = 400_000 * 1024 * 1024
    params_total = results["gpt2_xl"]["parameters"]["total"]
    training_flops = 6 * params_total * tokens
    peak = 0.5 * 495e12
    results["mfu_training_time"] = {
        "tokens": tokens,
        "training_flops": training_flops,
        "effective_flops_per_s": peak,
        "seconds": training_flops / peak,
        "hours": training_flops / peak / 3600,
        "days": training_flops / peak / 86400,
    }

    out_dir = os.path.join("experiments", "logs")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "accounting.json"), "w") as f:
        json.dump(results, f, indent=2)

    # 另外输出一份可读的 Markdown 汇总表
    lines = ["# Transformer / AdamW 资源核算", ""]
    for name in MODELS:
        r = results[name]
        lines.append("## " + name)
        lines.append(
            "- d_ff={}，参数量={:,}（{:.3f}B），fp32 显存={:.2f} GB".format(
                r["config"]["d_ff"],
                r["parameters"]["total"],
                r["parameters"]["total"] / 1e9,
                r["parameters"]["fp32_memory_gb"],
            )
        )
        lines.append(
            "- 前向 FLOPs（context {}）：{:.3e}（{:.1f} TFLOPs）".format(
                r["config"]["context_length"], r["forward_flops"]["total"], r["forward_flops"]["total_tflops"]
            )
        )
        props = r["forward_flops"]["proportions"]
        lines.append("- 占比：" + "，".join(f"{k}={v:.1%}" for k, v in props.items()))
        lines.append("")
    lines.append("## GPT-2 XL（context 16,384）")
    f = results["gpt2_xl_ctx16384"]["forward_flops"]
    lines.append(f"- 前向 FLOPs：{f['total']:.3e}（{f['total_tflops']:.1f} TFLOPs）")
    lines.append("- 占比：" + "，".join(f"{k}={v:.1%}" for k, v in f["proportions"].items()))
    lines.append("")
    lines.append("## AdamW 显存（GPT-2 XL，context 1024，float32）")
    lines.append("- 公式：" + mem["formula"])
    lines.append("- 固定开销（参数+梯度+优化器状态）：{:.2f} GB".format(mem["fixed_bytes"] / 1e9))
    lines.append("- 每个 batch 元素：{:.2f} GB".format(mem["per_batch_bytes"] / 1e9))
    lines.append("- 80 GB 内最大 batch size：{}".format(mem["max_batch_size_80gb"]))
    lines.append("")
    mfu = results["mfu_training_time"]
    lines.append("## MFU 训练时间估计")
    lines.append("- tokens={:.3e}，训练 FLOPs={:.3e}".format(mfu["tokens"], mfu["training_flops"]))
    lines.append("- 按 495 TFLOP/s 的 50% MFU：{:.1f} 小时（{:.1f} 天）".format(mfu["hours"], mfu["days"]))

    with open(os.path.join("experiments", "logs", "accounting.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
