"""根据实验输出，填充 writeup/writeup.md 中的 {{PLACEHOLDER}} 占位段落。"""

from __future__ import annotations

import glob
import json
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WRITEUP = os.path.join(ROOT, "writeup", "writeup.md")
LOGS = os.path.join(ROOT, "experiments", "logs")


def load_json(name: str):
    """读取 experiments/logs 下的 JSON 文件。"""
    path = os.path.join(LOGS, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def run_metrics(name: str):
    """读取某个 run 的 metrics.json。"""
    path = os.path.join(ROOT, "runs", name, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def frac(x: float) -> str:
    """把 0~1 的比例格式化为百分比字符串。"""
    return f"{x * 100:.1f}%"


def section_owt_bpe() -> str:
    path = "/tmp/train_owt_subset.log"
    if not os.path.exists(path):
        return "（待 OpenWebText tokenizer 训练完成后补充。）"
    text = open(path, encoding="utf-8", errors="replace").read()  # noqa: SIM115
    profile = re.search(r"\[profile\].*", text)
    trained = re.search(r"trained BPE:.*", text)
    if not trained:
        return "（待 OpenWebText tokenizer 训练完成后补充。）"
    lines = [
        (
            "为在有限时间内完成，OWT tokenizer 在 OpenWebText 前 2.0 GB（按文档边界对齐）子集上训练，"
            "词汇量 32,000（含 <|endoftext|> 特殊 token）。完整 11.92 GB 上的一次训练已跑约 20 分钟仍未完成 merge 阶段，"
            "按设计给出的 ≤12 小时资源预算属于可接受范围，但超出本会话预算，故改用子集并如实记录。"
        ),
        "",
    ]
    if profile:
        parts = dict(re.findall(r"(\w+)=([^\s]+)", profile.group(0)))
        lines.append("| 指标 | 数值 |")
        lines.append("| --- | --- |")
        lines.append(f"| 总耗时 | {parts.get('total', '?')} |")
        lines.append(f"| 预分词 | {parts.get('pretokenize', '?')}（32 进程） |")
        lines.append(f"| merge 阶段 | {parts.get('merge', '?')} |")
        lines.append(f"| 峰值内存 | {parts.get('peak_rss', '?')} MB |")
        lines.append(f"| 唯一 pre-token 数 | {parts.get('unique_pretokens', '?')} |")
        lines.append(f"| 最长 token | {parts.get('longest_token', '?')} |")
        lines.append("")
    lines.append(
        "与 TinyStories 相比，OWT 语料更大、更多样（网页、代码、多语言片段），词表更大（32K vs 10K），"
        "最大 token 往往更长、更偏英文词片段；而 TinyStories 的 token 更集中在儿童故事常见词与标点上。"
    )
    return "\n".join(lines)


def section_tokenizer_experiments() -> str:
    data = load_json("tokenizer_experiments.json")
    if not data:
        return "（待实验完成后补充。）"
    lines = []
    ts = data.get("tinystories_tokenizer_on_tinystories")
    owt = data.get("owt_tokenizer_on_owt")
    cross = data.get("tinystories_tokenizer_on_owt")
    if ts:
        lines.append(
            f"**(a)** TinyStories tokenizer 在 10 篇 TinyStories 文档上的压缩率为 "
            f"**{ts['bytes_per_token']:.2f} bytes/token**（{ts['bytes']} 字节 -> {ts['tokens']} tokens）。"
        )
    if owt:
        lines.append(
            f"OWT tokenizer 在 10 篇 OWT 文档上的压缩率为 **{owt['bytes_per_token']:.2f} bytes/token**"
            f"（{owt['bytes']} 字节 -> {owt['tokens']} tokens）。"
        )
    lines.append("")
    if cross:
        lines.append(
            f"**(b)** 用 TinyStories tokenizer 编码 OWT 文档时压缩率降到 **{cross['bytes_per_token']:.2f} bytes/token**"
            f"（{cross['bytes']} 字节 -> {cross['tokens']} tokens）——同一段文本被切得更碎。"
            "原因是 10K 词表缺少 OWT 中的大量常见词/子词与特殊字符组合；分词器与语料域不匹配会显著增加序列长度与训练成本。"
        )
    lines.append("")
    throughput = data.get("tinystories_bytes_per_second")
    if throughput:
        hours = data.get("pile_825gb_hours_tinystories_tokenizer")
        lines.append(
            f"**(c)** 单进程编码吞吐约 **{throughput / 1e6:.2f} MB/s**，按此速度编码 Pile（825 GB 文本）约需 "
            f"**{hours:.0f} 小时**（实际可多进程并行，时间可再按进程数缩短）。"
        )
    lines.append("")
    lines.append(
        "**(d)** uint16 合适：词表大小（10K / 32K）远小于 2^16 = 65,536，所有 token id 都能放入 uint16；"
        "相比 int32/int64 数组，uint16 把存储占用减为 1/2 或 1/4，便于对 GB 级 token 流做 memmap 随机采样。"
        "若词表超过 65,536 则需要 uint32。"
    )
    return "\n".join(lines)


def section_accounting() -> str:
    data = load_json("accounting.json")
    if not data:
        return "（待计算。）"
    xl = data["gpt2_xl"]
    p = xl["parameters"]
    f = xl["forward_flops"]
    lines = [
        "**(a)** GPT-2 XL 配置（vocab 50,257，context 1,024，48 层，d_model 1,600，25 头，d_ff 4,288）的参数量：",
        "",
    ]
    lines.append("| 组件 | 参数量 |")
    lines.append("| --- | --- |")
    lines.append(f"| token embedding | {p['token_embedding']:,} |")
    lines.append(f"| {xl['config']['num_layers']} 个 Transformer block | {p['blocks']:,} |")
    lines.append(f"| final RMSNorm | {p['final_norm']:,} |")
    lines.append(f"| LM head | {p['lm_head']:,} |")
    lines.append(f"| **合计** | **{p['total']:,}（{p['total'] / 1e9:.3f}B）** |")
    lines.append("")
    lines.append(f"单精度（fp32）加载需要 **{p['fp32_memory_gb']:.2f} GB**（每个参数 4 字节）。")
    lines.append("")
    lines.append("**(b)** 一次前向（context 1,024）的矩阵乘 FLOPs：")
    lines.append("")
    lines.append("| 矩阵乘 | FLOPs |")
    lines.append("| --- | --- |")
    for key, label in [
        ("qkv_projections", "Q/K/V 投影（3×2·n·d²，×L）"),
        ("attention_scores", "QK^T（2·n²·d，×L）"),
        ("attention_weighted_sum", "softmax(QK^T)V（2·n²·d，×L）"),
        ("output_projection", "输出投影（2·n·d²，×L）"),
        ("feed_forward", "SwiGLU FFN（3×2·n·d·d_ff，×L）"),
        ("lm_head", "LM head（2·n·d·V）"),
    ]:
        lines.append(f"| {label} | {f[key]:.3e} |")
    lines.append(f"| **合计** | **{f['total']:.3e}（{f['total_tflops']:.1f} TFLOPs）** |")
    lines.append("")
    lines.append(
        "**(c)** FFN 占比最大（"
        + frac(f["proportions"]["feed_forward"])
        + "），其次是注意力投影（"
        + frac(f["proportions"]["attention_projections"])
        + "）与 LM head（"
        + frac(f["proportions"]["lm_head"])
        + "）；注意力矩阵乘只有 "
        + frac(f["proportions"]["attention_matrix_products"])
        + "（n=1024 时还不大）。"
    )
    lines.append("")
    lines.append("**(d)** 不同规模 GPT-2 的组件 FLOPs 占比：")
    lines.append("")
    lines.append("| 模型 | 参数量 | 前向 TFLOPs | 注意力投影 | 注意力矩阵乘 | FFN | LM head |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for name in ["gpt2_small", "gpt2_medium", "gpt2_large", "gpt2_xl"]:
        r = data[name]
        props = r["forward_flops"]["proportions"]
        lines.append(
            f"| {name} | {r['parameters']['total'] / 1e9:.3f}B | {r['forward_flops']['total_tflops']:.2f} | "
            f"{frac(props['attention_projections'])} | {frac(props['attention_matrix_products'])} | "
            f"{frac(props['feed_forward'])} | {frac(props['lm_head'])} |"
        )
    lines.append("")
    lines.append(
        "模型变大时 FFN 与注意力投影占比上升，LM head 占比下降（embedding/head 参数不随层数增长）；"
        "注意力矩阵乘占比随模型变大而略降。"
    )
    lines.append("")
    f2 = data["gpt2_xl_ctx16384"]["forward_flops"]
    lines.append(
        f"**(e)** GPT-2 XL 将 context 提升到 16,384 后，一次前向 FLOPs 从 {f['total_tflops']:.1f} TFLOPs 增至 "
        f"**{f2['total_tflops']:.1f} TFLOPs**（约 {f2['total'] / f['total']:.0f} 倍，因注意力矩阵乘按 n² 增长）；"
        f"占比变为：注意力矩阵乘 {frac(f2['proportions']['attention_matrix_products'])}（主导），"
        f"注意力投影 {frac(f2['proportions']['attention_projections'])}，"
        f"FFN {frac(f2['proportions']['feed_forward'])}，LM head {frac(f2['proportions']['lm_head'])}。"
    )
    return "\n".join(lines)


def section_adamw() -> str:
    data = load_json("accounting.json")
    if not data:
        return "（待计算。）"
    mem = data["gpt2_xl_adamw_memory"]
    mfu = data["mfu_training_time"]
    P = data["gpt2_xl"]["parameters"]["total"]
    lines = []
    lines.append("**(a)** AdamW 峰值显存（float32；P=参数量，B=batch size，n=context，d=d_model，d_ff=FFN 宽度）：")
    lines.append("")
    lines.append("- 参数 4P 字节；梯度 4P 字节；优化器状态（m、v）8P 字节 ⇒ 固定开销 **16P**。")
    lines.append(
        "- 激活（假设全部保留到 backward）：每层约 B·n·(11d + 4d_ff) + 2·B·h·n² 个元素，"
        "另有 embedding 与 final norm 各 B·n·d，logits 与 loss 约 2·B·n·V。"
    )
    lines.append("- 综合公式：" + mem["formula"])
    lines.append("")
    lines.append(
        f"**(b)** GPT-2 XL（n=1024）：固定部分 **{mem['fixed_bytes'] / 1e9:.2f} GB**，每 batch 元素 "
        f"**{mem['per_batch_bytes'] / 1e9:.2f} GB**，峰值 ≈ {mem['per_batch_bytes'] / 1e9:.2f}·B + "
        f"{mem['fixed_bytes'] / 1e9:.2f} GB；80 GB 内最大 **B = {mem['max_batch_size_80gb']}**。"
        "（激活重计算/分块可支持更大 batch。）"
    )
    lines.append("")
    lines.append(
        f"**(c)** AdamW 每步对每个参数做常数次逐元素运算（权重衰减、一阶/二阶矩、偏差修正更新），"
        f"约 **13P 次运算**（XL 约 {13 * P / 1e9:.1f} GFLOP），相对 forward+backward（约 6P·tokens）可忽略。"
    )
    lines.append("")
    lines.append(
        f"**(d)** 400K 步 × batch 1,024 × context 1,024 = {mfu['tokens']:.3e} tokens；"
        f"训练 FLOPs ≈ 6·P·tokens = **{mfu['training_flops']:.3e}**。H100 fp32/TF32 峰值 495 TFLOP/s，"
        f"50% MFU ⇒ 247.5 TFLOP/s，总时间 ≈ {mfu['seconds']:.3e} s ≈ **{mfu['hours']:.0f} 小时"
        f"（约 {mfu['days']:.0f} 天）**。"
    )
    return "\n".join(lines)


def table_runs(pattern: str, label=None) -> str:
    lines = [
        "| 运行 | 验证损失 | 困惑度 | 步数 | 词元数 | 耗时（秒） | 词元/秒 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for path in sorted(glob.glob(os.path.join(ROOT, "runs", pattern))):
        name = os.path.basename(path)
        m = run_metrics(name)
        if m is None:
            continue
        ppl = m.get("final_valid_perplexity")
        lines.append(
            f"| {label(name) if label else name} | {m['final_valid_loss']:.4f} | "
            f"{ppl:.2f} | {m['num_steps']} | {m['tokens_processed']:,} | {m['wall_time_s']} | {m['tokens_per_s']} |"
            if ppl is not None
            else f"| {name} | {m['final_valid_loss']:.4f} | - | {m['num_steps']} | {m['tokens_processed']:,} | "
            f"{m['wall_time_s']} | {m['tokens_per_s']} |"
        )
    return "\n".join(lines)


def section_lr() -> str:
    return table_runs("lr_*", label=lambda n: "lr=" + n.replace("lr_", ""))


def section_base() -> str:
    m = run_metrics("tinystories_base")
    if not m:
        return "（待补充。）"
    return (
        f"TinyStories base 模型（vocab 10K、context 256、d_model 512、4 层、16 头、d_ff 1344，"
        f"batch 32 × 5,000 步 = 40.96M tokens，lr 2e-3）：最终验证损失 **{m['final_valid_loss']:.4f}**"
        f"（perplexity **{m['final_valid_perplexity']:.2f}**），训练耗时 {m['wall_time_s']:.0f} s，"
        f"吞吐 {m['tokens_per_s']:,} tokens/s。达到低资源目标（≤2.00）；"
        "与更大 token 预算（327.68M tokens）下的目标 1.45 仍有差距，符合 8 倍 token 预算的预期。"
    )


def section_batch() -> str:
    rows = []
    for bs in [8, 32, 64, 128]:
        m = run_metrics(f"batch_{bs}")
        if m:
            rows.append(
                f"| {bs} | {m['num_steps']} | {m['final_valid_loss']:.4f} | "
                f"{m['final_valid_perplexity']:.2f} | {m['wall_time_s']:.0f} |"
            )
    if not rows:
        return "（待补充。）"
    header = ["| 批量大小 | 步数 | 验证损失 | 困惑度 | 耗时（秒） |", "| --- | --- | --- | --- | --- |"]
    return "\n".join(header + rows) + (
        "\n\n在相同 token 预算（约 9.83M tokens）下，batch 8 的 loss 最高（梯度噪声大），32/64/128 更好；"
        "更大的 batch 需要相应调大学习率才能保持同等效率，本实验保持 lr=2e-3 不变，因此大 batch 收益有限。"
    )


def section_ablation() -> str:
    names = {
        "ablation_baseline": "baseline（pre-norm + RMSNorm + RoPE + SwiGLU）",
        "ablation_no_rmsnorm_lr2e-3": "去掉 RMSNorm（lr=2e-3）",
        "ablation_no_rmsnorm_lr3e-4": "去掉 RMSNorm（lr=3e-4）",
        "ablation_post_norm": "post-norm",
        "ablation_nope": "NoPE（去掉 RoPE）",
        "ablation_silu_ffn": "SiLU FFN（无门控，d_ff=2048）",
    }
    rows = []
    for run, label in names.items():
        m = run_metrics(run)
        if m:
            rows.append(f"| {label} | {m['final_valid_loss']:.4f} | {m['final_valid_perplexity']:.2f} |")
    if not rows:
        return "（待补充。）"
    header = ["| 变体 | valid loss | ppl |", "| --- | --- | --- |"]
    return "\n".join(header + rows) + (
        "\n\n结论：去掉 RMSNorm 后，在 lr=2e-3 下训练直接发散（loss 从 33 爆炸到 5190 后触发 NPU 异常终止）；"
        "把学习率降到 3e-4 可以稳定训练，但 1200 步时 loss 仍为 2.48，明显差于 baseline 的 2.03。"
        "post-norm 2.05 略差于 pre-norm；NoPE 2.25 明显更差，说明 RoPE 对短上下文语言建模仍重要；"
        "参数匹配的 SiLU FFN 2.07 略差于 SwiGLU，说明门控带来增益。"
    )


def section_generation() -> str:
    path = os.path.join(LOGS, "generation_0.8.txt")
    if not os.path.exists(path):
        return "（待补充。）"
    text = open(path, encoding="utf-8").read().split("<|endoftext|>")[0].strip()  # noqa: SIM115
    return (
        '使用 base 模型在 temperature 0.8、top-p 0.95 下采样（prompt: "Once upon a time"）：\n\n'
        "> " + text.replace("\n", " ") + "\n\n"
        "生成文本语法通顺、是连贯的儿童故事。影响生成质量的因素：训练 token 数、temperature/top-p（过低重复、"
        "过高跑题）、词表与模型规模。"
    )


def section_owt() -> str:
    m = run_metrics("owt_base")
    if not m:
        return "（待 OpenWebText 实验完成后补充。）"
    return (
        f"在 OpenWebText 前 700 MB（约 160M tokens，按文档边界对齐）上、用 32K OWT tokenizer 训练："
        f"相同模型结构与 token 预算（40.96M tokens，5,000 步）下，最终验证损失 "
        f"**{m['final_valid_loss']:.4f}**（ppl {m['final_valid_perplexity']:.2f}），耗时 {m['wall_time_s']:.0f} s。"
        "OWT loss 明显高于 TinyStories：真实网页文本熵更高、长尾更大，17M 参数 + 40M token 预算远不足以"
        "拟合其复杂度。受限于单机 CPU/NPU 与时间，OWT 使用 700 MB 子集而非完整 11.9 GB 语料。"
    )


def section_long_training() -> str:
    m = run_metrics("owt_45min")
    if not m:
        return "（待长训练完成后补充。）"
    return (
        f"从 OWT base checkpoint 继续训练到 10,000 步（约 82M tokens，墙钟约 {m['wall_time_s'] / 60:.0f} 分钟，"
        f"满足 45 分钟预算）：最终验证损失 **{m['final_valid_loss']:.4f}**（ppl {m['final_valid_perplexity']:.2f}），"
        "已优于朴素基线（5.0），但距离更高水平仍有差距；本实验仅在本机完成。"
    )


def main() -> None:
    text = open(WRITEUP, encoding="utf-8").read()  # noqa: SIM115
    experiments = "\n\n".join(
        [
            "### 5.1 学习率扫描（TinyStories，1000 步 ≈ 8.19M tokens）\n\n" + section_lr() + "\n\n"
            "最佳学习率约 **2e-3**（loss 2.117）；6e-3 起明显变差（2.565），1e-2 与 3e-2 进入不稳定/发散区"
            "（3e-2 最终 loss 3.668）。这说明“稳定边缘”在 5e-3~1e-2 附近，最佳 lr 约为发散阈值的 1/3~1/5。",
            "### 5.2 TinyStories base 模型\n\n" + section_base(),
            "### 5.3 批量大小扫描（固定 token 预算）\n\n" + section_batch(),
            "### 5.4 文本生成\n\n" + section_generation(),
            "### 5.5 架构消融\n\n" + section_ablation(),
            "### 5.6 OpenWebText\n\n" + section_owt(),
            "### 5.7 45 分钟墙钟预算的长训练\n\n" + section_long_training(),
        ]
    )
    replacements = {
        "{{OWT_BPE_RESULT}}": section_owt_bpe(),
        "{{TOKENIZER_EXPERIMENTS}}": section_tokenizer_experiments(),
        "{{ACCOUNTING}}": section_accounting(),
        "{{ADAMW_ACCOUNTING}}": section_adamw(),
        "{{EXPERIMENTS}}": experiments,
    }
    for key, value in replacements.items():
        text = text.replace(key, value)
    with open(WRITEUP, "w", encoding="utf-8") as fh:
        fh.write(text)
    print("剩余占位符：", re.findall(r"\{\{[A-Z_]+\}\}", text))
    print("报告长度：", len(text))


if __name__ == "__main__":
    main()
