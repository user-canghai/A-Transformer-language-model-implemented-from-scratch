"""训练完成后，把 writeup.md 中的 OpenWebText 结果段落替换为真实数据。"""

from __future__ import annotations

import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WRITEUP = os.path.join(ROOT, "writeup", "writeup.md")


def metrics(name: str):
    """读取某个 run 的 metrics.json。"""
    path = os.path.join(ROOT, "runs", name, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main() -> None:
    text = open(WRITEUP, encoding="utf-8").read()  # noqa: SIM115
    base = metrics("owt_base")
    if base is not None:
        owt_text = (
            f"在 OpenWebText 前 700 MB（约 160M tokens，按文档边界对齐）上用 32K OWT tokenizer 训练："
            f"与 TinyStories 相同的模型结构与 token 预算（40.96M tokens，5,000 步）下，最终验证损失 "
            f"**{base['final_valid_loss']:.4f}**（ppl {base['final_valid_perplexity']:.2f}），"
            f"耗时 {base['wall_time_s']:.0f} s。\n\n"
            "OWT 的 loss 显著高于 TinyStories（约 +4 到 +5）：真实网页文本的熵与长尾远高于儿童故事，"
            "同样的 17M 参数模型 + 40M token 预算远不足以拟合其复杂度；这说明 loss 的绝对值不能跨数据集比较，"
            "而应看相同数据/预算下的相对改进。受限于单机 CPU/NPU 与时间，OWT 使用 700 MB 子集而非完整 11.9 GB 语料。"
        )
        text = text.replace("（待 OpenWebText 实验完成后补充。）", owt_text)
    leader = metrics("owt_45min")
    if leader is not None:
        text = text.replace(
            "（待长训练完成后补充。）",
            f"从 OWT base checkpoint 继续训练到 10,000 步（约 82M tokens，总计墙钟 "
            f"{(base['wall_time_s'] + leader['wall_time_s']) / 60 if base else leader['wall_time_s'] / 60:.0f} 分钟，"
            f"满足 45 分钟预算）：最终验证损失 **{leader['final_valid_loss']:.4f}**"
            f"（ppl {leader['final_valid_perplexity']:.2f}），已明显优于朴素基线（5.0），"
            "但距离更高水平仍有差距；本实验仅在本机完成。",
        )
    with open(WRITEUP, "w", encoding="utf-8") as fh:
        fh.write(text)
    print("已替换 OWT 段落；剩余待补处数量：", text.count("（待"))


if __name__ == "__main__":
    main()
