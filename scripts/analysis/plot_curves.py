"""把各训练 run 的学习曲线画成论文风格的图。"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 使用系统已安装的 Noto CJK 字体渲染中文标签
matplotlib.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

OUT_DIR = os.path.join("writeup", "figures")


def read_log(path: str) -> dict:
    """读取一个 run 的 log.csv，返回训练/验证损失序列。"""
    steps, train_loss, valid = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["train_loss"]:
                steps.append(int(row["step"]))
                train_loss.append(float(row["train_loss"]))
            if row["valid_loss"]:
                valid.append((int(row["step"]), float(row["valid_loss"])))
    return {"steps": steps, "train_loss": train_loss, "valid": valid}


def style_axes(ax, xlabel: str, ylabel: str, title: str | None = None) -> None:
    """统一的坐标轴风格：网格、去掉上/右边框。"""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_family(pattern: str, label_fn, filename: str, title: str) -> None:
    """把匹配 pattern 的一组 run 画在同一张图上。"""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for run_dir in sorted(glob.glob(pattern)):
        log_path = os.path.join(run_dir, "log.csv")
        if not os.path.exists(log_path):
            continue
        data = read_log(log_path)
        xs = [step for step, _ in data["valid"]] or data["steps"]
        ys = [loss for _, loss in data["valid"]] or data["train_loss"]
        ax.plot(xs, ys, marker="o", markersize=2.5, linewidth=1.4, label=label_fn(run_dir))
    style_axes(ax, "训练步数", "验证损失（每 token）", title)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, filename), bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, filename.replace(".pdf", ".png")), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("已写入", os.path.join(OUT_DIR, filename))


def main() -> None:
    plot_family(
        "runs/lr_*",
        lambda d: "学习率=" + os.path.basename(d).replace("lr_", ""),
        "lr_sweep.pdf",
        "TinyStories 学习率扫描（验证损失）",
    )
    plot_family(
        "runs/batch_*",
        lambda d: "batch=" + os.path.basename(d).replace("batch_", ""),
        "batch_sweep.pdf",
        "批量大小扫描（相同 token 预算）",
    )
    ablation_labels = {
        "ablation_baseline": "基线（pre-norm + RMSNorm + RoPE + SwiGLU）",
        "ablation_no_rmsnorm_lr2e-3": "去掉 RMSNorm（lr=2e-3）",
        "ablation_no_rmsnorm_lr3e-4": "去掉 RMSNorm（lr=3e-4）",
        "ablation_post_norm": "post-norm",
        "ablation_nope": "NoPE",
        "ablation_silu_ffn": "SiLU FFN（无门控）",
    }
    plot_family(
        "runs/ablation_*",
        lambda d: ablation_labels.get(os.path.basename(d), os.path.basename(d).replace("ablation_", "")),
        "ablations.pdf",
        "TinyStories 架构消融",
    )
    plot_family(
        "runs/tinystories_base", lambda d: "TinyStories 基线模型", "tinystories_curve.pdf", "TinyStories 基线模型"
    )
    plot_family("runs/owt_base", lambda d: "OpenWebText", "owt_curve.pdf", "OpenWebText 模型")
    if os.path.exists("runs/owt_base/log.csv") and os.path.exists("runs/tinystories_base/log.csv"):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for run_dir, label in [("runs/tinystories_base", "TinyStories"), ("runs/owt_base", "OpenWebText")]:
            data = read_log(os.path.join(run_dir, "log.csv"))
            xs = [step for step, _ in data["valid"]]
            ys = [loss for _, loss in data["valid"]]
            ax.plot(xs, ys, marker="o", markersize=2.5, linewidth=1.4, label=label)
        style_axes(ax, "训练步数", "验证损失（每 token）", "TinyStories 与 OpenWebText 对比")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "ts_vs_owt.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(OUT_DIR, "ts_vs_owt.png"), bbox_inches="tight", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
