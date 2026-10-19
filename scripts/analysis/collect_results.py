"""收集所有 run 的指标，汇总为 JSON 与写报告用的 Markdown 表格。"""

from __future__ import annotations

import glob
import json
import os

RUNS = "runs"


def load_metrics(run_dir: str) -> dict | None:
    """读取一个 run 的 metrics.json（若存在 config.json 也一并读取）。"""
    path = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        metrics = json.load(f)
    config_path = os.path.join(run_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            metrics["config"] = json.load(f)
    return metrics


def main() -> None:
    summary: dict[str, dict] = {}
    for run_dir in sorted(glob.glob(os.path.join(RUNS, "*"))):
        metrics = load_metrics(run_dir)
        if metrics is not None:
            summary[os.path.basename(run_dir)] = metrics

    lines = ["# 实验汇总（自动生成）", ""]
    groups = {
        "学习率扫描": "lr_",
        "批量大小扫描": "batch_",
        "架构消融": "ablation_",
        "主实验": ("tinystories_base", "owt_base", "owt_45min"),
    }
    for title, prefix in groups.items():
        lines.append("## " + title)
        lines.append("")
        lines.append("| 运行 | 验证损失 | 困惑度 | 步数 | 词元数 | 耗时（秒） | 词元/秒 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for name, metrics in summary.items():
            if isinstance(prefix, tuple):
                matched = name in prefix
            else:
                matched = name.startswith(prefix)
            if not matched:
                continue
            ppl = metrics.get("final_valid_perplexity")
            lines.append(
                "| {} | {:.4f} | {} | {} | {} | {} | {} |".format(
                    name,
                    metrics["final_valid_loss"],
                    f"{ppl:.2f}" if ppl else "-",
                    metrics["num_steps"],
                    "{:,}".format(metrics["tokens_processed"]),
                    metrics["wall_time_s"],
                    metrics["tokens_per_s"],
                )
            )
        lines.append("")

    os.makedirs(os.path.join("experiments", "logs"), exist_ok=True)
    with open(os.path.join("experiments", "logs", "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    markdown = "\n".join(lines)
    with open(os.path.join("experiments", "logs", "summary.md"), "w") as f:
        f.write(markdown + "\n")
    print(markdown)


if __name__ == "__main__":
    main()
