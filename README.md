# 从零实现的 Transformer 语言模型

本项目从零实现字节级 BPE 分词器与 Decoder-only Transformer 语言模型，并在 TinyStories /
OpenWebText 上完成训练、评估、文本生成与架构消融实验。核心组件只使用 PyTorch 的
Parameter、Module 容器与初始化工具实现，不依赖 torch.nn 中现成的层、损失函数或优化器。

## 功能

- 字节级 BPE 分词器：并行预分词、特殊 token、增量 pair 统计、流式编码、词表/merges 序列化
- 模型组件：Linear、Embedding、RMSNorm、SwiGLU、RoPE、因果多头自注意力、Transformer Block / LM
- 训练基础设施：交叉熵、AdamW、余弦学习率 + warmup、梯度裁剪、memmap 数据加载、checkpoint
- 采样解码：temperature 缩放与 top-p（nucleus）采样
- 实验脚本、学习曲线、完整报告与实验日志

## 目录结构

```
lm_basics/          核心实现（分词器、模型、优化器、训练、采样等）
scripts/            一键复现脚本
scripts/analysis/   分词器实验、资源核算、绘图、汇总等分析脚本
tests/              单元测试
writeup/            报告（Markdown / HTML / PDF）与图表
experiments/        实验日志与运行记录
```

## 环境

- Python >= 3.12
- 依赖见 `requirements.txt`；CPU 测试环境为 Python 3.13 + torch 2.11.0+cpu
- 训练环境（可选，昇腾 NPU）：Python 3.11 + torch 2.10 + torch_npu

安装示例：

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# CPU 版 torch 也可以直接从 PyTorch 官方源安装：
# .venv/bin/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
```

## 运行测试

```bash
.venv/bin/python -m pytest tests/ -q    # 47 passed, 1 xfailed
```

## 复现实验

1. 把 TinyStories 与 OpenWebText 的文本文件放到 `data/` 目录；
2. 训练分词器：`bash scripts/train_tinystories_tokenizer.sh`、`bash scripts/train_owt_tokenizer.sh`；
3. 分词：`bash scripts/tokenize_datasets.sh`；
4. 训练与实验：`bash scripts/run_tinystories_base.sh`、`run_lr_sweep.sh`、`run_batch_sweep.sh`、
   `run_ablations.sh`、`run_owt_base.sh`；
5. 采样：`bash scripts/generate_samples.sh`。

训练日志写入 `runs/<实验名>/`，指标文件为 `metrics.json`。

## 报告与实验日志

- 报告：`writeup/writeup.pdf`（同目录有 `writeup.md`）
- 实验日志：`experiments/EXPERIMENTS_LOG.md`
- 运行日志摘要：`experiments/logs/`

## 主要结果

| 实验 | 结果 |
| --- | --- |
| TinyStories BPE（词表 10K） | 31 s 训练完成，峰值内存 221 MB |
| TinyStories 语言模型（40.96M tokens） | 验证损失 1.6698（困惑度 5.31） |
| 学习率扫描 | 最优 2e-3，3e-2 发散 |
| 批量大小扫描（等 token 预算） | batch 32 最优 |
| 架构消融 | pre-norm + RMSNorm + RoPE + SwiGLU 最优 |
| OpenWebText 语言模型（40.96M tokens） | 验证损失 4.7067 |
| 45 分钟预算长训练（82M tokens） | 验证损失 4.5093 |
