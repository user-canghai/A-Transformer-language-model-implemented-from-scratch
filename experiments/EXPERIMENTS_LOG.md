# 实验日志（Experiment Log）

本文件记录该项目的全部实验：配置、命令、结果与结论。运行日志摘要保存在 `experiments/logs/run_logs/`（原始日志目录未随仓库保留），学习曲线在 `writeup/figures/`，完整报告见 `writeup/writeup.pdf`。

## 硬件与通用设置

- 训练设备：Ascend 910 NPU（2 卡），torch 2.10.0 + torch_npu 2.10.0（Python 3.11）。
- 单元测试环境：Python 3.13.7 + torch 2.11.0+cpu（满足 pyproject 的 >=3.12,<3.14）。
- 通用模型：vocab 10,000（OWT 为 32,000），context 256，d_model 512，4 层，16 头，d_ff 1344，
  RoPE θ=10000（约 17M 非嵌入参数，22.7M 含 embedding）。
- AdamW：β=(0.9, 0.95)，ε=1e-8，weight decay 0.01，梯度裁剪 1.0，cosine 衰减到 lr_min=1e-4（扫参用 1e-5）。
- 数据：TinyStories train 541,229,347 tokens / valid 5,465,883 tokens（10K tokenizer，uint16）。

## 1. BPE 分词器

| 实验 | 命令 | 结果 |
| --- | --- | --- |
| TinyStories tokenizer (10K) | `scripts/train_tinystories_tokenizer.sh` | 31.0 s，峰值 221 MB，最长 token b" accomplishment"（15B），59,933 个唯一 pre-token |
| OpenWebText tokenizer (32K) | `scripts/train_owt_tokenizer.sh`（子集版：2 GB） | 543.6 s（预分词 72.2 s + merge 471.3 s），峰值 5,975 MB，2,096,076 个唯一 pre-token，最长 token 为 64 个连字符 |
| 分词器实验 | `scripts/analysis/tokenizer_experiments.py` | TS 压缩率 4.15 B/token；OWT 4.39 B/token；TS tokenizer 编 OWT 3.30 B/token；吞吐 ~4 MB/s（Pile 约 56 h） |
| Toy SGD lr 实验 | `scripts/analysis/toy_sgd.py` | lr 0.1/1/10/100/1000：收敛速度递增，1000 发散 |

说明：完整 11.9 GB OWT 上的一次 tokenizer 训练已跑约 20 分钟仍未完成 merge 阶段（设计资源预算 ≤12 小时），
超出本会话预算后改为在 OWT 前 2 GB 子集上训练 32K tokenizer，并在报告/日志中如实记录。

## 2. 学习率扫描（TinyStories，1000 步 ≈ 8.19M tokens）

| lr | valid loss | ppl |
| --- | --- | --- |
| 3e-4 | 2.6088 | 13.58 |
| 1e-3 | 2.2046 | 9.07 |
| **2e-3** | **2.1171** | **8.31** |
| 3e-3 | 2.2005 | 9.03 |
| 6e-3 | 2.5653 | 13.00 |
| 1e-2 | 2.8503 | 17.29 |
| 3e-2 | 3.6684 | 39.19 |

结论：最佳 lr = 2e-3；6e-3 起明显变差，1e-2/3e-2 进入不稳定区（3e-2 学到高 loss，可视为发散）。
“稳定边缘”在 5e-3~1e-2，最佳 lr 约为其 1/3~1/5。曲线：`writeup/figures/lr_sweep.pdf`。

## 3. TinyStories base run

`scripts/run_tinystories_base.sh`：vocab 10K，context 256，d_model 512，4 层，16 头，d_ff 1344，
batch 32 × 5,000 步 = 40.96M tokens，lr 2e-3（warmup 200，cosine 到 5,000），NPU:1。

结果：valid loss **1.6698**（ppl 5.31），406 s，100,850 tokens/s。
满足低资源目标（≤2.00）。曲线：`writeup/figures/tinystories_curve.pdf`。

## 4. 批量大小扫描（固定 token 预算 ≈ 9.83M tokens）

| batch | steps | valid loss | ppl |
| --- | --- | --- | --- |
| 8 | 4800 | 2.3524 | 10.51 |
| 32 | 1200 | 2.0365 | 7.66 |
| 64 | 600 | 2.0830 | 8.03 |
| 128 | 300 | 2.2491 | 9.48 |

结论：小 batch（8）噪声大、loss 更差；32 最好；64/128 在未重调 lr 的情况下略差。
曲线：`writeup/figures/batch_sweep.pdf`。

## 5. 架构消融（TinyStories，1200 步 ≈ 9.83M tokens）

| 变体 | valid loss | ppl |
| --- | --- | --- |
| baseline（pre-norm + RMSNorm + RoPE + SwiGLU） | 2.0346 | 7.65 |
| post-norm | 2.0504 | 7.77 |
| 去掉 RMSNorm（lr=3e-4） | 2.4787 | 11.93 |
| 去掉 RMSNorm（lr=2e-3） | 发散（loss 33.8 → 5190，NPU 异常终止） | - |
| NoPE（去掉 RoPE） | 2.2498 | 9.49 |
| SiLU FFN（无门控，d_ff=2048） | 2.0739 | 7.96 |

结论：RMSNorm 对稳定性至关重要；pre-norm 优于 post-norm；RoPE 对短上下文仍有明显收益；SwiGLU 门控略有增益。
曲线：`writeup/figures/ablations.pdf`。

## 6. 文本生成

`scripts/generate_samples.sh`（base 模型，prompt "Once upon a time"）：

- temperature 0.8 / top-p 0.95：`experiments/logs/generation_0.8.txt`（连贯儿童故事）
- temperature 1.0：`experiments/logs/generation_T1.0.txt`
- temperature 0.7 / top-p 0.9：`experiments/logs/generation_T0.7.txt`
- greedy：`experiments/logs/generation_greedy.txt`

结论：语法通顺、主旨连贯；温度与 top-p 影响多样性/重复性。

## 7. OpenWebText

数据：OWT 前 700 MB（约 160M tokens，文档边界对齐），32K OWT tokenizer；模型与 token 预算同 TinyStories。

模型：vocab 32K，context 256，d_model 512，4 层，16 头，d_ff 1344，batch 32 × 5,000 步 = 40.96M tokens。

结果：valid loss **4.7067**（ppl 110.68），耗时 491 s，83,355.2 tokens/s。
loss 明显高于 TinyStories：网页语料熵高、长尾大，相同算力下更难拟合。
生成样例：experiments/logs/generation_owt.txt（网页/商品评测风格文本，句法基本通顺）。

## 8. 45 分钟墙钟预算长训练

从 OWT base checkpoint 继续训练到 10,000 步（82M tokens，总墙钟约 16 分钟 < 45 分钟预算），最终 valid loss **4.5093**（ppl 90.86），优于朴素基线 5.0；受限于单机 2 卡 NPU，该实验仅在本机完成。

## 9. 环境问题记录

- 远端 sshd 的 sftp subsystem 路径错误（指向不存在的 /usr/libexec/openssh/sftp-server），用软链接修复。
- PyPI 直连极慢（~11 KB/s），使用华为云 PyPI 镜像（~13 MB/s）；torch 2.11.0+cpu aarch64 wheel 来自
  download.pytorch.org（版本与 uv.lock 一致）。
- Python 3.13 由源码编译（源码包经华为镜像下载，启用 zlib/ssl/readline 等），用于满足 pyproject 要求；
  NPU 训练使用镜像自带 Python 3.11 + torch_npu 环境。
- 完整 OWT tokenizer 训练超时 → 使用 2 GB 子集（见第 1 节）。
