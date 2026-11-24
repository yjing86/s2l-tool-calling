# S2L for Tool Calling

Can a small model choose better fine-tuning examples by tracking **which function to call** and **which argument values to supply** separately? We compare the resulting subsets by how often a larger model generates an entirely correct call on an unseen request.

Across three paired training seeds, field-level selection improved exact-match accuracy on 2,000 held-out Turnstile requests by **1.53 percentage points** over whole-answer selection (95% paired interval +0.60 to +2.48). Uniform random selection scored 0.45 points higher on average (field minus random interval −1.42 to +0.53). On two fixed 100-question BFCL subsets, the equally weighted field-minus-whole contrast was −1.33 points (−3.00 to +0.33). The intervals describe task sampling for these trained models.

[中文实验结果与图表](docs/results.md) · [实验进展与运行记录](docs/experiment-log.md) · [中文研究方案](docs/research-plan.md) · [字段损失验证与复现](docs/field-loss-validation.md) · [论文阅读笔记](docs/references.md)

## 在哪里查看实验记录

[实验日志](docs/experiment-log.md)记录实际运行、修复和已核验的结果。[研究方案](docs/research-plan.md)给出对照和判分规则。训练、打分、生成与分析的源程序位于 [scripts](scripts/)。

本地课程项目位于课程目录的 `CS260D/s2l-tool-calling/`，同一仓库在 GPU 持久盘上的位置为 `/workspace/cs260d/repo/`。题号与划分清单位于仓库下的 `local/prepared/`，GPU 单步验证和基模摸底在 `local/runs/`；正式 A/B/C 运行在 `local/course-run/`，其中 `training/` 保存快照，`selection/` 保存各组题号和聚类统计，`evaluation/` 保存逐题预测，`receipts/` 保存阶段耗时与校验值。本机私有备份位于 `local/runpod/`。服务器安装日志、标准输出和 GPU 遥测位于 `/workspace/cs260d/logs/`。Git 保存实验代码和 `docs/experiment-log.md`。模型、原始数据和逐题预测保存在 Git 忽略的目录中，下载与归档需单独进行。

## Experiment

A Qwen3.5-0.8B proxy undergoes one LoRA fine-tuning run on 4,000 candidates. We score each reference answer at five checkpoints (steps 0, 10, 30, 80 and 250) and select 1,000 requests per method. Each Qwen3.5-2B target learns from complete reference calls with the same training-token budget.

| Group | Signal used to select training requests |
| --- | --- |
| A · Uniform random | No proxy scores |
| B · Whole-answer trajectory | One mean loss over all supervised answer tokens per checkpoint |
| C · Separate field trajectories | Two means per checkpoint, one for function names and one for argument values |
| B′ · Pooled fields, first additional control | One token-weighted mean over exactly the name and value positions used by C |

The primary outcome is the paired difference in **complete-call accuracy** between C and B. C versus B′ tests the value of keeping the fields separate after both selectors discard syntax and parameter keys. A second comparison charges each method for proxy training, scoring and target training, then lets the random baseline use the same GPU time.

## Training data and evaluation

The only training corpus is Amazon’s **Turnstile-Synthetic-Domains**. The pinned proxy and target weights are dated March 2, 2026. Amazon uploaded the public training data on August 3, 2026. We keep near-duplicate requests together in the same split and audit overlap in tool schemas. Those checks reduce the risk from previously published examples. Exposure to related templates during pretraining remains possible. Download the data directly from Amazon under CC BY-NC 4.0. [Amazon dataset snapshot](https://huggingface.co/datasets/amazon/Turnstile-Synthetic-Domains/tree/70bf9e9a6234e694efa36d1ec6c207c65a695f5b) · [0.8B snapshot](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17) · [2B snapshot](https://huggingface.co/Qwen/Qwen3.5-2B/tree/15852e8c16360a2fea060d615a32b45270f8a8fc)

Before training the target, we test it on 200 held-out single-call requests. The pilot measures headroom. The main task stays single-call, with a target of 2,000 held-out test requests and matched A/B/C runs at seeds 42, 43 and 44. We also reserve selected BFCL v4 single-turn categories for external evaluation. Scores will be reported by category. [Experiment protocol](docs/research-plan.md) · [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html)

## 下载实验材料

在项目根目录运行 `python3 scripts/download_inputs.py`，需要 Python 3 和 curl。脚本从发布者的固定版本下载约 6.74 GB 文件，逐个核对长度和哈希，通过后才移入正式路径。再次运行会跳过校验通过的文件。数据、模型与外测各有单独目录。

| 材料 | 本地位置 | 用途 |
|---|---|---|
| Amazon Turnstile | `local/data/turnstile/` | 唯一训练语料。保留原始 JSONL、API 定义、数据卡和许可信息。 |
| Qwen3.5-0.8B | `local/models/Qwen3.5-0.8B/` | 代理训练及五个时刻的损失打分。 |
| Qwen3.5-2B | `local/models/Qwen3.5-2B/` | 未微调基线及 A/B/C 等组的共同起点。 |
| BFCL v4 | `local/tools/bfcl/` | 官方评测器、题目和参考答案，仅用于外测。 |
| S2L 作者参考实现 | `local/tools/s2l-reference/` | 对照原论文的轨迹提取、聚类与抽样代码。[固定版本](https://github.com/BigML-CS-UCLA/S2L/tree/2bc84dda1a054d0ff0e704ae9658ee700d5c25fa) |
| Python 安装包 | `local/tools/wheels/` | ms-swift、Transformers、PEFT、qwen-vl-utils 的固定版本通用 wheel。 |

在新机器上可用以下命令获取 S2L 参考实现、BFCL 和通用安装包。BFCL 固定在方案使用的提交；主实验使用 `simple_python`、`multiple`；`parallel_multiple` 留给后续研究，具体启用条件和盲测抽样规则见[实验方案](docs/research-plan.md)。训练机上的 PyTorch、torchvision 和 CUDA 环境请按显卡镜像配置；本地缓存的四个 wheel 仍需配套依赖。[训练环境版本与复跑命令](docs/field-loss-validation.md)

```bash
mkdir -p local/tools/bfcl local/tools/wheels local/tools/s2l-reference
curl -fL --retry 4 https://codeload.github.com/BigML-CS-UCLA/S2L/tar.gz/2bc84dda1a054d0ff0e704ae9658ee700d5c25fa -o local/tools/s2l-reference.tar.gz
tar -xzf local/tools/s2l-reference.tar.gz -C local/tools/s2l-reference --strip-components=1
curl -fL --retry 4 https://codeload.github.com/ShishirPatil/gorilla/tar.gz/f7cf7359b7ac615a0b294831c5ba2bc95ee4a000 -o local/tools/bfcl.tar.gz
tar -xzf local/tools/bfcl.tar.gz -C local/tools/bfcl --strip-components=2 gorilla-f7cf7359b7ac615a0b294831c5ba2bc95ee4a000/berkeley-function-call-leaderboard
tar -xOf local/tools/bfcl.tar.gz gorilla-f7cf7359b7ac615a0b294831c5ba2bc95ee4a000/LICENSE > local/tools/bfcl/LICENSE
python3 -m pip download --no-deps --only-binary=:all: -d local/tools/wheels ms-swift==4.6.1 transformers==5.3.0 peft==0.18.1 qwen-vl-utils==0.0.14
```

## Run the field-loss check

From the repository root, the verifier can export the built-in order-tracking example with Python’s standard library. This creates one local JSONL file and requires no model download.

```bash
python3 scripts/verify_tool_call_loss.py --help
mkdir -p artifacts
python3 scripts/verify_tool_call_loss.py --write-swift-jsonl artifacts/demo.jsonl
```

The [validation guide](docs/field-loss-validation.md) contains the pinned installation and scoring commands, measured field losses on two Turnstile requests, and a one-step adapter check. The [results report](docs/results.md) includes the 2B comparisons, BFCL scores, and measured proxy and training costs.

## Repository contents

| Path | Purpose |
| --- | --- |
| [docs/research-plan.md](docs/research-plan.md) | Hypotheses, controls, interpretation and replication criteria |
| [docs/field-loss-validation.md](docs/field-loss-validation.md) | Reproduction commands, observed values and the scope of each check |
| [docs/references.md](docs/references.md) | Primary papers, method comparisons and official links |
| [scripts/verify_tool_call_loss.py](scripts/verify_tool_call_loss.py) | Runnable field-loss verifier |

Keep downloaded datasets and weights, run outputs and private working material in the ignored directories listed in [.gitignore](.gitignore).
