# A40 实验日志

本日志记录本轮实际执行的操作和所生成的证据。研究假设、正式对照和停止规则见[实验方案](research-plan.md)。远端日志使用 UTC，2026 年 11 月洛杉矶使用 PST（UTC−8）。下述 11 月 24 日 00:51 UTC 对应洛杉矶 11 月 23 日 16:51。

## 数据与软件

| 项目 | 实际值 | 可复核的证据 |
|---|---|---|
| 单卡环境 | NVIDIA A40，46,068 MiB，驱动 595.112.07，Python 3.12.3。驱动报告支持 CUDA 13.2，训练环境 PyTorch 为 2.8.0+cu128。 | GPU 持久盘的 `/workspace/cs260d/logs/preflight.txt`，每 30 秒一次的 `gpu.csv`，以及训练运行目录中的 `run.json` |
| 模型快照 | Qwen3.5‑0.8B `2fc06364715b967f1860aea9cf38778875588b17`，Qwen3.5‑2B `15852e8c16360a2fea060d615a32b45270f8a8fc`。两份快照均在 Pod 完成校验。 | `logs/download.txt` |
| 数据快照 | Turnstile `70bf9e9a6234e694efa36d1ec6c207c65a695f5b`。分组种子 31415。输入和输出合计上限 4,096 token，输出最多 512 token。 | 项目下 `local/prepared/manifest.json`、`rows.audit.jsonl` |
| 实验代码 | 初始方案 `b923536`，训练器 `5939492`，数据及推理代码 `6598260`，打分、选样与 BFCL 程序 `fae15e0`。 | 每次实际运行的参数、文件哈希和版本另存于该轮运行目录。 |

### 冻结的评测清单

Turnstile 的 8,728 条单调用候选中，29 条参数键不符合 schema，14 条超输出窗口，2 条在类型校验阶段失败。合格候选中另去掉 4 条同组重复样本，并将调试样本与正式划分隔离。保留 3 道调试题、200 道摸底题、4,000 道候选训练题、400 道开发题和 2,000 道封存测试题。合法但多余的组留在划分之外。训练池、摸底集及封存集文件的 SHA‑256 已在 Mac 和 GPU 两端核对一致。

| 划分 | SHA-256 |
|---|---|
| 训练池 | `fd0ee69020ff537165b0fc405883f33ea2269ccc14cd4a231f62e5aff32e6001` |
| 摸底集 | `525aebaa6ef83dce9152eabbb89ebe99b5614947eddd78358333606dd8da6476` |
| 封存测试集 | `000c29f46a06a35146d8d369e4ff763226bf66a690a7339680e32fc8b779407c` |

BFCL 外测预先冻结 `simple_python` 和 `multiple` 各 100 题。生成端使用仅含题面和工具的文件，SHA‑256 为 `626b8855e169b2c0aefef13517946ab50100e277f48d55ac461569c78ca695ad`。与训练池比较，规定的题面、工具名和带名签名匹配数均为零。官方 AST 评分器通过每类 6 项自编输入检验；把 200 条输出全部改成无法解析时，两类分母均保持 100。这些是评分器的功能检查。记录见 `local/bfcl/manifest.json`、`local/bfcl/validation.json`。

## 时间线

| UTC（2026-11-24） | 实际操作与结果 |
|---|---|
| 00:51 | GPU、驱动、磁盘、Python 和原始包版本写入预检日志。代码和实验输出分别位于 `/workspace/cs260d/repo` 及其 `local/runs`，安装、作业控制台和显卡遥测位于 `/workspace/cs260d/logs`。 |
| 00:54–00:56 | 第一次 0.8B 单步训练与安装重叠，缺失 `torch.torch_version`，保留 `logs/smoke1.log`。安装完成后在单独目录 `local/runs/proxy-smoke2` 重跑。第 49 行，训练一步，loss 0.1016、梯度范数 4.352、有效步用时约 4.45 秒。ms-swift 成功写出 `checkpoint-1`。 |
| 00:56 | 重新加载同一个 0.8B adapter，通过 ms-swift 的实际模板复算第 49 行。1,438 个输入 token 中有 79 个受监督位置。整答手算 0.0222907234，模型自带 loss 0.0222907215，绝对差约 1.9×10⁻⁹。函数名 6 个 token，平均 0.006271；参数值 12 个，平均 0.074211；其余 61 个，平均 0.013653。证据为 `logs/gpu-probe.log`。 |
| 00:56 | 2B 在同一调试样本完成一次训练并保存 LoRA。运行目录 `local/runs/target-smoke`，记录 1,438 个输入、79 个受监督词元、峰值张量显存 5,256,627,712 字节。单题计时不能直接外推完整训练吞吐。 |
| 00:57 | 原始 2B 摸底生成进入 vLLM 初始化，准备文件和逐题提示词元已生成。启动日志为 `logs/pilot.log`，完成后的逐题生成与评分写入 `local/runs/pilot-base`。 |

**模板口径。** 早期 HF 模板的第 49 行有 78 个受监督位置。本轮以正式 ms-swift 标签计算有 79 个。两次使用的权重也有区别，先前是未训练权重，本轮复算用了第 1 步 adapter。单个结果不能用来估计筛选优势；A/B/C 的字段损失及选样比较将共用同一模板、同一组代理快照。

**计时口径。** 首次安装、下载、失败重跑、训练、固定答案打分、生成及等待分开记录。正式卡时依赖实测，单轮训练内部报告的显存和耗时与完整系统用量分开列示。


### 环境修复与生成重试

正式训练的 Transformers 版本锁定为 5.3.0。尝试加入 Flash Linear Attention 0.6 时，依赖解析会升级 Transformers，可能同时改变模型代码和模板行为。将候选加速包改为 0.4.2 后重新验证数值与吞吐。采用哪种内核，由相同输入的校验和 A40 基准确定。

vLLM 0.19.1 首次启动时对模拟视觉输入调用 Qwen3VLProcessor，报错 `ValueError: Failed to apply Qwen3VLProcessor`。本任务只使用文本，重试时启用 `language_model_only=True`，并明确 `max_lora_rank=16`。首次及重试的控制台分别记录于 `logs/pilot.log` 和 `logs/pilot2.log`。输出目录为 `local/runs/pilot-base`。只有完整落盘的 `predictions.jsonl`、`metrics.json` 与实际题目数吻合时才登记准确率。
