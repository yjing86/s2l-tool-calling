# 工具调用的分项损失验证

2026-11-19 · [验证脚本](../scripts/verify_tool_call_loss.py) · [实验方案](research-plan.md)

**验证结果。** 在一道自编题和两道 Turnstile 题上，Qwen3.5-0.8B 的一次前向计算能分别给出函数名和参数值的损失。原始 2B 摸底和各组选样比较仍待运行。下文保留先前的测量记录，仓库未保存当时的原始日志、权重或 adapter。

## 1. 先回答训练问题

**0.8B 和 2B 训练时都只优化一个量：整段标准工具调用答案上，有监督的 token 的平均交叉熵。** 用户的请求和候选工具说明作为上下文，padding 也不参加平均。函数名、参数名、参数值、调用格式以及被模板标为答案的结束符都能参与训练。各组 2B 始终用相同的训练目标。[ms-swift 的工具调用训练及标签示例](https://github.com/modelscope/ms-swift/blob/v4.6.1/docs/source_en/Instruction/Agent-support.md)

举例，用户说“查订单 4821”，标准调用是 `track_order(order_id="4821")`。模型按顺序预测答案里的每个 token。每到一个位置，模型给标准 token 的概率越低，这个位置的 loss 就越高。训练程序先得到一串这样的数，然后取平均，用这个平均数做反向传播。**只拿到日志里的平均数，已经无法知道到底错在函数名还是订单号；在取平均之前保留这串数，就能分别查看。** [Qwen3.5 模型输出说明](https://huggingface.co/docs/transformers/model_doc/qwen3_5) · [PyTorch 的逐位置交叉熵](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy)

| 同一串逐 token loss 可以怎么算 | 用途 | 改模型权重吗？ |
|---|---|---|
| 所有有监督答案 token 取平均 | 训练 0.8B 或 2B 的目标；冻结权重后也可以用它给每道题打总分 | 训练时更新；打分时不更新 |
| 只对函数名所在的 token 取平均 | 观察小模型在这一时刻对正确工具名有多确定 | 不更新 |
| 只对参数值所在的 token 取平均 | 观察小模型在给定标准前文时，对正确订单号、地点等有多确定 | 不更新 |

因此不用让优化器在三个 loss 中选一个，也不用把两个分项 loss 各做一次反向传播。**一个训练步骤中，本来就有每个答案位置的预测。为了比较同一道题在四个时刻的变化，我们另外读取四个冻结的模型快照，对同一批题打分。** 这是额外的计算开销。

## 2. 哪些步骤用现成框架？哪些由我们写？

| 工作 | 实际入口 |
|---|---|
| 用本地权重训练 LoRA、按步保存 adapter | ModelScope 的 **ms-swift 命令行** `swift sft`。macOS 上请使用安装了 ms-swift 的 Python 虚拟环境中的 `swift`。 |
| 还原每道训练题的输入和哪些位置算 loss | 若用 ms-swift 训练，Python 端调用同版本的 `get_processor`、`get_template`、`template.set_mode("train")`、`template.encode(...)` 取得真实 `input_ids` 和 `labels`。Hugging Face 原生模板单独验证。 |
| 重载模型、算逐位置 loss | Python 的 Transformers、PyTorch；有 adapter 时再用 PEFT 的 `PeftModel.from_pretrained`。脚本用 `model.eval()` 和 `torch.inference_mode()` 打分。 |
| 找到函数名和参数值的位置、每题输出三个平均数 | 这部分由我们写，见配套脚本。模型提供预测概率所需的 logits，框架的常规训练日志不会替我们生成这张逐题表。 |
| 按学习轨迹挑题（计划） | 拟用 scikit-learn 的 K-means，对每题收集四个时刻的函数名 loss 和四个时刻的参数值 loss，拼成八个数，再聚类。 |

不需要开通付费模型 API，也不用把训练题发给托管聊天服务。首次取得模型、数据和依赖需要下载；加载已下载的权重后可以在本地运行。[ms-swift](https://github.com/modelscope/ms-swift/tree/v4.6.1) · [PEFT adapter 加载接口](https://huggingface.co/docs/peft/package_reference/peft_model)

### 怎样避免把 loss 算错一位？

模型位置 0 的输出预测位置 1 的 token，位置 1 的输出预测位置 2，依此类推。若标准答案的一部分位于位置 50，就要取位置 49 给出的预测。配套脚本按这个对应关系计算。下面三行中的 `target_positions` 是训练标签里有监督的位置：

```python
scores = logits[0, target_positions - 1]
gold = input_ids[0, target_positions]
token_loss = F.cross_entropy(scores.float(), gold, reduction="none")
```

Qwen3.5 会把调用渲染成 `<function=track_order>` 和 `<parameter=order_id>` 等标签。找字段时只在标准答案里找；输入前面的工具说明也写着 `track_order`，不能算进去。我们按实际 tokenizer 的字符范围定位，结构化参数只取里面的值，不把 JSON 键当作值。同一个 token 如果既含符号又含字段字符，如 `=a`，本脚本把它计入该字段并保留整枚 token 的 loss，不能把一个 token 的概率再拆开。没参数的调用将参数值 loss 记为 `null`。正式做八维聚类时，可预先把无参数题单独分层、在该层按函数名的四维轨迹取样，两种选样方法给这一层相同的名额；不要把缺失填成“零难度”。[Qwen 官方工具模板](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/chat_template.jinja)

这个打分方式叫 **teacher forcing**。衡量订单号时，模型能看到此前的**标准**函数名和参数名。因此“参数值 loss 低”只说明给对前文后它比较有把握。要评估它自己能否先选对工具、再填对参数，仍须遮住完整答案，让模型独立生成，再用考试评分器核对。

## 3. 用真实权重和真实工具题得到什么？

首轮记录使用 PyTorch 2.11.0、Transformers 5.3.0、Hugging Face 原生聊天模板和 CPU BF16。固定官方 Qwen3.5-0.8B 版本 `2fc06364715b967f1860aea9cf38778875588b17`，当时核对了权重文件的 SHA-256。该版本 `model.safetensors` 的官方摘要是 `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`。表内为平均交叉熵，使用自然对数。Turnstile 行号从 0 开始。[官方权重版本](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17)

| 测试题 | 输入总 token（含答案） | 受监督答案 token | 函数名 token | 参数值 token | 整答 loss | 函数名 loss | 参数值 loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| 自编短题，查订单 4821 | 313 | 31 | 2 | 4 | 0.00338948 | 0.00008970 | 0.00006344 |
| Turnstile 第 42 行，查配送＋找优惠券 | 1,702 | 151 | 11 | 40 | 0.027895 | 0.007999 | 0.081193 |
| Turnstile 第 49 行，按偏好推荐锻炼计划 | 1,438 | 78 | 6 | 12 | 0.093611 | 0.009124 | 0.159913 |

短题同时让 Transformers 用官方 `labels` 接口计算训练 loss，返回 **0.0033894795924425125**，与手工取整段答案的平均值一致。第 42 行有两次调用，参考答案包含订单号、顾客编号、城市、商品列表和日期字典；保留日期及五个干扰工具。第 49 行是一道单调用题。两题都成功分出了函数名和参数值。它们证明实现能处理这些题型，不能用两题的数值推断全库准确率或选样收益。[第 42 行](https://huggingface.co/datasets/amazon/Turnstile-Synthetic-Domains/viewer/default/train?row=42) · [第 49 行](https://huggingface.co/datasets/amazon/Turnstile-Synthetic-Domains/viewer/default/train?row=49)

## 4. 怎样在项目里用这套打分？

假设 4,000 题训练一遍，每 16 题做一次优化器更新，共约 250 次更新。可保存初始、约三分之一、约三分之二和最后四个时刻。初始快照直接用原权重。每个训练后快照都加载“相同的原权重＋该时刻保存的 adapter”。逐一用相同顺序和模板扫描 4,000 道候选训练题，得到 16,000 次逐题前向。**同一次前向同时产出整答、函数名和参数值的分数**，原版 S2L 和分项版本共享这四遍计算。

| 选题方法 | 交给 K-means 的数字 | 最终 2B 学什么 |
|---|---|---|
| 原版 S2L 对照 | 每题四个整答平均 loss | 被选出的 1,000 道题的完整标准调用 |
| 分项选样 | 每题四个函数名平均 loss 加四个参数值平均 loss | 同样学 1,000 道题的完整标准调用 |
| 随机对照 | 从同池随机抽取 | 同样学 1,000 道题的完整标准调用 |

固定每组选题数、2B 起点和评分规则，按实验方案匹配目标训练的非填充 token 预算，另记答案监督 token 数。特征尺度会影响欧氏距离，具体的标准化策略、参照组和比较顺序见[主实验方案与数据划分](research-plan.md)。

## 5. M1 Pro 上的试跑记录

实测硬件是 14 核 GPU 的 Apple M1 Pro、16 GB 统一内存。PyTorch 2.11.0 成功分配 MPS 张量。下表来自早期本地试跑记录。临时环境与检查点未随仓库保存，耗时及检查点写入情况来自当时记录。

| 已做的事情 | 观察到的结果 | 可以据此判断什么 |
|---|---|---|
| Metal 上给未微调的 0.8B 打分 | FP16、batch 1，两道 1,702 / 1,438 token 的真实题，预热后的前向耗时分别为 **2.172 / 2.024 秒**，关闭了 CPU 算子回退。 | 可以在本机开发逐题打分器。仅把约 2 秒乘以 16,000 次，四遍打分就约 9 小时；两道题只能估数量级，尚未包含训练和考试生成。 |
| Metal 上训练 0.8B LoRA | 同一短题用 FP16 反向传播时出现 NaN 梯度。改 BF16 后，313-token 短题的一次优化器更新用时 **7.063 秒**，检查到的 LoRA 梯度范数约 0.215。 | 短题可以完成一次更新。尚无长题持续训练、2B 训练或整套实验用时的可靠测量。 |
| ms-swift 命令行训练与存档 | 用真实 0.8B 在 CPU FP32 下验证，ms-swift 4.6.1、PEFT 0.18.1、LoRA rank 2，442,368 个可训练参数。确认第 1 步 loss 为 **0.00453959**，梯度范数约 0.378，写出了 `checkpoint-1`。 | 该记录验证了训练入口和首次保存；只恢复到计划三步中的第一步，不能据此声称已在真实权重上完成四时刻轨迹。正式 rank 16 与这个小测试不同。 |
| 2B 和方法比较 | 这次只完成了代理模型的小样本验证；2B 的 200 题摸底与三组微调仍待运行。 | 2B 的基模准确率、选样收益和正式吞吐均待实测。 |

**正式数据比演示题长。** 从官方全库按固定种子各抽了 100 道单、双调用题，保留每道题的干扰工具。含答案的平均长度分别约 **1,514、1,830 tokens**，两组中位数分别为 1,530、1,804。测试中用来跑短题的 1,024-token 设置会截断所有这 200 道题。正式试跑要先统计实际分词长度，用保留完整工具和答案的设置测显存与耗时。另有 1,030 道带“双不同工具”标签的题实际重复调用同一工具，真正两个不同工具的候选有 5,336 道，还要进一步去重。[数据与参考答案](https://huggingface.co/datasets/amazon/Turnstile-Synthetic-Domains)

**训练标签还需要复核一处。** 用 ms-swift Python SDK 手工指定 `loss_scale="default"` 时，自编订单题有 36 个受监督 token。第 3 节 Hugging Face 短题规则得到 31 个。CLI 实际保存的 `loss_scale` 是 `default+ignore_empty_think`，会屏蔽空的 thinking 前缀。即使输入 token 相同，分母也可能不同。脚本用 ms-swift 4.6.1 的配置转换方法，从该次训练保存的 `args.json` 读取模板设置。上表的数值来自 HF 路径。在正式打分前，还要从训练端取得一题的 `input_ids` 和 `labels`，逐位置与验证脚本比对，再用相同权重核对整答 loss。[ms-swift 模板参数转换](https://github.com/modelscope/ms-swift/blob/v4.6.1/swift/arguments/base_args/template_args.py#L172-L199)

### 如何复跑一个小样例

以下命令从仓库根目录运行，适用于 bash/zsh。先用 Python 标准库检查入口，并将自编订单题导出到本机的新目录。导出文件含一个用户回合以及标准工具调用，脚本会拒绝覆盖已有文件。

```bash
python3 scripts/verify_tool_call_loss.py --help
mkdir -p local
SMOKE_DIR=$(mktemp -d local/smoke.XXXXXX)
python3 scripts/verify_tool_call_loss.py --write-swift-jsonl "$SMOKE_DIR/demo.jsonl"
```

**复现 HF 前向。** 在单独的 Python 环境中安装测量使用的两个依赖。通过 Hugging Face CLI 取固定版本的模型和 Amazon Turnstile 的原始文件。模型和训练数据保存在本地，后面的 `--model` 和 `--turnstile-dir` 均可指向已有的离线副本。复现用数据版本的第 42、49 行已与文中工具名核对。数据使用遵循 Amazon 标注的 CC BY-NC 4.0 许可。[固定数据版本](https://huggingface.co/datasets/amazon/Turnstile-Synthetic-Domains/tree/70bf9e9a6234e694efa36d1ec6c207c65a695f5b) · [数据卡和许可](https://huggingface.co/datasets/amazon/Turnstile-Synthetic-Domains) · [HF CLI 文档](https://huggingface.co/docs/huggingface_hub/guides/cli)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
MODEL=local/models/Qwen3.5-0.8B
DATA=local/data/turnstile
hf download Qwen/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 --local-dir "$MODEL"
hf download amazon/Turnstile-Synthetic-Domains data.jsonl api_definitions.json \
  --repo-type dataset --revision 70bf9e9a6234e694efa36d1ec6c207c65a695f5b \
  --local-dir "$DATA"

# 内置短题同时核对手算平均数和 HF 的 labels 接口。
python scripts/verify_tool_call_loss.py --model "$MODEL" --dtype bfloat16 --check-hf
# Turnstile 两题仅计算受监督位置需要的 logits。
python scripts/verify_tool_call_loss.py --model "$MODEL" --dtype bfloat16 --turnstile-dir "$DATA"
```

每行输出一条 JSON。`id` 对应题号，`input_tokens` 包括提示和答案，`answer_tokens` 是所有受监督位置的数量。`function`、`value`、`other` 是三组位置各自的平均 loss，分母见对应的 `*_tokens`。`all` 是全部受监督 token 的平均 loss。`--check-hf` 额外输出 `hf_loss` 并按绝对误差 0.0001 检查一致性。用 `--examples` 可读入自备的 JSONL，每行含 `id`、`tools` 和 `messages`。最后一个消息须为单个 assistant 回合，含非空 `tool_calls`、空文本和对象形式的 `function.arguments`。未提供输入文件时，使用内置订单题。

**可选的 ms-swift 单步检查。** 在同一环境中加入训练依赖，沿用上面的 `MODEL` 和 `SMOKE_DIR`。首轮记录的 torchvision 是 0.26.0，与 PyTorch 2.11.0 配套。Qwen3.5 的 processor 在这条纯文本训练路径上也检查 qwen-vl-utils。若仅用 HF 模板给已有 adapter 打分，只需额外安装 PEFT。[ms-swift 工具训练文档](https://github.com/modelscope/ms-swift/blob/v4.6.1/docs/source_en/Instruction/Agent-support.md)

```bash
python -m pip install -r requirements.txt 'ms-swift==4.6.1' 'peft==0.18.1' \
  'torchvision==0.26.0' 'qwen-vl-utils==0.0.14'
# CPU 单步检查，rank 2 和 max_length 1024 仅用于上面的自编短题。
swift sft --model "$MODEL" --model_type qwen3_5 --template qwen3_5 \
  --agent_template qwen3_5 --dataset "$SMOKE_DIR/demo.jsonl" --use_hf true \
  --tuner_type lora --target_modules up_proj down_proj --lora_rank 2 --lora_alpha 4 \
  --loss_scale default --enable_thinking false --use_cpu true --torch_dtype float32 \
  --bf16 false --fp16 false --attn_impl eager --gradient_checkpointing false \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --max_steps 1 --save_steps 1 --logging_steps 1 --learning_rate 0.0001 \
  --max_length 1024 --split_dataset_ratio 0 --report_to none --check_model false \
  --add_version false --output_dir "$SMOKE_DIR/run"

# 同一题重载本次 adapter，用原训练配置生成监督标签。
python scripts/verify_tool_call_loss.py --model "$MODEL" --adapter "$SMOKE_DIR/run/checkpoint-1" \
  --encoder swift --swift-run-args "$SMOKE_DIR/run/args.json" --check-hf
```

## 6. 原论文和本实验的微调方式

**作者公开的 S2L 复现代码使用全参数 SFT。** 它直接加载预训练模型，将其交给 Hugging Face Trainer 微调。作者演示的代理和目标训练沿用同一入口，代码没有为该方案冻结主干或插入 LoRA。论文附录 B.3 另记载实验使用 4 张 48 GB RTX A6000 和 FSDP。FSDP 描述多卡怎样分摊训练状态，判断全参的依据是模型加载及训练代码。[模型加载](https://github.com/BigML-CS-UCLA/S2L/blob/2bc84dda1a054d0ff0e704ae9658ee700d5c25fa/utils.py#L283-L288) · [训练和保存](https://github.com/BigML-CS-UCLA/S2L/blob/2bc84dda1a054d0ff0e704ae9658ee700d5c25fa/schedule_base.py#L150-L165) · [论文附录 B.3](https://arxiv.org/html/2403.07384v2#A2.SS3)

**本实验计划让 0.8B 和所有 2B 组统一使用 LoRA。** 全参 SFT 更新原模型的权重。LoRA 冻结主干，只更新插入指定线性层的低秩矩阵。两者都能用完整答案的交叉熵训练，也都能记录分项 loss。各 2B 组使用相同的可训练模块、rank 和训练预算，以便将成绩差异归因于选样。结论的适用范围是这些 LoRA 设置。

**推荐分工。** Mac 负责数据清理、模板和标签对齐、小例子的分项 loss 及评分器。在 NVIDIA 上先用有代表性的单、双调用长题，验证 0.8B 与 2B 的一个训练步、保存重载及开发集生成，再确定卡时上限。阶段性 checkpoint 和运行配置写入持久目录；三组正式计时均使用同型号显卡。
