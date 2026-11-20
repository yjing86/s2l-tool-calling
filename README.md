# S2L for Tool Calling

Can a small model choose better fine-tuning examples by tracking **which function to call** and **which argument values to supply** separately? We compare the resulting subsets by how often a larger model generates an entirely correct call on an unseen request.

[中文研究方案](docs/research-plan.md) · [实验执行步骤](docs/experiment-protocol.md) · [字段损失验证与复现](docs/field-loss-validation.md) · [论文阅读笔记](docs/references.md)

## Experiment

A Qwen3.5-0.8B proxy undergoes one LoRA fine-tuning run on 4,000 candidates. We score each reference answer at four checkpoints and select 1,000 requests per method. Each Qwen3.5-2B target learns from complete reference calls with the same training-token budget.

| Group | Signal used to select training requests |
| --- | --- |
| A · Uniform random | No proxy scores |
| B · Whole-answer trajectory | One mean loss over all supervised answer tokens per checkpoint |
| C · Separate field trajectories | Two means per checkpoint, one for function names and one for argument values |
| B′ · Pooled fields, first additional control | One token-weighted mean over exactly the name and value positions used by C |

The primary outcome is the paired difference in **complete-call accuracy** between C and B. C versus B′ tests the value of keeping the fields separate after both selectors discard syntax and parameter keys. A second comparison charges each method for proxy training, scoring and target training, then lets the random baseline use the same GPU time.

## Training data and evaluation

The only training corpus is Amazon’s **Turnstile-Synthetic-Domains**. The pinned proxy and target weights are dated March 2, 2026. Amazon uploaded the public training data on August 3, 2026. We keep near-duplicate requests together in the same split and audit overlap in tool schemas. Those checks reduce the risk from previously published examples. Exposure to related templates during pretraining remains possible. Download the data directly from Amazon under CC BY-NC 4.0. [Amazon dataset snapshot](https://huggingface.co/datasets/amazon/Turnstile-Synthetic-Domains/tree/70bf9e9a6234e694efa36d1ec6c207c65a695f5b) · [0.8B snapshot](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17) · [2B snapshot](https://huggingface.co/Qwen/Qwen3.5-2B/tree/15852e8c16360a2fea060d615a32b45270f8a8fc)

Before training the target, we test it on 200 held-out single-call requests. If it answers at least 180 correctly, the experiment switches to independent double-call requests. We also reserve selected BFCL v4 single-turn categories for external evaluation. Scores will be reported by category. [Experiment protocol](docs/experiment-protocol.md) · [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html)

## Run the field-loss check

From the repository root, the verifier can export the built-in order-tracking example with Python’s standard library. This creates one local JSONL file and requires no model download.

```bash
python3 scripts/verify_tool_call_loss.py --help
mkdir -p artifacts
python3 scripts/verify_tool_call_loss.py --write-swift-jsonl artifacts/demo.jsonl
```

For numerical scoring, install the versions in [requirements.txt](requirements.txt) and follow the pinned download and scoring commands in the [validation guide](docs/field-loss-validation.md). That guide reports measured field losses on two real Turnstile requests and a separate one-step adapter check. The full 2B comparison, BFCL results and NVIDIA GPU hours are pending.

## Repository contents

| Path | Purpose |
| --- | --- |
| [docs/research-plan.md](docs/research-plan.md) | Hypotheses, controls, interpretation and replication criteria |
| [docs/experiment-protocol.md](docs/experiment-protocol.md) | Data splits, model settings, sampling, scoring and compute accounting |
| [docs/field-loss-validation.md](docs/field-loss-validation.md) | Reproduction commands, observed values and the scope of each check |
| [docs/references.md](docs/references.md) | Primary papers, method comparisons and official links |
| [scripts/verify_tool_call_loss.py](scripts/verify_tool_call_loss.py) | Runnable field-loss verifier |

Keep downloaded datasets and weights, run outputs and private working material in the ignored directories listed in [.gitignore](.gitignore).
