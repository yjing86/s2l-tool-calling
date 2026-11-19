# S2L for Tool Calling

Does using a small model to select fine-tuning examples improve function calling when the selection step counts against the GPU budget?

This repository starts with the experiment protocol. Implementation and measurements are pending.

## Experiment

A Qwen3.5-0.8B proxy will record how the loss on each training answer changes during supervised fine-tuning. We will cluster those trajectories and sample a subset using the balanced allocation in SmallToLarge (S2L), Algorithm 1. A Qwen3.5-2B target will then train on that subset with LoRA.

| Comparison | Question |
| --- | --- |
| Unmodified target vs. fine-tuned target | How much room is there for fine-tuning? |
| S2L vs. uniform and stratified random sampling, with comparable target-training tokens | Does the proxy select more informative examples? |
| S2L vs. random training with the same total GPU time | Does the benefit justify training and scoring with the proxy? |

## Data and evaluation

We plan to use the Salesforce xLAM function-calling dataset. Our proposed split is approximately 4,000 training, 400 development, and 800 test examples, grouped by near-duplicate request templates. Each test prompt contains the request and candidate tool schemas. Reference calls are used only for scoring. Dataset access is available through the publisher's repository. [Salesforce xLAM dataset](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)

The primary metric is **complete tool-call accuracy**. A request counts as correct when all required calls match the reference function names and argument values. Malformed outputs, missing calls, and extra calls count as errors. For example, a prediction that swaps the departure and arrival airports fails even when it names the correct flight-search function. We will report paired accuracy differences alongside GPU hours for proxy training, trajectory scoring, and target fine-tuning.

The first milestone is to validate the native Qwen tool-call parser and measure the unmodified target on the development set. That pilot will determine whether the task has enough headroom and whether the planned single-GPU runs fit the budget. An external check on the BFCL single-turn categories is planned if time permits. [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)

## References

- Yang et al., *SmallToLarge (S2L)*, NeurIPS 2024. [Paper](https://arxiv.org/abs/2403.07384) · [Authors' implementation](https://github.com/YuYang0901/S2L)
- Qwen Team, Qwen3.5 model cards. [0.8B proxy](https://huggingface.co/Qwen/Qwen3.5-0.8B) · [2B target](https://huggingface.co/Qwen/Qwen3.5-2B)
