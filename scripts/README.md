# Experiment programs

For a new checkout, start with the [project quickstart](../README.md). The table follows the order in which a request moves through the experiment. Each Python file listed below accepts `--help` unless it is a training callback.

| Stage | Program | What it produces |
| --- | --- | --- |
| Fetch inputs | [Pinned downloader](../s2l_tool_calling/data/download_inputs.py) | Turnstile training data and both Qwen model snapshots, with publisher hashes verified |
| Prepare splits | [prepare_experiment.py](prepare_experiment.py) | Grouped train/dev/test requests and split fingerprints |
| Measure fields | [verify_tool_call_loss.py](verify_tool_call_loss.py) | Supervised token masks and separate losses for function names and argument values |
| Train | [train_experiment.py](train_experiment.py) and [experiment_callback.py](experiment_callback.py) | ms-swift LoRA adapters, exact token counts and checkpoint receipts |
| Score the pool | [score_trajectories.py](score_trajectories.py) | Teacher-forced losses at each proxy checkpoint |
| Select examples | [select_experiment.py](select_experiment.py) | Equal-size A/B/C training subsets and a selection manifest |
| Evaluate | [evaluate_experiment.py](evaluate_experiment.py) and [bfcl_subset.py](bfcl_subset.py) | Per-request predictions, exact-call metrics and official BFCL AST scores |
| Compare results | [summarize_course.py](summarize_course.py) and [summarize_bfcl_seeds.py](summarize_bfcl_seeds.py) | Seed-level effects, paired intervals and provenance checks |

## Experiment controllers

- [run_course_experiment.py](run_course_experiment.py) orders the main experiment. It benchmarks the GPU, fixes the design, trains the proxy and targets, then evaluates the locked models. [benchmark_gpu.py](benchmark_gpu.py) measures the local GPU.
- [run_full_reference.py](run_full_reference.py) trains a full-pool reference model. It uses the [token-budget callback](full_reference_hook.py) and [full-reference summary](summarize_full_reference.py).
- [run_crossed_seeds.py](run_crossed_seeds.py) varies selection and training seeds independently in a development-set diagnostic. [summarize_crossed_seeds.py](summarize_crossed_seeds.py) checks the matched comparisons.

## Diagnostics and figures

| Question | Code |
| --- | --- |
| Which examples and tools did each method select? | [Selection overlap and coverage](../s2l_tool_calling/analysis/describe_selection.py) |
| How did the proxy's losses evolve? | [Proxy dynamics figure](../s2l_tool_calling/analysis/plot_proxy_dynamics.py) |
| How much time did it take to reach each development accuracy? | [Training and selection cost](../s2l_tool_calling/analysis/summarize_dev_cost.py) |
| How do the trained arms compare on the internal test? | [Held-out score and effect plot](../s2l_tool_calling/analysis/plot_course_results.py) |
| Which fields or output representations contributed to errors? | [Field diagnostics](field_diagnostics.py) and [representation diagnostics](representation_diagnostics.py) |

Importable data and analysis programs live under [s2l_tool_calling](../s2l_tool_calling/). For example, `python -m s2l_tool_calling.analysis.plot_course_results --help` prints the inputs needed to redraw the internal test results. The earlier commands such as `python scripts/plot_course_results.py` call the same implementation. The experiment controllers, scoring programs and ms-swift callbacks retain their original source paths because the run manifests record those files' hashes.
