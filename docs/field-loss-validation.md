# Validating per-field loss for tool calls

November 19, 2026 · [Validation script](../scripts/verify_tool_call_loss.py) · [Research plan](research-plan.md)

One forward pass through Qwen3.5-0.8B gave separate losses for the function name and argument values on a hand-written example and two Turnstile examples. This note records the initial implementation check. At the time, the unadapted 2B pilot and the selection experiments had yet to run. The original smoke-test logs, weights and adapters are outside the public repository. The [results report](results.md) covers the subsequent experiments.

## 1. What loss do the models train on?

**Both the 0.8B proxy and the 2B target optimize the mean cross-entropy over the supervised tokens in the complete reference call.** The user's request and the tool descriptions provide context. Padding is excluded from the mean. Function names, argument names and values, call syntax, and any end token labeled as part of the answer can all contribute. Every 2B arm uses this same training objective. [ms-swift tool-call training and label examples](https://github.com/modelscope/ms-swift/blob/v4.6.1/docs/source_en/Instruction/Agent-support.md)

Suppose the user asks to track order 4821, and the reference call is `track_order(order_id="4821")`. At each answer position, the model assigns a probability to the next reference token. A lower probability gives that position a higher loss. The trainer takes the mean of those losses and uses that scalar for backpropagation. Keeping the losses **before averaging** lets us examine how much of the difficulty lies in the function name or the order number. A training log containing only the mean has lost this information. [Qwen3.5 model outputs](https://huggingface.co/docs/transformers/model_doc/qwen3_5) · [PyTorch per-position cross-entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy)

| Positions included in the mean | What the statistic measures | Effect on the weights |
| --- | --- | --- |
| All supervised answer tokens | The objective for either model during training, or a whole-answer score for one example at a frozen checkpoint | Updates weights during training. During scoring the weights stay frozen. |
| Tokens overlapping the function name | The proxy's confidence in the reference function name at that checkpoint | Scoring only |
| Tokens overlapping argument values | The proxy's confidence in values such as the order number or city, given the reference prefix | Scoring only |

A training step already produces predictions for every supervised answer position. The optimizer uses one scalar, and computing field-specific scores requires no additional backward pass. To measure how a particular example changes over training, we load five frozen checkpoints and score the same examples at each one. Those additional forward passes cost compute.

## 2. Which parts come from existing tools?

| Task | Implementation |
| --- | --- |
| Train a LoRA adapter from local weights and save it at specified steps | The ModelScope **ms-swift** command `swift sft`. On macOS, use the `swift` executable from the Python environment in which ms-swift is installed. |
| Reconstruct the training input and its supervised positions | For an ms-swift run, use the same library version and call `get_processor`, `get_template`, `template.set_mode("train")` and `template.encode(...)` to obtain the actual `input_ids` and `labels`. The Hugging Face native template has a separate check. |
| Reload a model and compute per-position loss | Transformers and PyTorch, plus PEFT's `PeftModel.from_pretrained` when using an adapter. The scoring script uses `model.eval()` and `torch.inference_mode()`. |
| Locate the function names and argument values and return three scores per example | The project script identifies spans and aggregates the model's logits at their supervised token positions. Standard training logs report only an aggregate. |
| Select examples from learning trajectories | The planned selector uses scikit-learn K-means. Five function-name losses and five argument-value losses form a ten-dimensional feature vector for one example. |

After the initial download of the weights, dataset and dependencies, scoring runs locally without a paid model API or a hosted chat service. [ms-swift](https://github.com/modelscope/ms-swift/tree/v4.6.1) · [PEFT adapter loading](https://huggingface.co/docs/peft/package_reference/peft_model)

### Aligning logits with the next token

The logits at position 0 predict the token at position 1. To score a reference token at position 50, take the logits at position 49. Here `target_positions` contains the positions that have supervised training labels.

```python
scores = logits[0, target_positions - 1]
gold = input_ids[0, target_positions]
token_loss = F.cross_entropy(scores.float(), gold, reduction="none")
```

The Qwen3.5 template renders calls with tags such as `<function=track_order>` and `<parameter=order_id>`. The scorer searches the **reference answer** for fields. The name `track_order` can also appear in the list of available tools in the prompt. Tokenizer character offsets identify the answer tokens overlapping each field. For structured arguments, the scorer keeps the leaf values and excludes JSON keys. A token such as `=a` can contain both syntax and a field character. When it overlaps the field, we assign the loss for that entire token to the field. The model supplies one probability for the token as a whole.

A call without arguments receives `null` for its argument-value loss. One proposed policy for a ten-dimensional selector was to put such calls in a separate stratum, use the five-dimensional function-name trajectory there, and assign both methods the same quota. Replacing a missing value with zero would treat missing supervision as perfect confidence. [Official Qwen tool template](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/chat_template.jinja)

These are **teacher-forced scores**. When predicting the order number, the model already sees the correct function name and argument name in its prefix. Low argument-value loss therefore measures confidence conditional on that prefix. To measure whether the model can choose the tool and produce its arguments on its own, we withhold the entire reference answer, generate a call and score the prediction.

## 3. Measurements on real model weights

The first measurements used PyTorch 2.11.0, Transformers 5.3.0, the native Hugging Face chat template and CPU BF16. We pinned Qwen3.5-0.8B to revision `2fc06364715b967f1860aea9cf38778875588b17` and verified the weights' SHA-256. The published digest for `model.safetensors` at this revision is `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`. Losses below are mean cross-entropies in natural-log units. Turnstile row numbers are zero-based. [Pinned model revision](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17)

| Example | Input tokens including answer | Supervised answer tokens | Function-name tokens | Argument-value tokens | Whole-answer loss | Function-name loss | Argument-value loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hand-written prompt, track order 4821 | 313 | 31 | 2 | 4 | 0.00338948 | 0.00008970 | 0.00006344 |
| Turnstile row 42, check a delivery and find coupons | 1,702 | 151 | 11 | 40 | 0.027895 | 0.007999 | 0.081193 |
| Turnstile row 49, recommend a workout plan based on preferences | 1,438 | 78 | 6 | 12 | 0.093611 | 0.009124 | 0.159913 |

On the short example, Transformers' own `labels` interface returned **0.0033894795924425125**, matching the manually averaged loss over the answer. Row 42 contains two calls. Its reference includes an order number, customer ID, city, product list and date dictionary. We retained the date and five distractor tools. Row 49 contains one call. We could isolate the function names and argument values in both examples. Two examples establish that the implementation handled those input structures. Dataset-wide accuracy and the effect of data selection require held-out generation experiments. [Row 42](https://huggingface.co/datasets/amazon/Turnstile-Synthetic-Domains/viewer/default/train?row=42) · [Row 49](https://huggingface.co/datasets/amazon/Turnstile-Synthetic-Domains/viewer/default/train?row=49)

## 4. Using these scores for sample selection

One pass over 4,000 examples with one optimizer update per 16 examples gives approximately 250 updates. Score the initial weights at step 0 and save adapters at steps 10, 30, 80 and 250. For each later checkpoint, load the same base weights with the adapter saved at that step. Score all 4,000 candidates in the same order with the same template. This gives 20,000 example-level forward evaluations. **Each forward evaluation supplies all three scores**, so both trajectory selectors share the cost of the five passes.

| Selection method | Input to the selector | Training target for the 2B model |
| --- | --- | --- |
| Whole-answer S2L baseline | Five mean whole-answer losses per example for K-means | Complete reference calls for the selected 1,000 examples |
| Field-level selection | Five mean function-name losses and five mean argument-value losses per example for K-means | Complete reference calls for the selected 1,000 examples |
| Uniform random baseline | A random sample from the same pool | Complete reference calls for the selected 1,000 examples |

Each arm has the same selection count, initial 2B weights and evaluation rules. The protocol matches the non-padding input-token budget for target training and separately records the number of supervised answer tokens. Feature scale affects Euclidean distance. The [research plan and split definitions](research-plan.md) specify the standardization, comparison arms and order of comparisons.

## 5. Early feasibility tests on an M1 Pro

The test machine was an Apple M1 Pro with a 14-core GPU and 16 GB of unified memory. PyTorch 2.11.0 successfully allocated an MPS tensor. Timings and checkpoint writes below come from the early local tests. The temporary environments and checkpoints were not archived in Git.

| Check | Measurement | What it established at the time |
| --- | --- | --- |
| Score the unadapted 0.8B model on Metal | With FP16, batch size 1 and CPU operator fallback disabled, warmed-up forward passes took **2.172 s and 2.024 s** on real inputs of 1,702 and 1,438 tokens, respectively. | The Mac could be used to develop the scorer. Extrapolating 2 seconds per example to 20,000 evaluations gives about 11 hours for scoring alone. Two examples supply only an order-of-magnitude estimate, excluding training and evaluation generation. |
| Train a 0.8B LoRA adapter on Metal | On the short example, FP16 backpropagation produced NaN gradients. With BF16, one optimizer update on the 313-token example took **7.063 s**. The observed LoRA gradient norm was about 0.215. | The device completed one short update. This check did not measure sustained training on long inputs, 2B training or total experiment time. |
| Train and save from the ms-swift CLI | CPU FP32 on the real 0.8B model, ms-swift 4.6.1, PEFT 0.18.1, LoRA rank 2 and 442,368 trainable parameters. Step 1 produced a loss of **0.00453959**, a gradient norm of about 0.378 and a `checkpoint-1` directory. | The record covers the training entry point and first save. Only step 1 of three planned steps was recovered. The five-checkpoint trajectory and the formal rank-16 setup required separate validation. |
| 2B model and comparison arms | Only the small proxy checks had finished at this stage. The 200-example base-model pilot and three target fine-tuning arms came later. | The early checks alone provided no measured base accuracy, selection gain or full-run throughput. |

**Input lengths matter for memory and timing.** We used fixed seeds to sample 100 single-call and 100 two-call examples from the official corpus, retaining each example's distractor tools. Mean sequence lengths including the answers were approximately **1,514 and 1,830 tokens**. The medians were 1,530 and 1,804 tokens, respectively. A 1,024-token limit, suitable for the short check, would truncate every one of these 200 examples. A representative hardware pilot should use complete tools and answers and measure the tokenized lengths. In the corpus, 1,030 rows labeled as using two different tools actually called the same tool twice. The pool with two genuinely distinct tools contained 5,336 examples before further deduplication. [Dataset and reference answers](https://huggingface.co/datasets/amazon/Turnstile-Synthetic-Domains)

**The training and scoring label masks need to agree.** Manually specifying `loss_scale="default"` in the ms-swift Python SDK gave 36 supervised tokens on the hand-written order example. The Hugging Face setup in Section 3 gave 31. The CLI saved `loss_scale` as `default+ignore_empty_think`, which masks an empty thinking prefix. The denominator can therefore differ even when the input tokens are identical. Our script uses the ms-swift 4.6.1 configuration conversion to restore template settings from the run's `args.json`. The Section 3 numbers use the Hugging Face path. The prescribed preflight before formal scoring was to export one example's `input_ids` and `labels` from the trainer, compare them position by position with the validation script, and then compare the whole-answer loss under identical weights. [ms-swift template argument conversion](https://github.com/modelscope/ms-swift/blob/v4.6.1/swift/arguments/base_args/template_args.py#L172-L199)

### Reproduce a small example

Run the following commands at the repository root in bash or zsh. The first commands use the Python standard library and export the hand-written order prompt into a new local directory. The file contains one user turn and its reference call. The script refuses to overwrite an existing file.

```bash
python3 scripts/verify_tool_call_loss.py --help
mkdir -p local
SMOKE_DIR=$(mktemp -d local/smoke.XXXXXX)
python3 scripts/verify_tool_call_loss.py --write-swift-jsonl "$SMOKE_DIR/demo.jsonl"
```

**Reproduce the Hugging Face forward pass.** Install the two measured dependencies in a separate Python environment, then use the Hugging Face CLI to obtain the pinned model and Amazon Turnstile files. The `--model` and `--turnstile-dir` options can also point to existing offline copies. We checked the tool identities in rows 42 and 49 at this dataset revision. The dataset is distributed by Amazon under CC BY-NC 4.0. [Pinned dataset revision](https://huggingface.co/datasets/amazon/Turnstile-Synthetic-Domains/tree/70bf9e9a6234e694efa36d1ec6c207c65a695f5b) · [Dataset card and license](https://huggingface.co/datasets/amazon/Turnstile-Synthetic-Domains) · [Hugging Face CLI documentation](https://huggingface.co/docs/huggingface_hub/guides/cli)

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

# Compare the manual mean and the HF labels interface on the built-in example.
python scripts/verify_tool_call_loss.py --model "$MODEL" --dtype bfloat16 --check-hf
# For the two Turnstile examples, compute logits only at supervised positions.
python scripts/verify_tool_call_loss.py --model "$MODEL" --dtype bfloat16 --turnstile-dir "$DATA"
```

The script prints one JSON record per line. `id` identifies the example. `input_tokens` counts prompt and answer tokens, and `answer_tokens` counts supervised positions. `function`, `value` and `other` are the mean losses for their respective positions, with denominators given by `*_tokens`. `all` is the mean over all supervised tokens. The `--check-hf` option also prints `hf_loss` and checks agreement with an absolute tolerance of 0.0001. Supply your own JSONL with `--examples`. Each record needs `id`, `tools` and `messages`. Its last message must be an assistant turn with nonempty `tool_calls`, empty text and object-valued `function.arguments`. With no input file, the script uses the built-in order example.

**Optional one-step ms-swift check.** In the same environment, add the training dependencies and reuse `MODEL` and `SMOKE_DIR` from above. The original test paired torchvision 0.26.0 with PyTorch 2.11.0. The Qwen3.5 processor also checks for qwen-vl-utils on this text-only training path. Scoring an existing adapter with the Hugging Face template requires only PEFT in addition to the initial dependencies. [ms-swift tool-training documentation](https://github.com/modelscope/ms-swift/blob/v4.6.1/docs/source_en/Instruction/Agent-support.md)

```bash
python -m pip install -r requirements.txt 'ms-swift==4.6.1' 'peft==0.18.1' \
  'torchvision==0.26.0' 'qwen-vl-utils==0.0.14'
# CPU smoke test. Rank 2 and max_length 1024 apply only to this short example.
swift sft --model "$MODEL" --model_type qwen3_5 --template qwen3_5 \
  --agent_template qwen3_5 --dataset "$SMOKE_DIR/demo.jsonl" --use_hf true \
  --tuner_type lora --target_modules up_proj down_proj --lora_rank 2 --lora_alpha 4 \
  --loss_scale default --enable_thinking false --use_cpu true --torch_dtype float32 \
  --bf16 false --fp16 false --attn_impl eager --gradient_checkpointing false \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --max_steps 1 --save_steps 1 --logging_steps 1 --learning_rate 0.0001 \
  --max_length 1024 --split_dataset_ratio 0 --report_to none --check_model false \
  --add_version false --output_dir "$SMOKE_DIR/run"

# Reload this adapter and regenerate the labels from the saved training configuration.
python scripts/verify_tool_call_loss.py --model "$MODEL" --adapter "$SMOKE_DIR/run/checkpoint-1" \
  --encoder swift --swift-run-args "$SMOKE_DIR/run/args.json" --check-hf
```

## 6. Fine-tuning in S2L and in this study

**The public S2L reference implementation uses full-parameter supervised fine-tuning (SFT).** It loads the pretrained model directly and passes it to Hugging Face Trainer. The authors use the same entry point for their example proxy and target runs. The model-loading and training code leaves the backbone trainable and does not insert LoRA adapters. Appendix B.3 also reports using FSDP with four 48 GB RTX A6000 GPUs. FSDP describes how training state is sharded across devices. The evidence for full-parameter updates comes from the implementation. [Model loading](https://github.com/BigML-CS-UCLA/S2L/blob/2bc84dda1a054d0ff0e704ae9658ee700d5c25fa/utils.py#L283-L288) · [Training and saving](https://github.com/BigML-CS-UCLA/S2L/blob/2bc84dda1a054d0ff0e704ae9658ee700d5c25fa/schedule_base.py#L150-L165) · [Paper appendix B.3](https://arxiv.org/html/2403.07384v2#A2.SS3)

**Our protocol uses LoRA for the 0.8B proxy and every 2B arm.** Full-parameter SFT updates the original model weights. LoRA freezes the backbone and learns low-rank updates to the specified linear layers. Both parameterizations can optimize complete-answer cross-entropy and support field-level scoring. Keeping the trainable modules, rank and budget fixed across 2B arms isolates the contribution of sample selection. Conclusions are specific to the tested LoRA settings.

**Hardware workflow.** The recommended split of work was to use the Mac for data preparation, checking template and label alignment, developing the example-level scorer and validating the evaluator. On an NVIDIA GPU, first test one training step for both model sizes on representative long single-call examples, save and reload the adapters, and generate development predictions before setting the time budget. Save checkpoints and configurations on persistent storage and time the formal arms on the same GPU type.
