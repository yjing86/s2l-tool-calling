# Which 1,000 tool calls should we train on?

Across five paired seeds, full-pool training at matched target tokens differed from the 25% random subset by +1.01 percentage points. At the 10% subset budget, field trajectories differed from uniform random by -0.49 points.

Ten-minute presentation for four speakers. Aim to finish at 9:35. Speaker assignments below are for rehearsal. Each person should describe their actual contribution when asked.

## 1 · What makes a useful training example? (0:00–1:10, speaker 1)

**Show** a request to set a reminder for 3 p.m. and a generated call containing the correct tool and the time `03:00`. Ask the audience to score the response before revealing the answer.

**Say:** “Would you trust this reminder? The model wrote valid JSON and chose the right tool, but it will wake you twelve hours early. Our evaluator gives that call zero credit. Now imagine choosing 1,000 training requests out of 4,000. A whole-answer average gives credit for every predictable token, even when the time is hard to predict. Could a small model select better training examples if we tracked the name and argument values separately? We expected this change to help the target model generate more correct calls.”

## 2 · What comes from S2L, and what did we change? (1:10–2:30, speaker 1)

**Show** two example learning curves with similar final loss and different early histories. Label them illustrative. Then show one whole-answer curve per example for B and two field curves for C.

**Say:** “SmallToLarge measures how an example’s loss changes while a small proxy model learns. It clusters examples with similar trajectories and samples across clusters. We kept that idea and changed what the proxy measures. At each checkpoint we separately average the losses on the correct function name and on the correct argument values. Imagine a request with a familiar function and an unfamiliar date expression. Two field means can preserve a distinction that one average hides. Whether the distinction helps a larger model is an empirical question.” [S2L paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/97fe251c25b6f99a2a23b330a75b11d4-Abstract-Conference.html)

## 3 · How do we isolate the value of selection? (2:30–4:25, speaker 2)

**Show** one pipeline: 4,000 Turnstile candidates → one 0.8B proxy → steps 0, 10, 30, 80, 250 → selectors → 1,000 examples per run → 2B LoRA fine-tuning. Label A uniform random, B five whole-answer features, C ten field features.

**Say:** “We trained the proxy once. At each checkpoint it saw the request and the correct answer prefix. One forward pass gave us token losses that we could average over different positions. Both trajectory selectors standardized their features, used ten clusters and distributed the 1,000 places as evenly as possible. The random baseline skipped the proxy. All target models learned the complete reference call with the same training objective and about three million nonpadding input tokens. C versus B asks whether the representation helps. C versus A asks whether it is worth computing at all. We scored generated calls on the same requests so each comparison is paired.” [Implementation and budgets](reproduce.md)

## 4 · Would our early result repeat? (4:25–6:30, speaker 3)

**Show** three numbers, one at a time: C minus B was +1.53 points on the original test with three seeds, +0.90 on that same test with five seeds, and +0.08 on new requests with the same five seeds. Then show the reserved-test scores: base 47.27%, A 55.19%, B 54.57%, C 54.65%.

**Say:** “The first three runs all favored separate fields. Had we stopped there, we might have recommended them. Adding two seeds weakened the result. We then used 1,500 previously unused request groups with the models and primary test fixed before opening. C beat B in two of five seeds. Across 7,500 paired answers C got six more right. The five-seed interval for the difference runs from −1.80 to +1.96 percentage points. The registered one-sided p-value was 0.456. Our experiment did not establish a repeatable improvement. It still allows modest gains or losses. In contrast, even random-subset fine-tuning improved on the untuned model by 7.92 points.” [Reserved-test figure and results](results.md)

## 5 · Did the selector choose too many long arguments? (6:30–8:00, speaker 4)

**Show** the observed training-set means of 19.58 argument-value tokens for B and 15.37 for A in the initial analysis. Beside them put the new-test long-minus-short effect of −0.15 points, with its five-seed 95% interval [−1.00, +0.70].

**Say:** “We could see a length bias in what the original selector picked. To test whether that explained the performance gap, we changed the sampling policy directly. We drew randomly within length bins, targeting means of roughly 12 or 24 argument-value tokens, then retrained at the same target-token budget. The long group trailed the short group by only 0.15 points on the new requests. Matching B’s length quotas in a random selector also produced accuracy close to uniform random. We can document the sampling bias. These controls leave its role in downstream accuracy unresolved.” [Sampling controls](results.md)

## 6 · What should we do with the next GPU hour? (8:00–9:35, speaker 4)

**Show** one recommendation, one boundary, and one concrete proposed analysis for the December 11 report.

**Say:** “For a single fine-tune at this budget, we would sample randomly. On the reserved requests it scored 55.19%, compared with 54.65% for our method, and avoided about 41 minutes of proxy work on an A40. Our evidence covers one target-model family and single-call requests. For the report, we propose using the existing predictions to separate values copied from the request from values requiring a transformation, like converting a relative date. We would define the categories before checking the method differences, annotate without seeing model outputs, report group sizes and paired differences, and mark the analysis exploratory. That would tell us where to look in a new study. The main result is already clear enough to guide a training decision: we have no measured return on the extra selection cost in this setting.”

## Rehearsal and individual defense

The rubric assigns 15 points to idea and motivation, 10 to related work, 20 to methodology, 20 to experiments and interpretation, 15 to critical thinking and future directions, 5 to communication and 15 to Q&A. The rubric permits negative results and assigns grades individually. Reserve the last 25 seconds for handoffs or an overrun. Each member should be able to explain both comparisons, the training loss, the paired uncertainty, their own contribution and one limitation without a teammate answering first. If asked about early opening, explain that the date was amended publicly after inspecting the old test and external results and before seeing reserved outcomes. [Opening record](round2-opening-amendment.md)

## Backup questions

- **Did C actually fix argument mistakes?** In cases where B and C already agreed on the correct tool and call count, C gained 42, 19 and 23 correct calls across the three seeds. Those cases account for 84 of the 92 net gains, and the same 2,000 requests recur across seeds. Incorrect values fell on every seed while missing argument keys became more common. This is a descriptive breakdown of paired errors. [Field error analysis](results.md#from-proxy-loss-to-downstream-errors)
- **Could we just train on everything?** At seed 42, the full-pool run reached 56.30% at matched target tokens, versus 55.90% for A. With roughly four times the tokens and two full passes it reached 59.45%. That comparison changes both exposure and training budget. [Full-pool checkpoints](../results/full-pool-reference.json)
- **Did you equalize total GPU time?** We equalized target input tokens and separately measured selection overhead. The shared proxy took 40.56 minutes to train and score once. A target subset fit took about 16 minutes, before evaluation. Across all five stages, the recorded GPU-dependent subprocess time was 19.08 A40-hours, including loading and I/O. The experiment leaves open how much accuracy random sampling could buy if it reinvested every saved proxy minute. [Compute accounting](../results/compute-time.json)
- **Why report a seed interval with only three seeds?** It makes sensitivity to training and selection randomness visible. On the separate 400-request development set, keeping the selection seed at 42 and changing only the training seed moved C−B from +4.25 to −3.25 pp. The crossed design tested two values of each seed and does not estimate population variance components. [Crossed-seed check](../results/crossed-seeds.json)
- **What does the external check cover?** The five-seed BFCL Live panel contains 256 single-candidate and 300 multiple-candidate single-turn requests. The official AST checker evaluates generated calls against references. An earlier three-seed check on 200 Non-Live requests gave a C−B difference of −1.33 pp averaged across categories. These panels use different requests. Neither measures downstream success after executing a tool. [Live results](results.md#external-evaluation-on-bfcl-live) · [Earlier Non-Live results](../results/bfcl-single-turn.json)
