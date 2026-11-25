# Which 1,000 Tool Calls Should We Train On?

**Ten-minute research talk · six slides.** Which 1,000 tool calls should we select from 4,000 candidates? The first three seeds suggested that separate field trajectories helped. The two additional seeds averaged −0.05 percentage points for C minus B. Across all five seeds on the same 2,000 requests, C averaged 0.90 points above B and 0.30 below uniform random sampling. Slides 3–5 revisit the initial three-seed results; slide 6 gives the five-seed follow-up. [Results and scoring protocol](results.md)

## 1. A correct tool can still produce a wrong call

**0:00–1:15 · 1 min 15 s**

**On screen.** Show this invented example in two steps. Reveal the expected call, then change `"15:00"` to `"03:00"` and ask the audience to score it. The JSON illustrates the fields and can be displayed without any dataset records.

> “Set a reminder for 3 p.m. on May 12, 2027.”

```json
{"name": "set_reminder", "arguments": {"date": "2027-05-12", "time": "15:00"}}
```

**Say.** “The model chose the right tool, supplied both required arguments and produced valid JSON. How much credit should it get for waking you at three in the morning? Under our complete-call metric, zero. We give credit when the output parses and its call count, function and complete argument object match the reference under the scorer's rules. We evaluate the generated call before execution.

“Now put yourself in the position of the person choosing the training data. A small model can assign high probability to the familiar tool name and punctuation yet struggle with the value that decides whether we wake you up. If we average over the whole response, will we still pick the examples that teach the larger model what it needs? That's the question we tested.” [What the internal scorer checks](results.md#experimental-comparison)

## 2. How a small model selects examples

**1:15–3:00 · 1 min 45 s**

**On screen.** Draw one row showing **4,000 candidates → 0.8B proxy → five measurements per candidate → 10 clusters → 1,000 selected → 2B target**. Under the measurements, label the five training steps **0, 10, 30, 80, 250**. Use the name and value curves in the [proxy dynamics figure](figures/proxy-dynamics.png) to illustrate why early checkpoints matter.

**Say.** “SmallToLarge, or S2L, follows the *same* example over training. At each checkpoint the proxy sees the request and the correct answer prefix. If it assigns a low probability to the next reference token, that token has high loss. Averaging the losses gives one number for the answer. Read the five measurements in order and you get a learning trajectory. An example that starts hard and becomes easy can finish with exactly the same loss as one that was easy all along. Clustering the curves lets us give examples from both patterns a place in the training set.” [Original S2L paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/97fe251c25b6f99a2a23b330a75b11d4-Abstract-Conference.html)

“Our change is about **which tokens supply that curve**. From one forward pass we can average the name tokens and the argument-value tokens separately. By the final checkpoint, 96% of candidates had very low name loss, compared with 30% for values, using the same 0.01 cutoff. We tested whether those two curves were useful for choosing data. We trained the proxy once, then reused its measurements. Both the proxy and every target still trained with the usual loss across the complete reference answer. Once a subset was picked, the 2B model saw ordinary requests paired with complete calls.” [Field-loss validation](field-loss-validation.md) · [Proxy measurements](results.md#from-proxy-loss-to-downstream-errors)

## 3. What does a fair comparison buy us?

**3:00–4:50 · 1 min 50 s**

**On screen.** Use the [held-out accuracy figure](figures/heldout-results.png). Show the unadapted 2B baseline first, then uncover A, B and C. Read the left panel as absolute accuracy and the right panel as a paired difference.

| Model or selector | How the 1,000 examples are chosen | Complete calls correct |
|---|---|---:|
| Base 2B | Before fine-tuning | 47.20% |
| A · Random | Uniform draw from the 4,000 candidates | 55.68% |
| B · Whole answer | Cluster five whole-answer means per candidate | 53.70% |
| C · Separate fields | Cluster five name means and five value means per candidate | 55.23% |

**Say.** “We grouped exact and near-duplicate requests before splitting to limit leakage across sets. For the two trajectory methods we standardized the features, clustered them and allocated places as evenly as possible across clusters. Only the selector changed. Each target started from the same Qwen3.5-2B weights, used the same LoRA settings and stopped on first reaching about three million non-padding input tokens. That budget includes the prompt. Every model answered the same 2,000 single-call requests. We repeated subset selection and training with three paired seeds and locked the test endpoints in advance.

“Here is what each comparison buys us. Fine-tuning on a random quarter of the pool lifted accuracy from 47.20% to 55.68%. C beat B by 2.40, 1.05 and 1.15 points in the three runs, averaging **+1.53 pp**. But picking data with C finished **0.45 pp below A** on average, with a paired request-group interval of **−1.42 to +0.53**. So we have evidence that the representation matters to this selector, and a cheap baseline that still led on average. Which part of the representation made the difference?” [Primary results](results.md#held-out-turnstile-results)

## 4. Where does the field gain come from?

**4:50–6:30 · 1 min 40 s**

**On screen.** Show **B, 53.70% → B′, 54.77% → C, 55.23%**. Under B′ write “pool name and value tokens into one mean per checkpoint.” Under C write “give each field its own mean.” The first two rows of the [follow-up interval figure](figures/followup-effects.png) show the paired contrasts. Differences use unrounded rates.

**Say.** “Moving directly from B to C makes two changes at once. First we remove other answer tokens, including argument keys and syntax. Then we give the name and values separate coordinates. We built the intermediate method B-prime to distinguish these decisions. B-prime scores exactly the tokens C scores. A value with many tokens can dominate its pooled mean. C gives the name and value their own coordinate at each checkpoint.

“Moving from B to B-prime gained **1.07 pp** across seeds, with a request-group interval of **+0.27 to +1.88**. Moving from B-prime to C gained a further **0.47 pp**, with a request-group interval from **−0.40 to +1.32**. The larger observed change came from where we measured loss. Three runs leave the extra benefit of keeping separate curves uncertain.

“There are two kinds of uncertainty here. Take the headline C−B gap. If these trained models faced a different mix of test requests, jointly resampling the requests gives a 95% interval of **+0.60 to +2.48 pp**. Now hold the requests fixed. To see how stable the result is across selection and training seeds, use the three paired seed differences. They give a 95% t interval of **−0.34 to +3.40 pp**. With only three pairs, that second interval is wide. Both refer to the same +1.53-point estimate.” [Three-seed field controls](../results/followup/phase1-results.json) · [Both intervals for the headline contrast](../results/followup/p05-intervals.csv)

## 5. Does the selector favor long or difficult arguments?

**6:30–8:40 · 2 min 10 s**

**On screen.** Build this table one row at a time. Use “length” for the number of tokens in the reference values. Use “difficulty” for how surprising those tokens were to the *initial 0.8B proxy*, measured by their average negative log-probability.

| Selector | Length-bin counts | Difficulty-bin counts | Accuracy | Difference from A |
|---|---|---|---:|---:|
| A · Random | Observed in a uniform draw | Observed in the same draw | 55.68% | — |
| B · Whole answer | Observed after trajectory selection | Observed after trajectory selection | 53.70% | −1.98 pp |
| E2 · Length control | Matched to B | Free to vary | 54.43% | −1.25 pp |
| E2b · Length and difficulty control | Matched to A | Matched to B | 55.37% | −0.32 pp |

**Say.** “Why did B trail a uniform draw? B selected examples averaging 19.58 argument-value tokens. A selected examples averaging 15.37. B also selected more examples that the initial proxy found difficult, even among values of the same length. A rare cluster can get many slots when we balance quotas across clusters. This gave us two possible explanations to investigate.” [Training-pool difficulty audit](../results/followup/difficulty/README.md)

“For E2, we copied the number of examples B took from each length bin and drew randomly within the bins. E2 finished below A in all three runs. Next we asked whether B's difficult values alone would reproduce its nearly two-point deficit. E2b kept A's counts by length bin but matched B's counts by difficulty bin. Its mean was just **0.32 pp below A**. The per-seed differences were **+0.40, −1.45 and +0.10 pp**. Both its request-group interval, **−1.20 to +0.58**, and its seed interval, **−2.78 to +2.15**, cover zero.

“Matching B's length quotas cost accuracy in the random-within-bin comparison. Matching its difficulty mix at A-like lengths produced a much smaller average deficit. Narrower claims are warranted here. Different examples and tools enter each subset, length can still vary inside a bin, and E2b versus E2 changes *both* margins. These comparisons describe selection policies. Separate causal effects of length and difficulty would require a different design. We developed these checks after seeing the original test results, and evaluated them on those same requests.” [Phase 1 controls](../results/followup/phase1-results.json) · [E2b counts and paired comparisons](../results/followup/e2b-results.json)

## 6. What can we take to the next experiment?

**8:40–10:00 · 1 min 20 s**

**On screen.** Show the five-seed results: A 54.95%, B 53.75%, C 54.65%. The December 5 evaluation will test C against B on 1,500 reserved requests.

**Say.** “We added two seeds on the same test requests. C minus B was +0.40 and −0.50 points. The initial gain shrank, and matching the length distribution of B in random sampling gave nearly the same five-seed accuracy as unrestricted random sampling. We cannot attribute the original deficit to length alone. Random sampling also saves the roughly 41 minutes of proxy training and scoring on one A40.

“We completed the main study, checked sensitivity to the training seed, and trained a full-pool reference. The follow-up added eleven runs for field, length and tool-coverage controls and three for the joint length–difficulty control. The internal finding has a boundary. On a separate 200-request Berkeley Function Calling Leaderboard check, the average difference between C and B was −1.33 points. The internal advantage did not carry over there.

“We have frozen 47 model configurations. Our primary comparison on the reserved requests asks whether C beats B across paired seeds. C versus random is a secondary comparison. For a single training run today I would use random sampling. Its accuracy on the original test and its lower cost justify that choice.” [Frozen evaluation protocol](round2-holdout.md) · [Measured GPU time](../results/compute-time.json)

## Backup questions

- **Did C actually fix argument mistakes?** In cases where B and C already agreed on the correct tool and call count, C gained 42, 19 and 23 correct calls across the three seeds. Those cases account for 84 of the 92 net gains, and the same 2,000 requests recur across seeds. Incorrect values fell on every seed while missing argument keys became more common. This is a descriptive breakdown of paired errors. [Field error analysis](results.md#from-proxy-loss-to-downstream-errors)
- **Could we just train on everything?** At seed 42, the full-pool run reached 56.30% at matched target tokens, versus 55.90% for A. With roughly four times the tokens and two full passes it reached 59.45%. That comparison changes both exposure and training budget. [Full-pool checkpoints](../results/full-pool-reference.json)
- **Did you equalize total GPU time?** We equalized target input tokens and separately measured selection overhead. The shared proxy took 40.56 minutes to train and score once. A target subset fit took about 16 minutes, before evaluation. Across all five stages, the recorded GPU-dependent subprocess time was 19.08 A40-hours, including loading and I/O. The experiment leaves open how much accuracy random sampling could buy if it reinvested every saved proxy minute. [Compute accounting](../results/compute-time.json)
- **Why report a seed interval with only three seeds?** It makes sensitivity to training and selection randomness visible. On the separate 400-request development set, keeping the selection seed at 42 and changing only the training seed moved C−B from +4.25 to −3.25 pp. The crossed design tested two values of each seed and does not estimate population variance components. [Crossed-seed check](../results/crossed-seeds.json)
- **What does the external check cover?** It uses 100 single-candidate and 100 multiple-candidate single-turn BFCL requests with official scoring. C−B was −2.00 and −0.67 pp in those slices. The equal-weight mean was −1.33 pp, with an exploratory 95% request interval of −3.00 to +0.33. Broader generalization and success in a live tool environment remain open. [External results and interpretation](results.md#external-tool-calling-evaluation) · [BFCL aggregate](../results/bfcl-single-turn.json)
