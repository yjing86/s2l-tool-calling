# How can a small model choose training examples for a larger one?

Suppose you have 4,000 tool-calling requests and can fine-tune a model on 1,000. What evidence would justify spending time choosing which 1,000 to keep? **SmallToLarge (S2L)** proposes a selection method. **Compute-Constrained Data Selection** asks how much compute that method consumes. **ToolACE-R** estimates the difficulty of a request by having a model actually answer it. **Chasing Random** tests whether a selection method's advantage survives changes in the dataset, evaluation and random sample. Each paper studies a different task and model. The results below are tied to the experiments that produced them. The worked examples are illustrative.

The four sections below link to the authors' papers and explain which experimental comparisons each paper motivated. Consult the published figures and equations for the original definitions and numerical results.

## 01 · SmallToLarge — what does it look like when a model learns an example?

**Yu Yang, Siddhartha Mishra, Jeffrey Chiang and Baharan Mirzasoleiman, NeurIPS 2024.** Start with Algorithm 1, then Figure 4 and Tables 1–2, Figures 8–9, and Appendix B.3.

### Why keep a curve for each example?

During fine-tuning, a model receives the prompt and the correct answer prefix and predicts the next reference token. A reference token assigned probability *p* contributes a loss of −log(*p*). Averaging over the answer tokens gives one loss for the example. Repeating that measurement on **the same example** throughout training produces a *learning trajectory*. A single checkpoint tells us how difficult the example is at that moment. Multiple checkpoints help distinguish examples the model already handled well from ones it learned during training.

Consider three illustrative trajectories measured at three checkpoints. Example A has losses [0.3, 0.2, 0.2], B has [2.4, 1.2, 0.2] and C has [2.5, 1.3, 0.3]. A and B finish with the same loss, yet B and C follow much more similar paths. If the training pool contains hundreds of examples like A and just a dozen like B, a uniform random sample will tend to preserve that imbalance. Allocating places by trajectory cluster gives the less common learning pattern more representation. No human labels the clusters. Questions about unrelated topics can end up together.

### How Algorithm 1 selects a subset

1. Fine-tune a small proxy model on the training pool and record each example's loss at several points. In the MathInstruct and MIMIC-III settings in Appendix B.3, the authors used Pythia-70M and recorded losses every **500 training iterations**. A separate sampling-density ablation uses five observations.
2. Treat each trajectory as a vector and compare vectors by Euclidean distance. The authors used Faiss K-means with **100 clusters and 20 clustering iterations** in the main setting. They also compared 50, 100 and 200 clusters. The 20 iterations update cluster centers.
3. Allocate the selection budget starting with the smallest cluster. At each cluster, divide the remaining places by the number of remaining clusters. Take the whole cluster if it contains fewer examples than its allocation. Otherwise, sample randomly within its allocation. For example, with cluster sizes 2, 20 and 100 and a budget of 12, the quotas become 2, 5 and 5. This preserves rare trajectories while allowing variation within a cluster.
4. Fine-tune the larger target on the selected **original examples and their reference answers**. The proxy supplies the list of examples. The practical question is whether learning patterns visible to the small model also matter to the larger one.

### What evidence supports the method?

The theoretical argument starts with the gradient each example contributes to a parameter update. If examples within a cluster induce sufficiently similar updates, it is possible to bound the discrepancy between the update from representatives and the full-data update. The authors discuss convergence to a neighborhood of the optimum under their assumptions. They also empirically compare trajectory grouping across small and large models. Whether those observations transfer to different tokenizers, LoRA fine-tuning or particular fields in a tool call requires downstream evaluation.

On MathInstruct, Pythia-70M selected roughly 30,000 examples for Pythia-410M, about **11%** of the original dataset. The selected subset matched full-data training on average across six math evaluations. Figure 4 and Table 1 hold target training steps fixed, so the smaller subset can be revisited. Table 2 holds the number of epochs fixed, so smaller subsets generally entail fewer updates. The first comparison addresses how many *distinct* examples are needed for a given number of updates. The second asks how much performance remains when each example gets the same number of passes and the subset gets fewer updates. The clinical summarization experiment applies the idea to a different output and compares three summarization metrics. Those scores use a different scale from exact tool-call accuracy.

Figure 8 studies trajectory length. Figure 9 fixes the length at four and varies the checkpoint positions. In the reported experiments, five closely spaced observations were more useful than observations distributed sparsely across training. **Our experiment measures steps 0, 10, 30, 80 and 250** to capture early changes while retaining the endpoint. Checkpoint spacing is a variable a follow-up experiment can test.

### What question does this leave for tool calling?

A model can choose the right tool and repeatedly fill in the wrong date. Averaging over many formatting tokens can obscure difficulty on that date. Group B in this project keeps the whole-answer trajectory. Group C keeps separate trajectories for the function name and argument values. A C advantage over B would show that changing the representation produced a more useful subset under these conditions. The B′ control pools exactly the same field tokens into one mean. Comparing C with B′ helps distinguish the effect of retaining separate field signals from the effect of excluding syntax and other tokens. [NeurIPS conference paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/97fe251c25b6f99a2a23b330a75b11d4-Abstract-Conference.html)

## 02 · Compute-Constrained Data Selection — when does selection pay for itself?

**Junjie Oscar Yin and Alexander M. Rush, ICLR 2025.** Read the cost breakdown in Table 1, the experimental settings in Table 2, the two budgets in Figure 5, and the break-even analysis in the appendix.

### Two budget choices answer different questions

Imagine that two systems train on 1,000 examples each. The system with carefully selected data scores 72, and the random baseline scores 70. Now suppose selection took two extra hours. What score could the random baseline have achieved if it had used those two hours for training? The first comparison measures the value of the data at a given target training budget. The second measures the performance of the entire strategy at a given *total* budget. The scores of 72 and 70 are illustrative.

The authors require that *selection cost + training cost on the selected examples ≤ total compute budget*. At fixed total compute, a more expensive selection step leaves fewer resources for target fine-tuning. To read the Pareto frontier, pick a budget on the horizontal axis and find the best performance among tested configurations that fit it. Compare methods at similar budgets before declaring one preferable for that budget.

### Why do selection methods cost different amounts?

| Method | How it scores a candidate | Main computational expense |
| --- | --- | --- |
| BM25 | Measures lexical overlap between the candidate and target examples. | Term statistics and retrieval, with little additional model compute. |
| Embed | Encodes examples as vectors and compares them to target examples using cosine similarity. | Running the encoder. The embeddings can be cached. |
| PPL | Uses a language model to estimate how difficult the candidate text is to predict, then selects according to the specified rule. | A forward pass on each candidate. Cost depends on the scoring model and the text length. |
| Gradient methods such as LESS | Compare the parameter-update directions for candidates and target examples. | Computing loss gradients and their representations costs more than a forward pass alone. |

The authors vary training-token budgets, selection methods and target sizes from 7B to 70B. Their evaluations cover knowledge on MMLU, reasoning on BBH and instruction following on IFEval. A change in the target capability can change which examples count as relevant. Figure 5 shows that an expensive method can perform well at a fixed amount of training data, while BM25 or Embed can lead at small or moderate total budgets because they leave more compute for training. A larger budget, or amortizing selection across downstream uses, can change the ordering.

The paper's approximate **5× and 10×** figures refer to the *target-to-selector model-size ratios* at which PPL and gradient selection, respectively, became compute-competitive in its analysis. Those values depend on the models, tasks and hardware studied. Our target-to-proxy ratio is 2B / 0.8B = 2.5. S2L also requires proxy training and repeated forward passes. Evaluating its cost requires measuring that pipeline directly.

### How should this project account for compute?

Record proxy training, five scoring passes over the full pool and target training. Measure GPU time on the same hardware and record development evaluation and shared costs separately. The research team's actual bill includes one proxy run shared by B and C. A team deploying C alone would have to pay the full proxy cost. Report both views when they answer different decisions.

An equal-time experiment could let the random baseline spend its savings on additional training from the full pool. Such an experiment needs the number of distinct examples that baseline actually sees. A development-accuracy curve against cumulative GPU time can mark the equal-training-token and equal-total-time checkpoints where both were measured. C could lead at the former and be caught by A at the latter. [ICLR conference paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5acb720a361eecb34ee62d356859d246-Abstract-Conference.html)

## 03 · ToolACE-R — how hard is a request for the model answering it?

**Xingshan Zeng and colleagues, AAAI 2026.** Read the training and inference halves of Figure 1, Equations 2–5, then Tables 3–6.

### Estimate difficulty from generated answers

Give a model the user request and candidate tool descriptions and let it generate the tool name and arguments. Sample several answers to the same request. Some answers may be fully correct, while others choose the right tool and omit an argument. Comparing each answer to the reference estimates difficulty **for that model at that point in training**. After training, another measurement may identify a different set of useful requests.

For requests with multiple calls, simply comparing predictions and references position by position can penalize an otherwise valid reordering. The paper computes pairwise matching scores for predicted and reference calls. It checks tool names first and, when they match, computes an intersection-over-union-like measure of argument overlap. The Hungarian algorithm chooses the assignment with the largest total score. Dividing by the larger of the predicted and reference call counts penalizes both missing and extra calls. One answer receives an overlap score *O* between 0 and 1. For *k* sampled answers, the difficulty is *D = 1 − mean(O)*. For example, a mean overlap of 0.75 across eight answers gives *D = 0.25*. This distinguishes repeatedly missing one argument from choosing the wrong tool, even when both answers fail strict exact match.

### What changes during training and inference?

The authors select requests in a difficulty range for the current stage, update the model and adjust the data as its abilities change. The main setting in Table 4 keeps **0 < D < 0.9**. When *D* is zero, all sampled answers matched completely. Near one, the answers rarely matched. The threshold controls the composition of that experiment's training set. The authors also construct self-correction dialogues containing an initial prediction, a request to check it and a revised answer. These teach the model to revisit an existing call.

At inference time a separate procedure asks the model to check its answer iteratively. It can stop when consecutive calls agree or when it reaches the iteration limit. Agreement signals that the model stopped changing its prediction, which can also happen if it repeats the same mistake. With a maximum of five iterations, the authors report about 2.4 on average.

### Reading Tables 3 and 6

For the BFCL single-turn aggregate in Table 3, base Llama 3.1 8B Instruct scores **76.83%**. The full training recipe with standard inference scores **86.33%**. Adding adaptive correction gives **86.83%**. Removing data selection yields **85.04%**. Also removing correction training data yields **84.42%**. The comparison against the base therefore includes the combined effects of fine-tuning and changes to training content. Adaptive inference correction accounts for a 0.50 percentage-point increment in this comparison. Table 5 changes the number of generations used to estimate difficulty from eight to one. Standard-inference performance with one sample is 85.96%, illustrating the trade-off between difficulty-estimation cost and downstream performance.

Table 6 holds the target at 8B. Using the target itself for selection gives 86.33%. Using Qwen2.5-3B and Qwen2.5-7B gives 85.50% and 85.97%, respectively. First-round selected counts are 120K, 125K and 134K in that order. Both the selector and the number of selected examples vary. Our study fixes the subset at 1,000 and matches target training tokens to isolate the effect of the selection signal more directly.

ToolACE-R observes what the model writes during free generation. Our proxy sees the correct answer prefix and scores the next reference tokens in specific fields. The right date may be easy to predict given the correct prefix, while the model might choose the wrong tool when generating freely and never reach the date field. This is why our endpoint evaluates the larger model's freely generated, complete calls. [Authors' paper and equations](https://arxiv.org/html/2504.01400v2)

## 04 · Chasing Random — does selection still beat a random draw under a different evaluation?

**Harshita Diddee and Daphne Ippolito, Findings of NAACL 2025.** Read Table 1, Figures 2–3, Table 2 on time and cost, and the limitations section.

### Why might a selector lose its advantage in a new setting?

Imagine a method that favors long responses. It might help on an evaluation that rewards complete explanations. Its advantage could disappear when the evaluation requires returning a single number in a strict format. The source dataset matters as well. Some instruction datasets cover broad topics, while others concentrate on a narrower set. The authors hold LLaMA-7B fixed and vary the sample budget and evaluation across four data sources, FLAN, Dolly, Evol and Alpaca.

They study four approaches. AlpaGasus asks GPT to score instruction pairs for quality. Longest takes the longest responses. Cherry builds a small set, trains a scoring model and filters by difficulty conditioned on the instruction. DEITA considers both quality and differences between examples. Comparisons include random subsets and full-data training. Training each selected set for three epochs means equal example counts can entail different token budgets, especially for Longest.

### What do Figures 2 and 3 measure?

Figure 2 compares a model judge's preference for answers from selected-data and random-data models on AlpacaEval. The authors sample evaluation requests to control judging costs, account for ties and compare source datasets at budgets of 1,000 and 10,000 examples. Some methods lead in individual settings, and their advantage changes with the source or budget. Figure 3 shifts to knowledge and reasoning benchmarks. Random selection looks comparatively weak on MMLU alone and more competitive across a collection of tasks. The capabilities represented in an evaluation help determine which selector looks strongest.

The authors report multiple random seeds and redraw some random subsets that overlap too much with method-selected subsets. Dolly's size also constrains sampling. These choices affect the meaning of a random baseline. In this project, we freeze uniform samples drawn without replacement and report overlap, keeping each sample even when it happens to resemble group C. Using the same held-out requests for every model allows us to count requests answered correctly by C alone and by B alone, showing how many individual outcomes support a score difference.

### Reading Table 2 and reporting negative findings

On Alpaca, the authors estimate about **$111.15** for Cherry selection and about **$15.60** for full-data training. The corresponding estimates on Dolly are about $32.07 and $5.20. These are estimates for the paper's configurations at the time. A smaller selected set, on its own, does not establish compute savings when selection costs can exceed the cost of full-data training.

The study covers general instruction following with one base model, and extensive human evaluation is expensive. Tool calling offers structured reference answers that make scoring more auditable. The paper identifies a risk worth testing on this task. If C beats B internally and ties on external BFCL data, report the two tasks and their denominators separately. If C and random look close, examine subset overlap and baseline headroom, then report the paired difference and interval. A wide interval can encompass several plausible effects and cannot establish equivalence. [ACL Anthology conference paper](https://aclanthology.org/2025.findings-naacl.103/)

## Four questions to check your understanding

| Question | Answer |
| --- | --- |
| Does clustering label examples as “dates,” “orders” and similar topics? | The inputs to clustering are learning trajectories. Tool or argument categories can be summarized afterward to interpret clusters. Topics do not enter S2L directly as trajectory coordinates. |
| Does C train two models with two different losses? | The proxy undergoes one ordinary fine-tuning run with complete-answer supervision. Field losses are measured at frozen checkpoints to select data. Each target model is still supervised on complete answers. |
| How can teacher-forced loss on an argument be low while the model answers the request incorrectly? | Scoring supplies the correct prefix. Free generation can fail earlier on tool choice, number of calls or structure. Each measure captures a different behavior. |
| Does matching target training volume settle the cost comparison? | A total-budget comparison also counts proxy training and five scoring passes and gives the random strategy access to the same total compute. |

## Additional reading on field signals and data provenance

- **Getting the Parameters Right.** Yu and colleagues use a linear probe on hidden states during argument-value generation to predict whether a value will be correct. They report in-domain AUC of 0.986, compared with 0.914 for token log probabilities in the same setting. They also use the probe to filter self-generated training answers and rerank inference candidates. Our project starts with reference calls and uses field losses across multiple proxy checkpoints to select requests. Its downstream endpoint is complete-call accuracy of a 2B model. [Probe experiments, Section 4](https://arxiv.org/html/2608.03071v1#S4)
- **AgentFlux and BalanceSFT.** AgentFlux trains separate LoRA adapters for tool selection and argument generation. BalanceSFT adjusts training weights for reasoning text and tool-call tokens and adds difficult examples that target errors. Both motivate studying the component parts of a call. Our primary comparison keeps the target training objective fixed for every 2B model and varies which examples enter training. [AgentFlux](https://arxiv.org/abs/2510.00229) · [BalanceSFT](https://aclanthology.org/2026.findings-acl.900/)
- **Data Turnstile.** Ramakrishnan and Sharma describe generating tool interactions from API definitions and validating those records. Our primary experiment uses Amazon's public Synthetic Domains data. We preserve the user request, system date and candidate tool descriptions and extract the reference call from `API_CALL`. The dataset card's scores depend on its stated model, training corpus and BFCL version. Our study separately measures the Qwen3.5-2B baseline and differences across selection methods. [Data generation paper](https://arxiv.org/abs/2607.29250) · [Amazon dataset and results table](https://huggingface.co/datasets/amazon/Turnstile-Synthetic-Domains)
