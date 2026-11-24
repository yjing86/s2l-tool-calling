# Why did trajectory selection trail random sampling?

**Post hoc follow-up protocol · November 24, 2026**

The original experiment found that field trajectories improved on whole-answer trajectories, while uniform random sampling remained competitive. We will test whether the selected examples' argument lengths, difficulty and tool frequencies help explain those rankings. The immediate scope is CPU analysis and up to eleven additional 2B fits under the existing GPU allowance. A method that beats random sampling is a possible outcome, not a condition for a useful study. [Original protocol](research-plan.md) · [Primary results](results.md) · [Execution log](experiment-log.md)

## 1. Observations and competing explanations

All primary arms selected 1,000 of the same 4,000 requests. The models answered the same 2,000 held-out Turnstile requests at a matched target training-token budget. Differences below are in percentage points (pp).

| Arm | Mean accuracy | Difference from B | Difference from A |
|---|---:|---:|---:|
| Unadapted 2B | 47.20% | — | — |
| A, uniform random | 55.68% | +1.98 | — |
| B, whole-answer trajectory | 53.70% | — | −1.98 |
| C, separate name/value trajectories | 55.23% | +1.53 | −0.45 |
| B′, pooled name/value tokens | 54.95% at seed 42 only | +2.25 at 42 | −0.95 at 42 |

C−B was positive at each of seeds 42, 43 and 44. Its fixed-seed mean paired 95% interval was [+0.60, +2.48] pp. The interval for C−A was [−1.42, +0.53] pp. At seed 42, C−B′ was +0.15 pp [−1.30, +1.55]. That single seed leaves room for gains and losses from keeping the two fields separate. E1 supplies the two missing independent target fits using the already selected lists. [Complete results](results.md) · [Machine-readable aggregate](../results/turnstile-single-call.json)

**Proxy diagnostics from the frozen 4,000-request pool.** The supervised answer's positions outside function names and argument values include formatting and parameter keys. Together they accounted for 73.95% of the tokens. Across the five checkpoints they supplied 16.76–24.56% of average whole-answer loss. Define W as the per-request whole-answer mean, and O as the sum of losses at other positions divided by the *whole-answer* token count. Across requests, cov(O,W)/var(W) ranged from 4.34% to 7.45%. This is an additive covariance decomposition under a particular partition. It cannot exclude an effect of removing those tokens on standardized features, cluster boundaries, selection or target performance. The correlation between W and whole-answer token count was +0.195 to +0.246. At step 250, name loss across requests had mean 0.00333 and standard deviation 0.02823. That pattern motivates inspecting each checkpoint's distribution, including its tail.

| Arm | Mean value tokens per selected request, across lists for seeds 42–44 | Distinct reference functions at seed 42 (953 in pool) |
|---|---:|---:|
| A | 15.37 | 603 |
| B | 19.58 | 585 |
| B′ | 18.16 | 578 |
| C | 17.42 | 555 |
| Full candidate pool | 15.73 | 953 |

The B′ subsets for all three seeds already exist. Only seed 42 provided an original target score. On that seed, both B and C chose the correct tool and call count on 1,964/2,000 requests. A wrong value occurred on 827/919 of B's requests whose first failure was in the arguments (90.0%); error types may overlap. Neither a low average proxy name loss nor high tool accuracy proves that all name features are uninformative. [Field-level validation](field-loss-validation.md) · [Error classification](../scripts/field_diagnostics.py)

Three explanations guide the analysis. **H1** is that quota-based sampling across loss clusters changes the distribution of argument lengths and difficulties, which may reduce performance under the training budget. **H2** is that exposure to the reference tool helps explain differences between A and C. A and C's distinct-function totals alone cannot isolate this effect. **H3** is that examples of intermediate difficulty may be more useful at this particular budget. Each is an explanation suggested *after* viewing the primary results. [Study on data difficulty and generalization](https://arxiv.org/abs/2605.12906)

## 2. Execution boundaries

Use only Amazon Turnstile for training. Reuse the frozen 0.8B loss sums and counts, the original Qwen3.5-2B initialization, LoRA settings, token budget of 3,005,228, greedy generation and complete-call scorer. K-means initialization uses seed 42. The selection and target training seeds are paired at 42, 43 and 44. The shared no-value stratum and its selected IDs apply to B′, E2 and E3; E4 draws from the whole pool. In the realized pool all 4,000 examples have at least one value token. [Selection implementation for the primary study](../scripts/select_experiment.py) · [Target training](../scripts/train_experiment.py) · [Generation and scoring](../scripts/evaluate_experiment.py)

Keep the archived primary runs read-only. The follow-up selector and GPU driver are separate entry points, `scripts/select_followup.py` and `scripts/run_followup.py`. Store new inputs, selected IDs, configurations, hashes, task receipts, outputs and failures beneath `local/followup/`. Record the arms, seeds, endpoints and decision criteria with timestamps before generating follow-up predictions. Stage 1 reuses an **already opened** test. Freezing new code and checkpoints prevents further selection on its scores but cannot restore independent confirmation. Reserve the new test in Section 5 without generating or reading scores on it. [Environment](environment.md) · [Reproduction guide](reproduce.md)

## 3. Phase 0 — analysis of the existing evidence

These five analyses require no target training. Preserve the tables, group definitions and denominators so that a reader can reconstruct the figures.

| Analysis | Inputs and calculation | What it can establish |
|---|---|---|
| P0-1 · Subset length and performance | Plot mean selected value-token count against held-out accuracy for the ten original fits. Repeat with total answer tokens. Show arm and seed on each point. Report Spearman correlation. | A draft threshold of ρ ≤ −0.6 flags a pattern worth following. Ten points share arms, a proxy and test requests, so correlation cannot establish a dose response. |
| P0-2 · Seen tools | Bin test requests by frequency of their reference tool in the selected training subset (0, 1, 2–3, ≥4). For A/C discordance, show counts for both saw the tool, only A, only C or neither. | A gain concentrated where only A saw the function would motivate an exposure explanation. Report group sizes and both directions of disagreement. |
| P0-3 · Cluster profiles | For each B/B′/C cluster at seed 42, report candidate and selected counts, mean value tokens, nested-request fraction, distinct functions and mean step-250 value loss. | Show whether small clusters with large per-example sampling rates contain longer or more difficult requests. Keep full and selected-cluster denominators distinct. |
| P0-4 · Test length | Use quartiles of reference value-token counts on the common test. Preserve ties and record actual bin boundaries, denominators, accuracy and paired arm differences by bin. | Identify where the observed gap concentrates. Performance by test length describes model behavior, not the cause of the selection effect. |
| P0-5 · Two sources of uncertainty | For C−B, B−A and C−A, report the paired near-duplicate-group bootstrap interval conditional on the three trained seeds. Also show mean ± t(0.975, 2) × seed SD / √3. | The second interval uses only three paired-seed contrasts, is sensitive to distributional assumptions, and does not simultaneously vary seeds and questions. Report both with their units. |

## 4. Phase 1 — up to eleven diagnostic target fits

Queue the first five fits before the six lower-priority fits. Each selected subset has 1,000 unique requests. Each fit has **one endpoint development evaluation on 400 requests and one evaluation on the original 2,000 test requests**. Use the matched-token checkpoint without tuning on its test scores.

| Control | New fits | Construction and interpretation |
|---|---:|---|
| E1 · Complete B′ | 2 | Fit the existing pooled-field subsets at seeds 43 and 44. Combine them with the original seed 42. Compare B′−B and C−B′ at each seed and on the fixed-seed mean. |
| E2 · Length-matched random | 3 | Bin value-bearing candidates by pool-wide deciles of value-token counts. Merge tied edges deterministically. At each seed take exactly B's count from each occupied bin, uniformly without replacement, then include the shared no-value IDs. Save edges, counts, IDs, achieved within-bin lengths, tool frequencies and overlap with B. |
| E3 · Residualized whole-answer loss | 3 | At each of five checkpoints regress whole-answer mean loss on an intercept, log(1 + value tokens), name/value token fraction and log(total supervised answer tokens). Use training-pool features alone. Standardize and cluster the five residuals, then apply B's quotas and sampling. Save coefficients, R², residual distributions and selected-set changes. |
| E4 · Tool-stratified random | 3 | Allocate the 1,000 seats proportionally to each reference function's frequency with largest-remainder rounding and a deterministic tie rule. Sample uniformly without replacement within each function. Record target and realized quotas, rare functions with zero quota, test-tool exposure and coverage. |

**E2 decision rule.** Prespecify ±1 pp as the equivalence margin for the fixed-seed mean of E2−B. The earlier design draft also considered a more conservative 95% sensitivity check. The actual pre-run registration for E2−B, E2−A and E3−A requires three completed seed pairs and the entire **unrounded 90% paired group interval** strictly inside (−1, +1) pp (two one-sided tests at 5%). Also publish the 95% interval for effect estimation. Report the paired difference and interval for E2−A, and counts for each seed. If E2 is equivalent to B, its mean is at least 1 pp below A and that direction repeats at all three seeds, the results are consistent with effects of B's length-bin composition and characteristics correlated with it. Bin matching preserves neither exact within-bin lengths nor all tool and difficulty distributions. If the interval crosses either margin, describe the observed differences and uncertainty. A similar point estimate alone cannot establish equivalence. E2 near A also leaves other within-bin and selection mechanisms open.

**E3 and E4 interpretation.** E3−B ≥ +1 pp is a diagnostic threshold worth pursuing. Conditioning away an association in the training features can also remove useful difficulty signals. Report uncertainty and the resulting distributions. Apply a prespecified equivalence test before describing E3 and A as equivalent. E4−A ≥ +1 pp would motivate checking how quotas, exposure and accuracy changed. Proportional allocation can assign zero slots to rare functions; any improvement measures the strategy as implemented. To attribute it to greater coverage, measure that coverage directly. Comparisons reuse a viewed test and include multiple diagnostic contrasts.

Present A, B, B′, C, E2, E3 and E4 by seed, with correct counts, denominators, differences against A and B, paired intervals and seed variability. Add the new fits to the P0-1 plot. Keep missing arms explicit if the budget binds. [Primary summary implementation](../scripts/summarize_course.py)

## 5. Phase 2 — independent evaluation of a new selector, conditional

Reserve an unopened internal test before any Phase 2 model selection. Start from the 8,728 raw single-call rows; 6,600 rows were assigned to pilot, candidate training, development and primary test, plus debug rows and other exclusions. Reuse all cleaning and near-duplicate grouping rules, and exclude any group represented in a used split or debug set. Request 1,500 eligible independent groups. Record actual eligibility, usable N, group/row IDs, exclusions and checksums before inference. The raw subtraction 8,728 − 6,600 does not imply 2,128 independent usable requests. If few groups survive, record the limitation and reassess precision before spending on tuning. [Split implementation](../scripts/prepare_experiment.py)

The following methods require a separate compute decision after Phase 1. Use the original 400-request development set for tuning. Freeze the chosen specifications and statistical comparisons before touching the new test. If several candidates will be evaluated, designate one primary method in advance or account for multiple comparisons.

- **E5 · Coverage and trajectory diversity.** From the training pool only, maximize f(S) = Σ_g √n_g(S) + λ Σ_c √n_c(S). The terms count selected requests by tool g and by cluster c of their five value-loss measurements. Square roots give diminishing marginal returns. Use greedy marginal selection, seeded tie breaks and a distinct cluster for no-value requests. Try λ in {0.5, 1, 2} and cluster counts in {10, 30} at seed 42 on development. Check whether different configurations actually select different IDs before paying for target fits. This is a concrete use of submodular maximization from lectures 6–8.
- **E6 · Moderate difficulty.** Estimate difficulty from the mean of five value losses or a time-weighted area; lock one definition and its tie handling before tuning. Uniformly sample between the 30th and 70th percentiles and keep the shared no-value stratum. An optional variant allocates proportional tool quotas first and draws intermediate-difficulty items within a tool. Prespecify how shortages redistribute. Development selects the variant.
- **Independent controls and gate.** Infer with the base model and A and C at all three seeds on the new test (seven existing checkpoints, no retraining). Compare the chosen method with matched A predictions using the same strict scorer. The proposed expansion gate requires a mean gain of at least +1 pp over A, a positive difference at all three seeds, and a paired 95% lower bound above zero on the new test. Otherwise report uncertainty and stop expansion under this gate. A failed superiority test does not establish equality.

## 6. Phase 3 — only after independent support and a new budget

| Extension | Diagnostic question and scope |
|---|---|
| Selection budgets of 10% (400) and 50% (2,000) | Compare A, C and the best new method at each fraction and three seeds (18 target fits). Recalculate tokens and time for each fraction. |
| Full-parameter 0.8B proxy | Pilot learning rates of 1e−5 to 2e−5, then repeat the five scoring passes. Prespecify the target comparisons and account for all proxy and scoring cost. |
| External evaluation | Compare A and the chosen method on BFCL. Label the already viewed 200 requests as a benchmark recheck. Reserve unopened IDs or categories for an independent external replication. |
| Corpus generalization | Keep Turnstile as the sole training source in this project. Any future corpus would require a separate release-date, contamination, licensing and budget review. |

## 7. Measured costs and admission order

Primary A40 task receipts include process startup. Across ten target fits, mean training time was 963.7 s (range 954.9–972.5). Across 30 development evaluations on 400 items, the mean was 217.9 s (184.8–258.5). Across ten 2,000-item adapted-model tests, the mean was 804.0 s (748.6–846.8). The original pretraining forecast of 981 s for development covered **three** development evaluations per arm. Charging that estimate for one endpoint and then adding a separate engine startup would overstate the cost here.

| Priority and scope | Target fits | Estimated A40 occupancy |
|---|---:|---:|
| P0 analysis and selection | 0 | CPU only |
| E1 followed by E2 | 5 | 5 × (963.7 + 217.9 + 804.0) / 3,600 = **2.76 h** |
| E3 and E4 after the priority fits | 6 | **3.31 h** additional |
| All eleven Phase 1 fits | 11 | **6.07 h** point forecast, excluding failures, incidental queue time and CPU-only analysis |
| Unopened-test controls, only when authorized | 0 | Seven inferences. For planning, seven × the observed 2,000-item adapter time gives **1.56 h**; the proposed test has at most 1,500 items, including a base checkpoint with different latency. Measure on the actual setup. |
| E5/E6 development search, conditional | Up to 6 E5 configurations and 1–2 E6 configurations at seed 42 | About **2.30–2.63 h** for 7–8 distinct fits plus one dev each at historical rates. Skip identical selected sets. |
| E5/E6 after tuning, conditional | Up to 6 target/seed cells for two methods × three seeds | A winning seed-42 fit can be reused if training inputs and settings match. Four additional fits plus dev would take about **1.31 h** if two such cells are reusable; six would take **1.97 h**. Add independent final-test inference separately. |
| Phase 3 | At least 18 for the full fraction grid, plus any admitted proxy and additional target runs | Estimate per token budget and configuration after the gate. |

Before admitting GPU work, include a **20% time margin** and preserve the original provider deadline's **ten-minute shutdown reserve**. The eleven point estimates imply 7.28 h with the 20% margin, plus the reserve. The priority five imply 3.31 h plus the reserve. Recalculate using current receipts before each admission. Never count the ten-minute reserve as spendable time. Record any budget skip and preserve partial evidence. Phase 2 tuning and Phase 3 are outside this immediate run.

## 8. Outputs and reporting

Archive Phase 0 tables/plots, each admitted phase's timestamped design, selected IDs, cluster/quota metadata, feature versions, training/dev/test receipts and paired summaries under the ignored follow-up directory. The length-versus-accuracy figure should show each arm and seed and identify the test used. Publish aggregate statistics and the code to reproduce them. Individual model generations and source examples stay in private artifacts. Report an improvement over A only at the test and budget actually evaluated. Comparable scores, wide intervals and a failure to exceed random sampling are publishable observations when their scope is stated.

To reproduce the proxy decomposition, read `scores/step-{0,10,30,80,250}.jsonl` from a completed primary run. At each step compute W = `all_sum/all_tokens` and O = `other_sum/all_tokens` for each ID. Report mean(O)/mean(W), cov(O,W)/var(W) and Pearson correlation of W with `all_tokens`. Join subset IDs to `value_tokens` from step 250, then extract the reference function from the corresponding training record's last assistant message. [Field scorer](../scripts/score_trajectories.py) · [Primary publication guide](references.md)

## Run the follow-up on an existing reproduction

Use the training environment and the primary run layout from the [reproduction guide](reproduce.md). The commands below analyze the opened test, reserve a distinct test set without generating predictions, select the training subsets and freeze the diagnostic protocol. Each command uses CPU only. Run the selector into a fresh output directory. Check that the prospective holdout's question IDs and near-duplicate groups are absent from every original split.

```bash
python -m s2l_tool_calling.analysis.followup_diagnostics register --out local/followup/phase0
python -m s2l_tool_calling.analysis.followup_diagnostics analyze --run local/course-run --out local/followup/phase0
python -m s2l_tool_calling.analysis.followup_length_slices \
  --course local/course-run --model local/models/Qwen3.5-2B --out local/followup/phase0/length-slices
python scripts/reserve_followup_test.py
python scripts/select_followup.py --primary local/course-run --out local/followup/phase1/selection
python scripts/run_followup.py freeze
python scripts/run_followup.py status
```

The A40 first completes the primary, full-pool and crossed-seed experiments and their evaluations. Verify and back up their terminal receipts. Then start the following commands manually, using the interpreter variables exported in the reproduction guide. The controller admits the five priority fits as a batch, followed by two batches of three. Using the maxima of observed stage times and 20% headroom, the initial admission estimates are about 3 h 27 min, 2 h 4 min and 2 h 4 min. Time and storage must cover a full batch. Each target has one endpoint dev evaluation and one 2,000-request test. The controller never uses the reserved 1,500 requests for these diagnostics.

```bash
python scripts/run_followup.py run --train-python "$TRAIN_PY" --eval-python "$VLLM_PY"
python scripts/summarize_followup.py --primary local/course-run --followup local/followup/phase1
```

[Timestamped diagnostic registration and file hashes](../results/followup/phase1-registration.json)

## 9. Review follow-up: distinguish proxy difficulty from argument length

This exploratory control was proposed after the original test was opened. The eleven Phase 1 fits and their registered contrasts retain their original priority. The three optional E2b training sets have passed a train-only construction audit. No E2b target-model score is available at registration.

### Why add a control?

Let L be the number of reference argument-value tokens. For E2b, let D be the **unadapted 0.8B proxy's mean negative log-likelihood (NLL) on those tokens at step 0**, computed as `value_sum / value_tokens` separately for each request. A model that assigns a lower probability to the reference value gets a higher D. D is a proxy-specific measure of predictability. The difficulty of choosing the function, the intrinsic complexity of the task and what the 2B target will learn are separate quantities. Later checkpoints and five-checkpoint averages may be used for sensitivity analyses. The selector uses only step 0.

Across seeds 42–44, average D was 0.394 in the full pool, 0.387 in A, 0.753 in B, 0.743 in C and 0.390 in E2. Subtracting the mean D of all candidate requests with the same exact L gave within-length mean residuals of −0.006 for A, +0.340 for B, +0.330 for C and −0.007 for E2. B and C both oversample high-D examples even at a given length. E2 matches B's *binned* length counts, yet B averaged 19.58 value tokens and E2 averaged 18.51. Matching bins leaves variation within them. C's target outperformed B while their proxy difficulty means were close, so a scalar difficulty score alone cannot explain the complete ranking. [Proxy scoring](../scripts/score_trajectories.py) · [Original paired outcomes](results.md)

### How to construct the three E2b subsets

| Quantity | Frozen rule | Verified outcome |
|---|---|---|
| Length margin | Match **the same seed's original A** length-bin counts. A is the paired random baseline. Pool-proportional counts were tested for feasibility only. | A's counts matched exactly for each seed. The maximum per-bin differences between A and a proportionally allocated pool target were 23, 17 and 16 examples. |
| Difficulty margin | Match **the same seed's original B** counts in bins of D. | B's counts matched exactly at all three seeds. |
| Feasible cells | Form positive-value bins using pool-wide inverse empirical CDF quantiles; bins are right-closed and ties stay together. Reuse B's no-value IDs. If needed, allocate A's positive length quotas proportionally to the remaining positive seats. Try 10 × 10 bins, then 5 × 5, then 2 × 2. Use the first grid for which integer maximum flow fills 1,000 seats at **all three** seeds, subject to pool cell capacities and the two margins. | All 4,000 candidates had values. All three seeds had flow 1,000 on the 10 × 10 grid. Every cell was occupied in the pool and the subset, and none was exhausted. |
| Sampling | Among feasible integer allocations, minimize the sum over cells of (selected count − expected pool-proportional count)² / max(expected pool-proportional count, 1). Draw uniformly without replacement within each cell using the registered seed stream. Freeze the IDs after a single draw. | Three distinct 1,000-ID lists were hashed and reconstructed. Their overlaps with paired B were 370, 367 and 353 requests. |

The exact margins, common capacity rule and single-draw policy were specified for list construction. The standardized mean differences (SMD) and maximum empirical CDF differences (KS distances) below audit continuous balance **after** that draw and before evaluating new target models. No numeric SMD/KS cutoff was preregistered for list construction. No seed was redrawn to improve balance. Positive SMD means the E2b mean is larger.

| Seed | Length vs A, SMD / KS | Difficulty vs B, SMD / KS | Difficulty vs A, SMD | Distinct functions, E2b / A |
|---:|---:|---:|---:|---:|
| 42 | +0.024 / 0.012 | −0.043 / 0.022 | +0.602 | 557 / 603 |
| 43 | +0.024 / 0.012 | −0.052 / 0.027 | +0.491 | 574 / 605 |
| 44 | +0.038 / 0.011 | −0.047 / 0.025 | +0.569 | 558 / 595 |

E2b achieved its prescribed bins, closely approximated continuous L in A and D in B by these diagnostics, and changed difficulty relative to A. Its lower tool coverage illustrates a remaining difference. Finite joint cells constrain how independent the draws can be, even when every cell is occupied. Keep overlaps, continuous means, quantiles and margins with the manifests. The 1.46 s CPU selection time with cached inputs excludes proxy scoring, target training and inference. [E2b selector](../scripts/select_difficulty_control.py)

### What the additional comparison can establish

- **E2b−A** compares two selection policies at A-like lengths, with high measured proxy difficulty deliberately overrepresented in E2b. A lower E2b score would be consistent with a cost of this selection policy at fixed target tokens. Tool identity, argument structure, semantic difficulty and the joint length/difficulty distribution can still change.
- **E2−A** changes the length-bin policy, and allows difficulty to follow naturally within bins. **E2−B** compares a random draw within B's length bins with B's original trajectory selection. **E2b−B** matches B's difficulty bins while changing length, joint cell allocation and the original clustering decisions. **E2b−E2** changes two margins. None of these differences estimates a pure length effect, pure difficulty effect or factorial interaction.
- An explicit 2 × 2 study could later impose A-like or B-like margins on **both** L and D using one randomized construction at every corner. Its B-length/B-difficulty cell would test whether matching both margins reproduces B's outcome without selecting by whole-answer cluster. The existing B, A and E2 are not three interchangeable randomized corners of that design. No such factorial is required for this diagnostic.

### Compare like test requests when interpreting tool exposure

For each pair and seed, stratify the **same test IDs** by whether the reference tool appeared in both training subsets, only the left subset, only the right subset, or neither. Compute left-only correct minus right-only correct within each stratum. The sums below count three evaluations of 2,000 distinct requests. They are not 6,000 independent test items.

| Left minus right | Both saw tool | Only left saw tool | Only right saw tool | Neither saw tool | Net correct |
|---|---:|---:|---:|---:|---:|
| A−B | +57 | +54 | −3 | +11 | +119 |
| B−C | −65 | +4 | −31 | 0 | −92 |
| A−C | +19 | +29 | −26 | +5 | +27 |

The unadapted 2B scored 56.29% on the A-only exposure stratum of A vs B and 39.66% on the B-only stratum. Those slices already differed before fine-tuning. In A vs C, the two exclusive-exposure strata together account for 3 of the 27 net extra correct evaluations. For C vs B, 65 of the 92 net gains occurred when both methods saw the tool. Method-specific frequency bins compare different mixes of tools and requests. Always show denominators, base scores and seed-specific paired counts. For intervals, resample the same request groups jointly across seeds; a tool-family cluster sensitivity analysis addresses a different source of dependence. Conditional and overall results describe associations. Neither isolates the effect of seeing a particular tool.

### GPU admission, analysis and independent evidence

Complete and verify the original study and its full-pool and crossed-seed checks first. Then finish all eleven previously registered Phase 1 target fits. Only after those obligations can the three E2b fits at 42, 43 and 44 be admitted together. Preserve the original deadline and ten-minute shutdown reserve, updated receipt-based forecasts and the 20% time margin. Mean timings from completed primary runs imply about **1.65 A40-hours** for three fits including development and original-test inference. The admission rule instead takes the slowest observed training, development and test stage separately and adds 20%. At registration those maxima were 972.46, 255.33 and 846.83 seconds per fit, giving **2 h 04 min 29 s** for a batch of three, before the common shutdown reserve. If all three do not fit, retain the CPU selection evidence and defer the GPU control. A partial run cannot stand for the three-seed comparison.

Before inference, lock the input IDs, target settings and matched checkpoints. The principal diagnostic is E2b−A at equal target tokens. Report E2b−B and E2b−E2 as secondary, show per-seed correct counts and discordances, and use the same paired request-group and separate seed-sensitivity intervals as Phase 1. Positive or negative findings on the already opened 2,000-request test remain post hoc. To make an independently tested superiority claim for a revised selector, lock the method and controls before observing predictions on the distinct, previously reserved 1,500-request test. The reserve supports a new-request test within Turnstile. It shares the underlying corpus and tool families. [Reserve construction](../scripts/reserve_followup_test.py)
