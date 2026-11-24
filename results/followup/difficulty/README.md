# Difficulty at a fixed argument length and the role of tool exposure

**Whole-answer and field selection both preferentially sampled examples that the proxy found difficult even among examples of the same length.** Matching B’s length-bin quotas with random draws (E2) produced a subset almost as easy for the proxy as unrestricted random sampling (A). This establishes how the training subsets differ. An effect on target-model accuracy requires a training contrast.

## What the proxy found difficult

The reference population is the frozen pool of 4,000 training candidates. Every candidate has at least one scalar argument-value token. At each of five saved proxy states, a candidate’s difficulty is the sum of teacher-forced negative log-likelihoods over these value tokens divided by their count. Each candidate gets equal weight. Step 0 is the original 0.8B proxy before fine-tuning. Step 10 is the first saved state after training starts. The proxy’s loss measures its difficulty predicting these particular values in context. Tool semantics, formatting and target-model learning can behave differently.

| Selection | Seeds | Mean value tokens | Step-0 value NLL | Same-length residual NLL | Bin residual NLL | In pool’s raw top 10% | In top 10% of own length |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full pool | — | 15.726 | 0.394 | 0.000 | 0.000 | 10.00% | 10.00% |
| A · Random | 42–44 | 15.370 | 0.387 | −0.006 | −0.007 | 9.47% | 9.85% |
| B · Whole answer | 42–44 | 19.580 | 0.753 | +0.340 | +0.359 | 26.30% | 27.54% |
| C · Separate fields | 42–44 | 17.419 | 0.743 | +0.330 | +0.348 | 26.43% | 25.78% |
| B′ · Pooled fields | 42 only | 18.292 | 0.826 | +0.416 | +0.432 | 30.10% | 30.72% |
| E2 · B-matched length bins | 42–44 | 18.513 | 0.390 | −0.007 | −0.004 | 10.10% | 10.28% |

For the **same-length residual**, subtract the mean loss among all training candidates with exactly that many value tokens, at the same checkpoint. The analogous bin residual uses the frozen E2 bins with right-inclusive cutoffs at 5, 7, 8, 10, 12, 15, 18, 23 and 29. At seed 42, the step-0 raw NLLs for A, B, C, B′ and E2 were 0.372, 0.763, 0.749, 0.826 and 0.386. B′ contributes one selection, so its mean has a different seed denominator.

For the **raw top 10%**, the cutoff was 1.042537 NLL. Of 4,000 rows, 399 exceeded it and one equaled it. For the within-length column, assign exactly 10% of the weight in each integer-length stratum to the highest losses, dividing the last available weight equally across boundary ties. There were 91 distinct lengths. In the 28 strata containing one example, that example gets 0.1 weight and has zero same-length residual. The group columns are percentages of selected candidates, including fractional boundary contributions. The five-checkpoint tables also report the upper decile after pooling the exact-length residuals. [Seed-level table](difficulty-by-seed.csv) · [Means at all checkpoints](difficulty-means.csv) · [Thresholds and ties](difficulty-thresholds.csv)

The sign persisted through step 250. At that checkpoint the same-length residual means were −0.004 for A, +0.188 for B, +0.178 for C, +0.205 for B′ and −0.004 for E2. B and C themselves use proxy losses for selection, so enrichment is a property of those methods. Their selected sets overlap, share the proxy, and reuse candidates across seeds.

## What E2 controls

In each of its three 1,000-example subsets, E2 matched B’s count in every registered value-length bin. E2’s exact mean length remained 1.067 tokens below B’s and 3.143 above A’s. Its same-length difficulty was close to A’s on all three seeds. E2 therefore controls a **coarse histogram of lengths**, while leaving appreciable within-bin variation in length and difficulty. A downstream E2−B difference could also reflect changes in tool identity or other properties. This audit uses E2 memberships only.

## Compare models on the same test questions

For each pair and seed, look up whether each test question’s reference function appeared in each training subset. The two models then receive exactly the same test questions within each of four exposure strata. Each table entry is the count correct only for the left model minus the count correct only for the right model, summed over seeds. Each pair has 6,000 evaluations of the same 2,000 distinct requests.

| Left minus right | Both saw function | Only left saw it | Only right saw it | Neither saw it | Total net wins | Full-test mean difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A−B | +57 | +54 | −3 | +11 | +119 | +1.98 pp |
| B−C | −65 | +4 | −31 | 0 | −92 | −1.53 pp |
| A−C | +19 | +29 | −26 | +5 | +27 | +0.45 pp |

For A vs. B, the common unadapted 2B model already scored 56.29% on the A-only exposure stratum and 39.66% on the B-only stratum. This shows that the two collections of questions differed in baseline performance. A also gained 57 net wins on questions whose function both models saw. For B vs. C, 65 of C’s 92 net wins occurred when both saw the function. Simple comparisons of one model’s accuracy on seen and unseen functions mix changes in request composition with any potential exposure effect. These observations support further controlled comparisons and do not separately identify causal effects of exposure, length, or difficulty. [Per-seed counts and exact-match rates](shared-exposure-by-seed.csv) · [Totals and baseline accuracy](shared-exposure-totals.csv)

## Reproduce the six aggregate files

From the repository root, use Python 3.12 with the pinned CPU selection dependencies. Set `PRIMARY_RUN` to your archived primary run if it lives outside `local/course-run`. The analysis reads the original five proxy scores, selected subsets, locked primary predictions and their receipts, plus the registered E2 selection under `local/followup/phase1`.

```bash
python3 -m pip install -r requirements/selection.txt
python3 -m s2l_tool_calling.analysis.followup_difficulty \
  --primary "${PRIMARY_RUN:-local/course-run}" --phase1 local/followup/phase1 \
  --out local/followup/phase0/difficulty-replay
```

The output consists of the five CSV tables linked above and the [metric definitions](diagnostic-definitions.json). The published tables contain counts, rates, thresholds and loss summaries. [Full reproduction guide](../../../docs/reproduce.md)
