# Round 2 internal evaluation

The registered panel has 47 endpoints: the base, 26 archived adapters and 20 new token-matched training endpoints. Every endpoint is evaluated on the same 1,500 reserved requests, for 70,500 generations. The earliest opening is **December 5, 2026 at 00:00 Los Angeles time (08:00 UTC)**. Complete the external BFCL protocol separately. [Round 2 registration](../results/followup/round2-registration.json)

## Before opening

Use the checkout containing the scripts and an ignored `local/` tree with the original artifacts. The following command inspects only the public registration.

```bash
python scripts/round2_holdout.py preflight
```

After all 20 new fits, stage the target snapshot in `local/models/Qwen3.5-2B/`, the original effective template arguments in `local/runs/target-smoke/args.json`, and the complete `local/course-run/`, `local/full-reference/`, `local/followup/phase1/`, `local/followup/difficulty-control/` and `local/followup/round2/` trees. The originals can reside in a different physical directory with the same structure; pass that directory as `--artifacts` consistently. Run the freeze before copying any reserved input to an evaluation machine.

```bash
python scripts/round2_holdout.py freeze
```

The command validates every historical adapter and its config against the pretraining registration, the 20 new endpoint locks against their weights, the base weights and template against their original checksums, and the selected training lengths against the frozen selection manifests. It writes a deterministic `local/followup/round2/holdout-evaluation/freeze.json` and its public commitment `results/followup/round2-holdout-freeze.json`. Publish the commitment to the project’s public main branch, keep the private freeze with the run, and record the 40-character published commit. The holdout file fingerprint in the earlier [reservation](../results/followup/phase1-registration.json) commits to the complete byte stream, including the ordering of IDs. The release checks those bytes and the ordered ID and group digests against the original reserve manifest.

## On or after the opening date

Run from a Git checkout with an up-to-date `origin/main`. Stage the existing private reservation `local/followup/holdout-v2/{test.jsonl,manifest.json}` on the evaluation host for this opening. Keep preregistered weights, source files and prompts fixed. Use the pinned training interpreter to encode the prompts once and the separate vLLM interpreter for greedy generation. The orchestration interpreter needs NumPy and SciPy for the final statistical summary. [Environment instructions](environment.md)

```bash
git fetch origin main
python scripts/round2_holdout.py run --anchor-commit "$FREEZE_COMMIT" \
  --training-python "$TRAIN_PY" --eval-python "$VLLM_PY"
```

The runner checks the date, public Git commitment, complete model panel, original split fingerprint and model files. It encodes 1,500 inputs once using the original template, and generates with a 4,096-token context and at most 512 output tokens. Each raw response is independently re-scored using the pinned complete-call judge. Genuine malformed or incorrect calls stay in the denominator. The progress log prints completed endpoint counts. Keep evaluation files and logs private throughout the batch.

For an interrupted run, use the same command, checkout, artifact tree and freeze commit. Finished endpoints are revalidated. A failed worker attempt remains under `holdout-evaluation/attempts/`; the next attempt uses the same weights, settings, input bytes and prompts. A mismatching published freeze, a changed complete output, missing predictions or an incomplete panel stop the workflow. When all 47 endpoints are verified, `holdout-evaluation/summary.json` records the aggregate results and input/output checksums. To reproduce it on a CPU machine with the complete archive:

```bash
python scripts/round2_holdout.py summarize --anchor-commit "$FREEZE_COMMIT"
```

The primary comparison is the preregistered one-sided paired test of C minus B across five seeds. The summary also reports seed-specific differences and their uncertainty, jointly resampled request-group intervals, the separate old/new seed means, and the prespecified descriptive contrasts. Per-request data and model outputs stay under `local/`. [Statistical implementation](../scripts/round2_holdout_stats.py)
