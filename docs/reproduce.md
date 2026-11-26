# Reproduce the single-call tool-selection experiment

This guide starts from the public repository and rebuilds the data splits, trains the proxy and target adapters, and scores the locked models. All inputs come from the publishers. Keep the clone at a stable absolute path for the duration of the run. The receipts record command lines and checkpoint paths.

## 1. Choose the level of reproduction

| Task | Hardware | Outcome |
| --- | --- | --- |
| Inspect the released result | CPU, Python standard library | The fixed aggregate results and a small scoring demo |
| Regenerate inputs | CPU, network, tokenizer dependencies | Verified public sources, frozen split files and an isolated smoke fixture |
| Repeat training and evaluation | One supported Linux NVIDIA GPU | New checkpoints, per-request predictions, paired intervals and timing records |

Run the command blocks from the repository root in one Bash shell. A new machine needs Git and curl. The original training and generation used one A40, CPython 3.12 and separate training, vLLM and CPU BFCL environments. The package versions and import checks are in the [environment guide](environment.md).

```bash
git clone https://github.com/yjing86/s2l-tool-calling.git
cd s2l-tool-calling
set -euo pipefail
git rev-parse HEAD
python3 -m s2l_tool_calling demo
python3 -m s2l_tool_calling results
```

## 2. Fetch the inputs and regenerate the splits

```bash
python3 -m s2l_tool_calling setup bfcl
python3 -m s2l_tool_calling download
```

The downloader verifies the Turnstile release and both Qwen3.5 snapshots against their pinned Hugging Face revisions and repairs any missing or mismatched download. The three snapshots total about 6.74 GB. Account separately for three environments, compiled caches, per-step records, predictions and optimizer checkpoints. The BFCL setup fetches the official pinned Git commit, installs its source tree under `local/tools/bfcl`, and checks the same question, reference and scorer hashes as the external evaluation. Follow the three-environment installation instructions now. For BFCL, the source-fetch step above already populated its directory. [Linux environments](environment.md)

```bash
TRAIN_PY="$PWD/local/envs/train/bin/python"
VLLM_PY="$PWD/local/envs/vllm/bin/python"
BFCL_PY="$PWD/local/envs/bfcl/bin/python"
COURSE="$PWD/local/course-run"
TARGET_SMOKE="$PWD/local/runs/target-smoke"

"$TRAIN_PY" -m s2l_tool_calling prepare
python3 -m s2l_tool_calling verify --prepared local/prepared
python3 -m s2l_tool_calling setup fixture
```

Preparation groups exact and near-duplicate requests, isolates audited examples, and draws one representative per eligible group. There are 3 debug, 200 pilot, 4,000 training-pool, 400 development and 2,000 held-out requests. The verifier compares all five split files and the row audit to published checksums. The complete preparation manifest also records the host's Python and package versions and can differ between hosts. The fixture is debug row 49, reconstructed in the original training format; its SHA-256 begins `ecdac2b4`. Its only training use here is the one-update smoke test. [Reference checksums and design](../results/reproduction-target.json)

## 3. Set the GPU time budget and run the smoke tests

Choose an absolute future UTC stop time within the allocation you control. To continue in a different shell, restore the same stop time and environment paths. The course runner reserves the final 600 seconds. Stopping a process leaves the provider's instance and billing under your control.

```bash
read -r -p 'Your GPU allocation stop time in UTC (YYYY-MM-DDTHH:MM:SSZ): ' DEADLINE
python3 -m s2l_tool_calling setup smoke \
  --training-python "$TRAIN_PY" --course "$COURSE" --deadline "$DEADLINE" --reserve-seconds 600
```

The smoke stage runs a single update on row 49 for each model. It produces `local/runs/proxy-smoke2` and `local/runs/target-smoke` with ms-swift arguments, step records and a verifiable LoRA adapter. The target's `args.json` is needed to encode the same chat prefix used for generation. The 200-question bare-target pilot is created automatically by the benchmark. To preview the smoke action without starting a GPU subprocess, add `--dry-run`.

## 4. Benchmark and freeze the design

```bash
run_primary() {
  "$TRAIN_PY" -m s2l_tool_calling run \
    --course "$COURSE" --deadline "$DEADLINE" --reserve-seconds 600 --micro-batch 4 \
    --training-python "$TRAIN_PY" --eval-python "$VLLM_PY" --bfcl-python "$BFCL_PY" \
    --proxy-smoke "$PWD/local/runs/proxy-smoke2" --target-smoke "$TARGET_SMOKE" \
    --pilot "$PWD/local/runs/pilot-base" --through "$1"
}
run_primary freeze
python3 -m s2l_tool_calling verify --prepared local/prepared --design "$COURSE/design.json"
```

The freeze stage checks both smoke adapters, audits first-token generation on three debug prompts, evaluates the bare target on the pilot, and times training and scoring on 64 pool items. It uses those timings and the remaining budget to choose the design. The reported design used micro-batch 4 with four gradient-accumulation steps (effective batch 16), a pool of 4,000, a subset of 1,000, 2,000 test requests, seeds 42–44, and the seed-42 B′ ablation. The equal-wall-time continuation was disabled. The raw driver's default micro-batch is 2; the command above explicitly selects 4.

A faster or slower machine or a different budget can change the frozen experiment. For example, the runner can reduce the exam to 1,500, fund fewer seed rounds, or fund equal-time continuation. If the verifier reports a difference, the file identifies the design you actually froze. A direct replication requires matching design fields and the later endpoints. To try a different allocation, launch a separate run directory and freeze its own budget. Preserve each design and its receipts.

## 5. Train, freeze controls and open the held-out evaluations

```bash
run_primary train
test -s "$COURSE/primary-progress.json"
```

The 0.8B proxy trains once on all 4,000 items. It scores the pool at steps 0, 10, 30, 80 and 250 for the full design. A draws at random, B samples clusters of whole-answer trajectories, C samples clusters of separate function-name and argument-value trajectories, and B′ pools the supervised tokens from both fields into one token-weighted mean at each checkpoint. Each 2B target starts from the same base weights and stops after crossing the same non-padding training-token goal. Development checks run every 50 updates and at the endpoint. `--through train` records these endpoints and returns before generating internal-test or BFCL predictions.

**Optional controls.** To reproduce the published procedure for the full-pool and crossed-seed investigations, run these two freeze commands now, before the blind evaluations. The full-pool control can also be frozen as soon as `--through proxy` has created the frozen pool and test files. The crossed-seed freeze requires the completed seed-42 and seed-43 B/C diagonal development endpoints. It records its exploratory origin and any blind prediction files already present. Freezing a control after viewing held-out results gives it a different evidential status.

```bash
full_reference() {
  "$TRAIN_PY" scripts/run_full_reference.py "$1" --primary "$COURSE" \
    --out "$PWD/local/full-reference" --training-python "$TRAIN_PY" \
    --eval-python "$VLLM_PY" --target-smoke "$TARGET_SMOKE"
}
crossed_seeds() {
  "$TRAIN_PY" scripts/run_crossed_seeds.py "$1" --primary "$COURSE" \
    --full "$PWD/local/full-reference" --out "$PWD/local/crossed-seeds" \
    --train-python "$TRAIN_PY" --eval-python "$VLLM_PY"
}
full_reference freeze
crossed_seeds freeze
```

Continue with the primary evaluation whether or not you included the optional controls.

```bash
run_primary all
test -s "$COURSE/receipts/summary.json"
python3 -m s2l_tool_calling verify --prepared local/prepared \
  --design "$COURSE/design.json" --locked "$COURSE/locked-models.json"
python3 -m s2l_tool_calling results --input "$COURSE/summary/internal.json"
"$BFCL_PY" scripts/summarize_bfcl_seeds.py --course "$COURSE" --out "$COURSE/summary/bfcl-seeds.json"
```

The primary lock contains the base, nine matched A/B/C endpoints and one B′ endpoint for the original design. The scheduler also rechecks remaining time before admitting later rounds. On the internal test, a correct answer requires a parseable response and the entire reference call to match. For BFCL, only frozen prompts are passed to greedy vLLM generation. Berkeley's official AST scorer checks 100 `simple_python` and 100 `multiple` items in a separate environment. Record both the internal result and the two individual BFCL category scores. Random seeds and exact inputs fix the comparison, while CUDA kernels, software builds and hardware can still affect numerical results. [Published results and uncertainty](results.md)

## 6. Run the optional controls within the frozen budget

```bash
full_reference fit
if test -s local/full-reference/full-models.json; then
  full_reference test
fi
crossed_seeds run
```

The seed-42 full-pool reference trains on all 4,000 examples and reports the matched-token, final and best-development endpoints on the internal test. Some endpoints can designate the same checkpoint. The diagnostic adds four off-diagonal B/C trainings, crossing selection seeds 42 and 43 with target training seeds 43 and 42. Each new adapter is evaluated once at the equal-token endpoint on the same 400 development questions. It yields no additional held-out or BFCL score. The original target smoke and the original working paths are needed for these stages.

The schedulers prioritize primary completion, then the full reference, then the crossed-seed diagnostic. They can return success while reporting a deferral or a budget skip. Read `local/full-reference/decisions` or `local/crossed-seeds/decisions` for timing decisions and the terminal receipts for completion. A full-reference timing skip can unlock the crossed diagnostic if the latter still fits. Another process holding a lock can defer execution; rerun the same action after that process completes and within the same deadline.

## 7. Inspect and retain the evidence

The CPU difficulty and exposure audit compares proxy loss at a fixed argument length and pairs models on the same held-out requests. Its report includes the command to regenerate the five aggregate tables and their metric definitions from the primary run and registered E2 subsets. [Difficulty and exposure audit](../results/followup/difficulty/README.md)

| Evidence | Location in the clone |
| --- | --- |
| Split identities, row audit and original sources | `local/prepared/manifest.json`, `local/prepared/rows.audit.jsonl`, `local/data/turnstile/` |
| Benchmark decision, planned and realized endpoints | `local/course-run/design.json`, `rounds/`, `locked-models.json` and `training-ledger.json` under the run directory |
| Proxy scores, subset IDs, adapters, per-step token counts | `local/course-run/scores/`, `selection/` and `training/` under the run directory |
| Raw generations, per-model scores and per-task receipts | `evaluation/{dev,test,bfcl}/`, `receipts/`, `logs/` and `ledger.jsonl` under the run directory |
| Primary result and audit bundle | `local/course-run/summary/internal.json`, `bfcl-seeds.json` and `aggregate-audit.zip` under `summary/` |
| Full-pool completion | `local/full-reference/summary/full-reference.json` and `local/full-reference/receipts/full-summary.json` |
| Crossed-seed completion | `local/crossed-seeds/summary.json` and `local/crossed-seeds/receipts/summary.json` |

On an interrupted run, inspect the last ledger event and the task log. Retry with the same absolute paths, interpreters, deadline, batch and output directory. The driver validates completed tasks against stored arguments and output hashes. An unclean interruption may require confirming that the child process has exited and preserving partial files before retrying. Keep the ignored `local/` archive on storage you control. The public checkout distributes code, synthetic examples, reference checksums and aggregate results. The raw Turnstile material remains subject to its publisher's CC BY-NC 4.0 terms. [Turnstile dataset card](https://huggingface.co/datasets/amazon/Turnstile-Synthetic-Domains)

**Verification scope.** The released A40 results supply the reference files and measured settings. An independent CPU replay regenerated all six split/audit files and both BFCL preparation files byte-for-byte. Given the original five proxy score files, it also regenerated all twelve ordered subset lists across the three seeds. A clean temporary checkout fetched the pinned BFCL tree from GitHub and reconstructed the original smoke input byte-for-byte. The recorded design and eleven locked endpoint labels match the public reference. A new clean-room GPU retrain from this public guide has yet to be run.

## 8. Rebuild the matched-token and 10% follow-up

This stage uses the completed earlier runs as inputs. It reuses the five proxy score files, the archived seed-42 reference adapter, the previous round's cached generation prompts, and its scored 25% controls. The public repository records their identities. The original weights and request-level generations stay in the private run archive. An exact audit requires that archive. A fresh retraining on another GPU can produce different weights even with the same seed.

Run from the clone root on Linux. Set `ARCHIVE` to the absolute directory that contains `course-run/`, `full-reference/` and `followup/` from the earlier stages. The example uses the default layout. Use the training and vLLM interpreters configured in section 2.

```bash
ARCHIVE="$PWD/local"
R3="$PWD/local/followup/round3-replay"
INPUT="$R3/inputs"
OLD="$ARCHIVE/followup/round2/holdout-evaluation"
mkdir -p "$INPUT/full10-s42"
cp "$ARCHIVE/course-run/frozen/train.jsonl" "$INPUT/train.jsonl"
cp "$ARCHIVE/course-run/training/s42-A/run.json" "$INPUT/reference-run.json"
cp "$ARCHIVE/followup/holdout-v2/test.jsonl" "$INPUT/test.jsonl"
cp "$OLD/prompts/"{prompts.jsonl,prepare.json} "$INPUT/"
cp "$ARCHIVE/full-reference/training/checkpoint-50/"{adapter_model.safetensors,adapter_config.json} "$INPUT/full10-s42/"
python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["provenance"]))' "$OLD/summary.json" > "$INPUT/holdout-provenance.json"
for seed in 42 43 44 45 46; do
    "$TRAIN_PY" scripts/select_experiment.py --train "$INPUT/train.jsonl" \
        --scores "$ARCHIVE/course-run/scores" --size 400 --clusters 10 \
        --seed "$seed" --cluster-seed 42 --out "$INPUT/selection/$seed"
done
python3 scripts/register_round3.py "$ARCHIVE" "$INPUT/selection" > "$R3/reconstructed-registration.json"
python3 -c 'import json,sys; assert json.load(open(sys.argv[1])) == json.load(open(sys.argv[2]))' \
    "$R3/reconstructed-registration.json" results/followup/round3-registration.json
```

The selector requires a fresh output directory for each seed. After preparing the inputs, run the following two commands in sequence. The first trains all 19 adapters and freezes 24 evaluation configurations. The second verifies the saved checkpoints and generates 1,500 calls per configuration. Returning to the same output directory reuses completed runs. Existing attempts are kept separately.

```bash
args=(--inputs "$INPUT" --out "$R3/output" --model "$ARCHIVE/models/Qwen3.5-2B"
      --train-python "$TRAIN_PY" --eval-python "$VLLM_PY")
"$TRAIN_PY" scripts/run_round3.py "${args[@]}" --train-only
"$TRAIN_PY" scripts/run_round3.py "${args[@]}"
"$TRAIN_PY" scripts/summarize_round3.py --run "$R3/output" --historical "$OLD" \
    --data "$INPUT/test.jsonl" --out "$R3/results.json"
```

Inspect `output/training/*/steps.jsonl` for the first optimizer update crossing each token threshold. Inspect `output/freeze.json` for the adapter identities and `output/predictions/*/config.json` for the actual prompts, generation settings and software versions. The last command regrades every raw response and checks the archived controls before calculating the paired effects. The model-by-seed comparison uses five pairs. The 36,000 new outputs cover 1,500 distinct test requests.
