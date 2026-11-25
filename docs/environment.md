# Reproduce the Linux GPU environments

The experiment ran on one NVIDIA A40 (46,068 MiB), Linux x86_64 with glibc 2.39, and CPython 3.12.3. The NVIDIA driver was 595.112.07. The training callback reported CUDA 12.8. Use a host with a working NVIDIA driver, BF16 support and a C/C++ build toolchain. Run the commands below from the repository root in the same Bash shell.

| Environment | Used for | Recorded versions |
| --- | --- | --- |
| Training | LoRA SFT, tokenizer and chat-template preparation, checkpoint scoring, k-means selection and plots | PyTorch 2.8.0, Transformers 5.3.0, ms-swift 4.6.1, PEFT 0.18.1, qwen-vl-utils 0.0.14, flash-linear-attention and fla-core 0.4.2, NumPy 2.5.5, scikit-learn 1.9.1 |
| Generation | Greedy vLLM decoding of prepared prompt token IDs, including LoRA adapters | vLLM 0.19.1, PyTorch 2.10.0, Transformers 5.24.0, NumPy 2.2.6 |
| BFCL scoring | Constructing the blind external prompt set and scoring predictions with Berkeley’s official AST checker on CPU | BFCL source revision listed below, PyTorch 2.15.0+cpu, NumPy 1.26.4, qwen-agent 0.0.34, soundfile 0.14.0 |

The root `requirements.txt` reproduces the earlier Hugging Face workstation field-loss check. For results from the A40, use the three Linux package lists below. Installing the environments separately preserves the PyTorch, Transformers and NumPy versions each stage used. [Workstation field-loss check](field-loss-validation.md)

## 1. Create isolated environments

Install CPython 3.12, its venv module, curl and tar on the host. The bootstrap below supplies a known version of the uv command-line tool. Each venv lives under the Git-ignored `local/` directory.

```bash
set -euo pipefail
nvidia-smi
mkdir -p local/envs
python3.12 -m venv local/envs/bootstrap
UV="$PWD/local/envs/bootstrap/bin/uv"
local/envs/bootstrap/bin/python -m pip install 'uv==0.9.9'
for env in train vllm bfcl; do
  "$UV" venv --python "$(command -v python3.12)" "$PWD/local/envs/$env"
done
TRAIN_PY="$PWD/local/envs/train/bin/python"
VLLM_PY="$PWD/local/envs/vllm/bin/python"
BFCL_PY="$PWD/local/envs/bfcl/bin/python"
```

## 2. Install the training and generation packages

The two Linux manifests pin the CUDA runtime packages resolved with the corresponding PyTorch wheels. The training manifest includes the final FLA 0.4.2 pin. In this run the model reported a fallback to its PyTorch implementation for an optional fast path. Adding extra attention or convolution kernels can change timings.

```bash
"$UV" pip install --python "$TRAIN_PY" --default-index https://pypi.org/simple -r requirements/linux-train.txt
"$UV" pip install --python "$VLLM_PY" --default-index https://pypi.org/simple -r requirements/linux-vllm.txt
"$UV" pip check --python "$TRAIN_PY"
"$UV" pip check --python "$VLLM_PY"
```

## 3. Install the official BFCL checker on CPU

Fetch the pinned source archive on a fresh clone. If this revision is already present in `local/tools/bfcl`, continue at the installation command. The source includes the four data files, checker and converter whose digests are verified by `scripts/bfcl_subset.py`. The local source version identifies the checker even when its build reports a fallback package version. [Pinned BFCL source](https://github.com/ShishirPatil/gorilla/tree/f7cf7359b7ac615a0b294831c5ba2bc95ee4a000/berkeley-function-call-leaderboard)

```bash
BFCL_REV=f7cf7359b7ac615a0b294831c5ba2bc95ee4a000
mkdir -p local/tools/bfcl
curl -fL --retry 4 "https://codeload.github.com/ShishirPatil/gorilla/tar.gz/$BFCL_REV" -o local/tools/bfcl.tar.gz
tar -xzf local/tools/bfcl.tar.gz -C local/tools/bfcl --strip-components=2 \
  "gorilla-$BFCL_REV/berkeley-function-call-leaderboard"
tar -xOf local/tools/bfcl.tar.gz "gorilla-$BFCL_REV/LICENSE" > local/tools/bfcl/LICENSE
```

Install the package's default dependencies and the recorded transitive versions. The CPU wheel comes from the PyTorch CPU index selected by uv. The explicit `soundfile` pin covers an import in qwen-agent. BFCL’s optional `oss_eval_vllm` extra pins vLLM 0.8.5 for Berkeley’s own serving path. This experiment generates predictions in the separate vLLM environment.

```bash
"$UV" pip install --python "$BFCL_PY" --torch-backend cpu --default-index https://pypi.org/simple \
  -r requirements/linux-bfcl.txt ./local/tools/bfcl
"$UV" pip check --python "$BFCL_PY"
```

## 4. Check the installation before training

These checks import the interfaces used by the experiment, verify the training CUDA runtime and confirm that BFCL accepts a correct argument and rejects an incorrect one. The synthetic BFCL example requires no credentials or model downloads.

```bash
"$TRAIN_PY" - <<'PY'
from importlib.metadata import version
import numpy, sklearn, torch
from swift import get_processor, get_template
from peft import PeftModel
from transformers import Qwen3_5ForConditionalGeneration
assert torch.version.cuda == '12.8' and torch.cuda.is_available()
assert torch.cuda.is_bf16_supported()
assert torch.__version__.split('+')[0] == '2.8.0'
assert [version(p) for p in ('ms-swift', 'transformers', 'peft')] == ['4.6.1', '5.3.0', '0.18.1']
assert (numpy.__version__, sklearn.__version__) == ('2.5.5', '1.9.1')
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
PY

"$VLLM_PY" - <<'PY'
import torch, transformers, vllm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
assert torch.__version__.split('+')[0] == '2.10.0'
assert (vllm.__version__, transformers.__version__) == ('0.19.1', '5.24.0')
print(torch.__version__, vllm.__version__, transformers.__version__)
PY

PYTHONPATH="$PWD/scripts:$PWD/local/tools/bfcl" HF_HUB_OFFLINE=1 "$BFCL_PY" - <<'PY'
from bfcl_subset import DEFAULT_ROOT, SOURCE_HASHES, digest, score_entry
for path, expected in SOURCE_HASHES.items():
    assert digest(DEFAULT_ROOT / 'bfcl_eval' / path) == expected, path
schema = {'name': 'order.track', 'parameters': {'type': 'dict', 'properties':
          {'order_id': {'type': 'string'}}, 'required': ['order_id']}}
truth = [{'order.track': {'order_id': ['4821']}}]
for value, expected in [('4821', True), ('9999', False)]:
    calls = [{'name': 'order_track', 'arguments': {'order_id': value}}]
    assert score_entry([schema], calls, truth, 'simple_python')['valid'] == expected
print('BFCL source hashes and both synthetic judgments passed')
PY
```

For a direct `swift sft` command, activate `local/envs/train` first. The course runner accepts `--training-python "$TRAIN_PY" --eval-python "$VLLM_PY" --bfcl-python "$BFCL_PY"` and puts each interpreter’s bin directory on PATH for its stage. Use the experiment instructions for data preparation, the training budget and the protocol arguments. [Repository instructions](../README.md)

## Restart a training Pod

From the project checkout, inspect the space available for restoring the packages. The preflight reads the two pinned package manifests and works on a Mac.

```bash
python3 scripts/restore_gpu_envs.py --plan
df -h /workspace /opt
```

On the Linux GPU host, run the following commands. The script first checks the historical environments at `/opt/trainenv` and `/opt/evalenv`. On a host with an empty container disk, it creates independent environments and a shared uv cache in Git-ignored `local/envs/round2` on the mounted project volume. The script checks available space before each installation. Choose `--env-root /opt/cs260d-envs` on both commands if the container disk is better suited for the packages; on a stop, those installations may need to be recreated. Keep enough space on the project volume for run artifacts. The first-stage receipt is available as soon as the training environment verifies.

```bash
python3 scripts/restore_gpu_envs.py --stage train
python3 scripts/restore_gpu_envs.py --stage vllm
```

The JSON output for each stage records the interpreter path for the controller, the pinned manifest hash, the CUDA version and the GPU model. On the host each check validates every pinned package, imports the interfaces used in that stage, and runs a small BF16 device calculation. `--check-only` repeats the checks without installing packages or writing files. The full installation logs and JSON verification records are written under the selected environment root. The training and generation checks do not load any model or evaluation examples.

## What was verified

The three manifests were compared package by package to the saved Linux installation logs, including the final training-package adjustments. The proxy score receipt confirms ms-swift, Transformers, PyTorch and PEFT. The selection receipt records the imported NumPy and scikit-learn versions. The generation receipt records vLLM, PyTorch and Transformers. Host-level packages also appear in the historical metadata dump, e.g. CUDA 12.9 and another NumPy build. The pins here follow the venv installation and the stage-specific runtime receipts.

On November 25, 2026 (UTC), `scripts/restore_gpu_envs.py` rebuilt both GPU environments on a fresh A40 Pod with Python 3.12.3. The training environment verified 143 pinned packages and the vLLM environment verified 179. Both passed `uv pip check`, imported their required interfaces, reported CUDA 12.8 and executed a BF16 matrix multiplication on the A40. Copies of both verification records and both installation logs were checked against their on-host SHA-256 hashes.

The pinned versions satisfy the Linux runtime dependency declarations in the four archived wheels (ms-swift, Transformers, PEFT and qwen-vl-utils) and the pinned BFCL project's 33 direct dependencies. The BFCL source hashes and two synthetic judgments were checked on a Mac CPU using the pinned Berkeley source. The fresh Pod check covered the training and generation environments. The historical record contains version pins and omits wheel hashes, source build tools and a container image digest.
