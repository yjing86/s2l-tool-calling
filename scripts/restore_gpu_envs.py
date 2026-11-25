"""Restore the pinned experiment environments on a restarted Linux GPU Pod."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
LEGACY = {"train": Path("/opt/trainenv"), "vllm": Path("/opt/evalenv")}
GIB = 1024**3


def manifest(stage):
    return ROOT / "requirements" / f"linux-{stage}.txt"


def fingerprint(stage):
    return hashlib.sha256(manifest(stage).read_bytes()).hexdigest()


def verify(stage):
    if sys.version_info[:2] != (3, 12) or platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Expected Linux x86_64 and CPython 3.12")
    pins = {}
    mismatches = []
    for raw in manifest(stage).read_text().splitlines():
        entry = raw.split("#", 1)[0].strip()
        if not entry:
            continue
        name, expected = entry.split("==")
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = "missing"
        pins[name] = actual
        if actual != expected and ("+" in expected or actual.split("+", 1)[0] != expected):
            mismatches.append(f"{name}: {actual} (expected {expected})")
    if mismatches:
        raise RuntimeError("Pin mismatches: " + ", ".join(mismatches))

    import torch
    if torch.version.cuda != "12.8" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A working BF16 GPU with the pinned CUDA 12.8 PyTorch build is required")
    if stage == "train":
        from swift import get_processor, get_template
        from transformers import Qwen3_5ForConditionalGeneration
        from peft import PeftModel
        from sklearn.cluster import KMeans
        libraries = ("torch", "transformers", "ms-swift", "peft", "scikit-learn")
    else:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        libraries = ("torch", "transformers", "vllm")
    if torch.ones((1, 1), dtype=torch.bfloat16, device="cuda").mm(
        torch.ones((1, 1), dtype=torch.bfloat16, device="cuda")
    ).item() != 1.0:
        raise RuntimeError("BF16 device calculation failed")
    return {
        "stage": stage, "python": sys.executable, "python_version": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0), "torch_cuda": torch.version.cuda,
        "versions": {name: pins[name] for name in libraries}, "packages_verified": len(pins),
        "requirements_sha256": fingerprint(stage),
    }


def probe(env_dir, stage):
    interpreter = env_dir / "bin/python"
    if not interpreter.is_file():
        return None
    outcome = subprocess.run(
        [str(interpreter), str(Path(__file__).resolve()), "_verify", stage],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    for line in reversed(outcome.stdout.splitlines()):
        if outcome.returncode == 0 and line.startswith("ENVIRONMENT_VERIFIED "):
            return json.loads(line[len("ENVIRONMENT_VERIFIED "):])
    print(f"Verification failed for {interpreter}:\n{outcome.stdout[-1600:]}", file=sys.stderr, flush=True)
    return None


def available_gib(path):
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free / GIB


def ensure_space(root, minimum):
    free = available_gib(root)
    if free < minimum:
        raise RuntimeError(f"{root} has {free:.1f} GiB free, requires {minimum:g} GiB before package installation")


@contextmanager
def exclusive(path):
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def restore(args, stage, root):
    target = root / f"{stage}-{fingerprint(stage)[:16]}"
    for candidate in (LEGACY[stage], target):
        existing = probe(candidate, stage)
        if existing is not None:
            return existing
    if args.check_only:
        raise RuntimeError(f"No verified {stage} interpreter found in {LEGACY[stage]} or {target}")
    ensure_space(root, args.min_free_gib)
    base = shutil.which("python3.12")
    if not base:
        raise RuntimeError("Install CPython 3.12 with venv support first")
    if target.exists() and not (target / "pyvenv.cfg").is_file() and any(target.iterdir()):
        raise RuntimeError(f"Existing non-venv directory at {target}; choose a different --env-root")
    uv = shutil.which("uv")
    if not uv:
        bootstrap = root / "uv-bootstrap"
        uv = bootstrap / "bin/uv"
        if not uv.is_file():
            subprocess.run([base, "-m", "venv", str(bootstrap)], check=True)
            subprocess.run([str(bootstrap / "bin/python"), "-m", "pip", "install", "uv==0.9.9"], check=True)
    uv = str(uv)
    env = os.environ.copy()
    for key in ("UV_INDEX", "UV_INDEX_URL", "UV_EXTRA_INDEX_URL", "UV_DEFAULT_INDEX", "UV_FIND_LINKS", "UV_NO_INDEX"):
        env.pop(key, None)
    env["UV_CACHE_DIR"] = str(root / "uv-cache")
    if not (target / "bin/python").is_file():
        subprocess.run([uv, "--no-config", "venv", "--python", base, str(target)], check=True, env=env)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log = root / f"{stage}-install-{stamp}.log"
    print(f"Installing {stage} packages. Output: {log}", flush=True)
    with log.open("x") as stream:
        calls = (
            [uv, "--version"],
            [uv, "--no-config", "pip", "install", "--python", str(target / "bin/python"),
             "--default-index", "https://pypi.org/simple", "-r", str(manifest(stage))],
            [uv, "--no-config", "pip", "check", "--python", str(target / "bin/python")],
        )
        for command in calls:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
            if result.returncode:
                raise RuntimeError(f"Environment installation failed; inspect {log}")
    restored = probe(target, stage)
    if restored is None:
        raise RuntimeError(f"{stage} did not pass the import and CUDA checks")
    restored["install_log"] = str(log)
    return restored


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "_verify" and sys.argv[2] in LEGACY:
        print("ENVIRONMENT_VERIFIED " + json.dumps(verify(sys.argv[2])), flush=True)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "vllm", "both"), default="both")
    parser.add_argument("--env-root", type=Path, default=ROOT / "local/envs/round2",
                        help="Managed environments and uv cache; use a mounted volume to retain them after stop")
    parser.add_argument("--min-free-gib", type=float, default=16,
                        help="Minimum measured free space before each install; historical reuse needs no install")
    parser.add_argument("--check-only", action="store_true", help="Probe installed environments; never install")
    parser.add_argument("--plan", action="store_true", help="Print intended paths, pins and disk space without touching the host")
    args = parser.parse_args()
    if not 0 < args.min_free_gib < float("inf"):
        parser.error("--min-free-gib must be finite and positive")
    root = args.env_root.resolve()
    stages = tuple(LEGACY) if args.stage == "both" else (args.stage,)
    if args.plan:
        for stage in stages:
            print(json.dumps({"stage": stage, "requirements_sha256": fingerprint(stage),
                              "legacy": str(LEGACY[stage]), "env_root": str(root),
                              "free_gib": round(available_gib(root), 2)}))
        return
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("Use a Linux x86_64 GPU host; --plan also works on a workstation")
    if args.check_only:
        for stage in stages:
            print(json.dumps(restore(args, stage, root)), flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    with exclusive(root / "restore.lock"):
        for stage in stages:
            record = restore(args, stage, root)
            destination = root / f"{stage}-verified.json"
            temporary = destination.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, indent=2) + "\n")
            temporary.replace(destination)
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
