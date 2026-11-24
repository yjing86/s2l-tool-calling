"""Launch a recorded single-GPU ms-swift tool-call LoRA run."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time

from verify_tool_call_loss import swift_row


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["smoke", "proxy", "target"], required=True)
    for arg in ["model", "data", "output"]:
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--pool", type=Path, help="Complete candidate pool, required for target token budget.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args()
    if args.stage == "target" and not args.pool:
        parser.error("--stage target requires --pool")
    batch = 1 if args.stage == "smoke" else 16
    if args.micro_batch < 1 or batch % args.micro_batch:
        parser.error("micro batch must divide the effective batch")
    swift = shutil.which("swift")
    if not swift:
        parser.error("Activate the ms-swift training environment first")
    model = args.model.resolve(strict=True)
    rows = [json.loads(s) for s in args.data.read_text().splitlines() if s.strip()]
    if args.stage == "smoke":
        rows = [max(rows, key=lambda r: len(json.dumps(r, ensure_ascii=False)))]
    ids = [str(r["id"]) for r in rows]
    if not rows or len(ids) != len(set(ids)):
        parser.error("Need a non-empty dataset with unique IDs")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    dataset = output / "train.swift.jsonl"
    with dataset.open("w") as stream:
        for row in rows:
            print(json.dumps(swift_row(row), ensure_ascii=False), file=stream)
    steps = math.ceil(len(rows) / batch)
    save_steps = sorted({max(1, round(steps * f)) for f in (0.04, 0.12, 0.32, 1)}) if args.stage == "proxy" else []
    plugin = Path(__file__).with_name("experiment_callback.py").resolve()
    cmd = [swift, "sft", "--model", str(model), "--dataset", str(dataset),
           "--model_type", "qwen3_5", "--template", "qwen3_5", "--agent_template", "qwen3_5",
           "--tuner_type", "lora", "--target_modules", "up_proj", "down_proj", "--freeze_vit", "true",
           "--lora_rank", "16", "--lora_alpha", "32", "--lora_dropout", "0.05",
           "--loss_scale", "default", "--enable_thinking", "false", "--use_hf", "true",
           "--use_cpu", "false", "--torch_dtype", "bfloat16", "--bf16", "true", "--fp16", "false",
           "--gradient_checkpointing", "true", "--attn_impl", "sdpa", "--optim", "adamw_torch",
           "--learning_rate", "0.0001", "--lr_scheduler_type", "constant_with_warmup", "--warmup_steps", "5" if args.stage != "smoke" else "0",
           "--split_dataset_ratio", "0", "--strict", "true", "--truncation_strategy", "delete",
           "--max_length", str(args.max_length), "--per_device_train_batch_size", str(args.micro_batch),
           "--gradient_accumulation_steps", str(batch // args.micro_batch),
           "--save_strategy", "no", "--logging_steps", "1", "--logging_nan_inf_filter", "false",
           "--seed", str(args.seed), "--data_seed", str(args.seed), "--report_to", "none",
           "--check_model", "false", "--add_version", "false", "--output_dir", str(output),
           "--external_plugins", str(plugin), "--callbacks", "tool_experiment"]
    if args.stage == "target":
        # The callback stops at the actual non-padding token target. Fail if this guard is ever hit.
        cmd += ["--num_train_epochs", "10"]
    else:
        cmd += ["--num_train_epochs", "1"]
    metadata = {"stage": args.stage, "seed": args.seed, "batch": batch, "planned_steps_one_epoch": steps,
                "save_steps": save_steps, "ids": ids, "data_sha256": sha256(args.data),
                "pool": str(args.pool.resolve()) if args.pool else None,
                "pool_sha256": sha256(args.pool) if args.pool else None,
                "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]},
                "python": sys.version, "platform": platform.platform(), "command": cmd,
                "sources_sha256": {p.name: sha256(p) for p in [Path(__file__), plugin, plugin.with_name("verify_tool_call_loss.py")]},
                "model": str(model), "model_config_sha256": sha256(model / "config.json"),
                "start_unix": time.time()}
    manifest = output / "run.json"
    write_json(manifest, metadata)
    print(shlex.join(cmd), flush=True)
    started = time.monotonic()
    env = dict(os.environ, TOOL_TRAIN_MANIFEST=str(manifest), PYTHONUNBUFFERED="1")
    with (output / "console.log").open("w") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        for line in proc.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        code = proc.wait()
    write_json(output / "exit.json", {"code": code, "elapsed_seconds": time.monotonic() - started, "end_unix": time.time()})
    if code:
        sys.exit(code)
    if not (output / "completed.json").exists():
        sys.exit("Trainer exited without the protocol completion record")


if __name__ == "__main__":
    main()
