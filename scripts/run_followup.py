"""Freeze and run the eleven post-hoc Phase 1 tool-selection controls on one GPU."""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import shlex
import signal
import subprocess
import time

from run_course_experiment import Protocol, args_parser, digest, read, rows, write
from run_crossed_seeds import WEIGHTS, WEIGHTS_SHA256, gen_protocol, normalized_training, token_boundary


BATCHES = {
    "Bprime_and_E2": [(43, "Bprime"), (44, "Bprime"), (42, "E2"), (43, "E2"), (44, "E2")],
    "E3": [(s, "E3") for s in (42, 43, 44)],
    "E4": [(s, "E4") for s in (42, 43, 44)],
}
ORDER = [cell for cells in BATCHES.values() for cell in cells]
STAGES = ("train", "dev", "test")
SPLITS = {"dev": 400, "test": 2000}
TEARDOWN_SECONDS = 15  # Protocol.task grants a terminated process group up to eight seconds to exit.
SOURCES = (
    "select_followup.py", "run_followup.py", "summarize_followup.py", "run_course_experiment.py",
    "run_crossed_seeds.py", "train_experiment.py", "experiment_callback.py", "evaluate_experiment.py",
    "verify_tool_call_loss.py", "select_experiment.py", "summarize_course.py",
)


def utc(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def label(seed, method):
    return f"s{seed}-{method}"


def immutable(path, obj):
    if path.exists():
        if read(path) != obj:
            raise ValueError(f"Registered content differs at {path}")
    else:
        write(path, obj)


def check_outputs(root, task, paths):
    """Map receipts by artifact basename, also supporting a relocated, read-only backup."""
    rec = read(root / "receipts" / (task + ".json"))
    if rec["task"] != task:
        raise ValueError(f"{task}: wrong receipt")
    for path in paths:
        expected = [h for name, h in rec["outputs"].items() if Path(name).name == path.name]
        if len(expected) != 1 or digest(path) != expected[0]:
            raise ValueError(f"{task}: artifact differs: {path}")
    return rec


def timing(path):
    value = read(path)["elapsed_s"]
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid task timing: {path}")
    return value


def footprint(directory):
    return sum(f.stat().st_size for f in directory.rglob("*") if f.is_file() and not f.is_symlink())


def train_protocol(manifest, runtime):
    return {
        "command": normalized_training(manifest["command"]),
        "packages": {k: manifest["packages"][k] for k in ("torch", "transformers", "peft", "ms-swift", "ms_swift")
                     if k in manifest["packages"]},
        "sources_sha256": manifest["sources_sha256"], "model_config_sha256": manifest["model_config_sha256"],
        "runtime": {k: runtime[k] for k in ("cuda", "gpu", "trainable_parameter_count", "token_goal", "starting_model_step")},
    }


def files_for_split(primary, runner, split):
    data = runner.prep / "dev.jsonl" if split == "dev" else primary / "frozen/test.jsonl"
    return data, primary / "prompts" / split / "prompts.jsonl"


def selection_spec(primary, runner, pool):
    by_id = {r["id"]: r for r in rows(pool)}
    if len(by_id) != 4000:
        raise ValueError("Expected 4,000 unique primary training candidates")
    base = runner.root / "selection"
    master = read(base / "manifest.json")
    pool_hash = digest(pool)
    if (master["train_sha256"] != pool_hash or master["pool_size"] != 4000 or master["size"] != 1000
            or master["seeds"] != [42, 43, 44] or master["steps"] != [0, 10, 30, 80, 250]
            or master["target_input_token_budget"] != 3005228 or master["cluster_seed"] != 42
            or master["clusters"] != 10 or master["script_sha256"] != digest(runner.scripts / "select_followup.py")
            or master["cluster_draw_source_sha256"] != digest(runner.scripts / "select_experiment.py")):
        raise ValueError("Selection manifest does not match the frozen study protocol or implementation")
    score_folder = primary / "scores"
    if master["score_config_sha256"] != digest(score_folder / "score-config.json"):
        raise ValueError("Selection score configuration differs")
    for step in master["steps"]:
        rec = read(score_folder / f"step-{step}.complete.json")
        if rec != master["score_receipts"][str(step)] or rec["sha256"] != digest(score_folder / f"step-{step}.jsonl"):
            raise ValueError(f"Registered proxy score receipt differs at step {step}")
    spec = {}
    for seed, method in ORDER:
        path = base / str(seed) / (method + ".jsonl")
        manifest_path = path.parent / "selection.json"
        metadata = read(manifest_path)
        wanted = metadata["subsets"][method]
        selected = rows(path)
        ids = [r["id"] for r in selected]
        checksum = digest(path)
        if (len(ids) != 1000 or len(set(ids)) != 1000 or ids != wanted["ids"]
                or master["selection_manifest_sha256"][str(seed)] != digest(manifest_path)
                or metadata["seed"] != seed or metadata["train_sha256"] != pool_hash
                or checksum != wanted["sha256"] or any(r != by_id.get(r["id"]) for r in selected)):
            raise ValueError(f"{path}: selected rows, order or recorded hashes differ")
        if method == "Bprime" and checksum != digest(primary / "selection" / str(seed) / "Bprime.jsonl"):
            raise ValueError(f"{path}: Bprime differs from the list produced before the primary test")
        spec[label(seed, method)] = {"sha256": checksum, "size": len(ids),
                                     "seed_manifest_sha256": digest(manifest_path)}
    return spec, digest(base / "manifest.json")


def build_spec(primary, runner, hard_cutoff):
    design = read(primary / "design.json")
    lock = read(primary / "locked-models.json")
    if ((design["pool"], design["subset"], design["test"], design["micro_batch"]) != (4000, 1000, 2000, 4)
            or design["seeds"] != [42, 43, 44] or not design["bprime"] or lock["equal_time"]):
        raise ValueError("Primary design does not match the released eleven endpoints")
    locked = lock["main"]
    required = {"base", "s42-Bprime"} | {label(s, g) for s in (42, 43, 44) for g in "ABC"}
    if set(locked) != required:
        raise ValueError("Unexpected primary locked endpoints")
    pool = primary / "frozen/train.jsonl"
    selections, manifest_sha = selection_spec(primary, runner, pool)
    sources = {s: digest(runner.scripts / s) for s in SOURCES}
    model_sha = digest(runner.big / WEIGHTS)
    if model_sha != WEIGHTS_SHA256:
        raise ValueError("Unexpected 2B weights")
    cfg_sha = digest(runner.big / "config.json")
    inputs = {}
    for split, size in SPLITS.items():
        data, prompts = files_for_split(primary, runner, split)
        check_outputs(primary, "encode-" + split, [prompts, prompts.with_name("prepare.json")])
        config = read(primary / "evaluation" / split / "base/config.json")
        controls = gen_protocol(config)
        items = rows(data)
        if (len(items) != size or len({r["id"] for r in items}) != size
                or controls["source_sha256"] != digest(data) or controls["prompts_sha256"] != digest(prompts)
                or controls["tokenizer_sha256"] != digest(runner.big / "tokenizer.json")
                or controls["template_sha256"] != digest(runner.big / "chat_template.jinja")
                or controls["script_sha256"] != sources["evaluate_experiment.py"]):
            raise ValueError(f"{split}: frozen split, prompt cache or generation code changed")
        inputs[split] = {"count": size, "generation": controls, "prepare_sha256": digest(prompts.with_name("prepare.json"))}
    timings, disk = {k: [] for k in STAGES}, {k: [] for k in STAGES}
    invariant = None
    for tag in sorted(required - {"base"}):
        d = primary / "training" / tag
        m, rt = read(d / "run.json"), read(d / "runtime.json")
        ident = train_protocol(m, rt)
        if (ident["model_config_sha256"] != cfg_sha or m["stage"] != "target"
                or rt["token_goal"] != 3005228 or rt["starting_model_step"] != 0
                or any(h != sources.get(k) for k, h in m["sources_sha256"].items())):
            raise ValueError(f"{tag}: primary target configuration changed")
        if invariant is not None and invariant != ident:
            raise ValueError("Primary training settings differ across controls")
        invariant = ident
        for stage in STAGES:
            task = "train-" + tag if stage == "train" else f"eval-{stage}-{locked[tag]}"
            location = d if stage == "train" else primary / "evaluation" / stage / locked[tag]
            timings[stage].append(timing(primary / "receipts" / (task + ".json")))
            disk[stage].append(footprint(location))
    return {
        "scope": "Post hoc diagnostic on the already-opened 2,000-request internal test. No new BFCL evaluation.",
        "cell_order": [label(s, m) for s, m in ORDER], "selections": selections,
        "batches": {k: [label(s, m) for s, m in cells] for k, cells in BATCHES.items()},
        "selection_manifest_sha256": manifest_sha, "primary_design_sha256": digest(primary / "design.json"),
        "primary_lock_sha256": digest(primary / "locked-models.json"), "pool_sha256": digest(pool),
        "models_sha256": {"target_weights": model_sha, "target_config": cfg_sha},
        "source_sha256": sources, "training": invariant, "splits": inputs,
        "endpoint": "First optimizer update crossing the common non-padding token goal; lock before development or test.",
        "analysis": {
            "contrasts": ["Bprime-B", "C-Bprime", "E2-B", "E2-A", "E3-B", "E3-A", "E4-A"],
            "outcome": "Complete-call exact match on the original internal held-out questions.",
            "intervals": "2,000 joint paired near-duplicate group resamples, seed 31415, type-7 95% and 90%; three paired seeds additionally get df=2 t intervals. Report observed partial coverage and every missing cell.",
            "equivalence": "E2-B, E2-A and E3-A only: require three paired seeds and the unrounded conditional 90% interval strictly inside (-1,+1) percentage points.",
            "diagnostics": "Length matching: E2 equivalent to B, mean E2-A <= -1 pp, and all three negative. Residualization: E3-B >= 1 pp and E3 equivalent to A. Tool stratification: mean E4-A >= 1 pp triggers follow-up.",
            "interpretation": "Post-hoc sensitivity on reused test data; unadjusted contrasts. Thresholds are exploratory decision rules. No score-based stopping.",
        },
        "admission": {"hard_cutoff_utc": utc(hard_cutoff), "teardown_reserve_seconds": TEARDOWN_SECONDS,
                      "time_multiplier": 1.2, "disk_multiplier": 1.2,
                      "gpu": invariant["runtime"]["gpu"],
                      "minimum_free_vram_fraction": inputs["test"]["generation"]["generation"]["gpu_memory_utilization"],
                      "historical_max_seconds": {k: max(v) for k, v in timings.items()},
                      "historical_max_bytes": {k: max(v) for k, v in disk.items()},
                      "rule": "Admit only an entire batch (5, 3, 3 cells), accounting for all unfinished stages, the largest observed stage times and 20% headroom. Defer an unaffordable batch and all subsequent batches in full. Once admitted, continue to the fixed endpoints subject to the hard cutoff; scores never affect this rule."},
    }


def preregister(primary, runner, cutoff, create):
    path = runner.root / "preregistration.json"
    if not path.exists() and not create:
        raise ValueError("Run freeze before starting Phase 1")
    spec = build_spec(primary, runner, cutoff)
    if path.exists():
        if read(path)["spec"] != spec:
            raise ValueError("Frozen Phase 1 plan, inputs or code changed")
    else:
        if any((runner.root / "training").glob("*")) or any((runner.root / "evaluation").glob("*")):
            raise ValueError("Existing training or evaluation artifacts predate registration")
        write(path, {"created_utc": utc(time.time()), "spec": spec})
    return spec


def dependency_receipts(primary, full, cross):
    records = {}
    for root, task, artifacts in (
        (primary, "summary", ["summary/internal.json", "summary/aggregate-audit.zip"]),
        (full, "full-summary", ["summary/full-reference.json"]),
        (cross, "summary", ["summary.json", "summary.md"]),
    ):
        if not (root / "receipts" / (task + ".json")).is_file():
            return None
        check_outputs(root, task, [root / f for f in artifacts])
        records[str(root)] = digest(root / "receipts" / (task + ".json"))
    return records


def budget(runner, spec, name):
    rule = spec["admission"]
    by_stage = {}
    needed_disk = 0
    for stage in STAGES:
        if (runner.root / "receipts" / f"{stage}-{name}.json").exists():
            continue
        history = [timing(p) for p in (runner.root / "receipts").glob(f"{stage}-s*.json")]
        by_stage[stage] = max([rule["historical_max_seconds"][stage], *history])
        needed_disk += rule["historical_max_bytes"][stage]
    estimate = sum(by_stage.values()) * rule["time_multiplier"]
    available = runner.end - time.time()
    return {"label": name, "utc": utc(time.time()), "missing_stage_seconds": by_stage,
            "required_seconds": estimate, "available_seconds": available,
            "fits_time": available >= estimate, "free_bytes": shutil.disk_usage(runner.root).free,
            "required_bytes": math.ceil(needed_disk * rule["disk_multiplier"])}


def batch_budget(runner, spec, batch):
    costs = [budget(runner, spec, label(seed, method)) for seed, method in BATCHES[batch]]
    seconds = sum(c["required_seconds"] for c in costs)
    available = runner.end - time.time()
    storage = sum(c["required_bytes"] for c in costs)
    free = shutil.disk_usage(runner.root).free
    return {"batch": batch, "utc": utc(time.time()), "cells": costs, "required_seconds": seconds,
            "available_seconds": available, "fits_time": available >= seconds,
            "required_bytes": storage, "free_bytes": free, "fits_storage": free >= storage}


def check_gpu(rule):
    """Read local scheduler-visible resources only after the previous experiments are complete."""
    query = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, check=True).stdout.strip().splitlines()
    jobs = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, check=True).stdout.strip()
    if len(query) != 1 or jobs:
        raise RuntimeError("A single idle GPU is required; another process may be using the allocation")
    gpu, total, free = [v.strip() for v in query[0].split(",")]
    if gpu != rule["gpu"] or float(free) < float(total) * rule["minimum_free_vram_fraction"]:
        raise RuntimeError(f"GPU differs from frozen capacity requirements: {query[0]}")
    return {"gpu": gpu, "total_mib": int(total), "free_mib": int(free)}


def protect_subprocesses(runner):
    """GNU timeout stays with the active child if its Python controller is interrupted."""
    timeout = shutil.which("timeout")
    if not timeout:
        raise RuntimeError("The GPU host requires GNU coreutils timeout for independent wall-time enforcement")
    version = subprocess.run([timeout, "--version"], capture_output=True, text=True, check=True)
    if "GNU coreutils" not in version.stdout:
        raise RuntimeError("Expected GNU coreutils timeout")
    for role in ("training_python", "eval_python"):
        exe = getattr(runner.a, role)
        if not exe.is_file() or not os.access(exe, os.X_OK):
            raise ValueError(f"Missing executable: {exe}")
        # The original venv must be first on PATH for the native ms-swift CLI.
        wrapper = runner.root / "runtime" / role / "python"
        content = (
            "#!/bin/sh\n"
            + "remaining=$(( " + str(math.floor(runner.end)) + " - $(date +%s) ))\n"
            + "test \"$remaining\" -gt 0 || exit 124\n"
            + "PATH=" + shlex.quote(str(exe.parent)) + ":\"$PATH\"; export PATH\n"
            + "exec " + shlex.quote(timeout) + " --kill-after=5s \"$remaining\" "
            + shlex.quote(str(exe)) + " \"$@\"\n"
        )
        if wrapper.exists():
            if wrapper.read_text() != content:
                raise ValueError(f"GPU subprocess launch settings changed: {wrapper}")
        else:
            wrapper.parent.mkdir(parents=True, exist_ok=True)
            wrapper.write_text(content)
            wrapper.chmod(0o700)
        setattr(runner.a, role, wrapper)


def run_cell(primary, runner, spec, seed, method):
    tag = label(seed, method)
    data = runner.root / "selection" / str(seed) / (method + ".jsonl")
    if digest(data) != spec["selections"][tag]["sha256"]:
        raise ValueError(f"{tag}: registered data changed")
    train = runner.root / "training" / tag
    runner.train("train-" + tag, "target", runner.big, data, train, seed, ["--pool", primary / "frozen/train.jsonl"])
    m, rt = read(train / "run.json"), read(train / "runtime.json")
    if (train_protocol(m, rt) != spec["training"] or m["seed"] != seed or m["stage"] != "target"
            or m["pool_sha256"] != spec["pool_sha256"] or m["data_sha256"] != spec["selections"][tag]["sha256"]):
        raise ValueError(f"{tag}: target configuration differs")
    bound = token_boundary(train, spec["training"]["runtime"]["token_goal"])
    adapter = train / f"checkpoint-{bound['step']}"
    key = tag + f"-step{bound['step']}"
    endpoint = {"label": tag, "seed": seed, "method": method, "checkpoint": key, **bound}
    immutable(runner.root / "locks" / (tag + ".json"), endpoint)
    scores, receipts = {}, {}
    for split, count in SPLITS.items():
        data, prompts = files_for_split(primary, runner, split)
        out = runner.root / "evaluation" / split / key
        ref = spec["splits"][split]
        if (digest(data) != ref["generation"]["source_sha256"] or digest(prompts) != ref["generation"]["prompts_sha256"]
                or digest(prompts.with_name("prepare.json")) != ref["prepare_sha256"]):
            raise ValueError(f"{tag}: {split} input cache changed")
        cmd = ["--model", runner.big, "--data", data, "--max-model-len", "4096", "--max-tokens", "512",
               "--prepared", prompts, "--out", out, "--adapter", adapter, "--enforce-eager"]
        task = f"{split}-{tag}"
        runner.task(task, runner.a.eval_python, "evaluate_experiment", cmd, [out / "metrics.json", out / "predictions.jsonl", out / "config.json"])
        c, metrics = read(out / "config.json"), read(out / "metrics.json")
        if (gen_protocol(c) != ref["generation"] or metrics["count"] != count
                or c["adapter_sha256"]["adapter_model.safetensors"] != bound["final_adapter_sha256"]):
            raise ValueError(f"{tag}: {split} generation configuration or adapter differs")
        scores[split] = {"checkpoint": key, "count": count, "correct": metrics["correct"]}
        receipts[split] = digest(runner.root / "receipts" / (task + ".json"))
    receipts["train"] = digest(runner.root / "receipts" / f"train-{tag}.json")
    cell = {"label": tag, "seed": seed, "method": method, "endpoint": endpoint, "scores": scores,
            "task_seconds": {stage: runner.cost(f"{stage}-{tag}") for stage in STAGES}, "receipt_sha256": receipts}
    immutable(runner.root / "cells" / (tag + ".json"), cell)
    print(tag, "verified", flush=True)
    return cell


def progress(runner, spec, state, reason, completed):
    wanted = spec["cell_order"]
    record = {"status": state, "reason": reason, "planned": wanted, "completed": list(completed),
              "missing": [n for n in wanted if n not in completed], "cutoff_utc": spec["admission"]["hard_cutoff_utc"]}
    write(runner.root / "execution.json", record)
    return record


def completed_cells(runner, spec):
    labels = {}
    for tag in spec["cell_order"]:
        path = runner.root / "cells" / (tag + ".json")
        if not path.is_file():
            continue
        cell = read(path)
        ep = read(runner.root / "locks" / (tag + ".json"))
        if ep != cell["endpoint"] or ep["label"] != tag:
            raise ValueError(f"{tag}: completed cell endpoint differs from pre-generation lock")
        train = runner.root / "training" / tag
        ck = train / f"checkpoint-{ep['step']}"
        if digest(ck / "adapter_model.safetensors") != ep["final_adapter_sha256"]:
            raise ValueError(f"{tag}: locked adapter changed")
        for stage in STAGES:
            task = f"{stage}-{tag}"
            if digest(runner.root / "receipts" / (task + ".json")) != cell["receipt_sha256"][stage]:
                raise ValueError(f"{tag}: completed {stage} receipt changed")
            if stage == "train":
                files = [train / f for f in ("completed.json", "args.json", "exit.json")]
            else:
                d = runner.root / "evaluation" / stage / ep["checkpoint"]
                files = [d / f for f in ("config.json", "metrics.json", "predictions.jsonl")]
            check_outputs(runner.root, task, files)
        labels[tag] = ep["checkpoint"]
    return labels


def execute(primary, full, cross, runner, spec):
    roots = (primary, full, cross)
    if dependency_receipts(*roots) is None:
        raise RuntimeError("Waiting for the primary, full-pool and crossed-seed terminal summary receipts")
    with ExitStack() as locks:
        for root in roots:
            f = locks.enter_context((root / ".lock").open("a"))
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checksums = dependency_receipts(*roots)
        if checksums is None:
            raise RuntimeError("An upstream terminal receipt disappeared")
        immutable(runner.root / "dependencies.json", checksums)
        completed = completed_cells(runner, spec)
        state, reason = "complete", "All eleven endpoints verified."
        guarded = False
        try:
            for batch, cells in BATCHES.items():
                admit = batch_budget(runner, spec, batch)
                has_work = any(c["missing_stage_seconds"] for c in admit["cells"])
                if has_work:
                    write(runner.root / "decisions" / f"{time.time_ns()}-{batch}.json", admit)
                    if not admit["fits_time"]:
                        state, reason = "partial_budget", f"Deferred remaining work from batch {batch}; insufficient time for the entire batch."
                        break
                    if not admit["fits_storage"]:
                        state, reason = "blocked_storage", f"Insufficient space for the entire {batch} batch."
                        break
                for seed, method in cells:
                    tag = label(seed, method)
                    if tag in completed:
                        continue
                    if not guarded:
                        gpu = check_gpu(spec["admission"])
                        write(runner.root / "decisions" / f"{time.time_ns()}-gpu.json", gpu)
                        protect_subprocesses(runner)
                        guarded = True
                    cell = run_cell(primary, runner, spec, seed, method)
                    completed[tag] = cell["endpoint"]["checkpoint"]
                    progress(runner, spec, "running", f"Processing admitted batch {batch}.", completed)
        except (subprocess.TimeoutExpired, TimeoutError, RuntimeError) as error:
            timeout = (isinstance(error, (subprocess.TimeoutExpired, TimeoutError)) or
                       ("exit 124" in str(error) or "exit 137" in str(error)))
            if not timeout or time.time() < runner.end - 2:
                progress(runner, spec, "failed", str(error), completed)
                raise
            state, reason = "partial_budget", "Reached the hard process cutoff while finishing an admitted batch."
        except KeyboardInterrupt:
            progress(runner, spec, "interrupted", "Controller interrupted; inspect receipts before resuming.", completed)
            raise
        record = progress(runner, spec, state, reason, completed)
        if state in ("complete", "partial_budget"):
            immutable(runner.root / "locked-models.json", completed)
        print(json.dumps(record), flush=True)
        if state == "blocked_storage":
            raise RuntimeError(reason)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("freeze", "status", "run"))
    p.add_argument("--primary", type=Path, default=Path("local/course-run"))
    p.add_argument("--full", type=Path, default=Path("local/full-reference"))
    p.add_argument("--cross", type=Path, default=Path("local/crossed-seeds"))
    p.add_argument("--out", type=Path, default=Path("local/followup/phase1"))
    p.add_argument("--train-python", type=Path, default=Path("/opt/trainenv/bin/python"))
    p.add_argument("--eval-python", type=Path, default=Path("/opt/evalenv/bin/python"))
    a = p.parse_args()
    primary, full, cross, dest = [v.resolve() for v in (a.primary, a.full, a.cross, a.out)]
    if len({primary, full, cross, dest}) != 4 or any(x in dest.parents or dest in x.parents for x in (primary, full, cross)):
        p.error("Use a separate output tree for the follow-up")
    design = read(primary / "design.json")
    conf = args_parser().parse_args([])
    conf.course, conf.training_python, conf.eval_python = dest, a.train_python.absolute(), a.eval_python.absolute()
    conf.deadline, conf.micro_batch = design["deadline"], design["micro_batch"]
    conf.reserve_seconds = max(600, design["reserve_seconds"])
    runner = Protocol(conf)
    cutoff = runner.end
    runner.end -= TEARDOWN_SECONDS
    if a.action == "status":
        reg = read(dest / "preregistration.json")["spec"]
        completed = list(completed_cells(runner, reg))
        next_name = next((n for n in reg["cell_order"] if n not in completed), None)
        print(json.dumps({"completed": completed, "next": next_name, "hard_cutoff_utc": utc(cutoff),
                          "upstreams_complete": dependency_receipts(primary, full, cross) is not None,
                          "budget_for_next_batch": next((batch_budget(runner, reg, b) for b, cells in BATCHES.items()
                                                         if next_name in [label(s, m) for s, m in cells]), None)}, indent=2))
        return
    def stop_controller(sig, frame):
        raise KeyboardInterrupt(f"signal {sig}")

    signal.signal(signal.SIGTERM, stop_controller)
    with (dest / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = preregister(primary, runner, cutoff, create=a.action == "freeze")
        if a.action == "freeze":
            print("Registered", len(spec["cell_order"]), "controls. No training or generation started.")
        else:
            execute(primary, full, cross, runner, spec)


if __name__ == "__main__":
    main()
