"""Freeze or resume the 20 registered Round 2 fits and their development and legacy-test evaluations.

    python scripts/run_round2.py freeze
    python scripts/run_round2.py run --hours 12
    python scripts/run_round2.py status

All paths in the portable protocol lock refer to the original experiment by digest.
The first token-budget crossing locks each adapter before generation. No other splits
or externally supplied evaluation inputs are accepted by this controller.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import time

from run_course_experiment import Protocol, args_parser, digest, read, rows, write
from run_crossed_seeds import WEIGHTS
from run_followup import (
    SPLITS, STAGES, budget, check_gpu, completed_cells, files_for_split,
    immutable, label, run_cell, utc,
)


REPO = Path(__file__).resolve().parents[1]
LOCAL = REPO / "local"
PUBLIC = REPO / "results/followup"
CELLS = {
    label(seed, method): (seed, method)
    for seeds, methods in [
        ((45, 46), ("A", "B", "C", "E2", "E2b")),
        (range(42, 47), ("L-short", "L-long")),
    ]
    for seed in seeds for method in methods
}


def verified_file(path, expected):
    if not path.is_file() or digest(path) != expected:
        raise ValueError(f"Missing or changed registered file: {path}")


def predecessor(local_name):
    native = LOCAL / local_name
    return native if native.is_dir() else LOCAL / "runpod/repo/local" / local_name


def verify_selection(root, manifest, pool):
    by_id = {row["id"]: row for row in rows(pool)}
    if len(by_id) != 4000 or set(manifest["cells"]) != set(CELLS):
        raise ValueError("Expected the complete 4,000-candidate pool and 20 selected subsets")
    selections = {}
    for tag, (seed, method) in CELLS.items():
        cell = manifest["cells"][tag]
        relative = f"{seed}/{method}.jsonl"
        if (cell["seed"], cell["method"], cell["size"], cell["path"]) != (seed, method, 1000, relative):
            raise ValueError(f"{tag}: incorrect selection metadata")
        path = root / relative
        verified_file(path, cell["sha256"])
        selected = rows(path)
        if (len(selected) != 1000 or len({r["id"] for r in selected}) != 1000
                or any(r != by_id.get(r["id"]) for r in selected)):
            raise ValueError(f"{tag}: selected examples differ from training pool")
        selections[tag] = {"sha256": cell["sha256"], "size": len(selected)}
    return selections


def build_spec(runner, primary, phase1, e2b):
    registration = PUBLIC / "round2-registration.json"
    rules = read(registration)
    if (len(rules["cell_order"]) != 20 or set(rules["cell_order"]) != set(CELLS)
            or rules["target_input_token_budget"] != 3005228):
        raise ValueError("The public registration does not describe the core 20 fits")
    old = phase1 / "preregistration.json"
    previous = e2b / "preregistration.json"
    verified_file(old, read(PUBLIC / "phase1-registration.json")["private_registration_sha256"])
    verified_file(previous, read(PUBLIC / "e2b-registration.json")["registration_sha256"])
    ref = read(old)["spec"]
    e2b_spec = read(previous)["spec"]
    if (e2b_spec["phase1_preregistration_sha256"] != digest(old)
            or any(ref[k] != e2b_spec[k] for k in ("training", "splits", "pool_sha256", "models_sha256"))
            or ref["training"]["runtime"]["token_goal"] != 3005228):
        raise ValueError("Inconsistent historical registrations")
    for name, source_hash in ref["source_sha256"].items():
        verified_file(runner.scripts / name, source_hash)
    for file, key in (("design.json", "primary_design_sha256"), ("locked-models.json", "primary_lock_sha256")):
        verified_file(primary / file, ref[key])
    pool = primary / "frozen/train.jsonl"
    verified_file(pool, ref["pool_sha256"])
    inputs = runner.root / "selection"
    manifest_path = inputs / "manifest.json"
    verified_file(manifest_path, rules["selection"]["manifest_sha256"])
    manifest = read(manifest_path)
    if manifest["target_input_token_budget"] != 3005228 or manifest["train_sha256"] != ref["pool_sha256"]:
        raise ValueError("Selection refers to a different training pool or token budget")
    verified_file(primary / "scores/score-config.json", manifest["score_config_sha256"])
    if set(manifest["score_receipts"]) != {"0", "10", "30", "80", "250"}:
        raise ValueError("Changed proxy scoring steps")
    for step, receipt in manifest["score_receipts"].items():
        if receipt != read(primary / "scores" / f"step-{step}.complete.json") or receipt["rows"] != 4000:
            raise ValueError(f"Changed proxy scoring receipt: step {step}")
        verified_file(primary / "scores" / f"step-{step}.jsonl", receipt["sha256"])
    selections = verify_selection(inputs, manifest, pool)
    if (set(rules["selection"]["cells"]) != set(CELLS)
            or any(rules["selection"]["cells"][tag]["sha256"] != selections[tag]["sha256"] for tag in CELLS)):
        raise ValueError("Publicly registered subset checksums differ")
    verified_file(runner.big / WEIGHTS, ref["models_sha256"]["target_weights"])
    verified_file(runner.big / "config.json", ref["models_sha256"]["target_config"])
    for split in SPLITS:
        expected = ref["splits"][split]
        generation = expected["generation"]
        data, cache = files_for_split(primary, runner, split)
        verified_file(data, generation["source_sha256"])
        verified_file(cache, generation["prompts_sha256"])
        verified_file(cache.with_name("prepare.json"), expected["prepare_sha256"])
        verified_file(runner.big / "tokenizer.json", generation["tokenizer_sha256"])
        verified_file(runner.big / "chat_template.jinja", generation["template_sha256"])
    sources = {**ref["source_sha256"], Path(__file__).name: digest(Path(__file__))}
    for name, expected in manifest["source_sha256"].items():
        candidate = runner.scripts / name
        verified_file(candidate, expected)
        sources[name] = expected
    admission = ref["admission"]
    return {
        "cell_order": rules["cell_order"], "public_registration_sha256": digest(registration),
        "phase1_preregistration_sha256": digest(old), "e2b_preregistration_sha256": digest(previous),
        "selection_manifest_sha256": digest(manifest_path), "source_sha256": sources,
        "pool_sha256": ref["pool_sha256"], "models_sha256": ref["models_sha256"],
        "training": ref["training"], "splits": ref["splits"], "selections": selections,
        "endpoint": ref["endpoint"],
        "admission": {k: admission[k] for k in (
            "historical_max_bytes", "historical_max_seconds", "minimum_free_vram_fraction",
            "time_multiplier", "disk_multiplier", "gpu",
        )},
    }


def freeze(runner, primary, phase1, e2b, create):
    dest = runner.root / "protocol-lock.json"
    if not create and not dest.is_file():
        raise ValueError("Run the CPU freeze before starting the queue")
    spec = build_spec(runner, primary, phase1, e2b)
    if dest.exists():
        if read(dest)["spec"] != spec:
            raise ValueError("Inputs, code or public registration changed after protocol freeze")
    else:
        if any((runner.root / p).exists() for p in ("training", "evaluation", "cells", "attempts", "receipts", "locks", "execution.json")):
            raise ValueError("Training or evaluation material predates the protocol freeze")
        write(dest, {"created_utc": utc(time.time()), "spec": spec})
    return spec


ENVIRONMENT_PROBE = """
import importlib.metadata as metadata
import json
import platform
import sys
import torch

names = sys.argv[1:]
active = {name: metadata.version(name) for name in names}
# The historical train_experiment.py also logs this dictionary. A system-site
# distribution encountered last can shadow the active venv's version here.
reported = {dist.metadata["Name"]: dist.version for dist in metadata.distributions()
            if dist.metadata["Name"] in {"torch", "transformers", "ms-swift", "ms_swift", "peft"}}
print(json.dumps({"python": sys.version, "platform": platform.platform(), "active": active,
                  "reported": reported, "torch_import": str(torch.__version__), "cuda": torch.version.cuda}))
"""


def host_protocol(runner, original):
    """Validate imported wheels and retain their host-specific reporting metadata."""
    spec = deepcopy(original)
    cmd = spec["training"]["command"]
    for flag, path in (("--model", runner.big), ("--external_plugins", runner.scripts / "experiment_callback.py")):
        cmd[cmd.index(flag) + 1] = str(path.resolve())
    spec["environment_observation"] = {}
    for role, python in (("training", runner.a.training_python), ("evaluation", runner.a.eval_python)):
        if role == "training":
            expected = dict(spec["training"]["packages"])
            # The frozen callback reported torch.version.cuda == 12.8; a second, system-level
            # torch+cu129 distribution was enumerated last in the old raw package dictionary.
            expected["torch"] = expected["torch"].split("+")[0]
        else:
            versions = spec["splits"]["dev"]["generation"]["versions"]
            expected = {k: v for k, v in versions.items() if k not in ("python", "platform")}
        observed = json.loads(subprocess.check_output([str(python), "-c", ENVIRONMENT_PROBE, *expected], text=True))
        if observed["torch_import"].split("+")[0] != expected["torch"].split("+")[0]:
            raise ValueError(f"{role}: imported torch version differs from the historical runtime")
        for package, version in expected.items():
            actual = observed["active"][package]
            if package == "torch" and role == "training":
                actual = actual.split("+")[0]
            if actual != version:
                raise ValueError(f"{role}: expected {package}=={version} in {python}, found {actual}")
        if role == "training":
            if observed["cuda"] != spec["training"]["runtime"]["cuda"]:
                raise ValueError("Active training PyTorch uses a different historical CUDA build")
            spec["training"]["packages"] = observed["reported"]
        else:
            if observed["python"].split()[0] != versions["python"].split()[0]:
                raise ValueError("Evaluation Python version differs from the frozen implementation")
            for split in SPLITS:
                record = spec["splits"][split]["generation"]["versions"]
                record.update(python=observed["python"], platform=observed["platform"])
        spec["environment_observation"][role] = observed
    return spec


def protect_workers(runner):
    """Use a stable argv across rentals, but enforce this invocation's absolute cutoff."""
    commands = {name: shutil.which(name) for name in ("timeout", "flock")}
    if not all(commands.values()):
        raise RuntimeError("GPU host requires GNU timeout and util-linux flock")
    if "GNU coreutils" not in subprocess.check_output([commands["timeout"], "--version"], text=True):
        raise RuntimeError("GPU host requires GNU timeout")
    runtime = runner.root / "runtime"
    runtime.mkdir(exist_ok=True)
    cutoff = runtime / "cutoff-unix"
    worker_lock = runtime / "worker.lock"
    with worker_lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        tmp = cutoff.with_suffix(".tmp")
        tmp.write_text(f"{math.floor(runner.end)}\n")
        tmp.replace(cutoff)
    for attr in ("training_python", "eval_python"):
        python = getattr(runner.a, attr)
        wrapper = runtime / attr / "python"
        script = (
            "#!/bin/sh\n"
            f"read -r end < {shlex.quote(str(cutoff))} || exit 124\n"
            'remaining=$((end - $(date +%s)))\n'
            'test "$remaining" -gt 0 || exit 124\n'
            f"PATH={shlex.quote(str(python.parent))}:\"$PATH\"; export PATH\n"
            f"exec {shlex.quote(commands['flock'])} --nonblock --no-fork {shlex.quote(str(worker_lock))} "
            f"{shlex.quote(commands['timeout'])} --kill-after=5s \"$remaining\" "
            f"{shlex.quote(str(python))} \"$@\"\n"
        )
        if wrapper.exists() and wrapper.read_text() != script:
            raise ValueError(f"Interpreter changed since first execution: {wrapper}")
        wrapper.parent.mkdir(exist_ok=True)
        wrapper.write_text(script)
        wrapper.chmod(0o700)
        setattr(runner.a, attr, wrapper)


def report(root, order, completed, state, reason):
    obj = {"status": state, "planned": order, "completed": [tag for tag in order if tag in completed],
           "missing": [tag for tag in order if tag not in completed], "reason": reason, "updated_utc": utc(time.time())}
    write(root / "execution.json", obj)
    return obj


def recover_orphans(runner):
    """Retry receiptless attempts only after the worker lock and OS process table are clear."""
    for path in sorted((runner.root / "attempts").glob("*.json")):
        intent = read(path)
        if intent["state"] != "started" or (runner.root / "receipts" / path.name).exists():
            continue
        command = intent["argv"]
        task, tag = path.stem.split("-", 1)
        if tag not in CELLS or task not in STAGES:
            raise ValueError(f"Unexpected attempt {path.stem}")
        directory = command[command.index("--output" if task == "train" else "--out") + 1]
        if runner.root not in Path(directory).parents:
            raise ValueError(f"Attempt points outside the run: {path}")
        processes = subprocess.check_output(["ps", "-eo", "args="], text=True).splitlines()
        if any(directory in process for process in processes):
            raise RuntimeError(f"A worker for {path.stem} may still be active; resume after it exits")
        # Protocol.task preserves the partial output tree before starting a fresh attempt.
        write(path, {**intent, "state": "failed"})
        runner.ledger("orphaned_attempt_verified_stopped", path.stem, output=directory)


def execute(runner, primary, spec, limit):
    order = spec["cell_order"]
    completed = completed_cells(runner, spec)
    state, note = "complete", "All registered fits and both legacy evaluations verified."
    launch = None
    count = 0
    try:
        for tag in order:
            if tag in completed:
                continue
            if limit is not None and count >= limit:
                state, note = "partial_batch", "Reached the requested invocation cell limit."
                break
            estimate = budget(runner, spec, tag)
            if not estimate["fits_time"] or estimate["free_bytes"] < estimate["required_bytes"]:
                state = "partial_time" if not estimate["fits_time"] else "partial_storage"
                note = f"Next cell {tag} requires {estimate['required_seconds']:.0f}s and {estimate['required_bytes']} bytes."
                break
            if launch is None:
                gpu = check_gpu(spec["admission"])
                launch = host_protocol(runner, spec)
                protect_workers(runner)
                recover_orphans(runner)
                write(runner.root / "sessions" / f"{time.time_ns()}.json",
                      {"gpu": gpu, "cutoff_utc": utc(runner.end), "limit": limit,
                       "protocol_lock_sha256": digest(runner.root / "protocol-lock.json"),
                       "observed_eval_versions": launch["splits"]["dev"]["generation"]["versions"],
                       "observed_training_command": launch["training"]["command"],
                       "environments": launch["environment_observation"]})
            report(runner.root, order, completed, "running", f"Executing {tag}.")
            cell = run_cell(primary, runner, launch, *CELLS[tag])
            completed[tag] = cell["endpoint"]["checkpoint"]
            count += 1
    except BaseException as exc:
        state = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report(runner.root, order, completed, state, str(exc))
        raise
    if state == "complete":
        completed = completed_cells(runner, spec)
        if set(completed) != set(order):
            raise ValueError("Rechecking all completed artifacts found missing endpoints")
        immutable(runner.root / "locked-models.json", completed)
    return report(runner.root, order, completed, state, note)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=("freeze", "run", "status"))
    p.add_argument("--out", type=Path, default=LOCAL / "followup/round2")
    p.add_argument("--primary", type=Path, default=predecessor("course-run"))
    p.add_argument("--phase1", type=Path, default=predecessor("followup/phase1"))
    p.add_argument("--e2b", type=Path, default=predecessor("followup/difficulty-control"))
    p.add_argument("--train-python", type=Path, default=Path("/opt/trainenv/bin/python"))
    p.add_argument("--eval-python", type=Path, default=Path("/opt/evalenv/bin/python"))
    cut = p.add_mutually_exclusive_group()
    cut.add_argument("--deadline", help="Absolute hard compute cutoff; include UTC offset or Z")
    cut.add_argument("--hours", type=float, help="Hard duration for this invocation, from now")
    p.add_argument("--max-cells", type=int, help="Optional maximum additional complete cells in this invocation")
    a = p.parse_args()
    end = time.time()
    if a.action == "run" and a.hours is None and a.deadline is None:
        p.error("run requires --hours or --deadline")
    if a.hours is not None:
        if not math.isfinite(a.hours) or a.hours <= 0:
            p.error("--hours must be positive and finite")
        end += 3600 * a.hours
    if a.deadline is not None:
        stamp = datetime.fromisoformat(a.deadline.replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            p.error("--deadline must include a timezone")
        end = stamp.timestamp()
    if a.max_cells is not None and a.max_cells < 1:
        p.error("--max-cells must be positive")
    dest = a.out.resolve()
    roots = [r.resolve() for r in (a.primary, a.phase1, a.e2b)]
    if any(dest == r or dest in r.parents or r in dest.parents for r in roots):
        p.error("Round 2 must use a separate output tree")
    conf = args_parser().parse_args([])
    conf.course, conf.micro_batch, conf.reserve_seconds = dest, 4, 0
    conf.deadline = datetime.fromtimestamp(end, timezone.utc).isoformat()
    conf.training_python, conf.eval_python = a.train_python.absolute(), a.eval_python.absolute()
    runner = Protocol(conf)
    if a.action == "status":
        spec = read(dest / "protocol-lock.json")["spec"]
        complete = completed_cells(runner, spec)
        print(json.dumps({"completed": list(complete), "planned": spec["cell_order"],
                          "execution": read(dest / "execution.json") if (dest / "execution.json").exists() else None}, indent=2))
        return
    def stop(sig, _frame):
        raise KeyboardInterrupt(f"signal {sig}")
    signal.signal(signal.SIGTERM, stop)
    with (dest / ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = freeze(runner, *roots, create=a.action == "freeze")
        if a.action == "freeze":
            print(f"Registered {len(spec['cell_order'])} fits, SHA-256 {digest(dest / 'protocol-lock.json')}")
        else:
            print(json.dumps(execute(runner, roots[0], spec, a.max_cells), indent=2))


if __name__ == "__main__":
    main()
