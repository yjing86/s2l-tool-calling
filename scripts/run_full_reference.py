"""Optional single-seed full-pool reference. No change to the primary course run.

Run `freeze` immediately; it reads hashes and protocol, does not read scores and
does not acquire a GPU. Use `fit` when the primary job is not running (prefer
after its outputs are backed up), then `test` after primary blind scoring ends.
Each command is run from the same repository root as the main experiment.
"""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import time

from run_course_experiment import Protocol, args_parser, digest, read, rows, write
from verify_tool_call_loss import swift_row


def setarg(cmd, flag, value):
    cmd[cmd.index(flag) + 1] = str(value)


def freeze(course, dest):
    specpath = dest / "frozen-reference.json"
    design = read(course / "design.json")
    if (design["pool"], design["subset"]) != (4000, 1000):
        raise ValueError("The 4,000-row reference is defined only for the 4,000/1,000 primary trial")
    specs = {
        "seed": 42,
        "pool_sha256": digest(course / "frozen/train.jsonl"),
        "primary_design_sha256": digest(course / "design.json"),
        "test_sha256": digest(course / "frozen/test.jsonl"),
        "criterion": "Full pool; 2x total non-padding tokens; actual first primary token crossing; dev every 50 + crossing + final. Best dev among base and saved checkpoints; ties earliest. No early stopping.",
        "endpoints": ["matched_tokens", "final", "best_dev"],
        "comparison": "Against locked seed42 A/B/C and raw 2B on the same internal test. BFCL excluded from this secondary reference.",
        "budget": "Primary completion takes priority. Use measured compute+eval cost, 30% padding, 10m minimum hard billing reserve; skip if insufficient.",
        "source_hashes": {
            p.name: digest(p) for p in [Path(__file__), Path(__file__).with_name("full_reference_hook.py")]
        },
    }
    if specpath.exists():
        saved = read(specpath)
        for k in specs:
            if saved[k] != specs[k]:
                raise ValueError(f"Frozen reference changed: {k}")
        return saved
    specs["freeze_unix"] = time.time()
    specs["primary_test_started_at_freeze"] = any((course / "evaluation/test").glob("*/predictions.jsonl"))
    specs["primary_external_started_at_freeze"] = any((course / "evaluation/bfcl").glob("*/predictions.jsonl"))
    write(specpath, specs)
    return specs


def train_native(primary, out, pool):
    """Run the known-good primary seed42 A native Swift command with a different input file."""
    ref = read(primary / "training/s42-A/run.json")
    if ref["stage"] != "target" or ref["seed"] != 42 or ref["pool_sha256"] != digest(pool):
        raise ValueError("Primary seed42 A manifest/pool mismatch")
    goal = read(primary / "training/s42-A/runtime.json")["token_goal"]
    items = rows(pool)
    if len(items) != 4000 or len({r["id"] for r in items}) != 4000:
        raise ValueError("Unexpected input pool")
    out.mkdir(parents=True, exist_ok=False)
    data = out / "train.swift.jsonl"
    with data.open("w") as f:
        for r in items:
            print(json.dumps(swift_row(r), ensure_ascii=False), file=f)
    cmd = list(ref["command"])
    hook = Path(__file__).with_name("full_reference_hook.py").resolve()
    for flag, val in (
        ("--dataset", data),
        ("--output_dir", out),
        ("--external_plugins", hook),
        ("--callbacks", "full_data_reference"),
    ):
        setarg(cmd, flag, val)
    meta = {
        **ref,
        "data_sha256": digest(pool),
        "pool": str(pool),
        "pool_sha256": digest(pool),
        "ids": [str(r["id"]) for r in items],
        "planned_steps_one_epoch": math.ceil(len(items) / ref["batch"]),
        "save_steps": [],
        "matched_token_goal": goal,
        "command": cmd,
        "start_unix": time.time(),
        "source_run_sha256": digest(primary / "training/s42-A/run.json"),
    }
    write(out / "run.json", meta)
    env = {**os.environ, "TOOL_TRAIN_MANIFEST": str(out / "run.json"), "PYTHONUNBUFFERED": "1"}
    started = time.monotonic()
    code = subprocess.call(cmd, env=env)
    write(out / "exit.json", {"code": code, "elapsed_seconds": time.monotonic() - started})
    if code or not (out / "completed.json").exists():
        raise SystemExit(code or 1)


def budget(primary, runner, design, action):
    """Only wall times, counts and file existence participate in the go/no-go decision."""
    if not (primary / "receipts/summary.json").exists() and not (primary / "primary-progress.json").exists():
        return {"proceed": False, "reason": "Primary training/dev or primary blind scoring still in progress."}
    estimate = design["estimate"]
    main_done = (primary / "receipts/summary.json").exists()
    train_s = max(read(primary / "receipts" / f"train-s42-{g}.json")["elapsed_s"] for g in "ABC")
    dev_s = max(
        [estimate["dev_eval_seconds"]]
        + [read(p)["elapsed_s"] for p in (primary / "receipts").glob("eval-dev-s42-*.json")]
    )
    test_s = max(
        [estimate["blind_seconds_per_model"]]
        + [read(p)["elapsed_s"] for p in (primary / "receipts").glob("eval-test-*.json")]
    )
    left = 0.0
    if not main_done:
        progress = read(primary / "primary-progress.json")
        routes = progress["schedules"]
        cap = design["equal_time_cap_seconds"] or 0
        cont = sum(max(0.0, cap - v["initial_cost"]) for g, v in routes.items() if not g.endswith("Bprime"))
        untested = len(progress["main"]) + (len(routes) if cap else 0)
        left = 1.3 * (cont + untested * estimate["blind_seconds_per_model"])
    latest = runner.end - time.time()
    commitments = runner.root / "receipts"
    frozen = runner.root / "full-models.json"
    distinct = len(set(read(frozen).values()) - {"base"}) if frozen.exists() else 3
    remaining_tests = max(0, distinct - len(list(commitments.glob("eval-test-full-*.json"))))
    blind = remaining_tests * test_s + 180
    left_dev = 12
    trained = (commitments / "full-train.json").is_file()
    if trained:
        steps = rows(runner.root / "training/steps.jsonl")
        left_dev = max(
            0, sum(bool(r["checkpoint_due"]) for r in steps) - len(list(commitments.glob("eval-dev-full-*.json")))
        )
    full = (0 if trained else 4.3 * train_s) + left_dev * dev_s if action == "fit" else 0
    required = 1.3 * (full + blind)
    return dict(
        proceed=(left + required < latest),
        primary_remaining_seconds=left,
        estimated_seconds=required,
        available_seconds=latest,
        assumed_full_training_seconds=4.3 * train_s,
        dev_per_checkpoint_seconds=dev_s,
        expected_dev_evals_upper_estimate=12,
        max_distinct_tests=remaining_tests,
        measured_primary_training_seconds=train_s,
        reason="Timing-only estimate, no primary result consulted",
    )


def entry():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["freeze", "fit", "test", "native"])
    p.add_argument("--primary", type=Path, default=Path("local/course-run"))
    p.add_argument("--out", type=Path, default=Path("local/full-reference"))
    p.add_argument("--training-python", type=Path, default=Path("/opt/trainenv/bin/python"))
    p.add_argument("--eval-python", type=Path, default=Path("/opt/evalenv/bin/python"))
    p.add_argument("--target-smoke", type=Path, default=Path("local/runs/target-smoke"))
    a = p.parse_args()
    prim = a.primary.resolve()
    dest = a.out.resolve()
    pool = prim / "frozen/train.jsonl"
    design = read(prim / "design.json")
    if a.action == "native":
        if os.environ.get("FULL_REFERENCE_AUTHORIZED_ROOT") != str(dest):
            raise RuntimeError("Use the budget-checked fit entry point")
        return train_native(prim, dest / "training", pool)
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / ".lock").open("a") as secondary:
        fcntl.flock(secondary, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.action == "freeze":
            print(json.dumps(freeze(prim, dest), indent=2))
            return
        if not (dest / "frozen-reference.json").exists():
            raise RuntimeError("Freeze the independent protocol first")
        freeze(prim, dest)
        with (prim / ".lock").open("a") as primary_lock:
            try:
                fcntl.flock(primary_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print("Skipped. Primary scheduler holds GPU slot.")
                return
            cfg = args_parser().parse_args([])
            cfg.course = dest
            cfg.training_python = a.training_python
            cfg.eval_python = a.eval_python
            cfg.target_smoke = a.target_smoke
            cfg.deadline = design["deadline"]
            cfg.micro_batch = design["micro_batch"]
            cfg.reserve_seconds = max(600, design["reserve_seconds"])
            runner = Protocol(cfg)
            if a.action == "test" and not (prim / "receipts/summary.json").exists():
                print("Skipped. Primary blind scoring takes priority.")
                return
            guard = budget(prim, runner, design, a.action)
            write(dest / "decisions" / f"{time.time_ns()}.json", guard)
            if not guard["proceed"]:
                print("Skipped.", json.dumps(guard))
                return
            # No optional subprocess runs into the budget reserved for primary blind testing.
            runner.end -= guard["primary_remaining_seconds"]
            if a.action == "fit":
                fit_reference(prim, dest, runner, guard)
            else:
                blind_reference(prim, dest, runner)


def fit_reference(prim, dest, runner, guard):
    out = dest / "training"
    pool = prim / "frozen/train.jsonl"
    os.environ["FULL_REFERENCE_AUTHORIZED_ROOT"] = str(dest)
    runner.task(
        "full-train",
        runner.a.training_python,
        "run_full_reference",
        ["native", "--primary", prim, "--out", dest],
        [out / "completed.json", out / "matched-step.json", out / "steps.jsonl", out / "exit.json"],
    )
    final_step = read(out / "completed.json")["final_step"]
    match = read(out / "matched-step.json")["step"]
    base_m = read(prim / "evaluation/dev/base/metrics.json")
    ckpts = [(0, "base", base_m["accuracy"])]
    costs = {"base": dict(input_tokens=0, answer_tokens=0, training_seconds=0, dev_seconds=0, selection_seconds=0)}
    devtime = 0
    rank = 0
    traces = rows(out / "steps.jsonl")
    train_total = runner.cost("full-train")
    start_overhead = max(0, train_total - traces[-1]["wall_seconds"])
    for row in traces:
        step = row["step"]
        if not row["checkpoint_due"]:
            continue
        name = f"full-s42-step{step}"
        folder = out / f"checkpoint-{step}"
        m = runner.evaluate("dev", name, runner.prep / "dev.jsonl", folder)
        rank += 1
        devtime += runner.cost("eval-dev-" + name)
        costs[name] = dict(
            input_tokens=row["input_tokens"],
            answer_tokens=row["answer_tokens"],
            training_seconds=min(train_total, start_overhead + row["wall_seconds"]),
            dev_seconds=devtime,
            selection_seconds=0,
        )
        ckpts.append((step, name, m["accuracy"]))
    final = f"full-s42-step{final_step}"
    matched = f"full-s42-step{match}"
    if final not in costs or matched not in costs:
        raise RuntimeError("Missing protocol checkpoint")
    locked = {"final": final, "matched_tokens": matched, "best_dev": max(ckpts, key=lambda v: v[2])[1]}
    write(
        dest / "full-costs.json",
        dict(
            checkpoints=costs,
            total_training_seconds=train_total,
            total_dev_seconds=devtime,
            dev_count=rank,
            curve=ckpts,
        ),
    )
    write(
        dest / "locked-adapters.json",
        {
            name: digest(out / f"checkpoint-{step}/adapter_model.safetensors")
            for step, name, acc in ckpts
            if step and name in locked.values()
        },
    )
    write(dest / "full-models.json", locked)
    print("Full reference locked:", locked)


def blind_reference(prim, dest, runner):
    locked = read(dest / "full-models.json")
    identities = read(dest / "locked-adapters.json")
    for name in set(locked.values()):
        if name == "base":
            continue
        step = int(name.split("step")[-1])
        adapter = dest / "training" / f"checkpoint-{step}"
        if digest(adapter / "adapter_model.safetensors") != identities[name]:
            raise ValueError("Changed reference adapter")
        runner.evaluate("test", name, prim / "frozen/test.jsonl", adapter)
    summary = dest / "summary"
    runner.task(
        "full-summary",
        runner.a.training_python,
        "summarize_full_reference",
        ["--primary", prim, "--secondary", dest, "--test", prim / "frozen/test.jsonl"],
        [summary / "full-reference.json"],
    )


if __name__ == "__main__":
    entry()
