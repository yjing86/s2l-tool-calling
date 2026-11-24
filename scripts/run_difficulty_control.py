"""Register or run the optional, all-three-seed E2b control after audited Phase 1 completion.

The forecast and freeze actions use only the CPU. Run requires independently checked
backups, an externally pinned attestation, and a fresh private provider balance record.
"""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import time
from itertools import combinations

from run_course_experiment import Protocol, args_parser, digest, read, rows, write
from run_crossed_seeds import WEIGHTS
from run_followup import (
    STAGES,
    TEARDOWN_SECONDS,
    check_gpu,
    check_outputs,
    completed_cells,
    footprint,
    immutable,
    label,
    run_cell,
    timing,
    utc,
)


SEEDS = (42, 43, 44)
ORDER = [label(s, "E2b") for s in SEEDS]
CURRENT_CUTOFF = "2026-11-24T20:29:00+00:00"
REGISTERED_LISTS = {
    "42": "8059a443e13d0582c88579df084417be5958d3bdb3e768b19f9fba9cdef1ee4e",
    "43": "bc8f7244b416266dba2b53fd13f82ab5a68ce3cee096a4a7ff239876cf92497a",
    "44": "fc17e38ac211d2caf28d9cd1f7a710fdf353d401318d5c4398bc3b81d90ab8e5",
}
NEW_SOURCES = (
    "select_difficulty_control.py",
    "run_difficulty_control.py",
    "difficulty_admission.py",
    "summarize_difficulty_control.py",
)


def as_time(value):
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("Specify an absolute UTC time with a timezone")
    return stamp.timestamp()


def science_hash(spec):
    return hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def phase1_selection_path(phase1_root, master_sha):
    """Locate the immutable selection even in a mirror which records symlinks as metadata."""
    base = Path(phase1_root).resolve()
    direct = base / "selection"
    pattern = r"\.selection-snapshot-" + re.escape(master_sha) + r"-[0-9a-f]{32}/tree"
    if direct.is_symlink():
        target = os.readlink(direct)
        if not re.fullmatch(pattern, target) or (base / target).resolve() != base / target:
            raise ValueError("Phase 1 link is outside its registered snapshot")
    if direct.is_dir():
        if digest(direct / "manifest.json") != master_sha:
            raise ValueError("Wrong Phase 1 master manifest")
        return direct
    if os.path.lexists(direct):
        raise ValueError("Invalid Phase 1 selection entry")
    candidates = []
    for tree in sorted(base.glob(f".selection-snapshot-{master_sha}-*/tree")):
        if not re.fullmatch(pattern, tree.relative_to(base).as_posix()):
            continue
        if tree.resolve() != tree or digest(tree / "manifest.json") != master_sha:
            raise ValueError("Invalid physical Phase 1 backup")
        master = read(tree / "manifest.json")
        for seed in SEEDS:
            d = tree / str(seed)
            metadata = d / "selection.json"
            if (
                metadata.resolve() != metadata
                or digest(metadata) != master["selection_manifest_sha256"][str(seed)]
            ):
                raise ValueError("Corrupt Phase 1 seed manifest")
            subsets = read(metadata)["subsets"]
            for arm in ("Bprime", "E2", "E3", "E4"):
                path = d / (arm + ".jsonl")
                if path.resolve() != path or digest(path) != subsets[arm]["sha256"]:
                    raise ValueError("Corrupt physical Phase 1 training subset")
        candidates.append(tree)
    if not candidates:
        raise ValueError("No complete pinned Phase 1 selection in this backup")
    return candidates[0]


def eligibility_spec(eligibility, primary, phase1, selector):
    manifest = read(eligibility / "manifest.json")
    pool = primary / "frozen/train.jsonl"
    indexed = {str(r["id"]): r for r in rows(pool)}
    phase1_selection = phase1_selection_path(phase1, manifest["sources"]["phase1_selection_manifest_sha256"])
    if (
        len(indexed) != 4000
        or manifest["status"] != "feasible"
        or manifest["seeds"] != list(SEEDS)
        or manifest["size"] != 1000
        or manifest["sources"]["train_sha256"] != digest(pool)
        or manifest["sources"]["step_0_sha256"] != digest(primary / "scores/step-0.jsonl")
        or manifest["common_target_input_token_budget"] != 3005228
        or manifest["source_sha256"] != digest(selector)
    ):
        raise ValueError("E2b train-only provenance differs from the frozen study")
    result = {}
    for seed in SEEDS:
        tag = label(seed, "E2b")
        d = eligibility / str(seed)
        meta = read(d / "selection.json")
        selected = rows(d / "E2b.jsonl")
        ids = [str(row["id"]) for row in selected]
        source_map = manifest["sources"]["primary_and_followup_subsets_sha256"][str(seed)]
        for arm in ("A", "B", "E2"):
            base = phase1_selection if arm == "E2" else primary / "selection"
            if digest(base / str(seed) / f"{arm}.jsonl") != source_map[arm]:
                raise ValueError(f"{tag}: matched reference {arm} changed")
        if (
            digest(d / "selection.json") != manifest["seed_manifest_sha256"][str(seed)]
            or digest(d / "E2b.jsonl") != REGISTERED_LISTS[str(seed)]
            or meta["sha256"] != REGISTERED_LISTS[str(seed)]
            or meta["seed"] != seed
            or len(ids) != len(set(ids))
            or len(ids) != 1000
            or meta["ids"] != ids
            or any(row != indexed.get(str(row["id"])) for row in selected)
        ):
            raise ValueError(f"{tag}: eligible subset changed")
        result[tag] = {
            "size": 1000,
            "sha256": REGISTERED_LISTS[str(seed)],
            "seed_manifest_sha256": digest(d / "selection.json"),
        }
    return manifest, result


def build_spec(primary, phase1, eligibility, runner, cutoff):
    path = phase1 / "preregistration.json"
    ref = read(path)["spec"]
    prior_cutoff = as_time(ref["admission"]["hard_cutoff_utc"])
    if (
        prior_cutoff != as_time(CURRENT_CUTOFF)
        or len(set(ref["cell_order"])) != 11
        or ref["primary_lock_sha256"] != digest(primary / "locked-models.json")
        or ref["primary_design_sha256"] != digest(primary / "design.json")
        or ref["pool_sha256"] != digest(primary / "frozen/train.jsonl")
        or ref["training"]["runtime"]["token_goal"] != 3005228
    ):
        raise ValueError("Unexpected original or Phase 1 source identity")
    for name, checksum in ref["source_sha256"].items():
        if digest(runner.scripts / name) != checksum:
            raise ValueError(f"Phase 1 source drift: {name}")
    eligible, selections = eligibility_spec(
        eligibility, primary, phase1, runner.scripts / "select_difficulty_control.py"
    )
    code = {**ref["source_sha256"], **{name: digest(runner.scripts / name) for name in NEW_SOURCES}}
    return {
        "scope": "Optional exploratory rerun using the previously opened internal test. Three E2b endpoints only.",
        "cell_order": ORDER,
        "selections": selections,
        "sources": eligible["sources"],
        "source_sha256": code,
        "primary_lock_sha256": ref["primary_lock_sha256"],
        "primary_design_sha256": ref["primary_design_sha256"],
        "phase1_preregistration_sha256": digest(path),
        "phase1_spec_sha256": science_hash(ref),
        "eligibility_manifest_sha256": digest(eligibility / "manifest.json"),
        "pool_sha256": ref["pool_sha256"],
        "models_sha256": ref["models_sha256"],
        "training": ref["training"],
        "splits": ref["splits"],
        "endpoint": ref["endpoint"],
        "phase1_cell_order": ref["cell_order"],
        "analysis": {
            "contrasts": ["E2b-A", "E2-A", "E2b-B", "E2b-E2"],
            "primary_question": "E2b minus A at matched length-bin counts and B's initial difficulty-bin counts.",
            "length_reference": "E2 minus A, reused from Phase 1. E2 matches B's length bins.",
            "exploratory": "E2b minus B is descriptive; E2b minus E2 changes both marginal targets.",
            "intervals": "2,000 joint paired held-out group resamples with fixed seeds and a three-seed t sensitivity.",
            "limitations": "Bin matching does not fix exact loss, tool identity or joint loss trajectories. No new test or BFCL; no equivalence or causal claim.",
        },
        "selection_design": {
            k: eligible[k]
            for k in (
                "primary_length_target",
                "difficulty_target",
                "chosen_requested_bins_per_axis",
                "joint_objective",
                "rng_namespace",
                "length_upper_bounds",
                "difficulty_upper_bounds",
            )
        },
        "admission": {
            **ref["admission"],
            "hard_cutoff_utc": utc(cutoff),
            "rule": "One indivisible three-seed batch. All eleven Phase 1 endpoints and upstream terminal runs require independent mirror and audit evidence. Use the largest observed stage times and sizes, 20% headroom, fresh cash runway and an idle A40. External interruptions retain receipts but permit no incomplete aggregate.",
        },
    }


def ensure_inputs(root, eligibility, selections):
    dest = root / "selection"
    names = [Path(str(seed)) / "E2b.jsonl" for seed in SEEDS]

    def verify(directory):
        if directory.is_symlink():
            raise ValueError("The separate E2b input directory must be an ordinary directory")
        for seed, name in zip(SEEDS, names):
            if digest(directory / name) != selections[label(seed, "E2b")]["sha256"]:
                raise ValueError(f"E2b input mismatch: {name}")

    if dest.exists():
        verify(dest)
        return
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        tree = Path(tmp) / "selection"
        tree.mkdir()
        for name in names:
            (tree / name).parent.mkdir()
            shutil.copyfile(eligibility / name, tree / name)
        verify(tree)
        tree.rename(dest)


def historical_maxima(ref, phase1, out):
    seconds = dict(ref["historical_max_seconds"])
    sizes = dict(ref["historical_max_bytes"])
    for base in (phase1, out):
        for stage in STAGES:
            for rec in (base / "receipts").glob(f"{stage}-s*.json"):
                tag = rec.stem.removeprefix(stage + "-")
                seconds[stage] = max(seconds[stage], timing(rec))
                if stage == "train":
                    folder = base / "training" / tag
                else:
                    lock = base / "locks" / (tag + ".json")
                    if not lock.is_file():
                        raise ValueError(f"Missing endpoint for timing {tag}")
                    folder = base / "evaluation" / stage / read(lock)["checkpoint"]
                sizes[stage] = max(sizes[stage], footprint(folder))
    return seconds, sizes


def estimate(spec, phase1, out, now):
    rule = spec["admission"]
    seconds, sizes = historical_maxima(rule, phase1, out)
    missing = {
        st: sum(not (out / "receipts" / f"{st}-{tag}.json").is_file() for tag in ORDER) for st in STAGES
    }
    fresh_s = len(ORDER) * sum(seconds.values()) * rule["time_multiplier"]
    remaining_s = sum(missing[k] * seconds[k] for k in STAGES) * rule["time_multiplier"]
    remaining_bytes = math.ceil(sum(missing[k] * sizes[k] for k in STAGES) * rule["disk_multiplier"])
    end = as_time(rule["hard_cutoff_utc"]) - TEARDOWN_SECONDS
    phase1_missing = {
        st: sum(not (phase1 / "receipts" / f"{st}-{tag}.json").is_file() for tag in spec["phase1_cell_order"])
        for st in STAGES
    }
    prior_s = sum(phase1_missing[k] * seconds[k] for k in STAGES) * rule["time_multiplier"]
    return {
        "hard_cutoff_utc": rule["hard_cutoff_utc"],
        "forecast_utc": utc(now),
        "observed_max_seconds": seconds,
        "observed_max_bytes": sizes,
        "missing_stage_counts": missing,
        "entire_fresh_batch_seconds": fresh_s,
        "remaining_batch_seconds": remaining_s,
        "required_bytes": remaining_bytes,
        "phase1_remaining_stage_counts": phase1_missing,
        "phase1_remaining_forecast_seconds": prior_s,
        "clock_seconds": max(0.0, end - now),
        "clock_admits_whole_remainder": now + remaining_s <= end,
        "latest_fresh_batch_start_utc": utc(end - fresh_s),
        "latest_phase1_plus_e2b_start_under_current_forecast_utc": utc(end - prior_s - remaining_s),
        "scope": "Forecast excludes ongoing original supplemental runs and predecessor verification delays. Confirm all gates at launch.",
    }


def admission_flags(estimate_, free_bytes, account_runway):
    required = estimate_["remaining_batch_seconds"]
    return {
        "clock": estimate_["clock_admits_whole_remainder"],
        "cash": account_runway >= required + TEARDOWN_SECONDS,
        "storage": free_bytes >= estimate_["required_bytes"],
    }


def record_status(runner, state, reason, completed):
    record = {
        "status": state,
        "reason": reason,
        "planned": ORDER,
        "completed": list(completed),
        "missing": [tag for tag in ORDER if tag not in completed],
    }
    write(runner.root / "execution.json", record)
    return record


def protect_children(runner):
    """Keep task argv stable on resume while shortening a previously admitted cash window."""
    timeout = shutil.which("timeout")
    if (
        not timeout
        or "GNU coreutils"
        not in subprocess.run([timeout, "--version"], check=True, capture_output=True, text=True).stdout
    ):
        raise RuntimeError("The execution host requires GNU timeout")
    limit = runner.root / "runtime/cutoff-unix"
    limit.parent.mkdir(exist_ok=True)
    previous = int(limit.read_text()) if limit.exists() else math.floor(runner.end)
    real_cutoff = min(previous, math.floor(runner.end))
    temp = limit.with_suffix(".tmp")
    temp.write_text(str(real_cutoff) + "\n")
    temp.replace(limit)
    runner.end = real_cutoff
    for role in ("training_python", "eval_python"):
        python = getattr(runner.a, role)
        if not python.is_file() or not os.access(python, os.X_OK):
            raise ValueError(f"Missing executable: {python}")
        wrapper = limit.parent / role / "python"
        content = (
            "#!/bin/sh\n"
            + "read -r end < "
            + shlex.quote(str(limit))
            + " || exit 124\n"
            + "remaining=$(( end - $(date +%s) ))\n"
            + 'test "$remaining" -gt 0 || exit 124\n'
            + "PATH="
            + shlex.quote(str(python.parent))
            + ':"$PATH"; export PATH\n'
            + "exec "
            + shlex.quote(timeout)
            + ' --kill-after=5s "$remaining" '
            + shlex.quote(str(python))
            + ' "$@"\n'
        )
        if wrapper.exists():
            if wrapper.read_text() != content:
                raise ValueError("A resumed subprocess must use the original environment")
        else:
            wrapper.parent.mkdir(parents=True, exist_ok=True)
            wrapper.write_text(content)
            wrapper.chmod(0o700)
        setattr(runner.a, role, wrapper)


def verify_resume_receipts(root):
    for tag in ORDER:
        for stage in STAGES:
            task = f"{stage}-{tag}"
            if not (root / "receipts" / (task + ".json")).exists():
                continue
            if stage == "train":
                base = root / "training" / tag
                files = [base / f for f in ("completed.json", "args.json", "exit.json")]
            else:
                checkpoint = read(root / "locks" / (tag + ".json"))["checkpoint"]
                base = root / "evaluation" / stage / checkpoint
                files = [base / f for f in ("config.json", "predictions.jsonl", "metrics.json")]
            check_outputs(root, task, files)


def register(primary, phase1, eligibility, runner, cutoff, create):
    path = runner.root / "preregistration.json"
    if not path.exists() and not create:
        raise ValueError("Freeze the separate E2b registration first")
    spec = build_spec(primary, phase1, eligibility, runner, cutoff)
    if path.exists():
        if read(path)["spec"] != spec:
            raise ValueError("Registered E2b study inputs, cutoff or source code changed")
    else:
        if any(
            (runner.root / d).exists()
            for d in ("training", "evaluation", "receipts", "cells", "admission.json")
        ):
            raise ValueError("Work predates the independent E2b registration")
        ensure_inputs(runner.root, eligibility, spec["selections"])
        write(path, {"created_utc": utc(time.time()), "spec": spec})
    ensure_inputs(runner.root, eligibility, spec["selections"])
    return spec


def execute(roots, runner, spec, attestation, attestation_sha, account):
    from difficulty_admission import account_seconds, verify_attestation

    primary, phase1 = roots["primary"], roots["phase1"]
    with ExitStack() as locks:
        for r in roots.values():
            handle = locks.enter_context((r / ".lock").open("r+b"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if digest(attestation) != attestation_sha:
            raise ValueError("The external backup/audit attestation does not match its supplied SHA-256")
        verified = verify_attestation(attestation, roots)
        verified_digest = digest(attestation)
        if verified_digest != attestation_sha:
            raise ValueError("Attestation changed during verification")
        immutable(
            runner.root / "dependencies.json", {"attestation_sha256": verified_digest, "evidence": verified}
        )
        if (
            digest(runner.big / WEIGHTS) != spec["models_sha256"]["target_weights"]
            or digest(runner.big / "config.json") != spec["models_sha256"]["target_config"]
        ):
            raise ValueError("Target model differs from the original A/B/E2 experiment")
        current = runner.root / "admission.json"
        if not current.exists() and any(
            (runner.root / d).exists() for d in ("training", "evaluation", "receipts", "cells")
        ):
            raise ValueError("GPU artifacts predate whole-batch admission")
        verify_resume_receipts(runner.root)
        completed = completed_cells(runner, spec)
        if len(completed) == len(ORDER):
            immutable(runner.root / "locked-models.json", completed)
            return record_status(runner, "complete", "All three fixed endpoints verified.", completed)
        now = time.time()
        prediction = estimate(spec, phase1, runner.root, now)
        cash = account_seconds(account, now, cutoff=as_time(spec["admission"]["hard_cutoff_utc"]))
        free = shutil.disk_usage(runner.root).free
        flags = admission_flags(prediction, free, cash)
        decision = {
            "budget": prediction,
            "available_cash_seconds": cash,
            "free_bytes": free,
            "fits": flags,
            "attestation_sha256": verified_digest,
            "account_observation_sha256": digest(account),
        }
        old = read(current) if current.is_file() else None
        cash_end = now + cash - TEARDOWN_SECONDS
        latest = min(runner.end, cash_end)
        if old is not None:
            prior_end = as_time(old["effective_subprocess_cutoff_utc"])
            guard = runner.root / "runtime/cutoff-unix"
            if guard.exists():
                prior_end = min(prior_end, int(guard.read_text()))
            flags["prior_subprocess_window"] = now + prediction["remaining_batch_seconds"] <= prior_end
            latest = min(prior_end, latest)
        write(runner.root / "decisions" / f"{time.time_ns()}.json", decision)
        if not all(flags.values()):
            state = "interrupted" if old else "deferred"
            return record_status(
                runner,
                state,
                "Entire remaining batch exceeded a verified limit: "
                + ", ".join(k for k, v in flags.items() if not v),
                completed,
            )
        gpu = check_gpu(spec["admission"])
        if old is None:
            immutable(
                current,
                {
                    "registered_utc": utc(now),
                    "effective_subprocess_cutoff_utc": utc(latest),
                    "attestation_sha256": verified_digest,
                    "decision": decision,
                    "gpu": gpu,
                },
            )
        runner.end = latest
        protect_children(runner)
        try:
            for seed in SEEDS:
                tag = label(seed, "E2b")
                if tag not in completed:
                    cell = run_cell(primary, runner, spec, seed, "E2b")
                    completed[tag] = cell["endpoint"]["checkpoint"]
                    record_status(runner, "running", "Whole three-seed batch admitted.", completed)
        except BaseException as error:
            record_status(
                runner, "interrupted", f"{type(error).__name__}: inspect receipts before resuming", completed
            )
            raise
        if set(completed) != set(ORDER):
            raise RuntimeError("No aggregate is permitted until all three complete")
        immutable(runner.root / "locked-models.json", completed)
        return record_status(runner, "complete", "All three fixed endpoints verified.", completed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("forecast", "freeze", "status", "run"))
    for role, path in (
        ("primary", "local/course-run"),
        ("full", "local/full-reference"),
        ("cross", "local/crossed-seeds"),
        ("phase1", "local/followup/phase1"),
        ("eligibility", "local/followup/e2b-eligibility"),
        ("out", "local/followup/difficulty-control"),
    ):
        p.add_argument("--" + role, type=Path, default=Path(path))
    p.add_argument("--train-python", type=Path, default=Path("/opt/trainenv/bin/python"))
    p.add_argument("--eval-python", type=Path, default=Path("/opt/evalenv/bin/python"))
    p.add_argument(
        "--new-rental-cutoff-utc",
        help="A separately authorized future rental's hard cutoff. Use a fresh output tree.",
    )
    p.add_argument(
        "--attestation",
        type=Path,
        help="Verified private backup/audit receipt from the independent backup host",
    )
    p.add_argument("--attestation-sha256", help="Digest printed by the independent backup host")
    p.add_argument(
        "--account-observation",
        type=Path,
        help="Fresh private provider balance/rate observation, JSON or JSONL",
    )
    a = p.parse_args()
    roots = {k: getattr(a, k).resolve() for k in ("primary", "full", "cross", "phase1")}
    eligibility, out = a.eligibility.resolve(), a.out.resolve()
    paths = [*roots.values(), eligibility, out]
    if len(set(paths)) != len(paths) or any(
        x in y.parents or y in x.parents for x, y in combinations(paths, 2)
    ):
        p.error("Provide mutually separate predecessor, eligibility and E2b directories")
    cutoff = as_time(a.new_rental_cutoff_utc or CURRENT_CUTOFF)
    if a.new_rental_cutoff_utc and (
        cutoff <= as_time(CURRENT_CUTOFF) or time.time() < as_time(CURRENT_CUTOFF)
    ):
        p.error(
            "The current rental cannot be extended. A new rental registration is available after its cutoff."
        )
    if a.action == "run" and not all((a.attestation, a.attestation_sha256, a.account_observation)):
        p.error("run requires --attestation, --attestation-sha256 and --account-observation")
    prior = read(roots["phase1"] / "preregistration.json")["spec"]
    if a.action == "forecast":
        quick = {
            "admission": {**prior["admission"], "hard_cutoff_utc": utc(cutoff)},
            "phase1_cell_order": prior["cell_order"],
        }
        print(json.dumps(estimate(quick, roots["phase1"], out, time.time()), indent=2))
        return
    if a.action == "status":
        frozen = read(out / "preregistration.json")["spec"]
        print(
            json.dumps(
                {
                    "execution": read(out / "execution.json") if (out / "execution.json").exists() else None,
                    "forecast": estimate(frozen, roots["phase1"], out, time.time()),
                },
                indent=2,
            )
        )
        return
    conf = args_parser().parse_args([])
    conf.course = out
    conf.micro_batch = read(roots["primary"] / "design.json")["micro_batch"]
    conf.training_python, conf.eval_python = a.train_python.absolute(), a.eval_python.absolute()
    conf.deadline, conf.reserve_seconds = utc(cutoff), TEARDOWN_SECONDS
    runner = Protocol(conf)

    def stop(sig, frame):
        raise KeyboardInterrupt(f"Signal {sig}")

    signal.signal(signal.SIGTERM, stop)
    with (runner.root / ".lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = register(roots["primary"], roots["phase1"], eligibility, runner, cutoff, a.action == "freeze")
        if a.action == "freeze":
            print(
                f"Registered {len(spec['cell_order'])} E2b endpoints at {out/'preregistration.json'}. CPU only."
            )
        else:
            result = execute(roots, runner, spec, a.attestation, a.attestation_sha256, a.account_observation)
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
