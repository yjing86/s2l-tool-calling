"""Freeze/run the development-only B/C selection-seed × target-training-seed diagnostic."""

import argparse
from contextlib import ExitStack
import fcntl
import json
from pathlib import Path
import time

from run_course_experiment import Protocol, args_parser, digest, read, rows, write

SEEDS = (42, 43)
GROUPS = ("B", "C")
CROSS = [(s, t, g) for s, t in ((42, 43), (43, 42)) for g in GROUPS]
WEIGHTS = "model.safetensors-00001-of-00001.safetensors"
WEIGHTS_SHA256 = "aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1"
CONFIG_KEYS = (
    "model_revision",
    "tokenizer_sha256",
    "template_sha256",
    "source_sha256",
    "prompts_sha256",
    "stop_token_ids",
    "greedy",
    "dtype",
    "seed",
    "language_model_only",
    "max_lora_rank",
    "script_sha256",
)
GENERATION_ARGS = ("max_tokens", "max_model_len", "batch_size", "enforce_eager", "gpu_memory_utilization")


def normalized_training(command):
    """Only these paths and the two intentionally coupled training-side seeds may change."""
    args = list(command[1:])
    for flag in ("--output_dir", "--dataset", "--seed", "--data_seed"):
        args[args.index(flag) + 1] = "<varies>"
    return args


def gen_protocol(config):
    return {
        **{k: config[k] for k in CONFIG_KEYS},
        "generation": {k: config["args"][k] for k in GENERATION_ARGS},
        "versions": config["versions"],
    }


def token_boundary(out, goal):
    steps = rows(out / "steps.jsonl")
    complete = read(out / "completed.json")
    runtime = read(out / "runtime.json")
    if not steps or runtime["token_goal"] != goal or runtime["starting_model_step"] != 0:
        raise ValueError(f"{out}: different start or token goal")
    before = steps[-2]["input_tokens"] if len(steps) > 1 else 0
    after = steps[-1]["input_tokens"]
    if not (
        before < goal <= after
        and after == complete["input_tokens"]
        and steps[-1]["step"] == complete["final_step"]
    ):
        raise ValueError(f"{out}: wrong first-crossing checkpoint")
    return {
        "step": complete["final_step"],
        "target_tokens": goal,
        "actual_input_tokens": after,
        "actual_answer_tokens": complete["answer_tokens"],
        "overshoot_tokens": after - goal,
        "last_update_tokens": after - before,
        "final_adapter_sha256": complete["adapter_sha256"][str(complete["final_step"])],
    }


def deterministic_spec(primary, runner):
    design = read(primary / "design.json")
    if not set(SEEDS).issubset(design["seeds"]):
        raise ValueError("Frozen primary lacks seed42 or 43")
    dev = runner.prep / "dev.jsonl"
    pool = primary / "frozen/train.jsonl"
    current_hash = digest(runner.big / WEIGHTS)
    if current_hash != WEIGHTS_SHA256:
        raise ValueError("Model weights differ from pinned verified 2B")
    dev_ids = [str(r["id"]) for r in rows(dev)]
    if len(set(dev_ids)) != 400 or len(dev_ids) != 400:
        raise ValueError("Expected 400 unique frozen dev IDs")
    frozen = {
        "purpose": "After seeing a sign reversal in the seed42/43 diagonal development C−B, separate fixed list and training-side seed dependence within this 2×2 grid.",
        "scope": "Post hoc development-only mechanism diagnostic; no test or BFCL generation, primary conclusions unchanged. Training seed controls both LoRA initialization and the trainer data seed/order.",
        "selection_groups": list(GROUPS),
        "seed_grid": list(SEEDS),
        "new_runs": [{"selection": s, "training": t, "group": g} for s, t, g in CROSS],
        "readout": "Evaluate each final equal-token checkpoint exactly once on the original 400 dev questions; never use the dev peak for endpoint choice.",
        "analysis": "Report the within-cell C−B differences and their 2×2 mean; selection effect, target-training effect, and difference-in-differences with joint near-duplicate-group bootstrap on the same 400 dev IDs. Intervals condition on these observed lists/weights.",
        "selection": {},
        "known_diagonal_dev": {},
        "design_sha256": digest(primary / "design.json"),
        "pool_sha256": digest(pool),
        "dev_sha256": digest(dev),
        "data_manifest_sha256": digest(runner.prep / "manifest.json"),
        "model_weights_sha256": current_hash,
        "model_config_sha256": digest(runner.big / "config.json"),
        "deadline": design["deadline"],
        "reserve_seconds": max(600, design["reserve_seconds"]),
        "budget_rule": "Measured worst diagonal training and endpoint dev wall seconds times missing runs, plus 30%; additional fixed 10min hard-deadline reserve.",
        "sources_sha256": {
            f: digest(runner.scripts / f)
            for f in (
                "train_experiment.py",
                "experiment_callback.py",
                "verify_tool_call_loss.py",
                "evaluate_experiment.py",
                "run_crossed_seeds.py",
                "summarize_crossed_seeds.py",
                "summarize_course.py",
                "run_course_experiment.py",
            )
        },
    }
    prompt_receipt = read(primary / "receipts/encode-dev.json")
    (prompt_hash,) = [v for k, v in prompt_receipt["outputs"].items() if Path(k).name == "prompts.jsonl"]
    cache = primary / "prompts/dev/prompts.jsonl"
    if cache.exists() and digest(cache) != prompt_hash:
        raise ValueError("Primary dev inference prefix file changed")
    frozen["dev_prompts_sha256"] = prompt_hash
    invariant = None
    for s in SEEDS:
        sel = primary / "selection" / str(s)
        report = read(sel / "selection.json")
        if report["train_sha256"] != frozen["pool_sha256"] or report["seed"] != s:
            raise ValueError("Selection source mismatch")
        for g in GROUPS:
            tag = f"s{s}-{g}"
            src = sel / f"{g}.jsonl"
            lst = report["subsets"][g]
            data_hash = digest(src)
            ids = [str(row["id"]) for row in rows(src)]
            if (
                lst["sha256"] != data_hash
                or ids != list(map(str, lst["ids"]))
                or len(set(ids)) != design["subset"]
                or len(ids) != len(set(ids))
            ):
                raise ValueError(f"{tag}: subset hash, IDs or size changed")
            out = primary / "training" / tag
            manifest = read(out / "run.json")
            runtime = read(out / "runtime.json")
            if (
                manifest["data_sha256"] != data_hash
                or manifest["pool_sha256"] != frozen["pool_sha256"]
                or manifest["seed"] != s
                or manifest["stage"] != "target"
            ):
                raise ValueError(f"{tag}: incorrect diagonal training run")
            expected_sources = {k: frozen["sources_sha256"][k] for k in manifest["sources_sha256"]}
            if (
                manifest["sources_sha256"] != expected_sources
                or manifest["model_config_sha256"] != frozen["model_config_sha256"]
            ):
                raise ValueError("Primary used different train code or model configuration")
            bound = token_boundary(out, runtime["token_goal"])
            name = tag + f"-step{bound['step']}"
            conf = read(primary / "evaluation/dev" / name / "config.json")
            m = read(primary / "evaluation/dev" / name / "metrics.json")
            if (
                conf["adapter_sha256"]["adapter_model.safetensors"] != bound["final_adapter_sha256"]
                or conf["source_sha256"] != frozen["dev_sha256"]
                or m["count"] != 400
            ):
                raise ValueError("Primary dev evaluation differs from final equitoken checkpoint or split")
            if conf["script_sha256"] != frozen["sources_sha256"]["evaluate_experiment.py"]:
                raise ValueError("Generation script differs from primary")
            ident = {
                "train_command": normalized_training(manifest["command"]),
                "token_goal": runtime["token_goal"],
                "trainable_parameters": runtime["trainable_parameter_count"],
                "cuda": runtime["cuda"],
                "gpu": runtime["gpu"],
                "training_packages": {
                    k: manifest["packages"][k]
                    for k in ("torch", "transformers", "ms_swift", "ms-swift", "peft")
                    if k in manifest["packages"]
                },
                "dev_protocol": gen_protocol(conf),
            }
            if invariant is None:
                invariant = ident
            elif invariant != ident:
                raise ValueError("Primary diagonal training or generation settings differ")
            frozen["selection"][tag] = {
                "file_sha256": data_hash,
                "selection_manifest_sha256": digest(sel / "selection.json"),
            }
            frozen["known_diagonal_dev"][tag] = {
                "checkpoint": name,
                "correct": m["correct"],
                "count": 400,
                **bound,
            }
    frozen["invariant"] = invariant
    return frozen


def freeze(primary, runner):
    spec = deterministic_spec(primary, runner)
    path = runner.root / "frozen.json"
    if path.exists():
        prev = read(path)
        if spec != prev["spec"]:
            raise ValueError("Frozen specification or source files changed")
    else:
        blind_counts = {
            k: sum(1 for _ in (primary / "evaluation" / k).glob("*/predictions.jsonl")) for k in ("test", "bfcl")
        }
        write(path, {"frozen_unix": time.time(), "primary_blind_files_at_freeze": blind_counts, "spec": spec})
    return spec


def full_ready(full):
    summary = full / "summary/full-reference.json"
    done = full / "receipts/full-summary.json"
    if summary.is_file() and done.is_file():
        checksums = read(done)["outputs"]
        if digest(summary) in [v for k, v in checksums.items() if Path(k).name == "full-reference.json"]:
            return "full reference complete"
        raise ValueError("Full reference summary does not match receipt")
    if summary.is_file() and (full / "ledger.jsonl").exists():
        if any(r.get("event") == "done" and r.get("task") == "full-summary" for r in rows(full / "ledger.jsonl")):
            return "full reference completed in ledger"
    decisions = sorted((full / "decisions").glob("*.json"))
    if decisions:
        d = read(decisions[-1])
        if (
            d.get("proceed") is False
            and isinstance(d.get("estimated_seconds"), (int, float))
            and isinstance(d.get("available_seconds"), (int, float))
            and d["estimated_seconds"] + d.get("primary_remaining_seconds", 0) >= d["available_seconds"]
            and d.get("reason", "").startswith("Timing-only estimate")
        ):
            return "full reference explicitly skipped for insufficient budget"
    return None


def main_ready(primary):
    done = primary / "receipts/summary.json"
    if done.exists() and read(done)["task"] == "summary":
        return True
    return False


def forecast(primary, runner, spec):
    times = []
    dev = []
    for tag, rec in spec["known_diagonal_dev"].items():
        times.append(read(primary / "receipts" / f"train-{tag}.json")["elapsed_s"])
        dev.append(read(primary / "receipts" / ("eval-dev-" + rec["checkpoint"] + ".json"))["elapsed_s"])
    if min(times + dev) <= 0:
        raise ValueError("Invalid measured stage duration")
    tleft = dleft = 0
    for s, t, g in CROSS:
        name = f"{g}-selection{s}-target{t}"
        if not (runner.root / "receipts" / ("train-" + name + ".json")).is_file():
            tleft += 1
        if not (runner.root / "receipts" / ("dev-" + name + ".json")).is_file():
            dleft += 1
    estimate = 1.3 * (tleft * max(times) + dleft * max(dev))
    remaining = runner.end - time.time()
    return {
        "proceed": remaining > estimate,
        "remaining_before_hard_cutoff_s": remaining,
        "estimated_with_30pct_s": estimate,
        "training_runs_remaining": tleft,
        "dev_runs_remaining": dleft,
        "observed_training_seconds": times,
        "observed_endpoint_dev_seconds": dev,
    }


def run(primary, full, runner, spec):
    locked = {}
    pool = primary / "frozen/train.jsonl"
    dev = runner.prep / "dev.jsonl"
    cache = primary / "prompts/dev/prompts.jsonl"
    if digest(cache) != spec["dev_prompts_sha256"]:
        raise RuntimeError("Primary dev prompt cache changed")
    for s, t, g in CROSS:
        name = f"{g}-selection{s}-target{t}"
        out = runner.root / "training" / name
        decision = forecast(primary, runner, spec)
        write(runner.root / "decisions" / f"{time.time_ns()}.json", decision)
        if not decision["proceed"]:
            print("Skipped remaining cells: insufficient budget.", json.dumps(decision))
            return
        src = primary / "selection" / str(s) / f"{g}.jsonl"
        if digest(src) != spec["selection"][f"s{s}-{g}"]["file_sha256"]:
            raise RuntimeError("Frozen list changed")
        runner.train("train-" + name, "target", runner.big, src, out, t, ["--pool", pool])
        man = read(out / "run.json")
        runtime = read(out / "runtime.json")
        inv = spec["invariant"]
        if not (
            man["seed"] == t
            and man["data_sha256"] == digest(src)
            and man["pool_sha256"] == spec["pool_sha256"]
            and normalized_training(man["command"]) == inv["train_command"]
            and man["sources_sha256"] == {k: spec["sources_sha256"][k] for k in man["sources_sha256"]}
            and runtime["gpu"] == inv["gpu"]
            and runtime["cuda"] == inv["cuda"]
            and runtime["trainable_parameter_count"] == inv["trainable_parameters"]
            and all(man["packages"].get(k) == v for k, v in inv["training_packages"].items())
        ):
            raise RuntimeError("Crossed training diverged from primary protocol")
        bound = token_boundary(out, inv["token_goal"])
        step = bound["step"]
        ck = out / f"checkpoint-{step}"
        key = name + f"-step{step}"
        scores = runner.root / "evaluation/dev" / key
        argv = [
            "--model",
            runner.big,
            "--data",
            dev,
            "--max-model-len",
            "4096",
            "--max-tokens",
            "512",
            "--prepared",
            cache,
            "--out",
            scores,
            "--adapter",
            ck,
            "--enforce-eager",
        ]
        runner.task(
            "dev-" + name,
            runner.a.eval_python,
            "evaluate_experiment",
            argv,
            [scores / "metrics.json", scores / "predictions.jsonl"],
        )
        conf = read(scores / "config.json")
        metrics = read(scores / "metrics.json")
        if (
            gen_protocol(conf) != inv["dev_protocol"]
            or conf["adapter_sha256"]["adapter_model.safetensors"] != bound["final_adapter_sha256"]
            or metrics["count"] != 400
        ):
            raise RuntimeError("Crossed evaluation diverged from primary dev conditions")
        write(
            runner.root / "cells" / (name + ".json"),
            {
                "selection_seed": s,
                "target_seed": t,
                "group": g,
                "correct": metrics["correct"],
                "n": metrics["count"],
                "data_sha256": digest(src),
                "predictions_sha256": digest(scores / "predictions.jsonl"),
                **bound,
            },
        )
        locked[f"select{s}-train{t}-{g}"] = key
    write(runner.root / "crossed-models.json", locked)
    runner.task(
        "summary",
        runner.a.training_python,
        "summarize_crossed_seeds",
        ["--primary", primary, "--crossed", runner.root, "--dev", dev],
        [runner.root / "summary.json", runner.root / "summary.md"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("--primary", type=Path, default=Path("local/course-run"))
    parser.add_argument("--full", type=Path, default=Path("local/full-reference"))
    parser.add_argument("--out", type=Path, default=Path("local/crossed-seeds"))
    parser.add_argument("--train-python", type=Path, default=Path("/opt/trainenv/bin/python"))
    parser.add_argument("--eval-python", type=Path, default=Path("/opt/evalenv/bin/python"))
    a = parser.parse_args()
    primary = a.primary.resolve()
    full = a.full.resolve()
    design = read(primary / "design.json")
    conf = args_parser().parse_args([])
    conf.deadline = design["deadline"]
    conf.reserve_seconds = max(600, design["reserve_seconds"])
    conf.training_python = a.train_python
    conf.eval_python = a.eval_python
    conf.micro_batch = design["micro_batch"]
    conf.course = a.out
    runner = Protocol(conf)
    with ExitStack() as stack:
        lock = stack.enter_context((runner.root / ".lock").open("a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.action == "freeze":
            spec = freeze(primary, runner)
            print("Frozen development-only protocol.", json.dumps(spec["known_diagonal_dev"]))
            return
        if not (runner.root / "frozen.json").exists():
            parser.error("Call freeze before run")
        for path in (primary, full):
            if not path.is_dir():
                print("Deferred: upstream experiment missing", path)
                return
            upstream = stack.enter_context((path / ".lock").open("a"))
            try:
                fcntl.flock(upstream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print("Deferred: upstream scheduler holds GPU.", path)
                return
        safeguard = full_ready(full)
        if not main_ready(primary) or not safeguard:
            print(
                "Deferred: primary completion and a verified full-reference completion/budget decision required."
            )
            return
        spec = freeze(primary, runner)
        write(
            runner.root / "dependencies.json",
            {
                "time": time.time(),
                "full_reference": safeguard,
                "primary": "completed",
                "accessible_splits": ["train", "dev"],
            },
        )
        run(primary, full, runner, spec)


if __name__ == "__main__":
    main()
