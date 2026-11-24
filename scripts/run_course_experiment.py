"""Run the single-GPU course protocol, keeping per-task receipts and a hard wall deadline."""

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def digest(path):
    with Path(path).open("rb") as f:
        h = hashlib.sha256()
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
        return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def subset(src, dest, count):
    """Choose by IDs alone, keep source order in the resulting dataset."""
    lines = src.read_text().splitlines()
    if count > len(lines):
        raise ValueError("Requested subset exceeds split size")
    if count != len(lines):
        ranked = sorted(
            range(len(lines)),
            key=lambda i: hashlib.sha256(("31415|" + str(json.loads(lines[i])["id"])).encode()).hexdigest(),
        )
        lines = [lines[i] for i in sorted(ranked[:count])]
    payload = "\n".join(lines) + "\n"
    if dest.exists():
        if dest.read_text() != payload:
            raise RuntimeError(f"Frozen subset changed: {dest}")
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp")
        tmp.write_text(payload)
        tmp.replace(dest)


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--course", type=Path, default=Path("local/course-run"))
    p.add_argument("--deadline", default="2026-11-24T20:39:00Z", help="Absolute provider billing deadline UTC.")
    p.add_argument("--reserve-seconds", type=int, default=600)
    p.add_argument("--training-python", type=Path, default=Path("/opt/trainenv/bin/python"))
    p.add_argument("--eval-python", type=Path, default=Path("/opt/evalenv/bin/python"))
    p.add_argument(
        "--bfcl-python", type=Path, help="Pinned BFCL with official checker dependencies; defaults to training env"
    )
    p.add_argument("--proxy-smoke", type=Path, default=Path("local/runs/proxy-smoke2"))
    p.add_argument("--target-smoke", type=Path, default=Path("local/runs/target-smoke"))
    p.add_argument("--pilot", type=Path, default=Path("local/runs/pilot-base"))
    p.add_argument("--micro-batch", type=int, choices=[1, 2, 4, 8, 16], default=2)
    p.add_argument("--through", choices=["bench", "freeze", "proxy", "train", "all"], default="all")
    return p


class Protocol:
    def __init__(self, a):
        self.a = a
        self.root = a.course.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.scripts = Path(__file__).resolve().parent
        self.repo = self.scripts.parent
        self.prep = self.repo / "local/prepared"
        self.small = self.repo / "local/models/Qwen3.5-0.8B"
        self.big = self.repo / "local/models/Qwen3.5-2B"
        self.end = dt.datetime.fromisoformat(a.deadline.replace("Z", "+00:00")).timestamp() - a.reserve_seconds
        self.models = {"base": None}
        self.base_dev = None

    def ledger(self, event, name, **values):
        with (self.root / "ledger.jsonl").open("a") as f:
            print(json.dumps(dict(event=event, task=name, unix=time.time(), **values)), file=f, flush=True)

    def task(self, name, python, tool, args, outputs, *, adopt=False):
        # Resolve scripts against the repository; the virtualenv bin on PATH selects the matching swift CLI.
        cmd = [str(python), str(self.scripts / (tool + ".py")), *map(str, args)]
        outputs = list(map(Path, outputs))
        stamp = self.root / "receipts" / (name + ".json")
        if stamp.exists():
            receipt = read(stamp)
            if receipt["argv"] != cmd or any(
                not f.is_file() or digest(f) != receipt["outputs"].get(str(f)) for f in outputs
            ):
                raise RuntimeError(f"Completed task {name} changed. Inspect before reusing")
            return receipt["elapsed_s"]
        # Keep crash evidence. A failed subprocess can restart in its original path
        # after a fix, while completed siblings remain protected by their receipts.
        intent = self.root / "attempts" / (name + ".json")
        script_hash = digest(Path(cmd[1]))
        if intent.exists():
            last = read(intent)
            if last["argv"] != cmd:
                raise RuntimeError(f"Arguments changed for an interrupted task: {name}")
            if last.get("state") == "failed" and time.time() < self.end:
                if (
                    tool in ("train_experiment", "evaluate_experiment")
                    and "--hf-audit" not in cmd
                    and not adopt
                    and "--prepare-only" not in cmd
                ):
                    # These jobs use create-only outputs; archived partial results can be examined later.
                    arg = "--output" if tool == "train_experiment" else "--out"
                    partial = Path(cmd[cmd.index(arg) + 1])
                    if partial.exists():
                        archived = self.root / "partial-attempts" / name / str(time.time_ns())
                        archived.parent.mkdir(parents=True, exist_ok=True)
                        partial.rename(archived)
                        self.ledger(
                            "archive_incomplete",
                            name,
                            old_code=last.get("script_sha256"),
                            new_code=script_hash,
                            location=str(archived),
                        )
            elif last.get("state") == "started" and not all(o.is_file() for o in outputs):
                raise RuntimeError(
                    f"Unclean shutdown during {name}. Verify child stopped and preserve partial artifacts before retrying"
                )
        if adopt and all(o.is_file() for o in outputs):
            elapsed = self.elapsed_external(outputs[0].parent)
            self.ledger("adopted", name, source=str(outputs[0].parent), elapsed_s=elapsed)
        else:
            if self.end <= time.time():
                raise TimeoutError("Budget window has ended; stop the Pod")
            write(intent, {"state": "started", "argv": cmd, "script_sha256": script_hash})
            self.ledger("begin", name, argv=cmd, script_sha256=script_hash)
            start = time.monotonic()
            env = dict(
                os.environ, PATH=str(python.parent) + ":" + os.environ.get("PATH", ""), PYTHONUNBUFFERED="1"
            )
            log = self.root / "logs" / (name + ".log")
            log.parent.mkdir(exist_ok=True)
            with log.open("a", buffering=1) as f:
                proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    proc.wait(timeout=max(0.01, self.end - time.time()))
                    if proc.returncode:
                        raise RuntimeError(f"{name} failed with exit {proc.returncode}; see {log}")
                except BaseException as exc:
                    if proc.poll() is None:
                        os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=8)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                    write(intent, {"state": "failed", "argv": cmd, "script_sha256": script_hash})
                    self.ledger("failed", name, elapsed_s=time.monotonic() - start, reason=str(exc))
                    raise
            elapsed = time.monotonic() - start
        receipt = dict(
            task=name,
            argv=cmd,
            script_sha256=script_hash,
            outputs={str(f): digest(f) for f in outputs},
            elapsed_s=elapsed,
        )
        write(stamp, receipt)
        self.ledger("done", name, elapsed_s=elapsed)
        return elapsed

    @staticmethod
    def elapsed_external(out):
        if (out / "exit.json").exists():
            return read(out / "exit.json")["elapsed_seconds"]
        return read(out / "metrics.json")["wall_seconds"]

    def train(self, name, stage, model, data, out, seed=42, extra=(), adopt=False):
        if adopt and (out / "completed.json").exists():
            m = read(out / "run.json")
            if (
                m["stage"] != stage
                or m["data_sha256"] != digest(data)
                or m["seed"] != seed
                or m["model_config_sha256"] != digest(model / "config.json")
                or read(out / "exit.json")["code"]
            ):
                raise RuntimeError(f"Existing training run {out} differs from requested task")
        args = [
            "--stage",
            stage,
            "--model",
            model,
            "--data",
            data,
            "--output",
            out,
            "--seed",
            seed,
            "--micro-batch",
            1 if stage == "smoke" else self.a.micro_batch,
            *extra,
        ]
        self.task(
            name,
            self.a.training_python,
            "train_experiment",
            args,
            [out / "completed.json", out / "args.json", out / "exit.json"],
            adopt=adopt,
        )
        completed = read(out / "completed.json")
        for step, sha in completed["adapter_sha256"].items():
            if digest(out / f"checkpoint-{step}/adapter_model.safetensors") != sha:
                raise ValueError(f"{out}: missing or changed adapter at step {step}")
        return completed

    def evaluate(self, split, name, data, adapter=None, out=None):
        external = split == "bfcl"
        extra = ["--generate-only"] if external else []
        cache = self.root / "prompts" / split
        common = ["--model", self.big, "--data", data, "--max-model-len", "4096", "--max-tokens", "512", *extra]
        self.task(
            "encode-" + split,
            self.a.training_python,
            "evaluate_experiment",
            [*common, "--swift-run-args", self.a.target_smoke / "args.json", "--prepare-only", "--out", cache],
            [cache / "prepare.json", cache / "prompts.jsonl"],
        )
        out = out or self.root / "evaluation" / split / name
        if adapter is not None:
            extra = ["--adapter", adapter]
        else:
            extra = []
        # An externally scheduled pilot can be reused after checking split, model and bare-base status.
        adopt = split == "pilot" and (out / "metrics.json").exists()
        if adopt:
            cfg = read(out / "config.json")
            if (
                cfg["source_sha256"] != digest(data)
                or cfg["adapter_sha256"]
                or cfg["prompts_sha256"] != digest(cache / "prompts.jsonl")
            ):
                raise RuntimeError("Pilot was produced using different inputs")
        cmd = [*common, "--prepared", cache / "prompts.jsonl", "--out", out, *extra, "--enforce-eager"]
        self.task(
            "eval-" + split + "-" + name,
            self.a.eval_python,
            "evaluate_experiment",
            cmd,
            [out / "metrics.json", out / "predictions.jsonl"],
            adopt=adopt,
        )
        if split == "debug":
            self.task(
                "verify-debug-" + name,
                self.a.training_python,
                "evaluate_experiment",
                [*common, "--prepared", cache / "prompts.jsonl", "--out", out, *extra, "--hf-audit"],
                [out / "first-token-audit.json"],
            )
        return read(out / "metrics.json")

    def bench(self):
        for smoke, model in ((self.a.proxy_smoke, self.small), (self.a.target_smoke, self.big)):
            manifest = read(smoke / "completed.json")
            if (
                digest(model / "config.json") != read(smoke / "run.json")["model_config_sha256"]
                or read(smoke / "exit.json")["code"]
            ):
                raise ValueError("Smoke run identity mismatch")
            for step, sha in manifest["adapter_sha256"].items():
                if digest(smoke / f"checkpoint-{step}/adapter_model.safetensors") != sha:
                    raise ValueError("Smoke adapter hash mismatch")
        debug = self.prep / "debug.jsonl"
        self.evaluate("debug", "base", debug)
        self.evaluate("debug", "smoke-2b", debug, self.a.target_smoke / "checkpoint-1")
        self.evaluate("pilot", "base", self.prep / "pilot.jsonl", out=self.a.pilot)
        bench = self.root / "bench"
        data = bench / "pool64.jsonl"
        subset(self.prep / "train.jsonl", data, 64)
        for name, model in (("proxy", self.small), ("target", self.big)):
            self.train("bench-" + name, "proxy", model, data, bench / name)
        args = [
            "--model",
            self.small,
            "--train",
            data,
            "--out",
            bench / "score",
            "--swift-run-args",
            self.a.proxy_smoke / "args.json",
            "--checkpoint",
            "0=base",
        ]
        self.task(
            "bench-score",
            self.a.training_python,
            "score_trajectories",
            args,
            [bench / "score/step-0.complete.json"],
        )
        return bench

    def freeze(self, bench):
        path = self.root / "design.json"
        if path.exists():
            frozen = read(path)
            if (
                frozen["deadline"] != self.a.deadline
                or frozen["reserve_seconds"] != self.a.reserve_seconds
                or frozen["micro_batch"] != self.a.micro_batch
            ):
                raise ValueError("Budget or batch differs from the frozen design")
            return frozen
        training_rates = {}
        for name in ("target", "proxy"):
            records = rows(bench / name / "steps.jsonl")
            training_rates[name] = (records[-1]["wall_seconds"] - records[0]["wall_seconds"]) / (
                records[-1]["examples"] - records[0]["examples"]
            )
        sc = rows(bench / "score/timings.jsonl")
        score_per = sum(r["seconds"] for r in sc) / 64
        base = read(self.a.pilot / "metrics.json")
        gen = (base["wall_seconds"] - base["engine_load_seconds"]) / base["count"]
        overhead = max(10, base["engine_load_seconds"])

        def ev(n):
            return n * gen + overhead

        remaining = self.end - time.time()
        for pool in (4000, 2000):
            n = pool // 4
            common = pool * (training_rates["proxy"] + 5 * score_per) + 300
            train = 2 * n * training_rates["target"] + 90
            devs = math.ceil((2 * n / 16) / 50)
            dev = devs * ev(400)
            for exam in (2000, 1500):
                blind = ev(exam) + ev(200)
                for groups in (10, 9, 6, 3):
                    minimum = 1.3 * (common + ev(400) + (groups + 1) * blind + groups * (train + dev))
                    if minimum <= remaining:
                        break
                else:
                    continue
                # Reserve blind generation for the independent compute-budget snapshot if affordable.
                slots = groups - (groups == 10)
                extension = (remaining - minimum) / (1.3 * slots) - blind
                equal_budget = common + train + dev + min(600, extension / 3)
                enable = extension > common + ev(400) + 180
                design = dict(
                    pool=pool,
                    subset=n,
                    test=exam,
                    seeds=[42, 43, 44][: min(3, groups // 3)],
                    bprime=groups == 10,
                    paired_groups=slots,
                    minimum_forecast_seconds=minimum,
                    equal_time_cap_seconds=equal_budget if enable else None,
                    estimate=dict(
                        common=common,
                        target_tps_examples=1 / training_rates["target"],
                        score_seconds_per_row=score_per,
                        generation_seconds_per_row=gen,
                        engine_load_seconds=overhead,
                        dev_eval_seconds=ev(400),
                        blind_seconds_per_model=blind,
                        expected_target_train_seconds=train,
                        expected_dev_per_group_seconds=dev,
                    ),
                    frozen_unix=time.time(),
                    remaining_seconds=remaining,
                    deadline=self.a.deadline,
                    reserve_seconds=self.a.reserve_seconds,
                    micro_batch=self.a.micro_batch,
                    manifest_sha256=digest(self.prep / "manifest.json"),
                )
                write(path, design)
                return design
        raise RuntimeError(
            "Measured costs exceed the remaining budget for even the reduced paired A/B/C experiment; inspect benchmark before running"
        )

    def formal(self, cfg):
        a = self.a
        if cfg["micro_batch"] != a.micro_batch or cfg["manifest_sha256"] != digest(self.prep / "manifest.json"):
            raise ValueError("Frozen hardware/data settings differ")
        train = self.root / "frozen/train.jsonl"
        test = self.root / "frozen/test.jsonl"
        subset(self.prep / "train.jsonl", train, cfg["pool"])
        subset(self.prep / "test.jsonl", test, cfg["test"])
        bf = self.root / "frozen/bfcl"
        self.task(
            "freeze-bfcl",
            a.bfcl_python or a.training_python,
            "bfcl_subset",
            ["prepare", "--train", train, "--out", bf],
            [bf / "manifest.json", bf / "prompts.jsonl"],
        )
        self.task(
            "annotation-packet",
            a.training_python,
            "field_diagnostics",
            [
                "template",
                "--data",
                test,
                "--out",
                self.root / "frozen/annotation.jsonl",
                "--size",
                min(200, cfg["test"]),
            ],
            [self.root / "frozen/annotation.jsonl"],
        )
        lockpath = self.root / "locked-models.json"
        if lockpath.exists():
            if a.through == "all":
                frozen_models = read(self.root / "model-paths.json")
                self.blind(read(lockpath), frozen_models, test, bf)
            return
        proxy = self.root / "training/proxy"
        complete = self.train("proxy", "proxy", self.small, train, proxy)
        if a.through == "proxy":
            return
        steps = [0] + sorted(map(int, complete["adapter_sha256"]))
        scores = self.root / "scores"
        checkpoints = ["--checkpoint", "0=base"]
        for step in steps[1:]:
            checkpoints += ["--checkpoint", f"{step}={proxy}/checkpoint-{step}"]
        self.task(
            "score",
            a.training_python,
            "score_trajectories",
            [
                "--model",
                self.small,
                "--train",
                train,
                "--out",
                scores,
                "--swift-run-args",
                proxy / "args.json",
                *checkpoints,
            ],
            [scores / f"step-{n}.complete.json" for n in steps],
        )
        common = self.cost("proxy") + self.cost("score")
        dev = self.prep / "dev.jsonl"
        base_acc = self.evaluate("dev", "base", dev)["accuracy"]
        main = {"base": "base"}
        schedules = {}
        models = {"base": None}
        # Run every primary arm before spending any spare time on continuation.
        for seed in cfg["seeds"]:
            selected = self.root / "selection" / str(seed)
            self.task(
                "select-" + str(seed),
                a.training_python,
                "select_experiment",
                [
                    "--train",
                    train,
                    "--scores",
                    scores,
                    "--out",
                    selected,
                    "--seed",
                    seed,
                    "--steps",
                    *steps,
                    "--size",
                    cfg["subset"],
                ],
                [selected / "selection.json", *[selected / f"{g}.jsonl" for g in ("A", "B", "C", "Bprime")]],
            )
        groups = [(s, g) for s in cfg["seeds"] for g in "ABC"]
        if cfg["bprime"]:
            groups.append((42, "Bprime"))
        actual_groups = []
        for seed, group in groups:
            label = f"s{seed}-{group}"
            if (group == "A" and seed != 42) or group == "Bprime":
                # Allocation changes only at complete paired-round boundaries, using time alone.
                round_path = self.root / "rounds" / (label + ".json")
                if not round_path.exists():
                    count = 1 if group == "Bprime" else 3
                    observed = max(v["own_training_and_dev_seconds"] for v in schedules.values())
                    minimum = 1.3 * (
                        count * observed + (len(main) + count) * cfg["estimate"]["blind_seconds_per_model"]
                    )
                    write(
                        round_path,
                        {
                            "proceed": self.end - time.time() >= minimum,
                            "remaining": self.end - time.time(),
                            "required_seconds": minimum,
                            "rule": "reserve remaining blind tests before adding a full training round",
                        },
                    )
                if not read(round_path)["proceed"]:
                    break
            actual_groups.append((seed, group))
            data = self.root / "selection" / str(seed) / (group + ".jsonl")
            run = self.root / "training" / label
            comp = self.train("train-" + label, "target", self.big, data, run, seed, ["--pool", train])
            final = comp["final_step"]
            states = [r for r in rows(run / "steps.jsonl") if r["checkpoint_due"]]
            score_cost = 0.0
            history = [{"step": 0, "name": "base", "accuracy": base_acc, "cost": 0.0}]
            for state in states:
                step = state["step"]
                name = label + f"-step{step}"
                adapter = run / f"checkpoint-{step}"
                result = self.evaluate("dev", name, dev, adapter)
                score_cost += self.cost("eval-dev-" + name)
                history.append(
                    dict(
                        step=step,
                        name=name,
                        accuracy=result["accuracy"],
                        cost=(common if group != "A" else 0) + state["wall_seconds"] + score_cost,
                    )
                )
                models[name] = str(adapter)
            main[label] = label + f"-step{final}"
            if main[label] not in models:
                raise RuntimeError("Missing equitoken evaluation")
            schedules[label] = dict(
                history=history,
                own_training_and_dev_seconds=self.cost("train-" + label) + score_cost,
                initial_cost=self.cost("train-" + label) + score_cost + (common if group != "A" else 0),
                parent=str(run / f"checkpoint-{final}"),
                data=str(train if group == "A" else data),
            )
        if a.through == "train":
            write(
                self.root / "primary-progress.json", dict(main=main, schedules=schedules, training_complete=True)
            )
            return
        eq = {}
        for seed, group in actual_groups:
            if group == "Bprime" or not cfg["equal_time_cap_seconds"]:
                continue
            label = f"s{seed}-{group}"
            route = schedules[label]
            spent = route["initial_cost"]
            cap = cfg["equal_time_cap_seconds"]
            history = route["history"]
            parent = Path(route["parent"])
            patience = 0
            reference = history[-1]["accuracy"]
            chunk = 0
            if spent > cap:
                route["status"] = "initial target exceeded frozen time cap"
                continue
            while patience < 3:
                allowed = cap - spent
                task = f"continue-{label}-{chunk+1}"
                decision = self.root / "continuation" / (task + ".json")
                completed_chunk = (self.root / "receipts" / (task + ".json")).is_file()
                # Never cannibalize the primary blind-test reserve for a continuation.
                remaining_blind = (
                    (len(set(main.values())) + len(eq) + 1) * cfg["estimate"]["blind_seconds_per_model"] * 1.3
                )
                available = min(allowed, self.end - time.time() - remaining_blind)
                eval_s = cfg["estimate"]["dev_eval_seconds"] * 1.25
                update_s = 16 / (cfg["estimate"]["target_tps_examples"]) * 1.25
                n = min(50, int((available - eval_s - 90) / update_s))
                if not decision.exists() and n < 1:
                    break
                chunk += 1
                out = self.root / "training" / task
                if decision.exists():
                    d = read(decision)
                    n = d["steps"]
                    if not completed_chunk and time.time() >= d["deadline_unix"]:
                        break
                else:
                    d = {
                        "steps": n,
                        "allowed_seconds": allowed,
                        "start_unix": time.time(),
                        "parent": str(parent),
                        "deadline_unix": time.time() + available - eval_s - 30,
                    }
                    write(decision, d)
                comp = self.train(
                    task,
                    "continue",
                    self.big,
                    Path(route["data"]),
                    out,
                    seed,
                    ["--resume-from", parent, "--additional-steps", n, "--deadline-unix", d["deadline_unix"]],
                )
                step = comp["final_step"]
                name = label + f"-step{step}"
                parent = out / f"checkpoint-{step}"
                result = self.evaluate("dev", name, dev, parent)
                spent += self.cost(task) + self.cost("eval-dev-" + name)
                history.append(dict(step=step, name=name, accuracy=result["accuracy"], cost=spent))
                models[name] = str(parent)
                if result["accuracy"] >= reference + 0.01:
                    reference = result["accuracy"]
                    patience = 0
                else:
                    patience += 1
                if n < 50 or comp.get("deadline_reached"):
                    break
            eligible = [h for h in history if h["cost"] <= cap]
            best = max(eligible, key=lambda h: h["accuracy"])
            eq[label] = best["name"]
            route.update(
                spent=spent,
                cap=cap,
                eligible_best=best,
                status="patience" if patience == 3 else "budget",
                threshold=min(1, base_acc + 0.03),
                first_threshold=next((h for h in eligible if h["accuracy"] >= min(1, base_acc + 0.03)), None),
            )
        lock = {"main": main, "equal_time": eq}
        lockpath = self.root / "locked-models.json"
        write(
            self.root / "model-paths.json",
            {n: ([p, digest(Path(p) / "adapter_model.safetensors")] if p else None) for n, p in models.items()},
        )
        write(
            self.root / "training-ledger.json",
            dict(
                common_proxy_and_scoring_seconds=common,
                routes=schedules,
                equal_time_status=(
                    "evaluated if present in lock"
                    if cfg["equal_time_cap_seconds"]
                    else "omitted by preflight budget"
                ),
            ),
        )
        write(lockpath, lock)
        self.blind(lock, read(self.root / "model-paths.json"), test, bf)

    def blind(self, lock, models, test, bf):
        main, eq = lock["main"], lock["equal_time"]
        a = self.a
        for name in dict.fromkeys([*main.values(), *eq.values()]):
            adapter = Path(models[name][0]) if models[name] else None
            if adapter and digest(adapter / "adapter_model.safetensors") != models[name][1]:
                raise ValueError(f"Frozen adapter changed: {name}")
            self.evaluate("test", name, test, adapter)
            if name in main.values():
                self.evaluate("bfcl", name, bf / "prompts.jsonl", adapter)
                out = self.root / "evaluation/bfcl" / name
                self.task(
                    "score-bfcl-" + name,
                    a.bfcl_python or a.training_python,
                    "bfcl_subset",
                    [
                        "score",
                        "--predictions",
                        out / "predictions.jsonl",
                        "--manifest",
                        bf / "manifest.json",
                        "--out",
                        out,
                    ],
                    [out / "bfcl-summary.json", out / "bfcl-scores.jsonl"],
                )
        for label in main:
            if not label.endswith("-C"):
                continue
            pre = label[:-1]
            external = self.root / "evaluation/bfcl"
            comparison = self.root / "bfcl-comparisons" / label
            self.task(
                "compare-bfcl-" + label,
                a.bfcl_python or a.training_python,
                "bfcl_subset",
                [
                    "compare",
                    "--baseline",
                    external / main[pre + "B"] / "bfcl-scores.jsonl",
                    "--candidate",
                    external / main[label] / "bfcl-scores.jsonl",
                    "--out",
                    comparison,
                ],
                [comparison / "bfcl-comparison.json"],
            )
            folder = self.root / "evaluation/test"
            out = self.root / "field-diagnostics" / label
            self.task(
                "field-diagnostics-" + label,
                a.training_python,
                "field_diagnostics",
                [
                    "analyze",
                    "--data",
                    test,
                    "--out",
                    out,
                    "--b",
                    folder / main[pre + "B"],
                    "--c",
                    folder / main[label],
                ],
                [out / "field-diagnostics.json"],
            )
        self.task(
            "summary",
            a.training_python,
            "summarize_course",
            ["--course", self.root, "--test", test],
            [self.root / "summary/internal.json", self.root / "summary/aggregate-audit.zip"],
        )

    def cost(self, key):
        return read(self.root / "receipts" / (key + ".json"))["elapsed_s"]


def main():
    a = args_parser().parse_args()
    if a.reserve_seconds < 0:
        raise ValueError("Negative billing reserve")
    if not (Path.cwd() / "scripts/run_course_experiment.py").is_file():
        raise ValueError("Run from the repository root")
    runner = Protocol(a)

    def interrupted(sig, frame):
        raise KeyboardInterrupt(f"signal {sig}")

    signal.signal(signal.SIGTERM, interrupted)
    with (runner.root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for file, sha in read(runner.prep / "manifest.json")["sha256"].items():
            if digest(runner.prep / file) != sha:
                raise ValueError(f"Split/manifest mismatch: {file}")
        if time.time() >= runner.end:
            raise TimeoutError("Absolute deadline reached")
        bench = runner.bench()
        if a.through == "bench":
            return
        cfg = runner.freeze(bench)
        print(json.dumps(cfg, indent=2), flush=True)
        if a.through != "freeze":
            runner.formal(cfg)


if __name__ == "__main__":
    main()
