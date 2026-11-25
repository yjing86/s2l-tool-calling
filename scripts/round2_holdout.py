"""Freeze and run the single preregistered Round 2 internal evaluation.

preflight reads the public registration only. freeze binds the complete 47-endpoint
panel without inspecting the reserve. Publish the printed freeze SHA before release.
run and summarize verify that public commitment and the calendar before reading
the private reserve. See docs/round2-holdout.md for commands and failure recovery.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from evaluate_experiment import BASE_REVISION, fingerprint, judge, row_reader
from run_crossed_seeds import WEIGHTS, WEIGHTS_SHA256

REPO = Path(__file__).resolve().parents[1]
PUBLIC = Path("results/followup")
REG = PUBLIC / "round2-registration.json"
ANCHOR = PUBLIC / "round2-holdout-freeze.json"
SCRIPTS = ["round2_holdout.py", "round2_holdout_stats.py", "evaluate_experiment.py", "verify_tool_call_loss.py"]
RELEASE = datetime(2026, 12, 5, 8, tzinfo=timezone.utc)  # midnight in Los Angeles
OUT = Path("followup/round2/holdout-evaluation")
TRAIN = Path("followup/round2")
HIST = Path("followup/phase1")
MODEL = Path("models/Qwen3.5-2B")
ARGS = Path("runs/target-smoke/args.json")
ADAPTER = "adapter_model.safetensors"
CONFIG = "adapter_config.json"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def write_once(path, value):
    """Publish a complete payload once; concurrent identical retries are harmless."""
    data = encoded(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        try:
            os.link(tmp.name, path)
        except FileExistsError:
            require(path.read_bytes() == data, f"Existing receipt or freeze differs: {path.name}")


def utc_now():
    return datetime.now(timezone.utc)


def require_release(now):
    require(now.tzinfo is not None and now >= RELEASE,
            f"Holdout is sealed until {RELEASE.isoformat()} (Dec 5 in Los Angeles).")


def registration():
    reg = read(REPO / REG)
    cfg = reg["holdout_v2"]
    opens = datetime.fromisoformat(cfg["not_before_utc"].replace("Z", "+00:00"))
    require(opens == RELEASE, "Registration opening time differs")
    expected_old = {"base", "full-s42-step125", "full-s42-step500"}
    expected_old.update(f"s{s}-{m}" for s in (42, 43, 44)
                        for m in ("A", "B", "C", "Bprime", "E2", "E2b", "E3", "E4"))
    expected_new = {f"s{s}-{m}" for s in (45, 46) for m in ("A", "B", "C", "E2", "E2b")}
    expected_new.update(f"s{s}-{m}" for s in range(42, 47) for m in ("L-short", "L-long"))
    panel = cfg["panel_order"]
    require(len(reg["cell_order"]) == 20 and set(reg["cell_order"]) == expected_new,
            "Incomplete new training roster")
    require(len(cfg["existing_endpoints"]) == 27 and
            {r["id"] for r in cfg["existing_endpoints"]} == expected_old, "Historical roster differs")
    require(len(panel) == 47 and set(panel) == expected_new | expected_old, "Evaluation roster differs")
    return reg


def lengths(local, spec):
    old = local / HIST / "selection"
    old_spec = read(old / "manifest.json")
    require(spec["phase1_selection_sha256"] == fingerprint(old / "manifest.json"),
            "Historical selection manifest changed")
    result = {}
    for seed in range(42, 47):
        for method in ("A", "E2", "L-short", "L-long"):
            tag = f"s{seed}-{method}"
            if tag in spec["cells"]:
                mean = spec["cells"][tag]["mean_value_tokens"]
            else:
                src = old / str(seed) / "selection.json"
                require(fingerprint(src) == old_spec["selection_manifest_sha256"][str(seed)],
                        f"Historical length provenance mismatch: {tag}")
                d = read(src)
                metrics = d["primary_controls"]["A"] if method == "A" else d["subsets"][method]
                require(metrics["requests"] == 1000, f"Length denominator differs: {tag}")
                mean = metrics["token_totals"]["value_tokens"] / metrics["requests"]
            require(type(mean) in (float, int) and math.isfinite(mean) and mean > 0, f"Invalid mean: {tag}")
            result[tag] = mean
    return result


def adapter_files(path):
    files = {p.name: fingerprint(p) for p in path.glob("adapter*") if p.is_file()}
    require(set(files) == {ADAPTER, CONFIG}, f"Expected pinned LoRA weights and configuration in {path}")
    return files


def resolve_panel(artifacts, reg):
    cfg = reg["holdout_v2"]
    old = {x["id"]: x for x in cfg["existing_endpoints"]}
    new = read(artifacts / TRAIN / "locked-models.json")
    require(set(new) == set(reg["cell_order"]), "All 20 registered new training locks are required")
    execution = read(artifacts / TRAIN / "execution.json")
    require(execution["status"] == "complete" and set(execution["completed"]) == set(new)
            and not execution["missing"], "Training records are incomplete")
    panel = []
    for tag in cfg["panel_order"]:
        if tag == "base":
            panel.append({"id": tag, "checkpoint": tag, "relative_adapter": None, "adapter_sha256": {}})
            continue
        if tag in old:
            entry = old[tag]
            cp = entry["checkpoint"]
            expected = entry["adapter_sha256"]
            if tag.startswith("full-"):
                directory = Path("full-reference/training")
            else:
                seed, method = tag.split("-", 1)
                if method in ("A", "B", "C") or tag == "s42-Bprime":
                    origin = Path("course-run")
                elif method == "E2b":
                    origin = Path("followup/difficulty-control")
                else:
                    origin = HIST
                directory = origin / "training" / tag
        else:
            lock = read(artifacts / TRAIN / "locks" / f"{tag}.json")
            cp = lock["checkpoint"]
            require(lock["label"] == tag and cp == new[tag] == f"{tag}-step{lock['step']}"
                    and lock["target_tokens"] == 3005228, f"New endpoint lock differs: {tag}")
            expected = lock["final_adapter_sha256"]
            directory = TRAIN / "training" / tag
        prefix = "full-s42" if tag.startswith("full-") else tag
        match = re.fullmatch(re.escape(prefix) + r"-step([1-9][0-9]*)", cp)
        require(match is not None, f"Invalid locked checkpoint: {tag}")
        relative = directory / f"checkpoint-{match.group(1)}"
        digests = adapter_files(artifacts / relative)
        require(digests[ADAPTER] == expected, f"Wrong LoRA weights: {tag}")
        if tag in old:
            require(digests[CONFIG] == entry["adapter_config_sha256"], f"Wrong historical LoRA config: {tag}")
        panel.append({"id": tag, "checkpoint": cp, "relative_adapter": relative.as_posix(),
                      "adapter_sha256": digests})
    require(len({e["adapter_sha256"].get(ADAPTER) for e in panel}) == len(panel),
            "Two registered endpoints resolve to the same weights")
    return panel


def pinned_versions(metadata):
    return {key: value for key, value in metadata.items() if key not in ("platform", "python")}


def build_freeze(artifacts, reg):
    selections_path = artifacts / TRAIN / "selection/manifest.json"
    require(fingerprint(selections_path) == reg["selection"]["manifest_sha256"], "Pretraining selection manifest differs")
    spec = read(selections_path)
    sources = {name: fingerprint(REPO / "scripts" / name) for name in SCRIPTS}
    old_protocol = read(REPO / "results/turnstile-single-call.json")["internal_generation_protocol"]
    require(sources["evaluate_experiment.py"] == old_protocol["script_sha256"], "Original scorer changed")
    model = artifacts / MODEL
    files = {p.name: fingerprint(p) for p in model.iterdir() if p.is_file()
             and p.suffix in {".safetensors", ".json", ".txt", ".jinja"}}
    require(files[WEIGHTS] == WEIGHTS_SHA256 and files["chat_template.jinja"] == old_protocol["template_sha256"]
            and files["tokenizer.json"] == old_protocol["tokenizer_sha256"], "Pinned target weights/template differ")
    reserve = read(REPO / PUBLIC / "phase1-registration.json")["prospective_holdout"]
    require(reserve["requests"] == reg["holdout_v2"]["requests"] == 1500 and
            reserve["sha256"] == reg["holdout_v2"]["original_file_sha256"],
            "Invalid published reserve fingerprint")
    prep = read(artifacts / "course-run/prompts/test/prepare.json")
    evaluation = read(artifacts / "course-run/evaluation/test/base/config.json")
    require(prep["script_sha256"] == evaluation["script_sha256"] == old_protocol["script_sha256"]
            and fingerprint(artifacts / ARGS) == "7cc9b2ea92a8054b9dad6782ab210ff89b96ebf4f9642068f5f4c7a594f1e876",
            "Original template evidence or effective training arguments changed")
    return {"schema": 1, "not_before_utc": reg["holdout_v2"]["not_before_utc"],
            "registration_sha256": fingerprint(REPO / REG), "reserve_sha256": reserve["sha256"],
            "requests": 1500, "selection_sha256": fingerprint(selections_path), "source_sha256": sources,
            "preparation_versions": pinned_versions(prep["prepare_versions"]), "template_kwargs": prep["template_kwargs"],
            "evaluation_versions": pinned_versions(evaluation["versions"]),
            "target_files_sha256": files, "template_args_sha256": fingerprint(artifacts / ARGS),
            "target_revision": BASE_REVISION, "length_means": lengths(artifacts, spec),
            "panel": resolve_panel(artifacts, reg)}


def anchor_document(frozen, sha):
    return {"schema": 1, "panel_count": len(frozen["panel"]), "private_freeze_sha256": sha,
            "pretraining_registration_sha256": frozen["registration_sha256"],
            "source_sha256": frozen["source_sha256"], "reserve_sha256": frozen["reserve_sha256"],
            "not_before_utc": frozen["not_before_utc"]}


def verify_anchor(commit, anchor):
    """The given commit must carry this exact digest on the fetched public main branch."""
    require(commit and re.fullmatch("[0-9a-f]{40}", commit), "Supply the full public freeze commit")
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "refs/remotes/origin/main"],
                   cwd=REPO, check=True)
    result = subprocess.run(["git", "show", f"{commit}:{ANCHOR.as_posix()}"], cwd=REPO,
                            check=True, stdout=subprocess.PIPE)
    require(json.loads(result.stdout) == anchor, "Published freeze commitment differs")
    reg = subprocess.check_output(["git", "show", f"{commit}:{REG.as_posix()}"], cwd=REPO)
    require(hashlib.sha256(reg).hexdigest() == anchor["pretraining_registration_sha256"],
            "Published registration differs from private freeze")


def inspect_released(data, frozen):
    """Only the public date/commit gate calls this in the real workflow."""
    require(fingerprint(data) == frozen["reserve_sha256"], "Reserve file differs from public commitment")
    rows = row_reader(data)
    groups = {str(r["id"]): str(r["group_id"]) for r in rows}
    require(len(rows) == len(groups) == len(set(groups.values())) == frozen["requests"] and
            all(r["group_id"] not in (None, "") for r in rows), "Reserve membership or groups differ")
    metadata = read(data.parent / "manifest.json")
    def ordered(key):
        return hashlib.sha256("".join(f"{r[key]}\n" for r in rows).encode()).hexdigest()
    require(metadata["test_sha256"] == frozen["reserve_sha256"]
            and metadata["ordered_ids_sha256"] == ordered("id")
            and metadata["ordered_groups_sha256"] == ordered("group_id")
            and metadata["selected_groups"] == frozen["requests"], "Reservation metadata differs")
    return {str(r["id"]): r for r in rows}, groups


@contextmanager
def exclusive(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".gate.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield lock.fileno()


def task_args(data, model, out, *, prompts=None, adapter=None, template_args=None):
    command = [str(REPO / "scripts/evaluate_experiment.py"), "--model", str(model), "--data", str(data),
               "--out", str(out), "--max-model-len", "4096", "--max-tokens", "512"]
    if prompts is None:
        return [*command, "--prepare-only", "--swift-run-args", str(template_args)]
    if adapter is not None:
        command.extend(["--adapter", str(adapter)])
    return [*command, "--prepared", str(prompts), "--enforce-eager"]


def launch(program, args, folder, lock_fd):
    env = {**os.environ, "PATH": f"{program.parent}:{os.environ.get('PATH', '')}"}
    with (folder / "process.log").open("x") as stream:
        subprocess.run([str(program), *args], env=env, stdout=stream, stderr=subprocess.STDOUT,
                       pass_fds=(lock_fd,), check=True)


def ensure_task(out, name, execute, validate):
    """Only promote complete validated attempts, preserving failed attempts for audit."""
    final = out / name
    if final.exists():
        return validate(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    prefix = name.replace("/", "-") + "."
    for candidate in sorted((out / "attempts").glob(prefix + "*")):
        # Power loss between validation and promotion can leave a fully usable
        # attempt. Incomplete or unparseable attempts remain available for audit.
        try:
            result = validate(candidate)
        except (FileNotFoundError, ValueError, KeyError, TypeError):
            continue
        candidate.rename(final)
        return result
    tmp = Path(tempfile.mkdtemp(prefix=prefix, dir=out / "attempts"))
    execute(tmp)
    result = validate(tmp)
    tmp.rename(final)
    return result


def verify_preparation(folder, frozen, reference):
    prompts = folder / "prompts.jsonl"
    meta = read(folder / "prepare.json")
    require(meta["prompts_sha256"] == fingerprint(prompts) and
            meta["source_sha256"] == frozen["reserve_sha256"] and
            meta["script_sha256"] == frozen["source_sha256"]["evaluate_experiment.py"] and
            meta["tokenizer_sha256"] == frozen["target_files_sha256"]["tokenizer.json"] and
            meta["template_sha256"] == frozen["target_files_sha256"]["chat_template.jinja"] and
            pinned_versions(meta["prepare_versions"]) == frozen["preparation_versions"] and
            meta["template_kwargs"] == frozen["template_kwargs"] and
            meta["args"]["prepare_only"] and not meta["args"]["generate_only"], "Prompt preparation differs")
    observed = [str(r["id"]) for r in row_reader(prompts)]
    require(observed == list(reference), "Prepared prompt IDs/order differ")
    return meta["prompts_sha256"]


def audit_predictions(folder, endpoint, frozen, prompt_sha, reference):
    cfg = read(folder / "config.json")
    args = cfg["args"]
    require(cfg["source_sha256"] == frozen["reserve_sha256"] and cfg["prompts_sha256"] == prompt_sha
            and cfg["adapter_sha256"] == endpoint["adapter_sha256"] and cfg["model_revision"] == BASE_REVISION
            and cfg["script_sha256"] == frozen["source_sha256"]["evaluate_experiment.py"]
            and cfg["tokenizer_sha256"] == frozen["target_files_sha256"]["tokenizer.json"]
            and cfg["template_sha256"] == frozen["target_files_sha256"]["chat_template.jinja"]
            and pinned_versions(cfg["versions"]) == frozen["evaluation_versions"]
            and cfg["greedy"] and cfg["seed"] == 42 and cfg["dtype"] == "bfloat16" and cfg["language_model_only"]
            and cfg["stop_token_ids"] == [248046] and cfg["max_lora_rank"] == 16
            and args["enforce_eager"] and not args["generate_only"] and args["max_tokens"] == 512
            and args["max_model_len"] == 4096 and args["batch_size"] == 32
            and args["gpu_memory_utilization"] == 0.85, f"Generation protocol differs: {endpoint['id']}")
    values, counts = {}, Counter()
    for pred in row_reader(folder / "predictions.jsonl"):
        key = str(pred["id"])
        require(key in reference and key not in values and
                str(pred["group_id"]) == str(reference[key]["group_id"]), "Incorrect/duplicate request")
        result = judge(reference[key], pred["raw_output"])
        require(type(pred["correct"]) is bool and pred["correct"] == result["correct"]
                and pred["reason"] == result["reason"], "Raw generation fails independent re-scoring")
        values[key] = result["correct"]
        counts[result["reason"]] += 1
    metrics = read(folder / "metrics.json")
    require(values.keys() == reference.keys() and metrics["count"] == len(reference)
            and metrics["correct"] == sum(values.values()) and metrics["reason_counts"] == dict(counts),
            "Incomplete or inconsistent generation metrics")
    return values


def released_run(args, reg):
    # Calendar and public integrity checks precede even stat/open of the private reserve.
    require_release(utc_now())
    out = args.artifacts / OUT
    frozen = read(out / "freeze.json")
    freeze_sha = fingerprint(out / "freeze.json")
    anchor = anchor_document(frozen, freeze_sha)
    verify_anchor(args.anchor_commit, anchor)
    require(build_freeze(args.artifacts, reg) == frozen, "Staged weights, inputs or code differ from published freeze")
    receipt = {"public_commit": args.anchor_commit, "private_freeze_sha256": freeze_sha}
    with exclusive(out) as lock_fd:
        if args.action == "summarize":
            require(read(out / "opened.json") == receipt, "A matching run must already be open")
            require(all((out / "predictions" / e["id"]).is_dir() for e in frozen["panel"]),
                    "Refusing to summarize an incomplete endpoint panel")
        else:
            write_once(out / "opened.json", receipt)
        data = args.artifacts / "followup/holdout-v2/test.jsonl"
        references, groups = inspect_released(data, frozen)
        model = args.artifacts / MODEL
        (out / "attempts").mkdir(exist_ok=True)
        if args.action == "summarize":
            prompt_sha = verify_preparation(out / "prompts", frozen, references)
        else:
            def prepare(path):
                launch(args.training_python, task_args(data, model, path,
                       template_args=args.artifacts / ARGS), path, lock_fd)
            prompt_sha = ensure_task(out, "prompts", prepare, lambda p: verify_preparation(p, frozen, references))
        values, prediction_hashes = {}, {}
        for index, endpoint in enumerate(frozen["panel"], start=1):
            tag = endpoint["id"]
            adapter = args.artifacts / endpoint["relative_adapter"] if endpoint["relative_adapter"] else None
            destination = out / "predictions" / tag
            def verify(folder):
                return audit_predictions(folder, endpoint, frozen, prompt_sha, references)
            if args.action == "run":
                def generate(path):
                    launch(args.eval_python, task_args(data, model, path, prompts=out / "prompts/prompts.jsonl",
                           adapter=adapter), path, lock_fd)
                values[tag] = ensure_task(out, "predictions/" + tag, generate, verify)
                print(f"Verified endpoint {index}/{len(frozen['panel'])} ({tag})", flush=True)
            else:
                values[tag] = verify(destination)
            prediction_hashes[tag] = fingerprint(destination / "predictions.jsonl")
        # Fail closed on statistical dependencies, leaving complete inference reusable.
        from round2_holdout_stats import summarize
        summary = summarize(reg, values, groups, frozen["length_means"])
        summary["provenance"] = {**receipt, "reserve_sha256": frozen["reserve_sha256"],
                                 "prompts_sha256": prompt_sha, "predictions_sha256": prediction_hashes}
        write_once(out / "summary.json", summary)
        print(f"Full panel verified. Private summary SHA-256: {fingerprint(out / 'summary.json')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["preflight", "freeze", "run", "summarize"])
    parser.add_argument("--artifacts", type=Path, default=REPO / "local",
                        help="Staged local/ tree with historic runs, round2, pinned base, and original template args")
    parser.add_argument("--anchor-commit", help="Full Git commit containing the published panel freeze SHA-256")
    parser.add_argument("--training-python", type=Path, default=Path("/opt/trainenv/bin/python"))
    parser.add_argument("--eval-python", type=Path, default=Path("/opt/evalenv/bin/python"))
    args = parser.parse_args()
    reg = registration()
    if args.action == "preflight":
        print(json.dumps({"registered_endpoints": len(reg["holdout_v2"]["panel_order"]),
                          "registered_new_fits": len(reg["cell_order"]), "opens_utc": RELEASE.isoformat(),
                          "private_files_accessed": False, "preregistration_sha256": fingerprint(REPO / REG)}))
    elif args.action == "freeze":
        data = build_freeze(args.artifacts, reg)
        out = args.artifacts / OUT
        with exclusive(out):
            require(not (out / "opened.json").exists(), "Cannot freeze after opening")
            write_once(out / "freeze.json", data)
            sha = fingerprint(out / "freeze.json")
            write_once(REPO / ANCHOR, anchor_document(data, sha))
        print(f"Freeze SHA-256: {sha}. Publish {ANCHOR} before opening the reserve.")
    else:
        released_run(args, reg)


if __name__ == "__main__":
    main()
