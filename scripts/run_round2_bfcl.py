"""Run the registered 36-model BFCL Live panel after all 20 new fits have finished.

All commands run from the repository root. The external panel has its own private
artifacts. Its evaluation never reads the internal test questions.

    python scripts/run_round2_bfcl.py freeze
    python scripts/run_round2_bfcl.py generate --eval-python /opt/evalenv/bin/python --limit 1 --hours 0.5
    python scripts/run_round2_bfcl.py generate --eval-python /opt/evalenv/bin/python --hours 4
    python scripts/run_round2_bfcl.py score --bfcl-python /opt/bfclenv/bin/python
    python scripts/run_round2_bfcl.py summarize

Supply --artifacts to point at an equivalent local mirror. The frozen native
cache and registrations must accompany the adapters. Each generation attempt
uses an isolated directory; a resume rechecks the completed inputs and outputs.
"""

import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import bfcl_live as live
from round2_holdout import ensure_task, exclusive, read, require, resolve_panel, write_once
from run_crossed_seeds import WEIGHTS, WEIGHTS_SHA256


REPO = Path(__file__).resolve().parents[1]
PUBLIC = REPO / "results/followup"
SCRIPTS = REPO / "scripts"
REG = PUBLIC / "round2-registration.json"
PANEL = PUBLIC / "round2-bfcl-live-panel.json"
DATA = Path("followup/round2/bfcl-live")
MODEL = Path("models/Qwen3.5-2B")
MODEL_IDS = ["base"] + [f"s{s}-{m}" for s in range(42, 47) for m in ("A", "B", "C", "E2", "E2b", "L-short", "L-long")]
EVAL_SHA = "ebcf9078ade3153631985ef422c8279a175af3c179c40d4d899f96840147c730"
CACHE_SHA = "ac6b13126c7cbaf702eb41b83ba784019888b05f9101ae64af7ccf71dbcc04f0"


def verify_cache(root, panel):
    manifest, prompts = root / "panel/manifest.json", root / "panel/prompts.jsonl"
    live.checked_sha(manifest, panel["manifest_sha256"])
    live.checked_sha(prompts, panel["prompts_sha256"])
    meta = read(root / "native-cache/prepare.json")
    cache = root / "native-cache/prompts.jsonl"
    live.checked_sha(cache, CACHE_SHA)
    require(meta["prompts_sha256"] == CACHE_SHA and meta["source_sha256"] == panel["prompts_sha256"]
            and meta["script_sha256"] == EVAL_SHA and meta["template_sha256"] == live.TEMPLATE_SHA
            and meta["tokenizer_sha256"] == live.TOKENIZER_SHA and meta["args"]["prepare_only"]
            and meta["args"]["generate_only"], "Native cache provenance differs")
    items = read(manifest)["items"]
    require([(x["id"], x["input_tokens"]) for x in items] ==
            [(r["id"], len(r["prompt_token_ids"])) for r in live.rows(cache)], "Cache lengths or IDs differ")
    return items


def stage_lock(local):
    reg = read(REG)
    ext = reg["external_bfcl"]
    panel = read(PANEL)
    require(ext["panel_order"] == MODEL_IDS and ext["revision"] == live.REVISION
            and tuple(ext["categories"]) == live.CATEGORIES and ext["context_limit"] == 8192
            and ext["max_generation_tokens"] == 512, "External registration changed")
    items = verify_cache(local / DATA, panel)
    require(len(items) == sum(panel["filter_counts"][c]["selected"] for c in live.CATEGORIES),
            "External denominators differ")
    live.checked_sha(SCRIPTS / "evaluate_experiment.py", EVAL_SHA)
    live.checked_sha(local / MODEL / WEIGHTS, WEIGHTS_SHA256)
    live.checked_sha(local / MODEL / "tokenizer.json", live.TOKENIZER_SHA)
    live.checked_sha(local / MODEL / "chat_template.jinja", live.TEMPLATE_SHA)
    # Reuse the installed checkpoint verifier. Its resolver reads only adapter
    # bytes, training receipts and endpoint IDs. It has no test-data input.
    locked = {e["id"]: e for e in resolve_panel(local, reg)}
    return {
        "registration_sha256": live.digest(REG), "public_panel_sha256": live.digest(PANEL),
        "manifest_sha256": panel["manifest_sha256"], "prompts_sha256": panel["prompts_sha256"],
        "cache_sha256": CACHE_SHA, "source_sha256": {name: live.digest(SCRIPTS / name) for name in (
            "run_round2_bfcl.py", "bfcl_live.py", "bfcl_subset.py", "evaluate_experiment.py", "verify_tool_call_loss.py",
            "round2_holdout.py", "run_crossed_seeds.py",
        )}, "weights_sha256": WEIGHTS_SHA256, "panel": [locked[tag] for tag in MODEL_IDS],
    }


def child_run(python, command, folder, lock_fd, deadline):
    """Propagate the time limit to vLLM workers and preserve the attempt log."""
    if deadline is not None and deadline <= time.monotonic():
        raise TimeoutError("Requested external generation window has ended")
    started = datetime.now(timezone.utc)
    tick = time.monotonic()
    with (folder / "process.log").open("x") as log:
        proc = subprocess.Popen([str(python), *map(str, command)], cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, "PATH": f"{python.parent}:{os.environ.get('PATH', '')}"},
                                pass_fds=(lock_fd,), start_new_session=True)
        try:
            rc = proc.wait(timeout=max(0, deadline - time.monotonic()) if deadline is not None else None)
        except BaseException:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            raise
    if rc:
        raise subprocess.CalledProcessError(rc, command)
    write_once(folder / "command.json", {
        "argv": [str(python), *map(str, command)], "started_utc": started.isoformat(),
        "completed_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - tick,
    })


def check_generation(folder, ep, lock, items):
    cfg = read(folder / "config.json")
    args = cfg["args"]
    require(cfg["prompts_sha256"] == lock["cache_sha256"] and cfg["source_sha256"] == lock["prompts_sha256"]
            and cfg["adapter_sha256"] == ep["adapter_sha256"] and cfg["model_revision"] == live.BASE_REVISION
            and cfg["tokenizer_sha256"] == live.TOKENIZER_SHA and cfg["template_sha256"] == live.TEMPLATE_SHA
            and cfg["script_sha256"] == EVAL_SHA and cfg["seed"] == 42 and cfg["greedy"]
            and cfg["dtype"] == "bfloat16" and cfg["language_model_only"] and cfg["stop_token_ids"] == [248046]
            and cfg["max_lora_rank"] == 16 and args["generate_only"] and args["enforce_eager"]
            and (args["max_tokens"], args["max_model_len"], args["batch_size"], args["gpu_memory_utilization"])
            == (512, 8192, 32, 0.85), f"Generation settings changed for {ep['id']}")
    versions = cfg["versions"]
    require(versions.get("vllm") == "0.19.1" and versions["torch"].split("+")[0] == "2.10.0"
            and versions["transformers"] == "5.24.0", "Generation package versions changed")
    predictions = live.keyed(folder / "predictions.jsonl")
    require(set(predictions) == set(items) and read(folder / "metrics.json")["count"] == len(items),
            "Incomplete external generations")
    for key, pred in predictions.items():
        require(pred["input_tokens"] == items[key]["input_tokens"] and isinstance(pred["raw_output"], str)
                and pred["reason"] in ("generated", "parse"), f"Invalid prediction metadata for {key}")
    proof = {"endpoint": ep["id"], "adapter_sha256": ep["adapter_sha256"], "manifest_sha256": lock["manifest_sha256"],
             "predictions_sha256": live.digest(folder / "predictions.jsonl"), "config_sha256": live.digest(folder / "config.json"),
             "count": len(items)}
    write_once(folder / "verified.json", proof)
    return proof


def check_scores(folder, gen, panel_sha, items):
    meta = read(folder / "bfcl-live-summary.json")
    require(meta["panel_sha256"] == panel_sha and meta["predictions_sha256"] == gen["predictions_sha256"]
            and meta["generation_config_sha256"] == gen["config_sha256"], "Official scores and generation disagree")
    rows = live.keyed(folder / "bfcl-live-scores.jsonl")
    require(rows.keys() == items.keys() and all(type(r["correct"]) is bool and
            r["category"] == items[k]["category"] for k, r in rows.items()), "Wrong scoring coverage")
    for category in live.CATEGORIES:
        subset = [r for r in rows.values() if r["category"] == category]
        c = meta["categories"][category]
        correct = sum(r["correct"] for r in subset)
        require(c["n"] == len(subset) and c["correct"] == correct and c["accuracy"] == correct / len(subset),
                "Score rows and category counts disagree")
    write_once(folder / "verified.json", {"generation": gen, "scores_sha256": live.digest(folder / "bfcl-live-scores.jsonl"),
                                          "summary_sha256": live.digest(folder / "bfcl-live-summary.json")})
    return meta["categories"]


def aggregate(summaries):
    require(list(summaries) == MODEL_IDS, "The descriptive summary requires the complete registered panel")
    by_method = {}
    differences = {}
    for c in live.CATEGORIES:
        denominator = summaries["base"][c]["n"]
        require(denominator > 0 and all(report[c]["n"] == denominator for report in summaries.values()),
                "Compared models must share the same nonempty category")
        means = {"base": summaries["base"][c]["accuracy"]}
        for arm in ("A", "B", "C", "E2", "E2b", "L-short", "L-long"):
            values = [summaries[f"s{s}-{arm}"][c]["correct"] for s in range(42, 47)]
            means[arm] = sum(values) / (5 * denominator)
        by_method[c] = means
        differences[c] = {f"{a}-{b}": 100 * (means[a] - means[b]) for a, b in (
            ("C", "B"), ("C", "A"), ("B", "A"), ("E2", "A"), ("E2b", "A"), ("L-short", "L-long"),
        )}
    return {"endpoint_category_counts": summaries, "mean_category_accuracy": by_method,
            "descriptive_differences_pp": differences}


def queue(args):
    local = args.artifacts.resolve()
    root = local / DATA
    with exclusive(root / "controller") as lock_fd:
        lock = stage_lock(local)
        lock_path = root / "controller/endpoints.json"
        if args.action == "freeze":
            write_once(lock_path, lock)
            print(f"Frozen {len(lock['panel'])} external endpoints; panel SHA-256 {lock['manifest_sha256']}")
            return
        require(read(lock_path) == lock, "External endpoint lock differs")
        (root / "attempts").mkdir(exist_ok=True)
        items = {r["id"]: r for r in read(root / "panel/manifest.json")["items"]}
        deadline = time.monotonic() + 3600 * args.hours if args.action == "generate" else None
        fresh = 0
        summaries = {}
        for ep in lock["panel"]:
            tag = ep["id"]
            generation = root / "generation" / tag
            scores = root / "scores" / tag
            if args.action == "generate":
                is_new = not generation.exists()
                if is_new and (fresh >= args.limit or time.monotonic() >= deadline):
                    break

                def make_generation(tmp):
                    cmd = [SCRIPTS / "evaluate_experiment.py", "--model", local / MODEL,
                           "--data", root / "panel/prompts.jsonl", "--prepared", root / "native-cache/prompts.jsonl",
                           "--out", tmp, "--generate-only", "--max-model-len", "8192", "--max-tokens", "512",
                           "--enforce-eager"]
                    if ep["relative_adapter"]:
                        cmd.extend(["--adapter", local / ep["relative_adapter"]])
                    child_run(args.eval_python, cmd, tmp, lock_fd, deadline)

                ensure_task(root, f"generation/{tag}", make_generation,
                            lambda tmp: check_generation(tmp, ep, lock, items))
                fresh += int(is_new)
                print(f"Verified generation {tag}", flush=True)
                continue
            if not generation.is_dir() and args.action == "score":
                continue
            gen = check_generation(generation, ep, lock, items)
            if args.action == "score":
                is_new = not scores.exists()
                if is_new and fresh >= args.limit:
                    break

                def make_scores(tmp):
                    cmd = [SCRIPTS / "bfcl_live.py", "--bfcl-root", args.bfcl_root.resolve(), "score",
                           "--panel", root / "panel", "--panel-sha256", lock["manifest_sha256"],
                           "--generation", generation, "--out", tmp]
                    child_run(args.bfcl_python, cmd, tmp, lock_fd, None)

                ensure_task(root, f"scores/{tag}", make_scores,
                            lambda tmp: check_scores(tmp, gen, lock["manifest_sha256"], items))
                fresh += int(is_new)
                print(f"Verified scores {tag}", flush=True)
            else:
                summaries[tag] = check_scores(scores, gen, lock["manifest_sha256"], items)
        if args.action == "summarize":
            write_once(root / "summary.json", {"scope": "descriptive external category accuracy",
                                               "stage_lock_sha256": live.digest(lock_path), **aggregate(summaries)})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("freeze", "generate", "score", "summarize"))
    parser.add_argument("--artifacts", type=Path, default=REPO / "local")
    parser.add_argument("--eval-python", type=Path, default=Path("/opt/evalenv/bin/python"))
    parser.add_argument("--bfcl-python", type=Path, default=Path("/opt/bfclenv/bin/python"))
    parser.add_argument("--bfcl-root", type=Path, default=live.DEFAULT_ROOT)
    parser.add_argument("--limit", type=int, default=36, help="Maximum new endpoint tasks in this invocation")
    parser.add_argument("--hours", type=float, help="Generation GPU time window starting in this process")
    args = parser.parse_args()
    if args.limit < 1 or (args.action == "generate" and
                          (args.hours is None or not math.isfinite(args.hours) or args.hours <= 0)):
        parser.error("Use a positive --limit and supply a finite positive --hours for generation")
    # Terminate the whole model-worker group as well as its controller.
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    queue(args)


if __name__ == "__main__":
    main()
