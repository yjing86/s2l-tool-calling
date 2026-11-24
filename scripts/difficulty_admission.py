"""Verify the external backup and replay evidence required before the optional E2b batch.

Run `attest` on the backup machine after its four full-tree transfers have finished.
Transport the small proof to the training host, retain its printed SHA-256 separately,
and use `verify` there. Proofs contain digests and counts, never requests or scores.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from tempfile import TemporaryDirectory

from evaluate_experiment import judge
from types import SimpleNamespace

from run_followup import check_outputs, completed_cells
from summarize_course import digest, read_json


COURSES = {"primary": "course-run", "full": "full-reference", "cross": "crossed-seeds", "phase1": "followup/phase1"}
SPLIT_SIZES = {"dev": 400, "test": 2000}
SEEDS = (42, 43, 44)
PRIMARY = {"base", "s42-Bprime"} | {f"s{s}-{m}" for s in SEEDS for m in "ABC"}
PHASE1 = {f"s{s}-{m}" for s in SEEDS for m in ("Bprime", "E2", "E3", "E4")} - {"s42-Bprime"}
CROSS = {f"select{s}-train{t}-{m}" for s, t in ((42, 43), (43, 42)) for m in "BC"}
DEFAULT_CUTOFF = "2026-11-24T20:29:00Z"
MAX_OBSERVATION_AGE_S = 600
CASH_RESERVE_S = 600


def require(condition, message):
    if not condition:
        raise ValueError(message)


def object_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def timestamp(value):
    if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    when = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    require(when.utcoffset() is not None, "Times must specify a UTC offset")
    return when.timestamp()


def money(obj):
    try:
        value = Decimal(str(obj))
    except InvalidOperation as error:
        raise ValueError("Invalid provider monetary observation") from error
    require(value.is_finite() and value >= 0, "Invalid provider monetary observation")
    return value


def account_seconds(path, now, cutoff=None):
    """Return the prepaid runtime remaining after observed burn and a cash stop reserve."""
    text = Path(path).read_text().strip()
    try:
        obs = json.loads(text)
    except json.JSONDecodeError:
        obs = json.loads(text.splitlines()[-1])  # Append-only private observation log.
    require(isinstance(obs, dict) and bool(str(obs.get("source") or "").strip()), "Missing provider observation/source")
    now = timestamp(now)
    age = now - timestamp(obs["utc"])
    require(0 <= age <= MAX_OBSERVATION_AGE_S, "Refresh the account observation within the last ten minutes")
    require(timestamp(obs["hard_cutoff_utc"]) == timestamp(DEFAULT_CUTOFF if cutoff is None else cutoff),
            "Account observation belongs to a different registered cutoff")
    rate = money(obs["account_spend_usd_per_hour"])
    pod_rate = money(obs["displayed_project_gpu_usd_per_hour"])
    require(rate > 0 and pod_rate > 0, "Positive total-account and current Pod rates are required")
    # Include all contemporaneous account spend, even if the displayed single-Pod rate is lower.
    funded = 3600 * money(obs["balance_usd"]) / max(rate, pod_rate)
    available = funded - Decimal(str(age)) - CASH_RESERVE_S
    return max(0, int(available.to_integral_value(rounding=ROUND_FLOOR)))


def safe_rel(name):
    name = str(name)
    p = PurePosixPath(name)
    require(name and not p.is_absolute() and name == str(p) and name != "." and ".." not in p.parts and "\\" not in name,
            f"Invalid archive path {name!r}")
    return name


def validate_receipt(path, root, course, necessary):
    """Verify every physical file recorded by the completed full-tree transfer, independently of its status flag."""
    rec = read_json(path)
    require(rec["course"] == course and rec["status"] == "complete" and rec["full_tree"] is True
            and not rec.get("lock_stage", False) and not rec.get("closed_checkpoints", False), "Wrong mirror mode")
    files = rec["files"]
    require(isinstance(files, dict) and len(files) == rec["inventory_count"] and files, "Incomplete mirror inventory")
    require(set(necessary) <= set(files), f"{course}: mirror predates completed endpoints")
    for name, recorded in files.items():
        file = root / safe_rel(name)
        require(not file.is_symlink() and file.resolve() == file and file.is_file(), f"Missing plain backup file: {name}")
        require(file.stat().st_size == recorded["bytes"] and digest(file) == recorded["sha256"],
                f"Independent backup checksum mismatch: {course}/{name}")
    require(all(files[n]["sha256"] == h for n, h in necessary.items()), "Snapshot differs from audited outputs")
    return {"receipt_sha256": digest(path), "inventory_sha256": object_sha(files),
            "file_sizes_sha256": object_sha({n: r["bytes"] for n, r in files.items()}),
            "symbolic_links_sha256": object_sha(rec["symbolic_links"]), "files": len(files),
            "verified_bytes": sum(v["bytes"] for v in files.values())}


def live_inventory(root):
    """Compare the live names, byte lengths and link targets with the full-tree transfer inventory.

    Scientific output bytes are checked by collect(). The backup machine independently
    rehashes every file, including optimizer checkpoints.
    """
    root = Path(root).resolve()
    sizes, links = {}, {}
    for parent, dirs, files in os.walk(root, followlinks=False):
        directory = Path(parent)
        for name in [*dirs, *files]:
            f = directory / name
            if f.is_symlink():
                target = f.resolve()
                links[f.relative_to(root).as_posix()] = target.relative_to(root).as_posix() if target.is_relative_to(root) else "OUTSIDE"
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__") and not (directory / d).is_symlink()]
        for name in files:
            f = directory / name
            if f.is_file() and not f.is_symlink():
                sizes[f.relative_to(root).as_posix()] = f.stat().st_size
    return object_sha(sizes), object_sha(links)


def latest_transfer(directory, course, required):
    for path in sorted(directory.glob("bulk-*.json"), reverse=True):
        rec = read_json(path)
        if (rec.get("course") == course and rec.get("status") == "complete"
                and rec.get("full_tree") is True and set(required) <= set(rec.get("files", {}))):
            return path
    raise ValueError(f"No completed, terminal full-tree transfer of {course}")


def locked_name(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]+", value), "Invalid frozen model checkpoint")
    return value


def collect(roots):
    """Bind terminal run files and each raw prediction to its own generating receipt and lock."""
    paths = {k: Path(roots[k]).resolve() for k in COURSES}
    witness = {k: {} for k in COURSES}
    jobs = []

    def bind(study, name):
        name = safe_rel(name)
        root = paths[study]
        f = root / name
        require(f.is_file() and f.resolve() == f, f"Missing terminal file: {study}/{name}")
        witness[study][name] = digest(f)
        return f

    def terminal(study, task, files):
        root = paths[study]
        bind(study, f"receipts/{task}.json")
        check_outputs(root, task, [bind(study, name) for name in files])

    def predictions(study, split, model, task):
        folder = f"evaluation/{split}/{locked_name(model)}"
        names = [f"{folder}/{name}" for name in ("predictions.jsonl", "metrics.json")]
        terminal(study, task, names)
        cfg = bind(study, f"{folder}/config.json")
        require(read_json(cfg)["script_sha256"] == digest(Path(__file__).with_name("evaluate_experiment.py")),
                "Evaluation implementation differs from the archived scorer")
        jobs.append((study, split, model))

    terminal("primary", "summary", ["summary/internal.json", "summary/aggregate-audit.zip"])
    primary = read_json(bind("primary", "locked-models.json"))["main"]
    require(set(primary) == PRIMARY and len(set(primary.values())) == len(PRIMARY), "Missing primary endpoints")
    bind("primary", "frozen/test.jsonl")
    for model in primary.values():
        predictions("primary", "test", model, "eval-test-" + model)
        # Store the two official BFCL file hashes to bind the external replay evidence.
        prefix = f"evaluation/bfcl/{model}"
        terminal("primary", "score-bfcl-" + model, [prefix + "/bfcl-scores.jsonl", prefix + "/bfcl-summary.json"])
        bind("primary", prefix + "/predictions.jsonl")

    terminal("full", "full-summary", ["summary/full-reference.json"])
    full = read_json(bind("full", "full-models.json"))
    require(set(full) == {"final", "matched_tokens", "best_dev"} and full["final"] != "base"
            and full["matched_tokens"] != "base", "Full-pool reference did not finish")
    for model in sorted(set(full.values()) - {"base"}):
        predictions("full", "test", model, "eval-test-" + model)

    terminal("cross", "summary", ["summary.json", "summary.md"])
    cross = read_json(bind("cross", "crossed-models.json"))
    require(set(cross) == CROSS and len(set(cross.values())) == 4, "Missing crossed-seed endpoints")
    for model in cross.values():
        tag, suffix = locked_name(model).rsplit("-step", 1)
        require(suffix.isdigit(), "Invalid crossed endpoint")
        predictions("cross", "dev", model, "dev-" + tag)

    execution = read_json(bind("phase1", "execution.json"))
    prereg = read_json(bind("phase1", "preregistration.json"))["spec"]
    observed = read_json(bind("phase1", "locked-models.json"))
    require(execution["status"] == "complete" and execution["missing"] == []
            and execution["completed"] == execution["planned"] == prereg["cell_order"]
            and len(execution["completed"]) == 11 and set(observed) == set(execution["completed"]) == PHASE1,
            "All eleven Phase 1 endpoints are required")
    # The registered runner checks the adapters against their cell locks and all 33 task receipts.
    checked = completed_cells(SimpleNamespace(root=paths["phase1"]), prereg)
    require(checked == observed, "Phase 1 locks and completed cells differ")
    for tag, model in observed.items():
        for name in (f"receipts/train-{tag}.json", f"cells/{tag}.json", f"locks/{tag}.json"):
            bind("phase1", name)
        for split in SPLIT_SIZES:
            predictions("phase1", split, model, split + "-" + tag)
    return witness, jobs


def replay(folder, gold, source_sha):
    """Regrade raw strings against the independent frozen references and verify every stored verdict."""
    config = read_json(folder / "config.json")
    require(config["source_sha256"] == source_sha and config["args"]["generate_only"] is False, "Wrong split or mode")
    seen, reasons = set(), Counter()
    correct = 0
    with (folder / "predictions.jsonl").open() as lines:
        for line in lines:
            row = json.loads(line)
            key = str(row["id"])
            require(key not in seen and key in gold and row["group_id"] == gold[key]["group_id"], "Unpaired request")
            seen.add(key)
            result = judge(gold[key], row["raw_output"])
            require(all(k in row and object_sha(row[k]) == object_sha(v) for k, v in result.items()),
                    "Raw model output does not reproduce the saved verdict")
            correct += result["correct"]
            reasons[result["reason"]] += 1
    metrics = read_json(folder / "metrics.json")
    require(seen == set(gold) and metrics["count"] == len(seen) and metrics["correct"] == correct
            and metrics["reason_counts"] == reasons and metrics["accuracy"] == correct / len(seen),
            "Raw replay or aggregate disagrees with the completed evaluation")
    return len(seen)


def archived_bfcl_audits(directory, primary):
    lock = primary / "locked-models.json"
    receipts = {}
    for model in read_json(lock)["main"].values():
        path = directory / (model + ".json")
        audit = read_json(path)
        bfcl = audit["bfcl"]
        expected = primary / "evaluation/bfcl" / model
        require(audit["model"] == bfcl["model"] == model and audit["status"] == "complete"
                and audit["lock_sha256"] == bfcl["lock_sha256"] == digest(lock)
                and bfcl["full_summary_equal"] is True and bfcl["official_rows_replayed"] == bfcl["n"] == 200
                and bfcl["predictions_sha256"] == digest(expected / "predictions.jsonl")
                and bfcl["scores_sha256"] == digest(expected / "bfcl-scores.jsonl")
                and bfcl["receipt_sha256"]["score"] == digest(primary / "receipts" / f"score-bfcl-{model}.json"),
                f"Official BFCL replay did not verify: {model}")
        receipts[model] = digest(path)
    return object_sha(receipts)


def counts_for(jobs):
    counts = {k: Counter() for k in COURSES}
    for study, split, _ in jobs:
        counts[study][split] += SPLIT_SIZES[split]
    return {k: dict(v) for k, v in counts.items()}


def attest(mirror_root, receipt_dir, dev):
    roots = {k: (mirror_root / v).resolve() for k, v in COURSES.items()}
    witness, jobs = collect(roots)
    copies = {k: validate_receipt(latest_transfer(receipt_dir, name, witness[k]), roots[k], name, witness[k])
              for k, name in COURSES.items()}
    datasets = {"test": roots["primary"] / "frozen/test.jsonl", "dev": dev}
    gold = {}
    for split, path in datasets.items():
        with path.open() as f:
            records = [json.loads(line) for line in f]
        indexed = {str(row["id"]): row for row in records}
        require(len(records) == len(indexed) == SPLIT_SIZES[split], f"Wrong frozen {split} split")
        gold[split] = (indexed, digest(path))
    for study, split, model in jobs:
        replay(roots[study] / "evaluation" / split / model, *gold[split])
    return {"version": 1, "verifier_sha256": digest(Path(__file__)), "terminal_tree_sha256": {k: object_sha(v) for k,v in witness.items()},
            "mirror_receipt_sha256": {k: v["receipt_sha256"] for k,v in copies.items()}, "mirrors": copies,
            "primary_official_bfcl_audits_sha256": archived_bfcl_audits(receipt_dir / "model-audits", roots["primary"]),
            "replayed_requests": counts_for(jobs)}


def verify_attestation(path, roots):
    proof = read_json(Path(path))
    witness, jobs = collect(roots)
    require(proof["version"] == 1 and proof["verifier_sha256"] == digest(Path(__file__)), "Verifier version mismatch")
    require(proof["terminal_tree_sha256"] == {k: object_sha(v) for k,v in witness.items()}
            and proof["replayed_requests"] == counts_for(jobs), "Backup audit does not match the live endpoints")
    require(set(proof["mirrors"]) == set(COURSES) == set(proof["mirror_receipt_sha256"]), "Missing full-tree backups")
    for k, meta in proof["mirrors"].items():
        size_hash, link_hash = live_inventory(roots[k])
        require(size_hash == meta["file_sizes_sha256"] and link_hash == meta["symbolic_links_sha256"],
                f"{k}: live tree differs from the finalized external full-tree transfer")
        require(re.fullmatch(r"[0-9a-f]{64}", meta["inventory_sha256"]) and re.fullmatch(r"[0-9a-f]{64}", meta["receipt_sha256"])
                and meta["receipt_sha256"] == proof["mirror_receipt_sha256"][k]
                and meta["files"] >= len(witness[k]) and meta["verified_bytes"] > 0, f"Invalid {k} mirror audit")
    require(re.fullmatch(r"[0-9a-f]{64}", proof["primary_official_bfcl_audits_sha256"]), "Missing BFCL replay digest")
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="action", required=True)
    export = modes.add_parser("attest", help="Check the external CPU backup and write a compact, private proof")
    export.add_argument("--mirror-root", type=Path, required=True)
    export.add_argument("--receipt-dir", type=Path, required=True)
    export.add_argument("--dev", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True)
    verify = modes.add_parser("verify", help="Match the CPU backup proof to the run's current terminal artifacts")
    verify.add_argument("--runs-root", type=Path, required=True)
    verify.add_argument("--attestation", type=Path, required=True)
    verify.add_argument("--sha256", required=True, help="Pinned digest printed on the independent backup machine")
    args = parser.parse_args()
    if args.action == "verify":
        require(digest(args.attestation) == args.sha256, "Transported proof differs from the external copy")
        proof = verify_attestation(args.attestation, {k: args.runs_root / v for k,v in COURSES.items()})
    else:
        proof = attest(args.mirror_root, args.receipt_dir, args.dev)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=args.out.parent) as temp:
            staged = Path(temp) / "attestation.json"
            staged.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
            try:
                os.link(staged, args.out)
            except FileExistsError:
                require(not args.out.is_symlink() and read_json(args.out) == proof,
                        "Refusing to replace a different external attestation")
        print("Attestation SHA-256:", digest(args.out))
    print("Replayed internal requests by study and split:", json.dumps(proof["replayed_requests"], sort_keys=True))


if __name__ == "__main__":
    main()
