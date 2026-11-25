r"""Report the complete second-round panel on the old development and test sets.

After independent backup verification, supply the externally recorded SHA-256 of
the final proof. The backup watcher includes the source inventory filename in its
proof. For a standalone verify_round2_archive receipt, supply --inventory.

    python scripts/summarize_round2_legacy.py --proof-sha256 VERIFIED_DIGEST \
      --out results/followup/round2-legacy.json

This CPU report contains aggregate counts and differences. Both splits were
used previously. The prospective evaluation has a separate analysis.
"""

import argparse
from pathlib import Path
from statistics import mean

from difficulty_admission import replay
from run_course_experiment import digest, read, write
from summarize_course import read_model
from summarize_followup import receipt
from verify_round2_archive import CORE, SPLITS, indexed, require, safe_path, verify


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "results/followup"
REGISTRATION = PUBLIC / "round2-registration.json"
PROTOCOL_SHA = "efe8993ce4ad1e40ffebec5c684b225e3445ede4c0c335798d1b423bf05e86f5"
METHODS = ("A", "B", "C", "E2", "E2b", "L-short", "L-long")
SEEDS = (42, 43, 44, 45, 46)
CONTRASTS = (("C", "B"), ("C", "A"), ("E2", "A"), ("L-long", "L-short"))


def verify_backup(artifacts, dev, proof_path, proof_sha, inventory):
    require(digest(safe_path(proof_path)) == proof_sha, "Independent backup proof SHA-256 differs")
    proof = read(proof_path)
    require(proof["terminal_state"] == "complete" and proof["completed_cells"] == 20
            and proof["completed_tasks"] == 60
            and proof["regraded_predictions"] == {"dev": 8000, "test": 40000},
            "The archive must cover all 20 fits and both complete legacy evaluations")
    if inventory is None:
        name = proof.get("source_inventory")
        require(bool(name) and Path(name).name == name, "Supply the original --inventory file")
        inventory = proof_path.parent / name
    verified = verify(artifacts / "followup/round2", inventory, proof["source_inventory_sha256"],
                      PROTOCOL_SHA, REGISTRATION, artifacts / "course-run", dev)
    require(all(proof.get(k) == v for k, v in verified.items()), "Independent verification differs from its proof")


def historical_endpoint(artifacts, tag, frozen, old_models):
    method = tag.split("-", 1)[1]
    origin = {"E2": "followup/phase1", "E2b": "followup/difficulty-control"}.get(method, "course-run")
    folder = artifacts / origin
    cp = frozen["checkpoint"]
    old = old_models[tag]
    require((old["checkpoint"], old["adapter_sha256"]) == (cp, frozen["adapter_sha256"]),
            f"{tag}: historical results and preregistered adapter differ")
    training = folder / "training" / tag / ("checkpoint-" + cp.rsplit("-step", 1)[1])
    hashes = {name: digest(safe_path(training / name)) for name in ("adapter_model.safetensors", "adapter_config.json")}
    require(hashes == {"adapter_model.safetensors": frozen["adapter_sha256"],
                       "adapter_config.json": frozen["adapter_config_sha256"]}, f"{tag}: historical adapter changed")
    return folder, cp, hashes


def legacy_scores(artifacts, tag, sources, splits_sha, groups, frozen, published):
    folder, cp, hashes = historical_endpoint(artifacts, tag, frozen, published)
    primary = folder == artifacts / "course-run"
    result = {}
    for split in SPLITS:
        evaluation = safe_path(folder / "evaluation" / split / cp)
        label = f"eval-{split}-{cp}" if primary else f"{split}-{tag}"
        files = ["predictions.jsonl", "metrics.json"] + ([] if primary else ["config.json"])
        record = receipt(folder, label, [safe_path(evaluation / fn) for fn in files])
        config = read(safe_path(evaluation / "config.json"))
        require(config["adapter_sha256"] == hashes and config["source_sha256"] == splits_sha[split],
                f"{tag}: {split} used a different adapter or input")
        require(record["script_sha256"] == config["script_sha256"] == digest(ROOT / "scripts/evaluate_experiment.py"),
                f"{tag}: {split} scoring entry point changed")
        replay(evaluation, sources[split], splits_sha[split])
        scored, metrics, _ = read_model(evaluation, groups[split], splits_sha[split], {})
        if split == "test":
            prior = published[tag]
            require(metrics["correct"] == prior["correct"] and
                    digest(evaluation / "predictions.jsonl") == prior["predictions_sha256"],
                    f"{tag}: prior published test results differ")
        result[split] = scored
    return result


def collect(artifacts, sources, splits_sha, registration):
    root = artifacts / "followup/round2"
    current = read(root / "locked-models.json")
    historical = {row["id"]: row for row in registration["holdout_v2"]["existing_endpoints"]}
    old_models = {**read(PUBLIC / "phase1-results.json")["models"], **read(PUBLIC / "e2b-results.json")["models"]}
    groups = {split: {key: str(row["group_id"]) for key, row in data.items()} for split, data in sources.items()}
    outcomes = {split: {} for split in SPLITS}
    for seed in SEEDS:
        for method in METHODS:
            tag = f"s{seed}-{method}"
            if tag in CORE:
                cp = current[tag]
                lock = read(root / "locks" / (tag + ".json"))
                train = root / "training" / tag / f"checkpoint-{lock['step']}"
                hashes = {"adapter_model.safetensors": lock["final_adapter_sha256"],
                          "adapter_config.json": digest(safe_path(train / "adapter_config.json"))}
                for split in SPLITS:
                    evaluation = safe_path(root / "evaluation" / split / cp)
                    config = read(safe_path(evaluation / "config.json"))
                    require(config["adapter_sha256"] == hashes and config["source_sha256"] == splits_sha[split],
                            f"{tag}: {split} used a different adapter or input")
                    scored, _, _ = read_model(evaluation, groups[split], splits_sha[split], {})
                    outcomes[split][tag] = scored
            else:
                historical_scores = legacy_scores(artifacts, tag, sources, splits_sha, groups, historical[tag], old_models)
                for split, scored in historical_scores.items():
                    outcomes[split][tag] = scored
    return outcomes


def split_report(outcomes, ids):
    expected = {f"s{seed}-{method}" for method in METHODS for seed in SEEDS}
    require(bool(ids) and set(outcomes) == expected, "Incomplete 35-endpoint legacy panel")
    require(all(set(scores) == ids and all(type(x) is bool for x in scores.values())
                for scores in outcomes.values()), "Models must cover exactly the same requests with boolean outcomes")
    n = len(ids)
    methods = {}
    for method in METHODS:
        counts = {str(seed): sum(outcomes[f"s{seed}-{method}"].values()) for seed in SEEDS}
        methods[method] = {"by_seed": {s: {"correct": k, "n": n, "accuracy_pct": 100 * k / n}
                                       for s, k in counts.items()},
                           "mean_accuracy_pct": 100 * mean(counts.values()) / n}
    contrasts = {}
    for left, right in CONTRASTS:
        deltas = {}
        for seed in SEEDS:
            a, b = (outcomes[f"s{seed}-{m}"] for m in (left, right))
            left_only = sum(a[key] and not b[key] for key in ids)
            right_only = sum(b[key] and not a[key] for key in ids)
            deltas[str(seed)] = {"only_left_correct": left_only, "only_right_correct": right_only,
                                 "difference_pp": 100 * (left_only - right_only) / n}
        contrasts[f"{left} minus {right}"] = {"by_seed": deltas, "mean_difference_pp": mean(
            row["difference_pp"] for row in deltas.values())}
    return {"requests": n, "methods": methods, "paired_contrasts": contrasts}


def run(args):
    artifacts, dev = safe_path(args.artifacts).resolve(), safe_path(args.dev).resolve()
    dest = safe_path(args.out).resolve()
    require(dest.suffix == ".json" and not dest.is_relative_to(artifacts), "Write the JSON report outside the archived run")
    verify_backup(artifacts, dev, args.proof, args.proof_sha256, args.inventory)
    registration = read(REGISTRATION)
    splits_sha = {s: registration["data"]["original_split_sha256"][s + ".jsonl"] for s in SPLITS}
    paths = {"dev": dev, "test": artifacts / "course-run/frozen/test.jsonl"}
    sources = {s: indexed(paths[s], n, splits_sha[s]) for s, n in SPLITS.items()}
    outcomes = collect(artifacts, sources, splits_sha, registration)
    result = {"scope": "Descriptive results on previously used development and test requests.",
              "metric": "Complete-call accuracy. Percentage-point differences pair identical requests and seeds.",
              "seeds": list(SEEDS), "registration_sha256": digest(REGISTRATION),
              "verified_archive_proof_sha256": args.proof_sha256, "dataset_sha256": splits_sha,
              "splits": {s: split_report(outcomes[s], set(sources[s])) for s in SPLITS}}
    if dest.exists():
        require(read(dest) == result, "Existing destination contains other data. Choose a new report path.")
    else:
        write(dest, result)
    print(f"{dest} {digest(dest)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "local/runpod/repo/local")
    parser.add_argument("--dev", type=Path, default=ROOT / "local/prepared/dev.jsonl")
    parser.add_argument("--proof", type=Path, default=ROOT / "local/runpod/receipts/round2-final-archive.json")
    parser.add_argument("--proof-sha256", required=True)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
