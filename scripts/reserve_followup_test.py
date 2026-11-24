"""Reserve an outcome-blind, same-corpus follow-up set from unused Turnstile groups.

Uses the published split/audit checksums and the original conversion functions.
No model, predictions, tokenizer installation, GPU or network request is needed.
Running the same command again verifies the existing reservation byte for byte.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from prepare_experiment import AUDITED, ROOT, SNAPSHOTS, digest, example, file_hash, schema, write_rows


SALT = "turnstile-single-call-followup-v2"
CAP = 1500
SPLITS = ("debug", "pilot", "train", "dev", "test")
# Fixtures, error-audit examples discussed in the log, and the two tokenizer-failure cases.
REVIEWED = AUDITED | {3260, 3495, 38751, 28077, 93268}
REFERENCE = Path(__file__).resolve().parents[1] / "results" / "reproduction-target.json"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_hash(path, expected):
    if file_hash(path) != expected:
        raise ValueError(f"Checksum mismatch for {path}")


def read_audit(prepared, reference):
    for filename, expected in reference["sha256"].items():
        check_hash(prepared / filename, expected)
    audit = {}
    with (prepared / "rows.audit.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            i = int(row["id"])
            if i in audit:
                raise ValueError(f"Repeated ID in the primary audit: {i}")
            audit[i] = row
    used_groups = set()
    for split in SPLITS:
        seen = set()
        with (prepared / f"{split}.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                i, group = int(row["id"]), row["group_id"]
                if audit[i].get("split") != split or group != audit[i]["group_id"]:
                    raise ValueError(f"ID/group mismatch in {split}: {i}")
                if group in used_groups:
                    raise ValueError(f"Duplicate group in the primary splits: {group}")
                seen.add(i)
                used_groups.add(group)
        expected = {i for i, row in audit.items() if row.get("split") == split}
        if seen != expected or len(seen) != reference["counts"][split]:
            raise ValueError(f"Membership/count mismatch in {split}")
    return audit, used_groups


def reserve(source, prepared, out):
    reference = read_json(REFERENCE)
    original = read_json(prepared / "manifest.json")
    if original["source_revision"] != SNAPSHOTS[0][2] or original["sha256"] != reference["sha256"]:
        raise ValueError("The primary preparation manifest does not match the published inputs")
    audit, used_groups = read_audit(prepared, reference)
    source_path, defs_path = source / "data.jsonl", source / "api_definitions.json"
    check_hash(source_path, original["data_source_sha256"])
    check_hash(defs_path, original["definitions_sha256"])
    raw_defs = read_json(defs_path)
    definitions = {name: {k: schema(v[k]) for k in ("name", "description", "parameters")}
                   for name, v in raw_defs.items()}

    # The frozen audit already chose one representative per near-duplicate component
    # after applying turn, schema and token-window filters. Reuse those exact groups.
    reviewed_groups = {audit[i]["group_id"] for i in REVIEWED}
    unused = {i: row["group_id"] for i, row in audit.items() if row.get("reason") == "unused eligible group"}
    if len(set(unused.values())) != len(unused) or used_groups.intersection(unused.values()):
        raise ValueError("A primary group was reused or has more than one unused representative")
    candidates = [i for i, group in unused.items() if group not in reviewed_groups]
    ordered = sorted(candidates, key=lambda i: (digest(f"{SALT}|{unused[i]}"), i))
    chosen = ordered[:CAP]
    if not chosen:
        raise ValueError("No untouched eligible groups available")
    ids = set(chosen)
    selected = {}
    with source_path.open(encoding="utf-8") as stream:
        for i, line in enumerate(stream):
            if i not in ids:
                continue
            entry = audit[i]
            if digest(line.rstrip("\n")) != entry["source_row_sha256"]:
                raise ValueError(f"Frozen source-row checksum mismatch at {i}")
            row = json.loads(line)
            if row["interaction_template_name"] != "TURNS_1_APIS_1":
                raise ValueError(f"Unexpected interaction template at {i}")
            item = example(row, i, definitions)
            if item["messages"][-1]["tool_calls"][0]["function"]["name"] != entry["fn"]:
                raise ValueError(f"Reference function differs from the frozen audit at {i}")
            length, prefix = entry["full_tokens"], entry["prompt_tokens"]
            if not (0 < prefix < length <= original["max_context"]
                    and length - prefix <= original["max_output"]
                    and prefix + original["max_output"] <= original["max_context"]):
                raise ValueError(f"Frozen token counts outside the original limits at {i}")
            item["group_id"] = unused[i]
            selected[i] = item
    if set(selected) != ids:
        raise ValueError("Source rows missing from the reservation")

    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as scratch:
        stage = Path(scratch) / "complete"
        stage.mkdir()
        write_rows(stage / "test.jsonl", (selected[i] for i in chosen))
        manifest = {
            "purpose": "Post-primary, outcome-blind follow-up on the same Turnstile single-call distribution.",
            "scope": "Unused original request groups; tool schemas, task templates and the corpus can recur.",
            "selection": {"salt": SALT, "cap": CAP, "order": "SHA-256 of UTF-8 salt|group_id ascending; ties by numeric row ID"},
            "source_revision": original["source_revision"],
            "source_sha256": original["data_source_sha256"],
            "definitions_sha256": original["definitions_sha256"],
            "grouping": original["transforms"],
            "original_counts": reference["counts"],
            "original_split_sha256": {f"{s}.jsonl": reference["sha256"][f"{s}.jsonl"] for s in SPLITS},
            "original_audit_sha256": reference["sha256"]["rows.audit.jsonl"],
            "original_tokenizer_sha256": original["tokenizer_sha256"],
            "original_template_sha256": original["template_sha256"],
            "original_script_sha256": original["preparation_script_sha256"],
            "original_single_candidates": original["single_candidate_count"],
            "original_valid_single_rows": original["single_valid_count"],
            "original_audit_reasons": dict(sorted(Counter(row["reason"] for row in audit.values() if "reason" in row).items())),
            "original_split_groups_excluded": len(used_groups),
            "reviewed_id_registry": sorted(REVIEWED),
            "reviewed_groups": len(reviewed_groups),
            "unused_eligible_groups": len(unused),
            "unused_excluded_by_review": len(unused) - len(candidates),
            "selected_groups": len(chosen),
            "eligible_groups_remaining": len(candidates) - len(chosen),
            "overlap_original_groups": len({unused[i] for i in chosen} & used_groups),
            "overlap_reviewed_groups": len({unused[i] for i in chosen} & reviewed_groups),
            "ordered_ids_sha256": digest("".join(f"{i}\n" for i in chosen)),
            "ordered_groups_sha256": digest("".join(f"{unused[i]}\n" for i in chosen)),
            "test_sha256": file_hash(stage / "test.jsonl"),
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if not out.exists():
            try:
                stage.rename(out)
            except OSError:
                if not out.is_dir():
                    raise
        for name in ("test.jsonl", "manifest.json"):
            if stage.exists() and file_hash(stage / name) != file_hash(out / name):
                raise ValueError(f"Existing reservation differs at {out / name}; original files were preserved")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/turnstile")
    parser.add_argument("--prepared", type=Path, default=ROOT / "prepared")
    parser.add_argument("--out", type=Path, default=ROOT / "followup/holdout-v2")
    args = parser.parse_args()
    manifest = reserve(args.source, args.prepared, args.out)
    print(json.dumps({k: manifest[k] for k in (
        "unused_eligible_groups", "selected_groups", "eligible_groups_remaining",
        "overlap_original_groups", "overlap_reviewed_groups", "test_sha256",
    )}, indent=2))


if __name__ == "__main__":
    main()
