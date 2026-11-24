"""Check fresh split files and optional run manifests against published reference values."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


TARGET = Path(__file__).resolve().parents[1] / "results/reproduction-target.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check(prepared, design, locked):
    target = read(TARGET)
    manifest = read(prepared / "manifest.json")
    errors = []
    if manifest["counts"] != target["counts"]:
        errors.append("Split sizes differ from the published split sizes.")
    for filename, expected in target["sha256"].items():
        with (prepared / filename).open("rb") as source:
            digest = hashlib.sha256()
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected or manifest["sha256"].get(filename) != expected:
            errors.append(f"{filename}: split file or manifest differs from published SHA-256 {expected}")
    if design:
        actual = read(design)
        for key, expected in target["design"].items():
            if key not in actual or actual[key] != expected:
                errors.append(f"{key}: expected {expected!r}, got {actual.get(key)!r}")
    if locked:
        actual = read(locked)
        published = set(target["endpoints"])
        observed = set(actual["main"])
        if observed != published or actual.get("equal_time") != {}:
            errors.append(f"Endpoint design differs: missing {sorted(published - observed)}, "
                          f"additional {sorted(observed - published)}, equal_time={actual.get('equal_time')!r}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("local/prepared"),
                        help="Directory produced by the prepare command.")
    parser.add_argument("--design", type=Path, help="Optional design.json produced by run --through freeze.")
    parser.add_argument("--locked", type=Path, help="Optional locked-models.json from the completed run.")
    args = parser.parse_args()
    try:
        errors = check(args.prepared, args.design, args.locked)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        parser.exit(1, f"Could not verify inputs: {exc}\n")
    if errors:
        parser.exit(1, "\n".join(errors) + "\n")
    print("Verified the split sizes and all six published SHA-256 digests.")
    if args.design:
        print("Frozen run design matches the published configuration.")
    if args.locked:
        print("The set of locked evaluation endpoints matches the published run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
