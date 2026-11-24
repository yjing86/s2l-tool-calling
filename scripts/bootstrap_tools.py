"""Fetch the official BFCL inputs and reconstruct the two training smoke tests."""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
from tempfile import TemporaryDirectory
import time

from bfcl_subset import REVISION, SOURCE_HASHES, digest


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "local"
PREPARED = LOCAL / "prepared"
BFCL_URL = "https://github.com/ShishirPatil/gorilla.git"
BFCL_SUBDIR = "berkeley-function-call-leaderboard"
FIXTURE_SHA = "ecdac2b4bf3f347133fc47596eb53b7d7e699ca975ff45ccc6bd1c92e86b916f"


def verify_bfcl(path):
    for name in ("LICENSE", "pyproject.toml"):
        if not (path / name).is_file():
            raise ValueError(f"BFCL is missing {name} at {path}")
    for name, sha in SOURCE_HASHES.items():
        if digest(path / "bfcl_eval" / name) != sha:
            raise ValueError(f"BFCL source checksum differs for {name} at {path}")


def fetch_bfcl():
    dest = LOCAL / "tools/bfcl"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=dest.parent) as tmp:
            upstream = Path(tmp) / "gorilla"
            subprocess.run(["git", "init", "--quiet", str(upstream)], check=True)
            git = ["git", "-C", str(upstream)]
            subprocess.run([*git, "remote", "add", "origin", BFCL_URL], check=True)
            subprocess.run([*git, "sparse-checkout", "set", BFCL_SUBDIR], check=True)
            subprocess.run([*git, "fetch", "--quiet", "--depth=1", "--filter=blob:none",
                            "origin", REVISION], check=True)
            subprocess.run([*git, "checkout", "--quiet", "--detach", "FETCH_HEAD"], check=True)
            if subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip() != REVISION:
                raise ValueError("Git returned a different BFCL commit")
            package = upstream / BFCL_SUBDIR
            shutil.copyfile(upstream / "LICENSE", package / "LICENSE")
            verify_bfcl(package)
            package.rename(dest)
    verify_bfcl(dest)
    print(f"Verified BFCL inputs and scoring modules at {dest} (pinned commit {REVISION}).")


def smoke_fixture():
    path = PREPARED / "fixture49.jsonl"
    debug = PREPARED / "debug.jsonl"
    rows = [json.loads(line) for line in debug.read_text().splitlines()]
    (row,) = [r for r in rows if str(r["id"]) == "49"]
    # The original smoke fixture predates group IDs and uses Python's default JSON separators.
    payload = (json.dumps({k: row[k] for k in ("id", "tools", "messages")}) + "\n").encode()
    if hashlib.sha256(payload).hexdigest() != FIXTURE_SHA:
        raise ValueError("Smoke fixture reconstruction differs from the original")
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"An existing smoke fixture differs: {path}")
    else:
        with TemporaryDirectory(dir=path.parent) as tmp:
            new = Path(tmp) / path.name
            new.write_bytes(payload)
            new.replace(path)
    print(f"Verified debug fixture at {path} ({FIXTURE_SHA}).")
    return path


def smoke(args):
    import fcntl
    from run_course_experiment import Protocol, args_parser

    deadline = dt.datetime.fromisoformat(args.deadline.replace("Z", "+00:00"))
    if deadline.utcoffset() != dt.timedelta(0) or args.reserve_seconds < 0:
        raise ValueError("Use a UTC deadline with Z or +00:00 and a nonnegative reserve")
    if deadline.timestamp() - args.reserve_seconds <= time.time():
        raise ValueError("The GPU deadline minus its reserve must be in the future")
    # Keep the virtualenv's symlink path. Resolving it can select the system interpreter.
    train = Path(os.path.abspath(args.training_python.expanduser()))
    for cmd in (train, train.with_name("swift")):
        if not cmd.is_file() or not os.access(cmd, os.X_OK):
            raise ValueError(f"Missing training executable: {cmd}")
    config = args_parser().parse_args([])
    config.course = args.course
    config.deadline = args.deadline
    config.reserve_seconds = args.reserve_seconds
    config.training_python = train
    runner = Protocol(config)

    def interrupted(signum, frame):
        # Let Protocol.task terminate the trainer process group before releasing the lock.
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    with (runner.root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fixture = smoke_fixture()
        for label, model, out in (("proxy", runner.small, config.proxy_smoke),
                                  ("target", runner.big, config.target_smoke)):
            adopt = (out / "completed.json").is_file() and (out / "exit.json").is_file()
            runner.train(f"setup-smoke-{label}", "smoke", model, fixture, out, adopt=adopt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    plans = {
        "bfcl": f"Fetch BFCL commit {REVISION} into local/tools/bfcl and verify its scoring inputs.",
        "fixture": "Reconstruct held-aside debug row 49 from local/prepared/debug.jsonl and verify the fixture.",
        "smoke": "Run two one-update LoRA smoke tests on debug row 49 and record receipts in the course directory.",
    }
    for name, desc in plans.items():
        cmd = sub.add_parser(name, help=desc)
        cmd.add_argument("--dry-run", action="store_true", help="Print the action without downloads or GPU use.")
        if name == "smoke":
            cmd.add_argument("--training-python", required=True, type=Path)
            cmd.add_argument("--deadline", required=True, help="Your future absolute GPU stop time in UTC.")
            cmd.add_argument("--reserve-seconds", type=int, default=600)
            cmd.add_argument("--course", type=Path, default=LOCAL / "course-run")
    args = parser.parse_args()
    if getattr(args, "dry_run", False):
        print(plans[args.action])
        return
    if Path.cwd().resolve() != ROOT:
        parser.error("Run this command from the root of the cloned repository")
    try:
        if args.action == "bfcl":
            fetch_bfcl()
        elif args.action == "fixture":
            smoke_fixture()
        else:
            smoke(args)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"{args.action}: {exc}\n")


if __name__ == "__main__":
    main()
