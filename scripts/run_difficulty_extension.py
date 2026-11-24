"""Apply the dated rental extension to an already registered, three-seed E2b batch.

The original registration, selection hashes, executables and audit remain authoritative.
Only the operational cutoff is amended. A private, independently pinned authorization
is required for the CPU forecast and for every run or resume.
"""

import argparse
from copy import deepcopy
import fcntl
from hashlib import sha256
import json
from pathlib import Path
import signal
import time

import run_difficulty_control as frozen


REGISTRATION_SHA256 = "291b6a659e9b2dba2f8ac781ef4484105ccd6f40d30ac5cd2130fe286ee151be"
RUN_CUTOFF = "2026-11-24T21:30:00Z"
STOP_BY = "2026-11-24T21:40:00Z"
REPO = Path(__file__).resolve().parents[1]
LOCAL = REPO / "local"
OUT = LOCAL / "followup/difficulty-control"
AMENDMENT = "rental-amendment.json"
EFFECTIVE = "effective-rental-budget.json"


def verified_amendment(out, expected_sha):
    path = out / AMENDMENT
    registration = out / "preregistration.json"
    if path.is_symlink() or registration.is_symlink():
        raise ValueError("The registration and authorization must be ordinary files")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected_sha:
        raise ValueError(
            "The rental authorization differs from its externally supplied SHA-256"
        )
    if frozen.digest(registration) != REGISTRATION_SHA256:
        raise ValueError("Original E2b registration changed")
    auth = json.loads(raw)
    required = {
        "version": 1,
        "scope": "existing rental; E2b seeds 42, 43, 44 only",
        "base_registration_sha256": REGISTRATION_SHA256,
        "extension_script_sha256": frozen.digest(Path(__file__).resolve()),
        "original_hard_cutoff_utc": "2026-11-24T20:29:00Z",
        "hard_cutoff_utc": RUN_CUTOFF,
        "stop_by_utc": STOP_BY,
        "shutdown_reserve_seconds_after_compute": 600,
        "stop_early_after_verified_backup": True,
    }
    if any(auth.get(k) != v for k, v in required.items()):
        raise ValueError(
            "Wrong rental, scope, extension source, cutoff, or shutdown reserve"
        )
    ts = frozen.as_time(auth["authorization_recorded_utc"])
    if not ts < frozen.as_time(frozen.CURRENT_CUTOFF) or ts > time.time():
        raise ValueError("Invalid authorization timestamp for this rental")
    return frozen.read(registration)["spec"]


def with_extended_budget(original):
    if original["cell_order"] != frozen.ORDER or frozen.as_time(
        original["admission"]["hard_cutoff_utc"]
    ) != frozen.as_time(frozen.CURRENT_CUTOFF):
        raise ValueError("Wrong registered seed order or original cutoff")
    effective = deepcopy(original)
    effective["admission"]["hard_cutoff_utc"] = frozen.utc(frozen.as_time(RUN_CUTOFF))
    return effective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("forecast", "run"))
    parser.add_argument(
        "--amendment-sha256",
        required=True,
        help="SHA-256 of the backup host's private rental-amendment.json",
    )
    parser.add_argument(
        "--phase1", type=Path, help="Read-only Phase 1 mirror for a CPU forecast"
    )
    parser.add_argument("--attestation", type=Path)
    parser.add_argument("--attestation-sha256")
    parser.add_argument("--account-observation", type=Path)
    args = parser.parse_args()
    original = verified_amendment(OUT, args.amendment_sha256)
    effective = with_extended_budget(original)
    roots = {
        role: LOCAL / location
        for role, location in (
            ("primary", "course-run"),
            ("full", "full-reference"),
            ("cross", "crossed-seeds"),
            ("phase1", "followup/phase1"),
        )
    }
    if args.action == "forecast":
        print(
            json.dumps(
                frozen.estimate(
                    effective, args.phase1 or roots["phase1"], OUT, time.time()
                ),
                indent=2,
            )
        )
        return
    if args.phase1 or not all(
        (args.attestation, args.attestation_sha256, args.account_observation)
    ):
        parser.error(
            "run requires native predecessor roots, a pinned independent attestation and a fresh provider observation"
        )
    if time.time() >= frozen.as_time(RUN_CUTOFF):
        parser.error("The authorized E2b compute window has ended")
    config = frozen.args_parser().parse_args([])
    config.course = OUT
    config.micro_batch = frozen.read(roots["primary"] / "design.json")["micro_batch"]
    config.deadline, config.reserve_seconds = RUN_CUTOFF, frozen.TEARDOWN_SECONDS
    config.training_python = Path("/opt/trainenv/bin/python")
    config.eval_python = Path("/opt/evalenv/bin/python")
    runner = frozen.Protocol(config)

    def stop(sig, _frame):
        raise KeyboardInterrupt(f"Signal {sig}")

    signal.signal(signal.SIGTERM, stop)
    with (OUT / ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original = verified_amendment(OUT, args.amendment_sha256)
        registered = frozen.register(
            roots["primary"],
            roots["phase1"],
            LOCAL / "followup/e2b-eligibility",
            runner,
            frozen.as_time(frozen.CURRENT_CUTOFF),
            create=False,
        )
        if original != registered:
            raise ValueError(
                "Reconstructed registration does not match the original bytes"
            )
        effective = with_extended_budget(registered)
        marker = OUT / EFFECTIVE
        if not marker.exists() and any(
            (OUT / name).exists()
            for name in (
                "admission.json",
                "training",
                "evaluation",
                "cells",
                "receipts",
                "runtime",
            )
        ):
            raise ValueError(
                "E2b workload predates the separately authorized extension"
            )
        frozen.immutable(
            marker,
            {
                "amendment_sha256": args.amendment_sha256,
                "registration_sha256": REGISTRATION_SHA256,
                "original_spec_sha256": frozen.science_hash(original),
                "effective_spec_sha256": frozen.science_hash(effective),
                "only_changed_field": "admission.hard_cutoff_utc",
                "compute_cutoff_utc": RUN_CUTOFF,
                "rental_stop_by_utc": STOP_BY,
            },
        )
        result = frozen.execute(
            roots,
            runner,
            effective,
            args.attestation,
            args.attestation_sha256,
            args.account_observation,
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
