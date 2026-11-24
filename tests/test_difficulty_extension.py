"""Exercise the authorization overlay and its handoff to the frozen runner without any GPU."""

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_difficulty_extension as ext


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        self.temp = Path(self.context.enter_context(tempfile.TemporaryDirectory()))
        self.out = self.temp / "followup/difficulty-control"
        self.out.mkdir(parents=True)
        self.original = {
            "cell_order": list(ext.frozen.ORDER),
            "training": {"budget": 3005228},
            "admission": {
                "hard_cutoff_utc": ext.frozen.CURRENT_CUTOFF,
                "time_multiplier": 1.2,
            },
        }
        self.reg = self.out / "preregistration.json"
        self.reg.write_text(json.dumps({"spec": self.original}))
        pinned = ext.frozen.digest(self.reg)
        self.auth = {
            "version": 1,
            "scope": "existing rental; E2b seeds 42, 43, 44 only",
            "base_registration_sha256": pinned,
            "extension_script_sha256": ext.frozen.digest(Path(ext.__file__)),
            "hard_cutoff_utc": ext.RUN_CUTOFF,
            "original_hard_cutoff_utc": "2026-11-24T20:29:00Z",
            "stop_by_utc": ext.STOP_BY,
            "shutdown_reserve_seconds_after_compute": 600,
            "stop_early_after_verified_backup": True,
            "authorization_recorded_utc": "2026-11-24T18:41:00Z",
        }
        self.context.enter_context(patch.object(ext, "REGISTRATION_SHA256", pinned))
        self.context.enter_context(patch.object(ext, "OUT", self.out))
        self.context.enter_context(patch.object(ext, "LOCAL", self.temp))
        self.context.enter_context(
            patch.object(
                ext.time,
                "time",
                return_value=ext.frozen.as_time("2026-11-24T18:45:00Z"),
            )
        )
        (self.temp / "course-run").mkdir()
        (self.temp / "course-run/design.json").write_text('{"micro_batch":4}')
        self.context.enter_context(redirect_stdout(StringIO()))

    def freeze_auth(self):
        authfile = self.out / ext.AMENDMENT
        authfile.write_text(json.dumps(self.auth))
        return ext.frozen.digest(authfile)

    def test_sole_changed_field_and_unmodified_original(self):
        key = self.freeze_auth()
        original = ext.verified_amendment(self.out, key)
        expected = deepcopy(original)
        expected["admission"]["hard_cutoff_utc"] = ext.frozen.utc(
            ext.frozen.as_time(ext.RUN_CUTOFF)
        )
        self.assertEqual(ext.with_extended_budget(original), expected)
        self.assertEqual(original, self.original)
        self.assertEqual(ext.frozen.digest(self.reg), ext.REGISTRATION_SHA256)

    def test_tampered_budget_old_pin_and_wrong_scope(self):
        key = self.freeze_auth()
        self.auth["stop_by_utc"] = "2027-01-01T21:40:00Z"
        new = self.freeze_auth()
        for token in (key, new):
            with self.assertRaises(ValueError):
                ext.verified_amendment(self.out, token)
        self.auth["stop_by_utc"] = ext.STOP_BY
        self.auth["scope"] = "another model"
        with self.assertRaises(ValueError):
            ext.verified_amendment(self.out, self.freeze_auth())

    def test_expired_window_and_missing_proof_refused(self):
        common = ["entry", "run", "--amendment-sha256", self.freeze_auth()]
        with patch.object(sys, "argv", common), self.assertRaises(SystemExit):
            ext.main()
        with (
            patch.object(
                ext.time, "time", return_value=ext.frozen.as_time(ext.RUN_CUTOFF)
            ),
            patch.object(
                sys,
                "argv",
                common
                + [
                    "--attestation",
                    "a",
                    "--attestation-sha256",
                    "a",
                    "--account-observation",
                    "b",
                ],
            ),
            self.assertRaises(SystemExit),
        ):
            ext.main()

    def test_handoff_reuses_registered_audit_and_account_controls(self):
        key = self.freeze_auth()
        account, proof = self.temp / "proof.account", self.temp / "backup"
        args = [
            "entry",
            "run",
            "--amendment-sha256",
            key,
            "--attestation",
            str(proof),
            "--attestation-sha256",
            "external-proof-hash",
            "--account-observation",
            str(account),
        ]
        runner = SimpleNamespace(root=self.out)
        with (
            patch.object(sys, "argv", args),
            patch.object(ext.frozen, "Protocol", return_value=runner) as build,
            patch.object(
                ext.frozen, "register", return_value=self.original
            ) as registration,
            patch.object(
                ext.frozen, "execute", return_value={"status": "deferred"}
            ) as execute,
            patch.object(ext.signal, "signal"),
        ):
            ext.main()
        self.assertEqual(
            registration.call_args.args[4],
            ext.frozen.as_time(ext.frozen.CURRENT_CUTOFF),
        )
        self.assertEqual(registration.call_args.kwargs, {"create": False})
        self.assertEqual(build.call_args.args[0].deadline, ext.RUN_CUTOFF)
        self.assertEqual(
            build.call_args.args[0].reserve_seconds, ext.frozen.TEARDOWN_SECONDS
        )
        roots, active, spec, att, sha, obs = execute.call_args.args
        self.assertEqual(set(roots), {"primary", "full", "cross", "phase1"})
        self.assertIs(active, runner)
        self.assertEqual(spec, ext.with_extended_budget(self.original))
        self.assertEqual((att, sha, obs), (proof, "external-proof-hash", account))
        self.assertEqual(
            ext.frozen.read(self.out / ext.EFFECTIVE)["amendment_sha256"], key
        )
        self.assertEqual(ext.frozen.digest(self.reg), ext.REGISTRATION_SHA256)


if __name__ == "__main__":
    unittest.main()
