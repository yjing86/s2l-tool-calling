"""Check that the public package and legacy download command work in a fresh checkout."""
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from s2l_tool_calling.data import download_inputs


REPO = Path(__file__).resolve().parents[1]


class SourceCheckoutTests(unittest.TestCase):
    def test_preparation_still_uses_the_same_snapshot_pins(self):
        from scripts import download_inputs as legacy

        self.assertEqual(legacy.ROOT, REPO / "local")
        self.assertIs(legacy.SNAPSHOTS, download_inputs.SNAPSHOTS)
        self.assertIs(legacy.fetch, download_inputs.fetch)

    def test_download_help_works_offline_through_both_entry_points(self):
        with TemporaryDirectory() as tmp:
            # An empty PATH detects accidental calls to curl during argument parsing.
            env = dict(os.environ, PATH=tmp)
            legacy = subprocess.run(
                [sys.executable, "-I", str(REPO / "scripts/download_inputs.py"), "--help"],
                cwd=tmp, env=env, capture_output=True, text=True, check=True,
            )
            public = subprocess.run(
                [sys.executable, "-m", "s2l_tool_calling.data.download_inputs", "--help"],
                cwd=REPO, env=env, capture_output=True, text=True, check=True,
            )
        for result in (legacy, public):
            self.assertIn("--output-root", result.stdout)
            self.assertNotIn("Downloading", result.stdout)
