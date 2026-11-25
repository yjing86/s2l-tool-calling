import argparse
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import restore_gpu_envs as installer


class RestoreGpuEnvironmentsTest(unittest.TestCase):
    def test_plan_reads_both_lockfiles_without_creating_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "missing"
            output = io.StringIO()
            with patch.object(sys, "argv", ["restore", "--plan", "--env-root", str(dest)]), redirect_stdout(output):
                installer.main()
            self.assertIn(installer.fingerprint("train"), output.getvalue())
            self.assertIn(installer.fingerprint("vllm"), output.getvalue())
            self.assertFalse(dest.exists())

    def test_valid_legacy_environment_needs_no_installer_or_disk_space(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(check_only=False, min_free_gib=1000000)
            with patch.object(installer, "probe", return_value={"python": "/opt/trainenv/bin/python"}) as probe:
                out = installer.restore(args, "train", Path(tmp))
            self.assertEqual(out["python"], "/opt/trainenv/bin/python")
            probe.assert_called_once_with(Path("/opt/trainenv"), "train")
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_check_only_and_low_disk_do_not_start_install(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(installer, "probe", return_value=None):
            args = argparse.Namespace(check_only=True, min_free_gib=16)
            with self.assertRaisesRegex(RuntimeError, "No verified vllm"):
                installer.restore(args, "vllm", Path(tmp))
            args.check_only = False
            with patch.object(installer, "available_gib", return_value=3.5):
                with self.assertRaisesRegex(RuntimeError, "3.5 GiB free"):
                    installer.restore(args, "vllm", Path(tmp))
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_corrupt_legacy_reuses_verified_managed_environment(self):
        args = argparse.Namespace(check_only=True, min_free_gib=16)
        with tempfile.TemporaryDirectory() as tmp:
            managed = Path(tmp) / f"vllm-{installer.fingerprint('vllm')[:16]}"
            result = {"python": str(managed / "bin/python")}
            with patch.object(installer, "probe", side_effect=(None, result)) as probe:
                self.assertEqual(installer.restore(args, "vllm", Path(tmp)), result)
            self.assertEqual(probe.call_args.args, (managed, "vllm"))

    def test_failed_install_retries_same_environment_then_reuses_result(self):
        args = argparse.Namespace(check_only=False, min_free_gib=1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_uv = root / "fake-uv"
            fake_uv.write_text(
                "#!" + sys.executable + "\n"
                "import os, sys\nfrom pathlib import Path\n"
                "args=sys.argv[1:]\nroot=Path(__file__).parent\n"
                "with (root/'calls').open('a') as f: f.write(' '.join(args)+'\\n')\n"
                "if 'venv' in args:\n"
                " env=Path(args[-1]);(env/'bin').mkdir(parents=True);(env/'pyvenv.cfg').write_text('test');"
                "(env/'bin/python').symlink_to(sys.executable)\n"
                "if 'install' in args:\n"
                " if not (root/'retried').exists(): (root/'retried').touch();sys.exit(5)\n"
                " (root/'installed').touch()\n"
            )
            fake_uv.chmod(0o700)
            envdir = root / f"train-{installer.fingerprint('train')[:16]}"
            def probe(path, stage):
                if path == envdir and (root / "installed").exists():
                    return {"stage": stage, "python": str(path / "bin/python")}
                return None
            def executable(name):
                return str(fake_uv) if name == "uv" else sys.executable
            with patch.object(installer, "probe", side_effect=probe), patch.object(installer.shutil, "which", side_effect=executable):
                with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "installation failed"):
                    installer.restore(args, "train", root)
                inode = (envdir / "pyvenv.cfg").stat().st_ino
                with redirect_stdout(io.StringIO()):
                    restored = installer.restore(args, "train", root)
                self.assertTrue(Path(restored["install_log"]).is_file())
                calls = (root / "calls").read_text()
                self.assertEqual(calls.count("--no-config venv "), 1)
                self.assertEqual(calls.count("--no-config pip install "), 2)
                self.assertEqual(calls.count("--no-config pip check "), 1)
                self.assertEqual((envdir / "pyvenv.cfg").stat().st_ino, inode)
                self.assertEqual(installer.restore(args, "train", root)["python"], str(envdir / "bin/python"))
                self.assertEqual((root / "calls").read_text(), calls)


if __name__ == "__main__":
    unittest.main()
