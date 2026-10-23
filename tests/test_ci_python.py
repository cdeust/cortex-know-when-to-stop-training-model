"""Test interpreter discovery; fake executables model versions, not the product API."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

HELPER = Path(__file__).resolve().parents[1] / "scripts/ci_python.sh"


class SelectPythonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ci python ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.cache = self.root / "tool cache"

    def executable(self, path, valid=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\nexit " + ("0\n" if valid else "1\n"))
        path.chmod(0o755)
        return path

    def cached_python(self, architecture="arm64", complete=True, valid=True):
        installation = self.cache / "Python" / "3.12.10" / architecture
        executable = self.executable(installation / "bin/python3", valid)
        if complete:
            installation.with_suffix(".complete").touch()
        return executable

    def select(self, platform="macos", architecture="ARM64"):
        return subprocess.run(
            ["/bin/bash", "-c", 'source "$1"; select_ci_python', "test", str(HELPER)],
            env={
                **os.environ,
                "PATH": str(self.bin),
                "CI_OS": platform,
                "RUNNER_ARCH": architecture,
                "RUNNER_TOOL_CACHE": str(self.cache),
            },
            capture_output=True,
            text=True,
        )

    def test_complete_cache_with_spaces_wins_over_path(self):
        expected = self.cached_python()
        self.executable(self.bin / "python3.12", valid=False)
        result = self.select()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(expected))

    def test_x64_cache_uses_matching_architecture(self):
        expected = self.cached_python(architecture="x64")
        result = self.select(architecture="X64")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(expected))

    def test_incomplete_cache_is_rejected(self):
        self.cached_python(complete=False)
        result = self.select()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Python 3.12", result.stderr)

    def test_wrong_cached_version_is_rejected(self):
        self.cached_python(valid=False)
        result = self.select()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")

    def test_path_python312_is_supported(self):
        expected = self.executable(self.bin / "python3.12")
        result = self.select()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(expected))

    def test_wrong_path_version_is_rejected(self):
        self.executable(self.bin / "python3.12", valid=False)
        self.assertNotEqual(self.select().returncode, 0)

    def test_missing_python_is_an_error(self):
        result = self.select()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Python 3.12", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_linux_and_windows_keep_python3_preference(self):
        expected = self.executable(self.bin / "python3")
        self.executable(self.bin / "python")
        for platform in ("linux", "windows"):
            with self.subTest(platform=platform):
                result = self.select(platform)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), str(expected))

    def test_windows_keeps_python_fallback(self):
        expected = self.executable(self.bin / "python")
        result = self.select("windows")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(expected))


if __name__ == "__main__":
    unittest.main()
