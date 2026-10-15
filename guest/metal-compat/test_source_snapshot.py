#!/usr/bin/env python3
import json
from pathlib import Path
import runpy
import subprocess
import tempfile
import unittest

api = runpy.run_path(str(Path(__file__).with_name("source_snapshot.py")))


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.source = self.repo / "guest" / "bridge"
        self.source.mkdir(parents=True)
        self.run_git("init", "-q")
        self.run_git("config", "core.autocrlf", "false")
        self.run_git("config", "user.name", "Snapshot test")
        self.run_git("config", "user.email", "test@example.invalid")
        (self.source / "bridge.cpp").write_text("verified source\n", encoding="utf8")
        self.run_git("add", "guest/bridge/bridge.cpp")
        self.run_git("commit", "-qm", "Fixture")
        self.output = self.root / "export"

    def run_git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.repo), *args], stderr=subprocess.PIPE)

    def test_export_is_independent_and_detects_reverted_source(self):
        revision = api["export"](self.source, self.output)
        self.assertEqual(revision, self.run_git("rev-parse", "HEAD").decode().strip())
        (self.source / "bridge.cpp").write_text("stale editor buffer\n", encoding="utf8")
        self.assertEqual(api["verify"](self.output), revision)
        with self.assertRaisesRegex(ValueError, "uncommitted"):
            api["export"](self.source, self.root / "rejected")
        self.assertFalse((self.root / "rejected").exists())
        (self.output / "bridge.cpp").write_text("stale editor buffer\n", encoding="utf8")
        with self.assertRaisesRegex(ValueError, "Source changed"):
            api["verify"](self.output)

    def test_untracked_or_extra_inputs_and_existing_destination_are_rejected(self):
        api["export"](self.source, self.output)
        with self.assertRaises(FileExistsError):
            api["export"](self.source, self.output)
        (self.source / "extra.hpp").touch()
        with self.assertRaisesRegex(ValueError, "uncommitted"):
            api["export"](self.source, self.root / "rejected")
        (self.output / "extra.hpp").touch()
        with self.assertRaisesRegex(ValueError, "unexpected"):
            api["verify"](self.output)

    def test_manifest_cannot_select_parent_files(self):
        api["export"](self.source, self.output)
        path = self.output / api["MANIFEST"]
        manifest = json.loads(path.read_text())
        manifest["files"] = {"../bridge.cpp": "0" * 64}
        path.write_text(json.dumps(manifest), encoding="utf8")
        with self.assertRaises(ValueError):
            api["verify"](self.output)


if __name__ == "__main__":
    unittest.main()
