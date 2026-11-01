import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import plistlib
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("bridge_config", Path(__file__).parents[1] / "metal-compat-config.py")
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


class BridgeConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="reims-bridge-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        source = self.repo / "guest/metal-compat/ReimsMetalCompat.cpp"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"// authored fixture\n")
        def git(*args):
            return subprocess.check_output(["git", "-C", str(self.repo), *args], stderr=subprocess.DEVNULL)
        git("init")
        git("add", "guest/metal-compat/ReimsMetalCompat.cpp")
        git("-c", "user.name=Reims Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")
        revision = git("rev-parse", "HEAD").decode().strip()
        self.bundle = self.root / "ReimsMetalCompat.kext"
        for d in ("MacOS", "Resources"):
            (self.bundle / "Contents" / d).mkdir(parents=True, exist_ok=True)
        (self.bundle / "Contents/MacOS/ReimsMetalCompat").write_bytes(b"authored test executable")
        (self.bundle / "Contents/Info.plist").write_bytes(plistlib.dumps(dict(CFBundleIdentifier="io.reims.MetalCompat", CFBundleExecutable="ReimsMetalCompat")))
        self.provenance = self.bundle / "Contents/Resources/source-manifest.json"
        self.provenance.write_text(json.dumps(dict(schema=1, revision=revision, files={"ReimsMetalCompat.cpp": hashlib.sha256(source.read_bytes()).hexdigest()})))
        self.kexts = self.root / "OC/Kexts"
        lilu = self.kexts / "Lilu.kext/Contents/Info.plist"
        lilu.parent.mkdir(parents=True)
        lilu.write_bytes(plistlib.dumps(dict(CFBundleIdentifier="as.vit9696.Lilu", CFBundleVersion="1.7.2")))
        self.config = dict(Kernel=dict(Add=[dict(BundlePath="Lilu.kext", Enabled=True, Arch="x86_64", MinKernel="25.0.0", MaxKernel="25.99.99"), dict(BundlePath="Keep.kext", Enabled=True)], Patch=[dict(Comment="keep")], Quirks={}), NVRAM=dict(Add={bridge.APPLE:{"boot-args":'-v test="two  spaces" -reimsmetaloff', "csr-active-config":bytes(4)}}), PlatformInfo=dict(Generic=dict(SystemUUID="keep")))

    def test_global_preserves_identity_patches_and_sip(self):
        original = copy.deepcopy(self.config)
        result = bridge.configure(self.config, self.kexts, "ReimsMetalCompat-test.kext", "Global")
        self.assertEqual([e["BundlePath"] for e in result["Kernel"]["Add"]], ["Lilu.kext", "ReimsMetalCompat-test.kext", "Keep.kext"])
        entry = result["Kernel"]["Add"][1]
        self.assertEqual((entry["MinKernel"], entry["MaxKernel"]), ("25.6.0", "25.6.0"))
        self.assertEqual(result["Kernel"]["Patch"], original["Kernel"]["Patch"])
        self.assertEqual(result["PlatformInfo"], original["PlatformInfo"])
        self.assertEqual(result["NVRAM"]["Add"][bridge.APPLE]["csr-active-config"], bytes(4))
        self.assertIn('test="two  spaces"', result["NVRAM"]["Add"][bridge.APPLE]["boot-args"])
        self.assertTrue(result["NVRAM"]["Add"][bridge.APPLE]["boot-args"].endswith("-reimsmetal"))

    def test_routing_is_idempotent_and_diagnostic_removes_global_flag(self):
        first = bridge.configure(self.config, self.kexts, "ReimsMetalCompat-test.kext", "Global")
        expected = copy.deepcopy(first)
        self.assertEqual(bridge.configure(first, self.kexts, "ReimsMetalCompat-test.kext", "Global"), expected)
        result = bridge.configure(first, self.kexts, "ReimsMetalCompat-other.kext", "Diagnostic")
        self.assertNotIn("-reimsmetal", result["NVRAM"]["Add"][bridge.APPLE]["boot-args"])
        self.assertEqual(len(result["Kernel"]["Add"]), 3)

    def test_lilu_must_load_on_the_target_kernel(self):
        for update in (dict(Enabled=False), dict(MinKernel="25.7.0"), dict(MaxKernel="25.4.0"), dict(Arch="arm64")):
            with self.subTest(update=update):
                config = copy.deepcopy(self.config)
                config["Kernel"]["Add"][0].update(update)
                with self.assertRaises(ValueError):
                    bridge.configure(config, self.kexts, "ReimsMetalCompat-test.kext", "Global")

    def test_provenance_must_match_committed_sources(self):
        valid = bridge.verify_bundle(self.bundle, self.repo)
        self.assertRegex(valid, r"^ReimsMetalCompat-[a-f0-9]{16}\.kext$")
        manifest = json.loads(self.provenance.read_text())
        manifest["files"]["ReimsMetalCompat.cpp"] = "0" * 64
        self.provenance.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "does not match"):
            bridge.verify_bundle(self.bundle, self.repo)

    def test_failure_keeps_source_and_previous_candidate(self):
        source, target = self.root / "OC/config.plist", self.root / "candidate.plist"
        self.config["Kernel"]["Add"] = []
        source.write_bytes(plistlib.dumps(self.config))
        original = source.read_bytes()
        target.write_bytes(b"previous candidate")
        with self.assertRaises(ValueError):
            bridge.prepare(source, target, self.bundle, self.repo, "Global")
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(target.read_bytes(), b"previous candidate")

    def test_repeat_preserves_existing_bundles(self):
        source, target = self.root / "OC/config.plist", self.root / "candidate.plist"
        source.write_bytes(plistlib.dumps(self.config))
        name = bridge.prepare(source, target, self.bundle, self.repo, "Global")
        original = (self.kexts / name / "Contents/MacOS/ReimsMetalCompat").read_bytes()
        source.write_bytes(target.read_bytes())
        self.assertEqual(bridge.prepare(source, target, self.bundle, self.repo, "Global"), name)
        self.assertEqual((self.kexts / name / "Contents/MacOS/ReimsMetalCompat").read_bytes(), original)

    def run_installer(self, fail, wrong_image=False):
        pwsh = shutil.which("pwsh")
        if not pwsh:
            self.skipTest("PowerShell 7 is required for the Windows wrapper test")
        scripts = self.repo / "scripts/windows"
        scripts.mkdir(parents=True)
        real = Path(__file__).parents[1]
        for name in ("start.ps1", "environment.ps1", "usb.ps1", "prepare-metal-compatibility.ps1", "metal-compat-config.py"):
            shutil.copyfile(real / name, scripts / name)
        (scripts / "prepare-opencore.ps1").write_text('''param([string]$Version,[switch]$Rebuild)
$m=Get-Content -Raw "$env:REIMS_VM_ROOT_OVERRIDE/state/opencore.json" | ConvertFrom-Json
[IO.File]::WriteAllText($m.image,'rebuilt image')
if($env:REIMS_TEST_REBUILD_FAIL -eq '1'){throw 'authored rebuild failure'}
''')
        validate = scripts / "ocvalidate.ps1"
        validate.write_text("$global:LASTEXITCODE=0\n")
        source = self.root / "OC/config.plist"
        source.write_bytes(plistlib.dumps(self.config))
        image = self.root / "OpenCore.img"
        image.write_bytes(b"previous image")
        state = self.root / "state"
        state.mkdir()
        manifest = state / "opencore.json"
        manifest.write_text(json.dumps(dict(config=str(source), image=str(image), ocvalidate=str(validate), version="1.0.8")))
        originals = [p.read_bytes() for p in (source, image, manifest)]
        if wrong_image:
            (state / "config.json").write_text(json.dumps(dict(opencore=str(self.root / "other.img"))))
        env = dict(os.environ, REIMS_PROJECT_ROOT_OVERRIDE=str(self.repo), REIMS_VM_ROOT_OVERRIDE=str(self.root), REIMS_TEST_REBUILD_FAIL="1" if fail else "0")
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
        result = subprocess.run([pwsh, "-NoProfile", "-File", str(scripts / "prepare-metal-compatibility.ps1"), "-KextDirectory", str(self.bundle), "-Routing", "Global"], env=env, capture_output=True, text=True)
        if wrong_image:
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("different OpenCore image", result.stderr)
            self.assertEqual([p.read_bytes() for p in (source, image, manifest)], originals)
            self.assertEqual(list(state.glob("metal-compat-backup-*")), [])
            return
        if fail:
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("authored rebuild failure", result.stderr)
            self.assertEqual([p.read_bytes() for p in (source, image, manifest)], originals)
        else:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(plistlib.loads(source.read_bytes())["NVRAM"]["Add"][bridge.APPLE]["boot-args"].endswith("-reimsmetal"))
        backups = list(state.glob("metal-compat-backup-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual((backups[0] / "config.plist").read_bytes(), originals[0])
        self.assertEqual((backups[0] / "OpenCore.img").read_bytes(), originals[1])

    def test_installer_restores_image_and_config_on_rebuild_failure(self):
        self.run_installer(fail=True)

    def test_installer_keeps_backup_on_success(self):
        self.run_installer(fail=False)

    def test_installer_refuses_an_unselected_image(self):
        self.run_installer(fail=False, wrong_image=True)


if __name__ == "__main__":
    unittest.main()
