"""Prepare a Tahoe bridge configuration without replacing existing kexts."""
import argparse
import hashlib
import json
from pathlib import Path
import plistlib
import re
import shutil
import subprocess

APPLE = "7C436110-AB2A-4BBB-A880-FE41995C9F82"


def verify_bundle(bundle, repository):
    info = plistlib.loads((bundle / "Contents/Info.plist").read_bytes())
    if (info.get("CFBundleIdentifier") != "io.reims.MetalCompat"
            or info.get("CFBundleExecutable") != "ReimsMetalCompat"):
        raise ValueError("Expected the authored ReimsMetalCompat bundle")
    if not (bundle / "Contents/MacOS/ReimsMetalCompat").stat().st_size:
        raise ValueError("The bridge executable is empty")
    manifest = json.loads((bundle / "Contents/Resources/source-manifest.json").read_text())
    revision = manifest.get("revision", "")
    files = manifest.get("files", {})
    if (manifest.get("schema") != 1 or not re.fullmatch(r"[a-f0-9]{40}", revision)
            or not isinstance(files, dict) or "ReimsMetalCompat.cpp" not in files):
        raise ValueError("Missing committed bridge source provenance")
    paths = subprocess.check_output([
        "git", "-C", str(repository), "ls-tree", "-r", "--name-only",
        revision, "--", "guest/metal-compat/",
    ]).decode().splitlines()
    names = {p.removeprefix("guest/metal-compat/") for p in paths}
    if set(files) != names:
        raise ValueError("Source manifest must cover the complete committed snapshot")
    for name, expected in files.items():
        if not name or any(c in name for c in "/\\:") or name in (".", ".."):
            raise ValueError("Invalid source manifest filename")
        data = subprocess.check_output([
            "git", "-C", str(repository), "show",
            f"{revision}:guest/metal-compat/{name}",
        ])
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Source manifest does not match the repository: " + name)
    digest = hashlib.sha256()
    for item in sorted(bundle.rglob("*")):
        if item.is_symlink():
            raise ValueError("Bridge bundles must not contain symbolic links")
        if item.is_file():
            digest.update(item.relative_to(bundle).as_posix().encode() + b"\0")
            digest.update(hashlib.sha256(item.read_bytes()).digest())
    return "ReimsMetalCompat-" + digest.hexdigest()[:16] + ".kext"


def configure(config, kexts, name, routing):
    # The validated global desktop configuration is currently Darwin 25.6.0.
    entries = config["Kernel"]["Add"]
    lilu = next((e for e in entries if e.get("Enabled")
                 and e.get("BundlePath") == "Lilu.kext"), None)
    if not lilu:
        raise ValueError("Install and enable Lilu first (prepare-update-compatibility.ps1)")
    info = plistlib.loads((kexts / "Lilu.kext/Contents/Info.plist").read_bytes())
    version = tuple(int(v) for v in info["CFBundleVersion"].split("."))
    if info.get("CFBundleIdentifier") != "as.vit9696.Lilu" or version < (1, 7, 2):
        raise ValueError("The bridge needs Lilu 1.7.2 or newer")
    for field, lower in (("MinKernel", True), ("MaxKernel", False)):
        bound = lilu.get(field, "")
        if bound:
            value = tuple(int(v) for v in bound.split("."))
            if (lower and value > (25, 6, 0)) or (not lower and value < (25, 6, 0)):
                raise ValueError("Lilu's kernel range excludes Darwin 25.6.0")
    if lilu.get("Arch", "Any") not in ("Any", "x86_64"):
        raise ValueError("Lilu must be enabled for x86_64")
    def bridge(entry):
        path = entry.get("BundlePath", "")
        return path == "ReimsMetalCompat.kext" or (
            path.startswith("ReimsMetalCompat-") and path.endswith(".kext"))
    entries = [e for e in entries if not bridge(e)]
    position = entries.index(lilu) + 1
    entries.insert(position, dict(
        Arch="x86_64", BundlePath=name, Comment="Experimental Tahoe Metal bridge",
        Enabled=True, ExecutablePath="Contents/MacOS/ReimsMetalCompat",
        MinKernel="25.6.0", MaxKernel="25.6.0", PlistPath="Contents/Info.plist",
    ))
    config["Kernel"]["Add"] = entries
    config["Kernel"]["Quirks"]["DisableLinkeditJettison"] = True
    nvram = config["NVRAM"]["Add"].setdefault(APPLE, {})
    args = re.sub(r"(?<!\S)-reimsmetal(?:off)?(?!\S)", "", nvram.get("boot-args", "")).strip()
    nvram["boot-args"] = args + (" -reimsmetal" if routing == "Global" else "")
    return config


def prepare(source, target, bundle, repository, routing):
    kexts = source.parent / "Kexts"
    name = verify_bundle(bundle, repository)
    config = configure(plistlib.loads(source.read_bytes()), kexts, name, routing)
    destination = kexts / name
    if destination.exists():
        if verify_bundle(destination, repository) != name:
            raise ValueError("The existing versioned bridge bundle has changed")
    else:
        shutil.copytree(bundle, destination)
    target.write_bytes(plistlib.dumps(config, sort_keys=False))
    return name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("source", "target", "bundle", "repository"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--routing", choices=("Diagnostic", "Global"), required=True)
    args = parser.parse_args()
    print(prepare(args.source, args.target, args.bundle, args.repository, args.routing))
