#!/usr/bin/env python3
"""Export committed bridge sources and verify their build identity."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile

MANIFEST = "source-manifest.json"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(directory, *args):
    return subprocess.check_output(["git", "-C", str(directory), *args])


def export(source, destination):
    source = source.resolve()
    repository = Path(git(source, "rev-parse", "--show-toplevel").decode().strip())
    relative = source.relative_to(repository).as_posix()
    revision = git(source, "rev-parse", "HEAD").decode().strip()
    if git(repository, "status", "--porcelain", "--untracked-files=all", "--", relative):
        raise ValueError("Source has uncommitted changes; review and commit before exporting")
    archive = git(repository, "archive", "--format=zip", revision + ":" + relative)
    with zipfile.ZipFile(io.BytesIO(archive)) as entries:
        files = {}
        for entry in entries.infolist():
            if entry.is_dir():
                continue
            if any(c in entry.filename for c in "/\\:") or entry.filename in (".", "..", MANIFEST):
                raise ValueError("Unexpected source archive path: " + entry.filename)
            if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Source archives cannot contain symbolic links")
            files[entry.filename] = entries.read(entry)
    destination.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        (destination / name).write_bytes(content)
    manifest = {"schema": 1, "revision": revision,
                "files": {name: digest(content) for name, content in sorted(files.items())}}
    (destination / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    verify(destination)
    return revision


def verify(source):
    manifest = json.loads((source / MANIFEST).read_text(encoding="utf8"))
    if manifest.get("schema") != 1 or not isinstance(manifest.get("files"), dict) or not manifest["files"]:
        raise ValueError("Invalid source manifest")
    if {p.name for p in source.iterdir()} != set(manifest["files"]) | {MANIFEST}:
        raise ValueError("Source snapshot contains missing or unexpected files")
    for name, expected in manifest["files"].items():
        if not name or any(c in name for c in "/\\:") or name in (".", "..", MANIFEST):
            raise ValueError("Invalid source filename")
        path = source / name
        if path.is_symlink() or digest(path.read_bytes()) != expected:
            raise ValueError("Source changed: " + name)
    return manifest["revision"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    exporter = commands.add_parser("export")
    exporter.add_argument("destination", type=Path)
    checker = commands.add_parser("verify")
    checker.add_argument("source", type=Path)
    args = parser.parse_args()
    try:
        revision = (export(Path(__file__).parent, args.destination) if args.command == "export"
                    else verify(args.source))
        print("Verified source commit " + revision)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
