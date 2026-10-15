#!/usr/bin/env python3
"""Write manifest.json listing every file of a folder with its size and SHA-256.

Used to prove that the PC and the Jetson evaluated exactly the same photos.

Usage:
    python tools/make_manifest.py data/experiments/<ID>
"""
import argparse
import hashlib
import json
import os
import sys

MANIFEST_NAME = "manifest.json"
_CHUNK = 1 << 20


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(folder):
    """Return {"files": [{"path", "size", "sha256"}, ...]} sorted by path.

    Paths are relative to ``folder`` and use forward slashes. The root-level
    manifest.json is skipped so the manifest never lists itself.
    """
    entries = []
    for dirpath, _dirnames, filenames in os.walk(folder):
        for name in filenames:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, folder).replace(os.sep, "/")
            if rel == MANIFEST_NAME:
                continue
            entries.append(
                {"path": rel, "size": os.path.getsize(full), "sha256": _sha256(full)}
            )
    entries.sort(key=lambda e: e["path"])
    return {"files": entries}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", help="folder to list; manifest.json is written inside it")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.folder):
        sys.stderr.write("error: not a folder: {}\n".format(args.folder))
        return 1

    manifest = build_manifest(args.folder)
    out = os.path.join(args.folder, MANIFEST_NAME)
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print("{}: {} files".format(out, len(manifest["files"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
