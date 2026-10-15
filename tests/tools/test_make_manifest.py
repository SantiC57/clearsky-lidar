import json
import os

from tools.make_manifest import MANIFEST_NAME, build_manifest, main

# Expected hashes computed independently with: printf '<content>' | sha256sum
HELLO_SHA = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
WORLD_SHA = "e258d248fda94c63753607f7c4494ee0fcbe92f1a76bfdac795c9d84101eb317"
NESTED_SHA = "233562de1a0288b139c4fa40b7d189f806e906eeb048517aeb67f34ac0e2faf1"


def _write(path, data):
    with open(str(path), "wb") as f:
        f.write(data)


def test_two_files_with_known_hashes_sorted_by_path(tmp_path):
    _write(tmp_path / "b.txt", b"world\n")
    _write(tmp_path / "a.txt", b"hello")

    manifest = build_manifest(str(tmp_path))

    assert manifest["files"] == [
        {"path": "a.txt", "size": 5, "sha256": HELLO_SHA},
        {"path": "b.txt", "size": 6, "sha256": WORLD_SHA},
    ]


def test_empty_folder_gives_empty_file_list(tmp_path):
    assert build_manifest(str(tmp_path))["files"] == []


def test_two_runs_give_identical_manifest_that_excludes_itself(tmp_path):
    _write(tmp_path / "a.txt", b"hello")
    manifest_path = os.path.join(str(tmp_path), MANIFEST_NAME)

    assert main([str(tmp_path)]) == 0
    with open(manifest_path, "rb") as f:
        first = f.read()
    assert main([str(tmp_path)]) == 0
    with open(manifest_path, "rb") as f:
        second = f.read()

    assert first == second
    paths = [e["path"] for e in json.loads(second.decode("utf-8"))["files"]]
    assert paths == ["a.txt"]


def test_subfolder_files_use_relative_forward_slash_paths(tmp_path):
    (tmp_path / "raw").mkdir()
    _write(tmp_path / "raw" / "c.bin", b"nested")

    files = build_manifest(str(tmp_path))["files"]

    assert files == [{"path": "raw/c.bin", "size": 6, "sha256": NESTED_SHA}]


def test_missing_folder_returns_error_code(tmp_path):
    assert main([str(tmp_path / "does-not-exist")]) == 1
