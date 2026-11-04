"""Publish only checksum-verified files from the V3 public release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

REPO = "kontext-security/Merlin"
PREFIX = "v3-ungated/"


def verified_files(folder: Path) -> list[Path]:
    manifest_path = folder / "release_manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("Symlinked manifest")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest["runtime"] != "merlin-v3-ungated-20261104"
        or manifest["contains_private_examples"] is not False
    ):
        raise ValueError("Unexpected release")
    paths = []
    for name, expected in manifest["files"].items():
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Unsafe manifest path")
        p = folder / rel
        if any(part.is_symlink() for part in [p, *p.parents] if part != folder.parent):
            raise ValueError("Symlinked release path")
        data = p.read_bytes()
        if len(data) != expected["size"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise ValueError(f"Changed release file: {name}")
        paths.append(p)
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    if actual != set(manifest["files"]) | {"release_manifest.json"}:
        raise ValueError("Unmanifested files in release directory")
    return paths + [manifest_path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    files = verified_files(args.folder)
    print(f"Verified {len(files)} public files; private examples are excluded")
    if not args.publish:
        return
    api = HfApi()
    revision = api.model_info(REPO).sha
    card = Path(hf_hub_download(REPO, "README.md", revision=revision)).read_text()
    note = """<!-- merlin-v3-ungated-start -->
## V3 ungated native evaluation release

The [V3 native Go package](v3-ungated/README.md) is available in `v3-ungated/`.
It keeps the V3 weights, threshold and action-agreement filter while removing the
three restrictive eligibility gates. AgentDojo recall is 24.1% (85/352), with
20/868 benign cases flagged. See the package for coverage, private-data aggregate
results and limitations. No private evaluation examples are included.

The root checkpoint and Python usage below remain **legacy v0.1**, with their
original results. They are not the V3 runtime. This is a controlled evaluation
release, not a claim of uniform performance across deployments.
<!-- merlin-v3-ungated-end -->
"""
    if "<!-- merlin-v3-ungated-start -->" in card:
        raise ValueError("V3 notice already exists; review remote changes before republishing")
    pos = card.find("\n---", 4) + 4 if card.startswith("---\n") else 0
    card = card[:pos] + "\n\n" + note + "\n" + card[pos:]
    operations = [
        CommitOperationAdd(
            path_in_repo=PREFIX + p.relative_to(args.folder).as_posix(), path_or_fileobj=str(p)
        )
        for p in files
    ]
    operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.encode()))
    result = api.create_commit(
        REPO,
        operations=operations,
        commit_message="Add Merlin V3 ungated native evaluation release",
        parent_commit=revision,
    )
    print(result.commit_url)


if __name__ == "__main__":
    main()
