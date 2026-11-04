"""Stage the allowlisted V3 runtime/model release, never private evaluation records."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = (
    "weights.00.gz",
    "weights.01.gz",
    "weights.02.gz",
    "weights.03.gz",
    "weights.04.gz",
    "tokenizer.json.gz",
    "agreement.json.gz",
    "request_gate.json.gz",
)


NATIVE_FILES = (
    "cmd/merlin/main.go",
    "go.mod",
    "go.sum",
    "stepsafety/agreement.go",
    "stepsafety/context.go",
    "stepsafety/model/ATTRIBUTION.md",
    "stepsafety/model/LICENSE",
    "stepsafety/model/PROVENANCE.json",
    "stepsafety/model/native/PROVENANCE.json",
    "stepsafety/model/native/candidate.json",
    "stepsafety/model/native/manifest.json",
    "stepsafety/modelstore.go",
    "stepsafety/native_assets.go",
    "stepsafety/native_assets_nomodel.go",
    "stepsafety/native_backend.go",
    "stepsafety/native_linear_arm64.go",
    "stepsafety/native_linear_arm64.s",
    "stepsafety/native_linear_generic.go",
    "stepsafety/native_model.go",
    "stepsafety/native_weights.go",
    "stepsafety/native_workspace.go",
    "stepsafety/normalization.go",
    "stepsafety/packing.go",
    "stepsafety/request_gate.go",
    "stepsafety/runtime_artifacts.go",
    "stepsafety/runtime_store.go",
    "stepsafety/scoped.go",
    "stepsafety/tokenizer.go",
    "stepsafety/types.go",
    "stepsafety/unicode9.go",
    "stepsafety/unigram.go",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stage(assets: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Use a new output directory to prevent stale files entering the release")
    provenance = json.loads((ROOT / "native/stepsafety/model/native/manifest.json").read_text())
    pinned = {row["name"]: row for row in provenance["files"]}
    selected = [(ROOT / "native" / name, Path("native") / name) for name in NATIVE_FILES]
    selected += [
        (ROOT / "huggingface/v3/README.md", Path("README.md")),
        (ROOT / "results/v3-ungated.json", Path("evaluation.json")),
        (ROOT / "LICENSE", Path("LICENSE")),
        (ROOT / "NOTICE", Path("NOTICE")),
    ]
    for name in ASSETS:
        source = assets / name
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Missing regular asset: {name}")
        spec = pinned[name]
        if digest(source) != spec["sha256"] or source.stat().st_size != spec["size_bytes"]:
            raise ValueError(f"Asset checksum mismatch: {name}")
        selected.append((source, Path("native/stepsafety/model/native") / name))
    # Reject symlinks and sources outside the explicit public tree/asset allowlist.
    for source, _ in selected:
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Not a regular release file: {source.name}")
    output.mkdir(parents=True)
    for source, relative in selected:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    manifest = {
        "schema_version": 1,
        "runtime": "merlin-v3-ungated-20261104",
        "weights_candidate": "merlin-v3-precision-20261007",
        "contains_private_examples": False,
        "files": {
            str(rel): {"sha256": digest(output / rel), "size": (output / rel).stat().st_size}
            for _, rel in selected
        },
    }
    (output / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = stage(args.assets, args.output)
    print(f"Staged {len(result['files'])} allowlisted files")


if __name__ == "__main__":
    main()
