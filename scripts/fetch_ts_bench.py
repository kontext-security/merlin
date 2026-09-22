"""Download checksum-pinned TS-Bench assets without executing their contents."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--kind",
        action="append",
        choices=("eval", "train", "validation", "reference"),
        help="Asset kind to fetch; repeat as needed. Defaults to eval.",
    )
    args = parser.parse_args()
    kinds = set(args.kind or ["eval"])
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "manifests/ts_bench_assets.json").read_text())

    for asset in manifest["assets"]:
        if asset["kind"] not in kinds:
            continue
        destination = args.output / asset["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if (
            destination.is_file()
            and destination.stat().st_size == asset["size"]
            and sha256(destination) == asset["sha256"]
        ):
            print(f"verified   {asset['path']}")
            continue
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        request = urllib.request.Request(
            asset["source"], headers={"User-Agent": "Merlin/0.1 dataset fetcher"}
        )
        try:
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                temporary.open("wb") as output,
            ):
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            if temporary.stat().st_size != asset["size"] or sha256(temporary) != asset["sha256"]:
                raise ValueError(f"Integrity check failed for {asset['path']}")
            temporary.replace(destination)
            print(f"downloaded {asset['path']}")
        finally:
            temporary.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
