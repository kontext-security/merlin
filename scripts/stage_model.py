"""Build the exact directory uploaded to the Merlin Hugging Face model repo."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

MODEL_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("dist/Merlin"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        source = args.checkpoint / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, args.output / name)
    model_config_path = args.output / "config.json"
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    model_config.update(
        {
            "id2label": {"0": "safe", "1": "unsafe"},
            "label2id": {"safe": 0, "unsafe": 1},
        }
    )
    model_config_path.write_text(
        json.dumps(model_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.copy2(root / "huggingface/model/README.md", args.output / "README.md")
    shutil.copy2(root / "huggingface/model/merlin_config.json", args.output / "merlin_config.json")

    files = {
        path.name: {"sha256": sha256(path), "size": path.stat().st_size}
        for path in sorted(args.output.iterdir())
        if path.is_file()
    }
    (args.output / "release_manifest.json").write_text(
        json.dumps({"schema_version": 1, "files": files}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
