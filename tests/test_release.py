from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_model_release_config_matches_reference_package() -> None:
    config = json.loads((ROOT / "huggingface/model/merlin_config.json").read_text())
    assert config["default_threshold"] == 0.5
    assert config["preprocessing"]["max_length"] == 512
    assert config["preprocessing"]["field_budgets"] == {
        "request": 96,
        "history": 144,
        "action": 128,
        "schema": 128,
    }


def test_manifest_contains_only_checksum_pinned_https_assets() -> None:
    manifest = json.loads((ROOT / "manifests/ts_bench_assets.json").read_text())
    assert manifest["assets"]
    for asset in manifest["assets"]:
        assert asset["source"].startswith("https://media.githubusercontent.com/")
        assert len(asset["sha256"]) == 64
        assert asset["size"] > 0
