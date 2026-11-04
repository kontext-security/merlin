from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def test_staging_ignores_private_files_and_pins_assets(tmp_path, monkeypatch):
    stage = module("stage_native")
    public = tmp_path / "public"
    assets = tmp_path / "assets"
    for directory in ["native/stepsafety/model/native", "huggingface/v3", "results"]:
        (public / directory).mkdir(parents=True)
    for relative in ["LICENSE", "NOTICE", "huggingface/v3/README.md", "results/v3-ungated.json"]:
        (public / relative).write_text("{}")
    assets.mkdir()
    (assets / "weights.00.gz").write_bytes(b"weights")
    (assets / "private-transcripts.json").write_text("PRIVATE EXAMPLE MUST NOT LEAK")
    (public / "native/private-captures.json").write_text("PRIVATE EXAMPLE MUST NOT LEAK")
    pinned = {
        "files": [
            {
                "name": "weights.00.gz",
                "size_bytes": 7,
                "sha256": stage.digest(assets / "weights.00.gz"),
            }
        ]
    }
    (public / "native/stepsafety/model/native/manifest.json").write_text(json.dumps(pinned))
    monkeypatch.setattr(stage, "ROOT", public)
    monkeypatch.setattr(stage, "ASSETS", ("weights.00.gz",))
    release = tmp_path / "release"
    manifest = stage.stage(assets, release)
    assert not any("private" in name for name in manifest["files"])
    publish = module("publish_native")
    publish.verified_files(release)
    (release / "private.json").write_text("PRIVATE")
    with pytest.raises(ValueError, match="Unmanifested"):
        publish.verified_files(release)
    (assets / "weights.00.gz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        stage.stage(assets, tmp_path / "bad-release")
    with pytest.raises(ValueError, match="new output"):
        stage.stage(assets, release)
