"""Command-line interface for Merlin."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from merlin_guard.model import DEFAULT_MODEL_ID, MerlinGuard
from merlin_guard.preprocessing import Invocation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify one contextual tool invocation with Merlin."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="JSON input path. Reads a JSON object from stdin when omitted.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--threshold", type=float)
    return parser


def _load_input(path: Path | None) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8") if path else sys.stdin.read()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise TypeError("Input must be one JSON object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    value = _load_input(args.input)
    invocation = Invocation(
        user_request=str(value.get("user_request", "")),
        history=value.get("history", ""),
        current_action=value.get("current_action", ""),
        tool_descriptions=value.get("tool_descriptions", ""),
    )
    guard = MerlinGuard.from_pretrained(
        args.model,
        device=args.device,
        threshold=args.threshold,
        revision=args.revision,
    )
    print(json.dumps(guard.predict(invocation).to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
