"""Exact preprocessing used by the released Merlin checkpoint."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

MAX_LENGTH = 512
FIELD_BUDGETS = {
    "request": 96,
    "history": 144,
    "action": 128,
    "schema": 128,
}
FIELD_MARKERS = {
    "request": "[USER_REQUEST]",
    "history": "[INTERACTION_HISTORY]",
    "action": "[CURRENT_ACTION]",
    "schema": "[TOOL_DESCRIPTIONS]",
}

_MARKER_PREFIX = r"^[ \t]*(?:(?:\(\d+\)|\d+[.)])[ \t]*)?(?:\*{0,2})"
_THOUGHT_MARKER = re.compile(
    _MARKER_PREFIX + r"thought(?:\*{0,2})[ \t]*:", re.IGNORECASE | re.MULTILINE
)
_ACTION_MARKER = re.compile(
    _MARKER_PREFIX + r"action(?:\*{0,2})[ \t]*:", re.IGNORECASE | re.MULTILINE
)
_ACTION_INPUT_MARKER = re.compile(
    _MARKER_PREFIX + r"action[ \t_]+input(?:\*{0,2})[ \t]*:",
    re.IGNORECASE | re.MULTILINE,
)
_HISTORY_MARKER = re.compile(
    r"(?im)^(?:\s*\(\d+\)\s*(?P<numbered>Thought|Action Input|Action)\s*:"
    r"|\s*(?P<plain>Observation|Final Answer)\s*:)\s*"
)


@dataclass(frozen=True)
class Invocation:
    """One tool invocation and its surrounding context."""

    user_request: str
    history: str | Sequence[Mapping[str, Any]]
    current_action: str | Mapping[str, Any]
    tool_descriptions: str | Mapping[str, Any] | Sequence[Mapping[str, Any]]


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return _compact_json(value)


def _argument_value(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return ""
    try:
        parsed, end = json.JSONDecoder().raw_decode(stripped)
    except (json.JSONDecodeError, TypeError):
        return stripped
    return parsed if not stripped[end:].strip() else stripped


def normalize_history(history: str | Sequence[Mapping[str, Any]]) -> str:
    """Return deterministic Thought-free JSON history.

    Structured event lists are canonicalized directly. Raw TS-Bench ReAct history
    is parsed into tool/arguments/observation events and Thought/Final Answer text
    is discarded.
    """

    if not isinstance(history, str):
        return _compact_json(list(history))
    if not history.strip():
        return "[]"

    markers = list(_HISTORY_MARKER.finditer(history))
    events: list[dict[str, Any]] = []
    pending: dict[str, Any] = {}

    def flush() -> None:
        nonlocal pending
        if pending:
            events.append(pending)
            pending = {}

    if markers and history[: markers[0].start()].strip():
        events.append({"observation": history[: markers[0].start()].strip()})

    for index, marker in enumerate(markers):
        kind = (marker.group("numbered") or marker.group("plain") or "").casefold()
        end = markers[index + 1].start() if index + 1 < len(markers) else len(history)
        content = history[marker.end() : end].strip()
        if kind == "thought":
            continue
        if kind == "action":
            if "tool" in pending or "observation" in pending:
                flush()
            pending["tool"] = content
        elif kind == "action input":
            pending["arguments"] = _argument_value(content)
        elif kind == "observation":
            pending["observation"] = content
            flush()
        elif kind == "final answer":
            flush()

    flush()
    if not markers:
        events.append({"observation": history.strip()})
    return _compact_json(events)


def _parse_json_argument(value: str) -> str:
    candidate = value.strip()
    if not candidate:
        return ""
    if candidate.startswith("```"):
        opening_end = candidate.find("\n")
        closing_start = candidate.rfind("```")
        if opening_end >= 0 and closing_start > opening_end:
            candidate = candidate[opening_end + 1 : closing_start].strip()
    try:
        parsed, _ = json.JSONDecoder().raw_decode(candidate)
    except (json.JSONDecodeError, TypeError):
        return candidate
    return _compact_json(parsed)


def execution_only_action(action: str | Mapping[str, Any]) -> str:
    """Convert a raw ReAct step or structured tool call to Merlin's action field."""

    if not isinstance(action, str):
        tool_name = action.get("tool_name", action.get("name", action.get("tool", "")))
        arguments = action.get("arguments", action.get("input", {}))
        return f"[TOOL_NAME]\n{_as_text(tool_name)}\n[ARGUMENTS]\n{_as_text(arguments)}".strip()

    actions = list(_ACTION_MARKER.finditer(action))
    action_inputs = list(_ACTION_INPUT_MARKER.finditer(action))
    action_input = action_inputs[0] if action_inputs else None
    preceding = (
        [candidate for candidate in actions if candidate.start() < action_input.start()]
        if action_input is not None
        else []
    )
    action_marker = preceding[-1] if preceding else (actions[0] if actions else None)

    tool_name = ""
    if action_marker is not None:
        suffix = action[action_marker.end() :]
        tool_name = suffix.splitlines()[0].strip() if suffix else ""
    argument_text = action[action_input.end() :].strip() if action_input is not None else ""
    normalized_arguments = _parse_json_argument(argument_text)
    return f"[TOOL_NAME]\n{tool_name}\n[ARGUMENTS]\n{normalized_arguments}".strip()


def model_fields(invocation: Invocation) -> dict[str, str]:
    """Build the four model fields before token-budgeted packing."""

    return {
        "request": invocation.user_request,
        "history": normalize_history(invocation.history),
        "action": execution_only_action(invocation.current_action),
        "schema": _as_text(invocation.tool_descriptions),
    }


def _head(values: Sequence[int], budget: int) -> list[int]:
    return list(values[:budget])


def _head_tail(values: Sequence[int], budget: int) -> list[int]:
    if len(values) <= budget:
        return list(values)
    if budget == 0:
        return []
    head = (budget + 1) // 2
    tail = budget - head
    return [*values[:head], *values[-tail:]] if tail else list(values[:head])


def pack_invocation(tokenizer: Any, invocation: Invocation) -> dict[str, list[int]]:
    """Pack fields exactly as used for Merlin training and evaluation."""

    fields = model_fields(invocation)
    start_id = tokenizer.cls_token_id
    if start_id is None:
        start_id = tokenizer.bos_token_id
    separator_id = tokenizer.sep_token_id
    if separator_id is None:
        separator_id = tokenizer.eos_token_id
    if start_id is None or separator_id is None:
        raise ValueError("Tokenizer requires start and separator tokens")

    input_ids = [int(start_id)]
    for name in ("request", "history", "action", "schema"):
        marker_id = tokenizer.convert_tokens_to_ids(FIELD_MARKERS[name])
        if marker_id is None or marker_id == tokenizer.unk_token_id:
            raise ValueError(f"Tokenizer is missing field marker {FIELD_MARKERS[name]}")
        tokens = tokenizer.encode(fields[name], add_special_tokens=False)
        selected = (
            _head(tokens, FIELD_BUDGETS[name])
            if name == "request"
            else _head_tail(tokens, FIELD_BUDGETS[name])
        )
        input_ids.extend((int(marker_id), *selected, int(separator_id)))

    if len(input_ids) > MAX_LENGTH:
        raise ValueError(f"Packed sequence has {len(input_ids)} tokens, over {MAX_LENGTH}")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else separator_id
    attention_mask = [1] * len(input_ids)
    padding = MAX_LENGTH - len(input_ids)
    input_ids.extend([int(pad_id)] * padding)
    attention_mask.extend([0] * padding)
    return {"input_ids": input_ids, "attention_mask": attention_mask}
