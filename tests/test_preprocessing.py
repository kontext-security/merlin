from __future__ import annotations

import json

from merlin_guard.preprocessing import (
    FIELD_MARKERS,
    Invocation,
    execution_only_action,
    model_fields,
    normalize_history,
    pack_invocation,
)


class StubTokenizer:
    cls_token_id = 1
    bos_token_id = None
    sep_token_id = 2
    eos_token_id = None
    pad_token_id = 0
    unk_token_id = 3

    def __init__(self) -> None:
        self.markers = {marker: 10 + index for index, marker in enumerate(FIELD_MARKERS.values())}

    def convert_tokens_to_ids(self, value: str) -> int:
        return self.markers.get(value, self.unk_token_id)

    def encode(self, value: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [100 + index for index, _ in enumerate(value.split())]


def test_normalize_history_discards_thought_and_final_answer() -> None:
    raw = """(1) Thought: private reasoning
(1) Action: search_mail
(1) Action Input: {"query": "invoice"}
Observation: one result
Final Answer: done"""

    normalized = normalize_history(raw)

    assert json.loads(normalized) == [
        {"arguments": {"query": "invoice"}, "observation": "one result", "tool": "search_mail"}
    ]
    assert "private reasoning" not in normalized
    assert "Final Answer" not in normalized


def test_structured_history_is_canonicalized() -> None:
    assert normalize_history([{"tool": "b", "arguments": {"z": 1, "a": 2}}]) == (
        '[{"arguments":{"a":2,"z":1},"tool":"b"}]'
    )


def test_execution_only_action_accepts_raw_react() -> None:
    value = execution_only_action(
        'Thought: hidden\nAction: transfer\nAction Input: {"amount": 10, "to": "x"}'
    )
    assert value == '[TOOL_NAME]\ntransfer\n[ARGUMENTS]\n{"amount":10,"to":"x"}'
    assert "hidden" not in value


def test_execution_only_action_accepts_structured_call() -> None:
    value = execution_only_action({"name": "lookup", "arguments": {"id": 7}})
    assert value == '[TOOL_NAME]\nlookup\n[ARGUMENTS]\n{"id":7}'


def test_pack_invocation_is_fixed_length_and_field_marked() -> None:
    tokenizer = StubTokenizer()
    invocation = Invocation(
        user_request="find an invoice",
        history=[],
        current_action={"name": "search", "arguments": {"q": "invoice"}},
        tool_descriptions=[{"name": "search"}],
    )

    packed = pack_invocation(tokenizer, invocation)

    assert len(packed["input_ids"]) == 512
    assert len(packed["attention_mask"]) == 512
    assert packed["input_ids"][0] == tokenizer.cls_token_id
    assert all(marker_id in packed["input_ids"] for marker_id in tokenizer.markers.values())
    assert sum(packed["attention_mask"]) < 512


def test_model_fields_match_released_representation() -> None:
    fields = model_fields(
        Invocation(
            user_request="request",
            history=[{"tool": "read", "arguments": {"path": "a"}, "observation": "ok"}],
            current_action={"tool_name": "write", "arguments": {"path": "b"}},
            tool_descriptions={"write": {"description": "write a file"}},
        )
    )
    assert set(fields) == {"request", "history", "action", "schema"}
    assert "[TOOL_NAME]\nwrite" in fields["action"]
