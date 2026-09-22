from __future__ import annotations

import pytest

from merlin_guard.model import Verdict, calibrated_probability


def test_calibrated_probability_is_sigmoid_of_affine_margin() -> None:
    assert calibrated_probability(0.0, 1.0, 0.0) == pytest.approx(0.5)
    assert calibrated_probability(1000.0, 1.0, 0.0) == pytest.approx(1.0)
    assert calibrated_probability(-1000.0, 1.0, 0.0) == pytest.approx(0.0)


def test_verdict_serialization() -> None:
    verdict = Verdict(label="unsafe", unsafe_probability=0.9, threshold=0.5)
    assert verdict.unsafe
    assert verdict.to_dict() == {
        "label": "unsafe",
        "unsafe": True,
        "unsafe_probability": 0.9,
        "threshold": 0.5,
    }


def test_predict_many_rejects_non_positive_batch_size() -> None:
    from merlin_guard.model import MerlinGuard

    guard = MerlinGuard(
        model=None,
        tokenizer=None,
        coefficient=1.0,
        intercept=0.0,
        threshold=0.5,
        device="cpu",
    )
    with pytest.raises(ValueError, match="batch_size must be positive"):
        guard.predict_many([object()], batch_size=0)  # type: ignore[list-item]
