"""Local inference for Merlin."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin_guard.preprocessing import Invocation, pack_invocation

DEFAULT_MODEL_ID = "kontext-security/Merlin"


@dataclass(frozen=True)
class Verdict:
    label: str
    unsafe_probability: float
    threshold: float

    @property
    def unsafe(self) -> bool:
        return self.label == "unsafe"

    def to_dict(self) -> dict[str, str | float | bool]:
        return {
            "label": self.label,
            "unsafe": self.unsafe,
            "unsafe_probability": self.unsafe_probability,
            "threshold": self.threshold,
        }


def calibrated_probability(margin: float, coefficient: float, intercept: float) -> float:
    value = coefficient * margin + intercept
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _resolve_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class MerlinGuard:
    """A calibrated, local binary classifier for contextual tool invocations."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        coefficient: float,
        intercept: float,
        threshold: float,
        device: str,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.coefficient = coefficient
        self.intercept = intercept
        self.threshold = threshold
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_id: str | Path = DEFAULT_MODEL_ID,
        *,
        device: str = "auto",
        threshold: float | None = None,
        revision: str | None = None,
    ) -> MerlinGuard:
        from huggingface_hub import hf_hub_download
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        source = str(model_id)
        local_config = Path(source) / "merlin_config.json"
        if local_config.exists():
            config_path = local_config
        else:
            config_path = Path(hf_hub_download(source, "merlin_config.json", revision=revision))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        selected_device = _resolve_device(device)
        tokenizer = AutoTokenizer.from_pretrained(source, revision=revision)
        model = AutoModelForSequenceClassification.from_pretrained(source, revision=revision)
        model.to(selected_device)
        model.eval()
        calibration = config["calibration"]
        return cls(
            model,
            tokenizer,
            coefficient=float(calibration["coefficient"]),
            intercept=float(calibration["intercept"]),
            threshold=float(config["default_threshold"] if threshold is None else threshold),
            device=selected_device,
        )

    def predict(self, invocation: Invocation) -> Verdict:
        return self.predict_many([invocation])[0]

    def predict_many(
        self, invocations: Iterable[Invocation], *, batch_size: int = 16
    ) -> list[Verdict]:
        import torch

        items = list(invocations)
        if not items:
            return []
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        packed = [pack_invocation(self.tokenizer, item) for item in items]
        margins: list[float] = []
        for start in range(0, len(packed), batch_size):
            batch = packed[start : start + batch_size]
            inputs = {
                name: torch.tensor([row[name] for row in batch], dtype=torch.long).to(self.device)
                for name in ("input_ids", "attention_mask")
            }
            with torch.inference_mode():
                logits = self.model(**inputs).logits.detach().cpu().float()
            margins.extend(float(value) for value in logits[:, 1] - logits[:, 0])
        probabilities = [
            calibrated_probability(value, self.coefficient, self.intercept) for value in margins
        ]
        return [
            Verdict(
                label="unsafe" if probability >= self.threshold else "safe",
                unsafe_probability=probability,
                threshold=self.threshold,
            )
            for probability in probabilities
        ]
