from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


JsonDict = dict[str, Any]


@dataclass
class FingerprintTask:
    task_id: str
    method: str
    input_text: str
    target: str
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass
class FingerprintArtifact:
    fingerprint_id: str
    method: str
    base_model: str
    task_id: str
    prompt_text: str
    optimized_text: str
    target: str
    best_loss: float | None = None
    best_step: int | None = None
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass
class VerificationResult:
    method: str
    suspect_model: str
    fingerprint_id: str
    success: bool
    score: float
    base_model: str | None = None
    raw_output: str | None = None
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass
class MethodSummary:
    method: str
    metrics: JsonDict
    num_results: int
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)
