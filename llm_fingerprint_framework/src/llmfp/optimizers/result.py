from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OptimizationResult:
    best_ids: list[int]
    best_text: str
    best_loss: float
    best_step: int
    loss_history: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
