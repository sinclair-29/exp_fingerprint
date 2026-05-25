from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

import numpy as np


def success_rate(results: Iterable[dict[str, Any]]) -> float:
    rows = list(results)
    if not rows:
        return 0.0
    return sum(bool(row.get("success")) for row in rows) / len(rows)


def invalid_rate(results: Iterable[dict[str, Any]]) -> float:
    rows = list(results)
    if not rows:
        return 0.0
    return sum(bool(row.get("metadata", {}).get("invalid")) for row in rows) / len(rows)


def tpr_fpr_from_labeled_results(results: Iterable[dict[str, Any]]) -> dict[str, float | None]:
    rows = [row for row in results if "label" in row.get("metadata", {})]
    if not rows:
        return {"tpr": None, "fpr": None}
    positives = [row for row in rows if bool(row["metadata"]["label"])]
    negatives = [row for row in rows if not bool(row["metadata"]["label"])]
    tpr = success_rate(positives) if positives else None
    fpr = success_rate(negatives) if negatives else None
    return {"tpr": tpr, "fpr": fpr}


def bitwise_accuracy(reference_bits: list[int], predicted_bits: list[int]) -> float:
    if not reference_bits:
        return 0.0
    if len(reference_bits) != len(predicted_bits):
        raise ValueError("Reference and predicted bit lists must have the same length")
    matches = sum(int(a == b) for a, b in zip(reference_bits, predicted_bits))
    return matches / len(reference_bits)


def threshold_from_negative_accuracies(negative_accuracies: list[float] | None, z: float = 1.64, default: float = 0.7) -> float:
    if not negative_accuracies:
        return default
    arr = np.asarray(negative_accuracies, dtype=float)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return min(1.0, float(arr.mean()) + z * std)


def summarize_by_model(results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[(row.get("method", "unknown"), row.get("suspect_model", "unknown"))].append(row)
    summaries = []
    for (method, suspect_model), rows in sorted(grouped.items()):
        labeled = tpr_fpr_from_labeled_results(rows)
        summaries.append(
            {
                "method": method,
                "suspect_model": suspect_model,
                "n": len(rows),
                "success_rate": success_rate(rows),
                "invalid_rate": invalid_rate(rows),
                "tpr": labeled["tpr"],
                "fpr": labeled["fpr"],
            }
        )
    return summaries


def safe_mean(values: Iterable[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if not clean:
        return None
    return sum(clean) / len(clean)
