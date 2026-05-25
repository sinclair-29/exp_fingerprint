from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from llmfp.schemas import FingerprintArtifact, FingerprintTask, VerificationResult


class FingerprintMethod(ABC):
    name: str

    @abstractmethod
    def build_tasks(self, cfg: dict[str, Any]) -> list[FingerprintTask]:
        raise NotImplementedError

    @abstractmethod
    def construct(self, task: FingerprintTask, model_backend, cfg: dict[str, Any]) -> FingerprintArtifact | None:
        raise NotImplementedError

    @abstractmethod
    def verify(self, artifact: FingerprintArtifact | dict[str, Any], suspect_backend, cfg: dict[str, Any]) -> VerificationResult:
        raise NotImplementedError

    @abstractmethod
    def aggregate(self, results: list[VerificationResult | dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


def artifact_get(artifact: FingerprintArtifact | dict[str, Any], key: str, default=None):
    if isinstance(artifact, dict):
        return artifact.get(key, default)
    return getattr(artifact, key, default)


def result_dicts(results: list[VerificationResult | dict[str, Any]]) -> list[dict[str, Any]]:
    return [row.to_dict() if hasattr(row, "to_dict") else row for row in results]
