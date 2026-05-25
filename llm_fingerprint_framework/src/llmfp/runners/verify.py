from __future__ import annotations

from pathlib import Path
from typing import Any

from llmfp.core.io import load_jsonl, save_jsonl
from llmfp.core.model_backend import ModelBackend
from llmfp.registry import get_method


def run_verify(
    method_name: str,
    method_cfg: dict[str, Any],
    suspect_model_cfg: dict[str, Any],
    fingerprints_path: str | Path,
    out_path: str | Path,
):
    method = get_method(method_name)
    backend = ModelBackend.from_config(suspect_model_cfg)
    artifacts = load_jsonl(fingerprints_path)
    results = [method.verify(artifact, backend, method_cfg) for artifact in artifacts]
    save_jsonl(out_path, results)
    return results


