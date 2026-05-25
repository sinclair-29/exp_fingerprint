from __future__ import annotations

from pathlib import Path
from typing import Any

from llmfp.core.io import save_jsonl
from llmfp.core.model_backend import ModelBackend
from llmfp.registry import get_method


def run_construct(method_name: str, method_cfg: dict[str, Any], model_cfg: dict[str, Any], out_path: str | Path):
    method = get_method(method_name)
    backend = ModelBackend.from_config(model_cfg)
    artifacts = []
    for task in method.build_tasks(method_cfg):
        artifact = method.construct(task, backend, method_cfg)
        if artifact is not None:
            artifacts.append(artifact)
    save_jsonl(out_path, artifacts)
    return artifacts
