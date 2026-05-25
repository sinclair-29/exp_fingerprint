from __future__ import annotations

from pathlib import Path
from typing import Any

from llmfp.core.io import load_yaml
from llmfp.runners.construct import run_construct
from llmfp.runners.verify import run_verify


def run_benchmark(cfg: dict[str, Any]):
    output_dir = Path(cfg.get("output_dir", "results"))
    model_cfg = load_yaml(cfg["model_config"])
    suspect_cfg = load_yaml(cfg.get("suspect_model_config", cfg["model_config"]))
    outputs = []
    for item in cfg.get("methods", []):
        method_name = item["name"]
        method_cfg = load_yaml(item["config"])
        fp_path = output_dir / "fingerprints" / f"{method_name}.jsonl"
        run_path = output_dir / "runs" / f"{method_name}_verify.jsonl"
        artifacts = run_construct(method_name, method_cfg, model_cfg, fp_path)
        results = run_verify(method_name, method_cfg, suspect_cfg, fp_path, run_path)
        outputs.append({"method": method_name, "fingerprints": str(fp_path), "runs": str(run_path), "n": len(results), "n_artifacts": len(artifacts)})
    return outputs
