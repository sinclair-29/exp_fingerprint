from __future__ import annotations

from typing import Any

from llmfp.methods.base import artifact_get


REQUIRED_TOP_LEVEL_KEYS = [
    "run_id",
    "method",
    "source_model",
    "seed",
    "fingerprint_id",
    "fingerprint_spec",
    "generation",
    "verification",
    "stealthiness",
    "efficiency",
]


def default_raw_record(
    run_id: str,
    method: str,
    source_model: str,
    seed: int,
    fingerprint_id: int,
    artifact,
) -> dict[str, Any]:
    metadata = artifact_get(artifact, "metadata", {}) or {}
    base_prompt = (
        metadata.get("base_prompt")
        or metadata.get("instruction")
        or metadata.get("question")
        or metadata.get("query")
        or ""
    )
    full_query = metadata.get("full_query") or metadata.get("query") or artifact_get(artifact, "prompt_text", "")
    loss_curve = metadata.get("loss_history")
    optimizer = metadata.get("optimizer") or {}
    configured_steps = optimizer.get("epochs") if method == "plugae" else optimizer.get("num_steps")
    num_steps = len(loss_curve) - 1 if isinstance(loss_curve, list) and loss_curve else configured_steps
    return {
        "run_id": run_id,
        "method": method,
        "source_model": source_model,
        "seed": seed,
        "fingerprint_id": fingerprint_id,
        "fingerprint_spec": {
            "base_prompt": base_prompt,
            "target": artifact_get(artifact, "target"),
            "full_query": full_query,
            "adversarial_text": artifact_get(artifact, "optimized_text"),
            "method_specific": metadata,
        },
        "generation": {
            "success_on_source": None,
            "final_loss": artifact_get(artifact, "best_loss"),
            "best_step": artifact_get(artifact, "best_step"),
            "num_steps": num_steps,
            "loss_curve": loss_curve,
            "loss_curve_path": None,
            "method_specific": metadata,
        },
        "verification": [],
        "stealthiness": {
            "ppl_model": None,
            "full_prompt_log_ppl": None,
            "adv_part_log_ppl": None,
            "ppl_filter_pass": None,
        },
        "efficiency": {
            "generation_time_sec": None,
            "peak_gpu_memory_gb": None,
            "num_optimization_steps": num_steps,
            "num_forward": metadata.get("num_forward"),
            "num_backward": metadata.get("num_backward"),
            "verification_queries_per_model": None,
        },
    }


def verification_entry(
    result,
    model: str,
    model_role: str,
    modification_type: str = "none",
    negative_type: str = "none",
    condition: str = "default",
    system_prompt: str | None = None,
    sampling: dict[str, Any] | None = None,
    valid_for_method: bool = True,
    method_specific: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result_dict = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    metadata = dict(result_dict.get("metadata") or {})
    if method_specific:
        metadata.update(method_specific)
    return {
        "model": model,
        "model_role": model_role,
        "modification_type": modification_type,
        "negative_type": negative_type,
        "condition": condition,
        "system_prompt": system_prompt,
        "sampling": sampling or {},
        "output": result_dict.get("raw_output"),
        "score": result_dict.get("score"),
        "valid_for_method": valid_for_method,
        "method_specific": metadata,
    }


def validate_raw_record(record: dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in record]
    if missing:
        raise ValueError(f"Raw record missing required keys: {missing}")
