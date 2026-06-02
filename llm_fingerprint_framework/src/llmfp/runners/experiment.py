from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

from llmfp.core.io import append_jsonl, load_yaml
from llmfp.core.model_backend import ModelBackend
from llmfp.core.raw_records import default_raw_record, validate_raw_record, verification_entry
from llmfp.registry import get_method
from llmfp.schemas import VerificationResult


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    return resolved


def _load_yaml_path(path: str | Path) -> dict[str, Any]:
    return load_yaml(_resolve_path(path))


def _method_items(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    for item in cfg.get("methods", []):
        if isinstance(item, str):
            items.append({"name": item, "config": f"configs/methods/{item}.yaml"})
        else:
            items.append(dict(item))
    return items


def _load_method_config(item: dict[str, Any], seed: int, fingerprints_per_method: int | None) -> dict[str, Any]:
    method_cfg = _load_yaml_path(item["config"]) if item.get("config") else {}
    method_cfg.update(copy.deepcopy(item.get("overrides", {})))
    method_cfg["seed"] = seed
    if fingerprints_per_method is not None:
        method_cfg["num_fingerprints"] = fingerprints_per_method
        method_cfg["num_questions"] = fingerprints_per_method
        method_cfg["num_token_pairs"] = fingerprints_per_method
        method_cfg["num_queries"] = fingerprints_per_method
    return method_cfg


def _model_spec_items(items: list[Any] | None) -> list[dict[str, Any]]:
    specs = []
    for item in items or []:
        if isinstance(item, str):
            specs.append({"config": item})
        else:
            specs.append(dict(item))
    return specs


def _load_model_cfg(spec: dict[str, Any]) -> dict[str, Any]:
    if spec.get("config"):
        model_cfg = _load_yaml_path(spec["config"])
    else:
        model_cfg = copy.deepcopy(spec.get("model", spec))
    for key in ("model_role", "modification_type", "negative_type", "config", "valid_for_method"):
        model_cfg.pop(key, None)
    return model_cfg


class DeploymentBackend:
    def __init__(self, backend: ModelBackend, system_prompt: str | None = None):
        self.backend = backend
        self.system_prompt = system_prompt

    @property
    def name(self):
        return self.backend.name

    @property
    def template_name(self):
        return self.backend.template_name

    @property
    def tokenizer(self):
        return self.backend.tokenizer

    @property
    def model(self):
        return self.backend.model

    @property
    def device(self):
        return self.backend.device

    def load(self):
        self.backend.load()
        return self

    def _prompt(self, prompt_text: str) -> str:
        if not self.system_prompt:
            return prompt_text
        return f"{self.system_prompt}\n\n{prompt_text}"

    def generate(self, prompt_text: str, **kwargs):
        return self.backend.generate(self._prompt(prompt_text), **kwargs)

    def generate_first_token(self, prompt_text: str, **kwargs):
        return self.backend.generate_first_token(self._prompt(prompt_text), **kwargs)

    def first_token_logits(self, prompt_text: str):
        return self.backend.first_token_logits(self._prompt(prompt_text))


def _default_conditions() -> list[dict[str, Any]]:
    return [
        {"name": "default", "sampling": {}},
        {"name": "greedy", "sampling": {"do_sample": False, "temperature": 1.0}, "reported_sampling": {"do_sample": False, "temperature": 0.0}},
        {"name": "temperature_0_7", "sampling": {"do_sample": True, "temperature": 0.7}},
        {"name": "top_p_0_9", "sampling": {"do_sample": True, "top_p": 0.9}},
    ]


def _safe_run_id(cfg: dict[str, Any]) -> str:
    run_id = str(cfg.get("run_id") or f"run-{int(time.time())}")
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in run_id)


def _gpu_peak_gb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.max_memory_allocated() / (1024**3))
    except Exception:
        return None


def _reset_gpu_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        return None


def _protected_derivative_specs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    plugae_cfg = cfg.get("plugae") or {}
    derivatives = plugae_cfg.get("protected_derivatives") or {}
    specs: list[dict[str, Any]] = []
    if isinstance(derivatives, list):
        specs.extend(_model_spec_items(derivatives))
    else:
        for modification_type, items in derivatives.items():
            for item in _model_spec_items(items):
                item.setdefault("modification_type", modification_type)
                specs.append(item)
    return specs


def _verification_specs(cfg: dict[str, Any], method_name: str, source_backend: ModelBackend) -> list[dict[str, Any]]:
    specs = [
        {
            "backend": source_backend,
            "name": source_backend.name,
            "model_role": "source",
            "modification_type": "none",
            "negative_type": "none",
            "valid_for_method": True,
        }
    ]
    positive_specs = _model_spec_items(cfg.get("positive_suspects"))
    if method_name == "plugae":
        positive_specs.extend(_protected_derivative_specs(cfg))
    for item in positive_specs:
        item.setdefault("model_role", "positive")
        item.setdefault("modification_type", "unknown")
        item.setdefault("negative_type", "none")
        specs.append(item)
    for item in _model_spec_items(cfg.get("negative_suspects")):
        item.setdefault("model_role", "negative")
        item.setdefault("modification_type", "none")
        item.setdefault("negative_type", item.get("type", "unrelated"))
        specs.append(item)
    return specs


def _backend_cache_key(model_cfg: dict[str, Any]) -> str:
    return json.dumps(
        {
            "model_name_or_path": model_cfg.get("model_name_or_path"),
            "tokenizer_name_or_path": model_cfg.get("tokenizer_name_or_path"),
            "device": model_cfg.get("device"),
            "dtype": model_cfg.get("dtype"),
            "template": model_cfg.get("template"),
            "trust_remote_code": model_cfg.get("trust_remote_code"),
        },
        sort_keys=True,
    )


def _load_backend_for_spec(spec: dict[str, Any], cache: dict[str, ModelBackend]) -> ModelBackend:
    if "backend" in spec:
        return spec["backend"]
    model_cfg = _load_model_cfg(spec)
    key = _backend_cache_key(model_cfg)
    if key not in cache:
        cache[key] = ModelBackend.from_config(model_cfg)
    return cache[key]


def _unload_backend_cache(cache: dict[str, ModelBackend]) -> None:
    seen: set[int] = set()
    for backend in list(cache.values()):
        if id(backend) in seen:
            continue
        seen.add(id(backend))
        backend.unload()
    cache.clear()


def _tokenizer_contains_tokens_for_spec(spec: dict[str, Any], tokens: list[str]) -> bool:
    if "backend" in spec:
        backend = spec["backend"]
        backend.load()
        tokenizer = backend.tokenizer
    else:
        from transformers import AutoTokenizer

        model_cfg = _load_model_cfg(spec)
        tokenizer_ref = model_cfg.get("tokenizer_name_or_path") or model_cfg.get("model_name_or_path")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_ref,
            trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        )
    unk_id = getattr(tokenizer, "unk_token_id", None)
    added = getattr(tokenizer, "added_tokens_encoder", {}) or {}
    for token in tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None:
            return False
        if unk_id is not None and int(token_id) == int(unk_id) and token not in added:
            return False
    return True


def _condition_config(method_cfg: dict[str, Any], condition: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    sampling = dict(condition.get("sampling") or {})
    reported = dict(condition.get("reported_sampling") or sampling)
    merged = copy.deepcopy(method_cfg)
    merged.update(sampling)
    return merged, reported


def _valid_for_method(method_name: str, model_role: str, configured_valid: bool, result) -> bool:
    if not configured_valid:
        return False
    if method_name != "plugae" or model_role == "negative":
        return True
    result_dict = result.to_dict() if hasattr(result, "to_dict") else result
    metadata = result_dict.get("metadata") or {}
    return bool(metadata.get("suspect_contains_copyright_tokens"))


def _plugae_missing_token_result(artifact, backend_name: str, spec: dict[str, Any], model_role: str, valid_for_method: bool) -> VerificationResult:
    metadata = artifact.metadata if hasattr(artifact, "metadata") else artifact.get("metadata", {})
    copyright_tokens = list(metadata.get("copyright_tokens") or [])
    return VerificationResult(
        method="plugae",
        base_model=artifact.base_model if hasattr(artifact, "base_model") else artifact.get("base_model"),
        suspect_model=backend_name,
        fingerprint_id=artifact.fingerprint_id if hasattr(artifact, "fingerprint_id") else artifact.get("fingerprint_id"),
        success=False,
        score=0.0,
        raw_output=None,
        metadata={
            "copyright_tokens": copyright_tokens,
            "suspect_contains_copyright_tokens": False,
            "skipped_model_load": True,
            "skip_reason": "suspect_tokenizer_missing_copyright_tokens",
            "valid_for_method": valid_for_method,
            "model_role": model_role,
            "model_config": spec.get("config"),
        },
    )


def run_experiment(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    run_id = _safe_run_id(cfg)
    output_dir = _resolve_path(cfg.get("output_dir", "results"))
    raw_root = output_dir / "raw"
    seeds = [int(seed) for seed in cfg.get("seeds", [0])]
    fingerprints_per_method = cfg.get("fingerprints_per_method")
    if fingerprints_per_method is not None:
        fingerprints_per_method = int(fingerprints_per_method)
    conditions = list(cfg.get("deployment_conditions") or _default_conditions())
    records: list[dict[str, Any]] = []

    for source_item in _model_spec_items(cfg.get("source_models") or [cfg.get("model_config")]):
        source_model_cfg = _load_model_cfg(source_item)
        source_cache_key = _backend_cache_key(source_model_cfg)
        for method_item in _method_items(cfg):
            method_name = method_item["name"]
            method = get_method(method_name)
            for seed in seeds:
                backend_cache: dict[str, ModelBackend] = {}
                source_backend: ModelBackend | None = None
                method_cfg = _load_method_config(method_item, seed, fingerprints_per_method)
                tasks = method.build_tasks(method_cfg)
                try:
                    for fp_index, task in enumerate(tasks):
                        if method_name == "plugae":
                            if source_backend is not None:
                                source_backend.unload()
                            source_backend = ModelBackend.from_config(source_model_cfg)
                        elif source_backend is None:
                            source_backend = ModelBackend.from_config(source_model_cfg)
                            backend_cache[source_cache_key] = source_backend
                        _reset_gpu_peak()
                        started = time.perf_counter()
                        artifact = method.construct(task, source_backend, method_cfg)
                        generation_time = time.perf_counter() - started
                        if artifact is None:
                            continue
                        record = default_raw_record(run_id, method_name, source_backend.name, seed, fp_index, artifact)
                        record["efficiency"]["generation_time_sec"] = generation_time
                        record["efficiency"]["peak_gpu_memory_gb"] = _gpu_peak_gb()
                        verification_specs = _verification_specs(cfg, method_name, source_backend)
                        model_condition_counts: dict[str, int] = {}
                        for spec in verification_specs:
                            model_role = str(spec.get("model_role", "suspect"))
                            modification_type = str(spec.get("modification_type", "none"))
                            negative_type = str(spec.get("negative_type", "none"))
                            configured_valid = bool(spec.get("valid_for_method", True))
                            contains_plugae_tokens = True
                            if method_name == "plugae" and model_role != "source":
                                artifact_metadata = artifact.metadata if hasattr(artifact, "metadata") else artifact.get("metadata", {})
                                tokens = list(artifact_metadata.get("copyright_tokens") or [])
                                contains_plugae_tokens = _tokenizer_contains_tokens_for_spec(spec, tokens)
                            if method_name == "plugae" and model_role != "source" and not contains_plugae_tokens:
                                if spec.get("config"):
                                    backend_name = str(_load_model_cfg(spec).get("name") or spec.get("config"))
                                else:
                                    backend_name = str((spec.get("model") or spec).get("name") or (spec.get("model") or spec).get("model_name_or_path"))
                                for condition in conditions:
                                    condition_name = str(condition.get("name", "default"))
                                    _, reported_sampling = _condition_config(method_cfg, condition)
                                    valid = configured_valid and model_role != "positive"
                                    result = _plugae_missing_token_result(artifact, backend_name, spec, model_role, valid)
                                    record["verification"].append(
                                        verification_entry(
                                            result,
                                            model=backend_name,
                                            model_role=model_role,
                                            modification_type=modification_type,
                                            negative_type=negative_type,
                                            condition=condition_name,
                                            system_prompt=condition.get("system_prompt"),
                                            sampling=reported_sampling,
                                            valid_for_method=valid,
                                        )
                                    )
                                    model_condition_counts[backend_name] = model_condition_counts.get(backend_name, 0) + 1
                                continue
                            backend = _load_backend_for_spec(spec, backend_cache)
                            for condition in conditions:
                                condition_name = str(condition.get("name", "default"))
                                system_prompt = condition.get("system_prompt")
                                verify_cfg, reported_sampling = _condition_config(method_cfg, condition)
                                deployment_backend = DeploymentBackend(backend, system_prompt=system_prompt)
                                result = method.verify(artifact, deployment_backend, verify_cfg)
                                valid = _valid_for_method(method_name, model_role, configured_valid, result)
                                record["verification"].append(
                                    verification_entry(
                                        result,
                                        model=backend.name,
                                        model_role=model_role,
                                        modification_type=modification_type,
                                        negative_type=negative_type,
                                        condition=condition_name,
                                        system_prompt=system_prompt,
                                        sampling=reported_sampling,
                                        valid_for_method=valid,
                                    )
                                )
                                model_condition_counts[backend.name] = model_condition_counts.get(backend.name, 0) + 1
                        source_default = [
                            row
                            for row in record["verification"]
                            if row["model_role"] == "source" and row["condition"] == "default" and row["valid_for_method"]
                        ]
                        record["generation"]["success_on_source"] = bool(source_default and float(source_default[0].get("score") or 0.0) > 0.0)
                        record["efficiency"]["verification_queries_per_model"] = max(model_condition_counts.values()) if model_condition_counts else 0
                        validate_raw_record(record)
                        raw_path = raw_root / method_name / f"{run_id}.jsonl"
                        append_jsonl(raw_path, record)
                        records.append(record)
                finally:
                    _unload_backend_cache(backend_cache)
                    if source_backend is not None:
                        source_backend.unload()
    return records
