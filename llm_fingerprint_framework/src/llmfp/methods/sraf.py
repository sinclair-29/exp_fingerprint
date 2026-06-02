from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from llmfp.core.io import load_yaml
from llmfp.core.matching import exact_normalized_target, match_keyword_target
from llmfp.core.metrics import safe_mean, success_rate, tpr_fpr_from_labeled_results
from llmfp.core.model_backend import ModelBackend
from llmfp.methods.base import FingerprintMethod, artifact_get, result_dicts
from llmfp.schemas import FingerprintArtifact, FingerprintTask, VerificationResult


_PPL_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}


def _looks_like_local_path(value: str) -> bool:
    return value.startswith(("/", "~", "."))


def _resolve_config_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    return resolved


def _compute_prompt_ppl(prompt_text: str, cfg: dict[str, Any], device) -> float:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_ref = str(cfg.get("ppl_model_name_or_path", "gpt2"))
    cache_key = (model_ref, str(device))
    if cache_key not in _PPL_MODEL_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(model_ref)
        model = AutoModelForCausalLM.from_pretrained(model_ref)
        model.to(device)
        model.eval()
        _PPL_MODEL_CACHE[cache_key] = (tokenizer, model)
    tokenizer, model = _PPL_MODEL_CACHE[cache_key]
    encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)
    input_ids = encoded["input_ids"]
    if input_ids.shape[1] < 2:
        return 0.0
    with torch.no_grad():
        logits = model(**encoded).logits[:, :-1, :]
        labels = input_ids[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
    return float(math.exp(min(20.0, float(loss.item()))))


class SRAFMethod(FingerprintMethod):
    name = "sraf"

    DEFAULT_TARGETS = [
        {"id": "sraf-1", "base_prompt": "Read the reference table and answer with the hidden key.", "target": "cerulean harbor"},
        {"id": "sraf-2", "base_prompt": "Use the table cells as reference material and output the hidden key.", "target": "amber orbit"},
    ]

    def build_tasks(self, cfg: dict[str, Any]) -> list[FingerprintTask]:
        items = list(cfg.get("targets", self.DEFAULT_TARGETS))[: int(cfg.get("num_fingerprints", 2))]
        tasks = []
        for index, item in enumerate(items):
            target = item["target"]
            tasks.append(
                FingerprintTask(
                    task_id=str(item.get("id", f"sraf-{index + 1}")),
                    method=self.name,
                    input_text=item.get("base_prompt", self.DEFAULT_TARGETS[0]["base_prompt"]),
                    target=target,
                    metadata={"target_keywords": item.get("target_keywords", [target])},
                )
            )
        return tasks

    def _load_co_model_backends(
        self,
        cfg: dict[str, Any],
        primary_backend,
        task: FingerprintTask,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        specs = []
        skipped = []
        for item in cfg.get("co_model_configs", []) or []:
            if isinstance(item, str):
                path = item
                weight = 1.0
                required = True
            else:
                path = item.get("path") or item.get("config")
                weight = float(item.get("weight", 1.0))
                required = bool(item.get("required", True))
            if not path:
                raise ValueError("SRAF co_model_configs entries must provide a path")
            config_path = _resolve_config_path(path)
            if not config_path.exists():
                if required:
                    raise FileNotFoundError(f"SRAF co-model config does not exist: {config_path}")
                skipped.append({"path": str(config_path), "reason": "config_not_found"})
                continue
            co_cfg = load_yaml(config_path)
            missing_path = self._missing_local_model_path(co_cfg)
            if missing_path and not required:
                skipped.append({"path": str(config_path), "reason": f"model_path_not_found:{missing_path}"})
                continue
            try:
                cache = getattr(self, "_co_backend_cache", {})
                backend = cache.get(str(config_path))
                if backend is None:
                    backend = ModelBackend.from_config(co_cfg)
                    cache[str(config_path)] = backend
                    self._co_backend_cache = cache
            except Exception as exc:
                if required:
                    raise RuntimeError(f"Failed to load required SRAF co-model config {config_path}: {exc}") from exc
                skipped.append({"path": str(config_path), "reason": f"load_failed:{exc}"})
                continue
            self._validate_tokenizer_compatibility(primary_backend, backend, task)
            specs.append({"backend": backend, "weight": weight, "config_path": str(config_path)})
        return specs, skipped

    def _missing_local_model_path(self, cfg: dict[str, Any]) -> str | None:
        for key in ("model_name_or_path", "tokenizer_name_or_path"):
            value = cfg.get(key)
            if not value:
                continue
            text = str(value)
            if _looks_like_local_path(text) and not Path(text).expanduser().exists():
                return text
        return None

    def _validate_tokenizer_compatibility(self, primary_backend, co_backend, task: FingerprintTask) -> None:
        primary_tokenizer = primary_backend.tokenizer
        co_tokenizer = co_backend.tokenizer
        primary_vocab = getattr(primary_tokenizer, "vocab_size", len(primary_tokenizer))
        co_vocab = getattr(co_tokenizer, "vocab_size", len(co_tokenizer))
        if primary_vocab != co_vocab:
            raise ValueError(
                f"SRAF co-model tokenizer for {co_backend.name} is incompatible with {primary_backend.name}: "
                f"vocab sizes differ ({primary_vocab} != {co_vocab})"
            )
        samples = [
            task.input_text,
            task.target,
            "! ! ! !",
            "| Field | Value |\n|---|---|\n",
            "Segment 1 alpha beta 123",
        ]
        for text in samples:
            primary_ids = primary_tokenizer(text, add_special_tokens=False)["input_ids"]
            co_ids = co_tokenizer(text, add_special_tokens=False)["input_ids"]
            if primary_ids != co_ids:
                raise ValueError(
                    f"SRAF co-model tokenizer for {co_backend.name} is incompatible with {primary_backend.name}: "
                    f"different tokenization for sample {text!r}"
                )

    def construct(self, task: FingerprintTask, model_backend, cfg: dict[str, Any]) -> FingerprintArtifact:
        import torch

        from llmfp.core.candidate_filters import AsciiFilter, CompositeFilter, RetokenizationConsistencyFilter
        from llmfp.core.losses import MultiModelMultiTemplateTargetLoss
        from llmfp.core.prompt_builders import SRAFTablePromptBuilder
        from llmfp.optimizers.gcg import GCGOptimizer

        templates = list(cfg.get("templates", ["sraf_default", "sraf_zero_shot"]))
        table_segments = int(cfg.get("table_segments", 3))
        init_mutable = ("! " * int(cfg.get("mutable_len", 64))).strip()
        init_ids = model_backend.tokenizer(init_mutable, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        builder = SRAFTablePromptBuilder(
            model_backend.tokenizer,
            task.input_text,
            task.target,
            templates,
            table_segments=table_segments,
        )
        co_specs, skipped_co_specs = self._load_co_model_backends(cfg, model_backend, task)
        filters = []
        if not bool(cfg.get("allow_non_ascii", False)):
            filters.append(AsciiFilter())
        if bool(cfg.get("filter_retokenization", True)):
            filters.append(RetokenizationConsistencyFilter())
        optimizer = GCGOptimizer(
            model_backend=model_backend,
            tokenizer=model_backend.tokenizer,
            top_k=cfg.get("top_k", 64),
            search_width=cfg.get("search_width", 16),
            batch_size=cfg.get("batch_size", 16),
            allow_non_ascii=cfg.get("allow_non_ascii", False),
            filter_retokenization=cfg.get("filter_retokenization", True),
            seed=cfg.get("seed", 0),
        )
        result = optimizer.optimize(
            init_ids=init_ids,
            prompt_builder=builder,
            loss_fn=MultiModelMultiTemplateTargetLoss(backend_specs=co_specs),
            candidate_filter=CompositeFilter(*filters),
            num_steps=cfg.get("num_steps", 5),
            log_every=cfg.get("log_every", 10),
        )
        best_ids = torch.tensor(result.best_ids, dtype=torch.long)
        prompt_text = builder.build_prompt_text(best_ids, include_target=False, variant_index=0)
        segments = builder.mutable_segments(best_ids)
        ppl = _compute_prompt_ppl(prompt_text, cfg, model_backend.device) if cfg.get("compute_ppl", False) else None
        model_weights = [{"name": model_backend.name, "weight": 1.0}]
        model_weights.extend({"name": spec["backend"].name, "weight": spec["weight"]} for spec in co_specs)
        return FingerprintArtifact(
            fingerprint_id=f"sraf-{task.task_id}",
            method=self.name,
            base_model=model_backend.name,
            task_id=task.task_id,
            prompt_text=prompt_text,
            optimized_text=result.best_text,
            target=task.target,
            best_loss=result.best_loss,
            best_step=result.best_step,
            metadata={
                "base_prompt": task.input_text,
                "table_prompt": prompt_text,
                "mutable_segments": segments,
                "table_segments": table_segments,
                "templates": templates,
                "model_names": [item["name"] for item in model_weights],
                "model_weights": model_weights,
                "co_model_configs": [{"path": spec["config_path"], "weight": spec["weight"]} for spec in co_specs],
                "skipped_co_model_configs": skipped_co_specs,
                "target_keywords": task.metadata.get("target_keywords", [task.target]),
                "match_mode": cfg.get("match_mode", "exact_normalized"),
                "optimizer": result.metadata,
                "loss_history": result.loss_history,
                "ppl": ppl,
            },
        )

    def _match_output(self, output: str, target: str, metadata: dict[str, Any], cfg: dict[str, Any]) -> bool:
        match_mode = str(cfg.get("match_mode") or metadata.get("match_mode", "exact_normalized"))
        if match_mode == "exact_normalized":
            return exact_normalized_target(output, target)
        if match_mode == "contains_normalized":
            return match_keyword_target(output, target, metadata.get("target_keywords", []))
        if match_mode == "exact":
            return (output or "").strip() == (target or "").strip()
        raise ValueError(f"Unsupported SRAF match_mode: {match_mode}")

    def verify(self, artifact, suspect_backend, cfg: dict[str, Any]) -> VerificationResult:
        output = suspect_backend.generate(
            artifact_get(artifact, "prompt_text"),
            max_new_tokens=int(cfg.get("max_new_tokens", 48)),
            temperature=float(cfg.get("temperature", 1.0)),
            top_p=float(cfg.get("top_p", 1.0)),
            do_sample=bool(cfg.get("do_sample", False)),
        )
        metadata = artifact_get(artifact, "metadata", {})
        success = self._match_output(output, artifact_get(artifact, "target"), metadata, cfg)
        return VerificationResult(
            method=self.name,
            base_model=artifact_get(artifact, "base_model"),
            suspect_model=suspect_backend.name,
            fingerprint_id=artifact_get(artifact, "fingerprint_id"),
            success=success,
            score=1.0 if success else 0.0,
            raw_output=output,
            metadata={
                "match_mode": str(cfg.get("match_mode") or metadata.get("match_mode", "exact_normalized")),
                "ppl": None if not cfg.get("compute_ppl", False) else metadata.get("ppl"),
            },
        )

    def aggregate(self, results, cfg: dict[str, Any]) -> dict[str, Any]:
        rows = result_dicts(results)
        labeled = tpr_fpr_from_labeled_results(rows)
        return {"fsr": success_rate(rows), "fpr": labeled["fpr"], "ppl": safe_mean((row.get("metadata") or {}).get("ppl") for row in rows)}
