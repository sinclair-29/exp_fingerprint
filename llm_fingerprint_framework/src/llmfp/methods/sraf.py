from __future__ import annotations

from typing import Any

from llmfp.core.matching import match_keyword_target
from llmfp.core.metrics import success_rate, tpr_fpr_from_labeled_results
from llmfp.methods.base import FingerprintMethod, artifact_get, result_dicts
from llmfp.schemas import FingerprintArtifact, FingerprintTask, VerificationResult


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

    def construct(self, task: FingerprintTask, model_backend, cfg: dict[str, Any]) -> FingerprintArtifact:
        import torch

        from llmfp.core.candidate_filters import AsciiFilter, CompositeFilter, RetokenizationConsistencyFilter
        from llmfp.core.losses import MultiModelMultiTemplateTargetLoss
        from llmfp.core.prompt_builders import SRAFTablePromptBuilder
        from llmfp.optimizers.gcg import GCGOptimizer

        templates = list(cfg.get("templates", ["raw", "zero_shot"]))
        init_mutable = ("! " * int(cfg.get("mutable_len", 64))).strip()
        init_ids = model_backend.tokenizer(init_mutable, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        builder = SRAFTablePromptBuilder(model_backend.tokenizer, task.input_text, task.target, templates)
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
            loss_fn=MultiModelMultiTemplateTargetLoss(),
            candidate_filter=CompositeFilter(AsciiFilter(), RetokenizationConsistencyFilter()),
            num_steps=cfg.get("num_steps", 5),
            log_every=cfg.get("log_every", 10),
        )
        best_ids = torch.tensor(result.best_ids, dtype=torch.long)
        prompt_text = builder.build_prompt_text(best_ids, include_target=False, variant_index=0)
        segments = builder.mutable_segments(best_ids)
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
                "templates": templates,
                "model_names": [model_backend.name],
                "target_keywords": task.metadata.get("target_keywords", [task.target]),
                "optimizer": result.metadata,
                "loss_history": result.loss_history,
                "ppl": None,
            },
        )

    def verify(self, artifact, suspect_backend, cfg: dict[str, Any]) -> VerificationResult:
        output = suspect_backend.generate(
            artifact_get(artifact, "prompt_text"),
            max_new_tokens=int(cfg.get("max_new_tokens", 48)),
            temperature=float(cfg.get("temperature", 1.0)),
            top_p=float(cfg.get("top_p", 1.0)),
            do_sample=bool(cfg.get("do_sample", False)),
        )
        metadata = artifact_get(artifact, "metadata", {})
        success = match_keyword_target(output, artifact_get(artifact, "target"), metadata.get("target_keywords", []))
        return VerificationResult(
            method=self.name,
            base_model=artifact_get(artifact, "base_model"),
            suspect_model=suspect_backend.name,
            fingerprint_id=artifact_get(artifact, "fingerprint_id"),
            success=success,
            score=1.0 if success else 0.0,
            raw_output=output,
            metadata={"ppl": None if not cfg.get("compute_ppl", False) else metadata.get("ppl")},
        )

    def aggregate(self, results, cfg: dict[str, Any]) -> dict[str, Any]:
        rows = result_dicts(results)
        labeled = tpr_fpr_from_labeled_results(rows)
        return {"fsr": success_rate(rows), "fpr": labeled["fpr"], "ppl": None}
