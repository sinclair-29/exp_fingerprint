from __future__ import annotations

from pathlib import Path
from typing import Any

from llmfp.core.io import load_jsonl
from llmfp.core.matching import match_keyword_target
from llmfp.core.metrics import success_rate
from llmfp.methods.base import FingerprintMethod, artifact_get, result_dicts
from llmfp.schemas import FingerprintArtifact, FingerprintTask, VerificationResult


class ProFLingoMethod(FingerprintMethod):
    name = "proflingo"

    def build_tasks(self, cfg: dict[str, Any]) -> list[FingerprintTask]:
        data_path = Path(cfg.get("questions_path", "data/proflingo_questions.jsonl"))
        rows = load_jsonl(data_path)[: int(cfg.get("num_questions", 2))]
        tasks = []
        for index, row in enumerate(rows):
            question = row["question"]
            target = row["target"]
            keywords = row.get("target_keywords", [target])
            tasks.append(
                FingerprintTask(
                    task_id=str(row.get("id", f"q{index + 1}")),
                    method=self.name,
                    input_text=question,
                    target=target,
                    metadata={"target_keywords": keywords},
                )
            )
        return tasks

    def construct(self, task: FingerprintTask, model_backend, cfg: dict[str, Any]) -> FingerprintArtifact:
        import torch

        from llmfp.core.candidate_filters import (
            CompositeFilter,
            ProFLingoWordFragmentFilter,
            RetokenizationConsistencyFilter,
            TargetKeywordExclusionFilter,
        )
        from llmfp.core.losses import MultiTemplateTargetLoss
        from llmfp.core.prompt_builders import ProFLingoPrefixPromptBuilder
        from llmfp.optimizers.gcg import GCGOptimizer

        templates = list(cfg.get("templates", ["fastchat_zero_shot", "alpaca"]))
        init_prefix = ("x " * int(cfg.get("prefix_len", 32))).strip()
        init_ids = model_backend.tokenizer(init_prefix, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        builder = ProFLingoPrefixPromptBuilder(model_backend.tokenizer, task.input_text, task.target, templates)
        candidate_filter = CompositeFilter(
            ProFLingoWordFragmentFilter(),
            TargetKeywordExclusionFilter(task.metadata.get("target_keywords", [task.target])),
            RetokenizationConsistencyFilter(),
        )
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
            loss_fn=MultiTemplateTargetLoss(),
            candidate_filter=candidate_filter,
            num_steps=cfg.get("num_steps", 5),
            log_every=cfg.get("log_every", 10),
        )
        best_ids = torch.tensor(result.best_ids, dtype=torch.long)
        query = result.best_text + builder.query_suffix
        prompt_text = builder.build_prompt_text(best_ids, include_target=False, variant_index=0)
        return FingerprintArtifact(
            fingerprint_id=f"proflingo-{task.task_id}",
            method=self.name,
            base_model=model_backend.name,
            task_id=task.task_id,
            prompt_text=prompt_text,
            optimized_text=result.best_text,
            target=task.target,
            best_loss=result.best_loss,
            best_step=result.best_step,
            metadata={
                "question": task.input_text,
                "target_keywords": task.metadata.get("target_keywords", [task.target]),
                "prefix": result.best_text,
                "query": query,
                "templates": templates,
                "optimizer": result.metadata,
                "loss_history": result.loss_history,
            },
        )

    def verify(self, artifact, suspect_backend, cfg: dict[str, Any]) -> VerificationResult:
        from llmfp.core.templates import get_template

        metadata = artifact_get(artifact, "metadata", {})
        query = metadata.get("query") or artifact_get(artifact, "optimized_text", "") + " simply answer: " + metadata.get("question", "")
        template_name = cfg.get("suspect_template") or suspect_backend.template_name
        prompt_text = get_template(template_name).render(query, assistant_prefix="")
        attempts = int(cfg.get("num_attempts", 1))
        outputs = []
        success = False
        for _ in range(max(1, attempts)):
            output = suspect_backend.generate(
                prompt_text,
                max_new_tokens=int(cfg.get("max_new_tokens", 32)),
                temperature=float(cfg.get("temperature", 1.0)),
                top_p=float(cfg.get("top_p", 1.0)),
                do_sample=bool(cfg.get("do_sample", attempts > 1)),
            )
            outputs.append(output)
            if match_keyword_target(output, artifact_get(artifact, "target"), metadata.get("target_keywords", [])):
                success = True
                break
        return VerificationResult(
            method=self.name,
            base_model=artifact_get(artifact, "base_model"),
            suspect_model=suspect_backend.name,
            fingerprint_id=artifact_get(artifact, "fingerprint_id"),
            success=success,
            score=1.0 if success else 0.0,
            raw_output=outputs[-1] if outputs else "",
            metadata={"outputs": outputs, "trr": 1.0 if success else 0.0},
        )

    def aggregate(self, results, cfg: dict[str, Any]) -> dict[str, Any]:
        rows = result_dicts(results)
        return {"trr": success_rate(rows), "target_response_rate": success_rate(rows)}
