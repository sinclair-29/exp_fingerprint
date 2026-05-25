from __future__ import annotations

import random
from typing import Any

from llmfp.core.matching import match_trap_output
from llmfp.core.metrics import invalid_rate, success_rate, tpr_fpr_from_labeled_results
from llmfp.methods.base import FingerprintMethod, artifact_get, result_dicts
from llmfp.schemas import FingerprintArtifact, FingerprintTask, VerificationResult


class TRAPMethod(FingerprintMethod):
    name = "trap"

    def build_tasks(self, cfg: dict[str, Any]) -> list[FingerprintTask]:
        rng = random.Random(int(cfg.get("seed", 0)))
        answer_lengths = list(cfg.get("answer_lengths", [3]))
        tasks = []
        for index in range(int(cfg.get("num_fingerprints", 2))):
            n_digits = int(answer_lengths[index % len(answer_lengths)])
            target_digits = "".join(rng.choice("0123456789") for _ in range(n_digits))
            instruction = (
                f"Write a random string composed of {n_digits} digits. "
                "Your reply should only contain the random string."
            )
            tasks.append(
                FingerprintTask(
                    task_id=f"trap-{index:04d}",
                    method=self.name,
                    input_text=instruction,
                    target=target_digits,
                    metadata={"n_digits": n_digits, "target_digits": target_digits},
                )
            )
        return tasks

    def construct(self, task: FingerprintTask, model_backend, cfg: dict[str, Any]) -> FingerprintArtifact:
        import torch

        from llmfp.core.candidate_filters import CompositeFilter, TrapNumberFilter
        from llmfp.core.prompt_builders import TRAPSuffixPromptBuilder
        from llmfp.optimizers.gcg import optimize_suffix_target_string

        init_suffix = ("! " * int(cfg.get("suffix_len", 20))).strip()
        candidate_filter = CompositeFilter(TrapNumberFilter(model_backend.tokenizer))
        result = optimize_suffix_target_string(
            model_backend=model_backend,
            instruction=task.input_text,
            target=task.target,
            init_suffix=init_suffix,
            cfg=cfg,
            candidate_filter=candidate_filter,
        )
        builder = TRAPSuffixPromptBuilder(
            tokenizer=model_backend.tokenizer,
            instruction=task.input_text,
            target=task.target,
            template_name=model_backend.template_name,
        )
        best_ids = torch.tensor(result.best_ids, dtype=torch.long)
        prompt_text = builder.build_prompt_text(best_ids, include_target=False)
        return FingerprintArtifact(
            fingerprint_id=f"trap-{task.task_id}",
            method=self.name,
            base_model=model_backend.name,
            task_id=task.task_id,
            prompt_text=prompt_text,
            optimized_text=result.best_text,
            target=task.target,
            best_loss=result.best_loss,
            best_step=result.best_step,
            metadata={
                **task.metadata,
                "instruction": task.input_text,
                "full_prompt": prompt_text,
                "optimizer": result.metadata,
                "loss_history": result.loss_history,
            },
        )

    def verify(self, artifact, suspect_backend, cfg: dict[str, Any]) -> VerificationResult:
        prompt_text = artifact_get(artifact, "prompt_text")
        target = artifact_get(artifact, "target")
        raw_output = suspect_backend.generate(
            prompt_text,
            max_new_tokens=int(cfg.get("max_new_tokens", 32)),
            temperature=float(cfg.get("temperature", 1.0)),
            top_p=float(cfg.get("top_p", 1.0)),
            do_sample=bool(cfg.get("do_sample", False)),
        )
        success, invalid, parsed = match_trap_output(raw_output, target)
        return VerificationResult(
            method=self.name,
            base_model=artifact_get(artifact, "base_model"),
            suspect_model=suspect_backend.name,
            fingerprint_id=artifact_get(artifact, "fingerprint_id"),
            success=success,
            score=1.0 if success else 0.0,
            raw_output=raw_output,
            metadata={"invalid": invalid, "parsed_digits": parsed, "target_digits": target},
        )

    def aggregate(self, results, cfg: dict[str, Any]) -> dict[str, Any]:
        rows = result_dicts(results)
        labeled = tpr_fpr_from_labeled_results(rows)
        return {
            "success_rate": success_rate(rows),
            "invalid_rate": invalid_rate(rows),
            "tpr": labeled["tpr"],
            "fpr": labeled["fpr"],
        }
