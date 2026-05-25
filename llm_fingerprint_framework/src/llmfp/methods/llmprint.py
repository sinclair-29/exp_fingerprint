from __future__ import annotations

from pathlib import Path
from typing import Any

from llmfp.core.io import load_jsonl
from llmfp.core.matching import normalize_text
from llmfp.core.metrics import bitwise_accuracy, threshold_from_negative_accuracies, tpr_fpr_from_labeled_results
from llmfp.methods.base import FingerprintMethod, artifact_get, result_dicts
from llmfp.schemas import FingerprintArtifact, FingerprintTask, VerificationResult


class LLMPrintMethod(FingerprintMethod):
    name = "llmprint"

    def build_tasks(self, cfg: dict[str, Any]) -> list[FingerprintTask]:
        data_path = Path(cfg.get("token_pairs_path", "data/llmprint_token_pairs.jsonl"))
        rows = load_jsonl(data_path)[: int(cfg.get("num_token_pairs", 5))]
        return [
            FingerprintTask(
                task_id=str(row.get("id", f"p{index + 1}")),
                method=self.name,
                input_text="Randomly output a word from your vocabulary",
                target=f"{row['w_plus']}|{row['w_minus']}",
                metadata={"w_plus": row["w_plus"], "w_minus": row["w_minus"]},
            )
            for index, row in enumerate(rows)
        ]

    def _single_token_id(self, tokenizer, word: str) -> int | None:
        ids = tokenizer(word, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            return int(ids[0])
        return None

    def construct(self, task: FingerprintTask, model_backend, cfg: dict[str, Any]) -> FingerprintArtifact | None:
        import torch

        from llmfp.core.losses import TokenPreferenceLoss
        from llmfp.core.prompt_builders import LLMPrintSuffixPromptBuilder
        from llmfp.optimizers.gcg import GCGOptimizer

        tokenizer = model_backend.tokenizer
        w_plus = task.metadata["w_plus"]
        w_minus = task.metadata["w_minus"]
        plus_id = self._single_token_id(tokenizer, w_plus)
        minus_id = self._single_token_id(tokenizer, w_minus)
        if plus_id is None or minus_id is None:
            return None
        init_suffix = ("x " * int(cfg.get("suffix_len", 20))).strip()
        init_ids = tokenizer(init_suffix, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        builder = LLMPrintSuffixPromptBuilder(tokenizer, task.input_text, cfg.get("template", "raw"))
        optimizer = GCGOptimizer(
            model_backend=model_backend,
            tokenizer=tokenizer,
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
            loss_fn=TokenPreferenceLoss(plus_id, minus_id, alpha=cfg.get("alpha", 0.1), beta=cfg.get("beta", 1.0)),
            candidate_filter=None,
            num_steps=cfg.get("num_steps", 5),
            log_every=cfg.get("log_every", 10),
        )
        best_ids = torch.tensor(result.best_ids, dtype=torch.long)
        prompt_text = builder.build_prompt_text(best_ids, include_target=False)
        logits = model_backend.first_token_logits(prompt_text)
        reference_bit = 1 if float(logits[plus_id]) >= float(logits[minus_id]) else 0
        return FingerprintArtifact(
            fingerprint_id=f"llmprint-{task.task_id}",
            method=self.name,
            base_model=model_backend.name,
            task_id=task.task_id,
            prompt_text=prompt_text,
            optimized_text=result.best_text,
            target=task.target,
            best_loss=result.best_loss,
            best_step=result.best_step,
            metadata={
                "w_plus": w_plus,
                "w_minus": w_minus,
                "w_plus_id": plus_id,
                "w_minus_id": minus_id,
                "reference_bit": reference_bit,
                "optimizer": result.metadata,
                "loss_history": result.loss_history,
            },
        )

    def verify(self, artifact, suspect_backend, cfg: dict[str, Any]) -> VerificationResult:
        metadata = artifact_get(artifact, "metadata", {})
        plus_id = int(metadata["w_plus_id"])
        minus_id = int(metadata["w_minus_id"])
        reference_bit = int(metadata["reference_bit"])
        mode = cfg.get("verification_mode", "black_box")
        if mode == "gray_box":
            import torch

            logits = suspect_backend.first_token_logits(artifact_get(artifact, "prompt_text"))
            predicted_bit = 1 if float(logits[plus_id]) >= float(logits[minus_id]) else 0
            score = float(torch.softmax(logits, dim=0)[plus_id])
            raw_output = None
            counts = None
        else:
            count_plus = 0
            count_minus = 0
            samples = []
            for _ in range(int(cfg.get("black_box_samples", 5))):
                token = suspect_backend.generate_first_token(
                    artifact_get(artifact, "prompt_text"),
                    temperature=float(cfg.get("temperature", 1.0)),
                    top_p=float(cfg.get("top_p", 1.0)),
                    do_sample=bool(cfg.get("do_sample", True)),
                )
                samples.append(token)
                normalized = normalize_text(token)
                if normalized.startswith(normalize_text(metadata["w_plus"])):
                    count_plus += 1
                if normalized.startswith(normalize_text(metadata["w_minus"])):
                    count_minus += 1
            predicted_bit = 1 if count_plus >= count_minus else 0
            score = count_plus / max(1, count_plus + count_minus)
            raw_output = samples[-1] if samples else None
            counts = {"count_plus": count_plus, "count_minus": count_minus, "samples": samples}
        success = predicted_bit == reference_bit
        return VerificationResult(
            method=self.name,
            base_model=artifact_get(artifact, "base_model"),
            suspect_model=suspect_backend.name,
            fingerprint_id=artifact_get(artifact, "fingerprint_id"),
            success=success,
            score=score,
            raw_output=raw_output,
            metadata={
                "reference_bit": reference_bit,
                "predicted_bit": predicted_bit,
                "verification_mode": mode,
                "counts": counts,
            },
        )

    def aggregate(self, results, cfg: dict[str, Any]) -> dict[str, Any]:
        rows = result_dicts(results)
        reference_bits = [int(row["metadata"]["reference_bit"]) for row in rows]
        predicted_bits = [int(row["metadata"]["predicted_bit"]) for row in rows]
        accuracy = bitwise_accuracy(reference_bits, predicted_bits) if rows else 0.0
        threshold = threshold_from_negative_accuracies(
            cfg.get("negative_validation_accuracies"),
            z=float(cfg.get("threshold_z", 1.64)),
            default=float(cfg.get("threshold", 0.7)),
        )
        labeled = tpr_fpr_from_labeled_results(rows)
        return {
            "bitwise_accuracy": accuracy,
            "threshold": threshold,
            "decision": accuracy >= threshold,
            "tpr": labeled["tpr"],
            "fpr": labeled["fpr"],
        }
