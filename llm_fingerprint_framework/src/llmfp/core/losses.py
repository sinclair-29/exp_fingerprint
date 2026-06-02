from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from llmfp.core.prompt_builders import BuiltPrompt


def _target_string_loss(model_backend, built_prompts: list[BuiltPrompt]) -> torch.Tensor:
    losses = []
    for built in built_prompts:
        if built.target_ids is None or built.loss_slice is None:
            raise ValueError("TargetStringLoss requires target_ids and loss_slice")
        outputs = model_backend.model(inputs_embeds=built.input_embeds)
        logits = outputs.logits[:, built.loss_slice, :]
        batch_size = logits.shape[0]
        target = built.target_ids.unsqueeze(0).repeat(batch_size, 1)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            reduction="none",
        ).view(batch_size, -1).mean(dim=1)
        losses.append(loss)
    return torch.stack(losses, dim=0).mean(dim=0)


class TargetStringLoss:
    def compute(self, model_backend, built_prompts: list[BuiltPrompt]) -> torch.Tensor:
        return _target_string_loss(model_backend, built_prompts)


class MultiTemplateTargetLoss(TargetStringLoss):
    pass


class TokenPreferenceLoss:
    def __init__(self, w_plus_id: int, w_minus_id: int, alpha: float = 0.1, beta: float = 1.0):
        self.w_plus_id = int(w_plus_id)
        self.w_minus_id = int(w_minus_id)
        self.alpha = float(alpha)
        self.beta = float(beta)

    def compute(self, model_backend, built_prompts: list[BuiltPrompt]) -> torch.Tensor:
        losses = []
        for built in built_prompts:
            outputs = model_backend.model(inputs_embeds=built.input_embeds)
            logits = outputs.logits[:, built.logit_index, :]
            z_plus = logits[:, self.w_plus_id]
            z_minus = logits[:, self.w_minus_id]
            uniqueness = -F.logsigmoid(z_plus - z_minus) + self.alpha * torch.abs(z_plus - z_minus)
            keep = torch.ones_like(logits, dtype=torch.bool)
            keep[:, self.w_plus_id] = False
            keep[:, self.w_minus_id] = False
            other_logits = logits.masked_fill(~keep, -torch.inf)
            robustness = F.relu(torch.logsumexp(other_logits, dim=1) - z_plus)
            losses.append(uniqueness + self.beta * robustness)
        return torch.stack(losses, dim=0).mean(dim=0)


class MultiModelMultiTemplateTargetLoss(TargetStringLoss):
    def __init__(self, extra_model_backends: list | None = None, backend_specs: list[dict[str, Any]] | None = None):
        specs = list(backend_specs or [])
        for backend in extra_model_backends or []:
            specs.append({"backend": backend, "weight": 1.0})
        self.backend_specs = specs

    def compute(self, model_backend, built_prompts: list[BuiltPrompt]) -> torch.Tensor:
        return super().compute(model_backend, built_prompts)

    def _weighted_backend_specs(self, primary_backend) -> list[tuple[Any, float]]:
        specs: list[tuple[Any, float]] = [(primary_backend, 1.0)]
        for item in self.backend_specs:
            backend = item["backend"] if isinstance(item, dict) else item
            weight = float(item.get("weight", 1.0)) if isinstance(item, dict) else 1.0
            if weight <= 0:
                continue
            specs.append((backend, weight))
        return specs

    def compute_for_candidates(self, primary_backend, prompt_builder, candidate_ids: torch.Tensor) -> torch.Tensor:
        total_loss = None
        total_weight = 0.0
        for backend, weight in self._weighted_backend_specs(primary_backend):
            built = prompt_builder.build_inputs(backend, mutable_ids=candidate_ids)
            loss = _target_string_loss(backend, built).to(primary_backend.device)
            total_loss = loss * weight if total_loss is None else total_loss + loss * weight
            total_weight += weight
        if total_loss is None or total_weight <= 0:
            raise ValueError("MultiModelMultiTemplateTargetLoss requires at least one positive-weight backend")
        return total_loss / total_weight

    def compute_with_onehot(self, primary_backend, prompt_builder, onehot: torch.Tensor) -> torch.Tensor:
        total_loss = None
        total_weight = 0.0
        for backend, weight in self._weighted_backend_specs(primary_backend):
            embedding_layer = backend.model.get_input_embeddings()
            backend_onehot = onehot.to(device=backend.device, dtype=embedding_layer.weight.dtype)
            mutable_embeds = backend_onehot @ embedding_layer.weight
            built = prompt_builder.build_inputs(backend, mutable_embeds=mutable_embeds)
            loss = _target_string_loss(backend, built).to(primary_backend.device)
            total_loss = loss * weight if total_loss is None else total_loss + loss * weight
            total_weight += weight
        if total_loss is None or total_weight <= 0:
            raise ValueError("MultiModelMultiTemplateTargetLoss requires at least one positive-weight backend")
        return total_loss / total_weight
