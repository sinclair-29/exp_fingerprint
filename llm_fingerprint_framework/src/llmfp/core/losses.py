from __future__ import annotations

import torch
import torch.nn.functional as F

from llmfp.core.prompt_builders import BuiltPrompt


class TargetStringLoss:
    def compute(self, model_backend, built_prompts: list[BuiltPrompt]) -> torch.Tensor:
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
    def __init__(self, extra_model_backends: list | None = None):
        self.extra_model_backends = extra_model_backends or []

    def compute(self, model_backend, built_prompts: list[BuiltPrompt]) -> torch.Tensor:
        # The primary backend path remains differentiable and drives GCG gradients.
        # Extra-model losses can be added by higher-level runners for exact scoring;
        # one model is the default SRAF smoke-test case.
        return super().compute(model_backend, built_prompts)
