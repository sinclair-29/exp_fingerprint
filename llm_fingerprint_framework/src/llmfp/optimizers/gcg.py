from __future__ import annotations

import gc
import inspect
import random
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from llmfp.core.candidate_filters import RetokenizationConsistencyFilter
from llmfp.core.losses import TargetStringLoss
from llmfp.core.prompt_builders import TRAPSuffixPromptBuilder
from llmfp.optimizers.result import OptimizationResult


def _should_reduce_batch_size(exception: Exception) -> bool:
    statements = [
        "CUDA out of memory.",
        "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",
        "DefaultCPUAllocator: can't allocate memory",
    ]
    return isinstance(exception, RuntimeError) and len(exception.args) == 1 and any(s in exception.args[0] for s in statements)


def find_executable_batch_size(function: Callable | None = None, starting_batch_size: int = 128):
    if function is None:
        return lambda fn: find_executable_batch_size(fn, starting_batch_size)
    batch_size = starting_batch_size

    def wrapped(*args, **kwargs):
        nonlocal batch_size
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        params = list(inspect.signature(function).parameters.keys())
        if len(params) < len(args) + 1:
            raise TypeError(f"{function.__name__} must accept search_batch_size as its first argument")
        while True:
            if batch_size <= 0:
                raise RuntimeError("No executable batch size found")
            try:
                return function(batch_size, *args, **kwargs)
            except Exception as exc:
                if not _should_reduce_batch_size(exc):
                    raise
                batch_size //= 2
    return wrapped


class GCGOptimizer:
    def __init__(
        self,
        model_backend,
        tokenizer,
        top_k: int = 256,
        search_width: int = 512,
        batch_size: int = 128,
        allow_non_ascii: bool = False,
        filter_retokenization: bool = True,
        use_prefix_cache: bool = False,
        seed: int = 0,
    ):
        self.model_backend = model_backend
        self.tokenizer = tokenizer
        self.top_k = int(top_k)
        self.search_width = int(search_width)
        self.batch_size = int(batch_size)
        self.allow_non_ascii = bool(allow_non_ascii)
        self.filter_retokenization = bool(filter_retokenization)
        self.use_prefix_cache = bool(use_prefix_cache)
        self.seed = int(seed)
        self.device = model_backend.device
        self.model = model_backend.model
        self.embedding_layer = self.model.get_input_embeddings()
        self.not_allowed_ids = None if self.allow_non_ascii else self._get_nonascii_toks()

    def _get_nonascii_toks(self) -> torch.Tensor:
        def is_ascii(text: str) -> bool:
            return text.isascii() and text.isprintable()

        special_ids = set(self.tokenizer.all_special_ids or [])
        blocked = []
        vocab_size = getattr(self.tokenizer, "vocab_size", len(self.tokenizer))
        for token_id in range(vocab_size):
            text = self.tokenizer.decode([token_id], skip_special_tokens=False)
            if token_id in special_ids or not is_ascii(text):
                blocked.append(token_id)
        return torch.tensor(blocked, dtype=torch.long, device=self.device)

    def _set_seed(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _sample_ids(self, cur_ids: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        if self.not_allowed_ids is not None and self.not_allowed_ids.numel() > 0:
            grad[:, self.not_allowed_ids] = torch.inf
        top_k = min(self.top_k, grad.shape[1])
        topk_ids = (-grad).topk(top_k, dim=1).indices
        original = cur_ids.repeat(self.search_width, 1)
        n_tokens = cur_ids.shape[0]
        positions = torch.randint(0, n_tokens, (self.search_width,), device=self.device)
        sampled_topk = torch.randint(0, top_k, (self.search_width, 1), device=self.device)
        sampled_values = torch.gather(topk_ids[positions], dim=1, index=sampled_topk).squeeze(1)
        return original.scatter(1, positions.unsqueeze(1), sampled_values.unsqueeze(1))

    def _filter_candidates(self, ids: torch.Tensor, candidate_filter) -> torch.Tensor:
        filtered = ids
        if candidate_filter is not None:
            filtered = candidate_filter.filter_ids(filtered, self.tokenizer)
        if self.filter_retokenization and filtered.numel() > 0:
            filtered = RetokenizationConsistencyFilter().filter_ids(filtered, self.tokenizer)
        return filtered

    def _compute_gradient(self, cur_ids: torch.Tensor, prompt_builder, loss_fn) -> torch.Tensor:
        ids = cur_ids.unsqueeze(0)
        onehot = F.one_hot(ids, num_classes=self.embedding_layer.num_embeddings).to(self.device, self.model.dtype)
        onehot = onehot.detach().clone().requires_grad_(True)
        if hasattr(loss_fn, "compute_with_onehot"):
            loss = loss_fn.compute_with_onehot(self.model_backend, prompt_builder, onehot).mean()
        else:
            mutable_embeds = onehot @ self.embedding_layer.weight
            built = prompt_builder.build_inputs(self.model_backend, mutable_embeds=mutable_embeds)
            loss = loss_fn.compute(self.model_backend, built).mean()
        grad = torch.autograd.grad(outputs=[loss], inputs=[onehot])[0]
        return grad.squeeze(0)

    def _candidate_losses(self, prompt_builder, loss_fn, candidate_ids: torch.Tensor) -> torch.Tensor:
        return find_executable_batch_size(self._candidate_losses_batched, self.batch_size)(prompt_builder, loss_fn, candidate_ids)

    def _candidate_losses_batched(self, search_batch_size: int, prompt_builder, loss_fn, candidate_ids: torch.Tensor) -> torch.Tensor:
        losses = []
        for start in range(0, candidate_ids.shape[0], search_batch_size):
            batch_ids = candidate_ids[start: start + search_batch_size]
            with torch.no_grad():
                if hasattr(loss_fn, "compute_for_candidates"):
                    loss = loss_fn.compute_for_candidates(self.model_backend, prompt_builder, batch_ids)
                else:
                    built = prompt_builder.build_inputs(self.model_backend, mutable_ids=batch_ids)
                    loss = loss_fn.compute(self.model_backend, built)
                losses.append(loss.detach())
        return torch.cat(losses, dim=0)

    def optimize(
        self,
        init_ids,
        prompt_builder,
        loss_fn,
        candidate_filter=None,
        num_steps: int = 500,
        log_every: int = 10,
    ) -> OptimizationResult:
        self._set_seed()
        if not isinstance(init_ids, torch.Tensor):
            init_ids = torch.tensor(init_ids, dtype=torch.long)
        cur_ids = init_ids.detach().clone().to(self.device, dtype=torch.long).view(-1)
        prompt_builder.prepare(self.model_backend, cur_ids)

        with torch.no_grad():
            init_loss = float(self._candidate_losses(prompt_builder, loss_fn, cur_ids.unsqueeze(0))[0].item())
        best_ids = cur_ids.detach().clone()
        best_loss = init_loss
        best_step = 0
        loss_history = [init_loss]

        iterator = tqdm(range(1, int(num_steps) + 1), desc="gcg", leave=False)
        for step in iterator:
            grad = self._compute_gradient(cur_ids, prompt_builder, loss_fn)
            with torch.no_grad():
                sampled = self._sample_ids(cur_ids, grad)
                sampled = self._filter_candidates(sampled, candidate_filter)
                if sampled.numel() == 0:
                    loss_history.append(loss_history[-1])
                    continue
                losses = self._candidate_losses(prompt_builder, loss_fn, sampled)
                min_index = int(torch.argmin(losses).item())
                cur_ids = sampled[min_index].detach().clone()
                current_loss = float(losses[min_index].item())
                loss_history.append(current_loss)
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_ids = cur_ids.detach().clone()
                    best_step = step
                if log_every and step % log_every == 0:
                    iterator.set_postfix(loss=f"{current_loss:.4f}", best=f"{best_loss:.4f}")

        best_text = self.tokenizer.decode(best_ids.detach().cpu().tolist(), skip_special_tokens=False)
        return OptimizationResult(
            best_ids=best_ids.detach().cpu().tolist(),
            best_text=best_text,
            best_loss=best_loss,
            best_step=best_step,
            loss_history=loss_history,
            metadata={
                "top_k": self.top_k,
                "search_width": self.search_width,
                "batch_size": self.batch_size,
                "allow_non_ascii": self.allow_non_ascii,
                "filter_retokenization": self.filter_retokenization,
                "use_prefix_cache": self.use_prefix_cache,
                "seed": self.seed,
            },
        )


def optimize_suffix_target_string(
    model_backend,
    instruction: str,
    target: str,
    init_suffix: str,
    cfg: dict,
    candidate_filter=None,
) -> OptimizationResult:
    tokenizer = model_backend.tokenizer
    init_ids = tokenizer(init_suffix, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    builder = TRAPSuffixPromptBuilder(
        tokenizer=tokenizer,
        instruction=instruction,
        target=target,
        template_name=getattr(model_backend, "template_name", "raw"),
    )
    optimizer = GCGOptimizer(
        model_backend=model_backend,
        tokenizer=tokenizer,
        top_k=cfg.get("top_k", 256),
        search_width=cfg.get("search_width", 512),
        batch_size=cfg.get("batch_size", 128),
        allow_non_ascii=cfg.get("allow_non_ascii", False),
        filter_retokenization=cfg.get("filter_retokenization", True),
        use_prefix_cache=cfg.get("use_prefix_cache", False),
        seed=cfg.get("seed", 0),
    )
    return optimizer.optimize(
        init_ids=init_ids,
        prompt_builder=builder,
        loss_fn=TargetStringLoss(),
        candidate_filter=candidate_filter,
        num_steps=cfg.get("num_steps", 500),
        log_every=cfg.get("log_every", 10),
    )
