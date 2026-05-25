from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmfp.core.templates import ChatTemplate, get_template


def _resolve_device(device: str | None) -> torch.device:
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_dtype(dtype: str | None) -> torch.dtype | None:
    if dtype in (None, "auto"):
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


@dataclass
class ModelBackend:
    name: str
    model_name_or_path: str
    device: torch.device
    dtype: torch.dtype | None
    template_name: str = "raw"
    trust_remote_code: bool = False
    tokenizer_name_or_path: str | None = None

    model: Any | None = None
    tokenizer: Any | None = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "ModelBackend":
        backend = cls(
            name=str(cfg.get("name") or cfg.get("model_name_or_path")),
            model_name_or_path=str(cfg["model_name_or_path"]),
            tokenizer_name_or_path=cfg.get("tokenizer_name_or_path"),
            device=_resolve_device(cfg.get("device", "auto")),
            dtype=_resolve_dtype(cfg.get("dtype", "auto")),
            template_name=str(cfg.get("template", "raw")),
            trust_remote_code=bool(cfg.get("trust_remote_code", False)),
        )
        backend.load()
        return backend

    @property
    def template(self) -> ChatTemplate:
        return get_template(self.template_name)

    def load(self) -> "ModelBackend":
        if self.model is not None and self.tokenizer is not None:
            return self
        tokenizer_ref = self.tokenizer_name_or_path or self.model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_ref,
            trust_remote_code=self.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_kwargs: dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        if self.dtype is not None:
            model_kwargs["torch_dtype"] = self.dtype
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        return self

    def tokenize(self, text: str, add_special_tokens: bool = False) -> list[int]:
        self.load()
        return self.tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"]

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        self.load()
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @torch.no_grad()
    def target_nll_loss(self, prompt_text: str, target_text: str) -> float:
        self.load()
        prompt_ids = self.tokenize(prompt_text, add_special_tokens=False)
        target_ids = self.tokenize(target_text, add_special_tokens=False)
        if not target_ids:
            return 0.0
        input_ids = torch.tensor([prompt_ids + target_ids], dtype=torch.long, device=self.device)
        outputs = self.model(input_ids=input_ids)
        start = max(0, len(prompt_ids) - 1)
        stop = start + len(target_ids)
        logits = outputs.logits[:, start:stop, :]
        target = torch.tensor(target_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        return float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1)).item())

    @torch.no_grad()
    def first_token_logits(self, prompt_text: str) -> torch.Tensor:
        self.load()
        ids = self.tokenize(prompt_text, add_special_tokens=False)
        if not ids:
            raise ValueError("Prompt must contain at least one token")
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        return self.model(input_ids=input_ids).logits[0, -1, :].detach().float().cpu()

    @torch.no_grad()
    def first_token_probs(self, prompt_text: str) -> torch.Tensor:
        return torch.softmax(self.first_token_logits(prompt_text), dim=-1)

    @torch.no_grad()
    def generate(
        self,
        prompt_text: str,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = False,
    ) -> str:
        self.load()
        encoded = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(self.device)
        output_ids = self.model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
        )[0]
        new_ids = output_ids[encoded["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    def generate_first_token(
        self,
        prompt_text: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = False,
    ) -> str:
        return self.generate(
            prompt_text,
            max_new_tokens=1,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )
