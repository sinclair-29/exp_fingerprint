from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from llmfp.core.templates import get_template


@dataclass
class BuiltPrompt:
    input_embeds: torch.Tensor
    target_ids: torch.Tensor | None = None
    loss_slice: slice | None = None
    logit_index: int = -1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptVariantSpec:
    name: str
    fixed_pieces: list[str]
    target_text: str = ""


class SegmentedPromptBuilder:
    def __init__(self, tokenizer, variants: list[PromptVariantSpec], segment_count: int = 1):
        if segment_count < 1:
            raise ValueError("segment_count must be positive")
        self.tokenizer = tokenizer
        self.variants = variants
        self.segment_count = segment_count

    def prepare(self, model_backend, init_ids: torch.Tensor) -> None:
        return None

    def _encode(self, text: str, device: torch.device) -> torch.Tensor:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return torch.tensor(ids, dtype=torch.long, device=device)

    def _embed_text(self, model_backend, text: str, batch_size: int) -> torch.Tensor | None:
        ids = self._encode(text, model_backend.device)
        if ids.numel() == 0:
            return None
        embeds = model_backend.model.get_input_embeddings()(ids.unsqueeze(0))
        return embeds.repeat(batch_size, 1, 1)

    def _coerce_mutable(
        self,
        model_backend,
        mutable_ids: torch.Tensor | None = None,
        mutable_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if mutable_embeds is None:
            if mutable_ids is None:
                raise ValueError("Either mutable_ids or mutable_embeds must be provided")
            if mutable_ids.dim() == 1:
                mutable_ids = mutable_ids.unsqueeze(0)
            mutable_ids = mutable_ids.to(model_backend.device, dtype=torch.long)
            mutable_embeds = model_backend.model.get_input_embeddings()(mutable_ids)
        elif mutable_embeds.dim() == 2:
            mutable_embeds = mutable_embeds.unsqueeze(0)
        return mutable_ids, mutable_embeds

    def _split_tensor(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        lengths = [tensor.shape[1] // self.segment_count] * self.segment_count
        for i in range(tensor.shape[1] % self.segment_count):
            lengths[i] += 1
        chunks = []
        cursor = 0
        for length in lengths:
            chunks.append(tensor[:, cursor: cursor + length, :])
            cursor += length
        return chunks

    def _split_ids(self, ids: torch.Tensor | list[int]) -> list[list[int]]:
        values = ids.detach().cpu().tolist() if isinstance(ids, torch.Tensor) else list(ids)
        lengths = [len(values) // self.segment_count] * self.segment_count
        for i in range(len(values) % self.segment_count):
            lengths[i] += 1
        chunks = []
        cursor = 0
        for length in lengths:
            chunks.append(values[cursor: cursor + length])
            cursor += length
        return chunks

    def build_inputs(
        self,
        model_backend,
        mutable_ids: torch.Tensor | None = None,
        mutable_embeds: torch.Tensor | None = None,
    ) -> list[BuiltPrompt]:
        mutable_ids, mutable_embeds = self._coerce_mutable(model_backend, mutable_ids, mutable_embeds)
        batch_size = mutable_embeds.shape[0]
        chunks = self._split_tensor(mutable_embeds)
        built: list[BuiltPrompt] = []
        for variant in self.variants:
            if len(variant.fixed_pieces) != self.segment_count + 1:
                raise ValueError("fixed_pieces must have segment_count + 1 entries")
            pieces: list[torch.Tensor] = []
            for idx, fixed_text in enumerate(variant.fixed_pieces):
                fixed_embeds = self._embed_text(model_backend, fixed_text, batch_size)
                if fixed_embeds is not None:
                    pieces.append(fixed_embeds)
                if idx < self.segment_count:
                    pieces.append(chunks[idx])
            target_ids = self._encode(variant.target_text, model_backend.device)
            non_target_len = sum(piece.shape[1] for piece in pieces)
            if target_ids.numel() > 0:
                target_embeds = model_backend.model.get_input_embeddings()(target_ids.unsqueeze(0)).repeat(batch_size, 1, 1)
                pieces.append(target_embeds)
                loss_slice = slice(non_target_len - 1, non_target_len + target_ids.numel() - 1)
            else:
                target_ids = None
                loss_slice = None
            built.append(
                BuiltPrompt(
                    input_embeds=torch.cat(pieces, dim=1),
                    target_ids=target_ids,
                    loss_slice=loss_slice,
                    logit_index=-1,
                    metadata={"variant": variant.name},
                )
            )
        return built

    def build_prompt_text(self, mutable_ids: torch.Tensor | list[int], include_target: bool = False, variant_index: int = 0) -> str:
        variant = self.variants[variant_index]
        chunks = [
            self.tokenizer.decode(chunk, skip_special_tokens=False)
            for chunk in self._split_ids(mutable_ids)
        ]
        parts = []
        for idx, fixed_text in enumerate(variant.fixed_pieces):
            parts.append(fixed_text)
            if idx < self.segment_count:
                parts.append(chunks[idx])
        if include_target:
            parts.append(variant.target_text)
        return "".join(parts)

    def mutable_segments(self, mutable_ids: torch.Tensor | list[int]) -> list[str]:
        return [
            self.tokenizer.decode(chunk, skip_special_tokens=False)
            for chunk in self._split_ids(mutable_ids)
        ]

    def metadata(self) -> dict[str, Any]:
        return {
            "templates": [variant.name for variant in self.variants],
            "segment_count": self.segment_count,
        }


def _templated_variant(template_name: str, user_before: str, user_after: str, target: str) -> PromptVariantSpec:
    template = get_template(template_name)
    before, after = template.split_around_mutable(user_before, user_after)
    return PromptVariantSpec(name=template_name, fixed_pieces=[before, after], target_text=target)


class TRAPSuffixPromptBuilder(SegmentedPromptBuilder):
    def __init__(self, tokenizer, instruction: str, target: str, template_name: str = "raw"):
        self.instruction = instruction
        self.target = target
        super().__init__(
            tokenizer=tokenizer,
            variants=[_templated_variant(template_name, instruction + " ", "", target)],
            segment_count=1,
        )


class ProFLingoPrefixPromptBuilder(SegmentedPromptBuilder):
    def __init__(self, tokenizer, question: str, target: str, templates: list[str]):
        self.question = question
        self.query_suffix = " simply answer: " + question
        self.target = target
        super().__init__(
            tokenizer=tokenizer,
            variants=[_templated_variant(name, "", self.query_suffix, target) for name in templates],
            segment_count=1,
        )


class LLMPrintSuffixPromptBuilder(SegmentedPromptBuilder):
    def __init__(self, tokenizer, base_prompt: str = "Randomly output a word from your vocabulary", template_name: str = "raw"):
        self.base_prompt = base_prompt
        super().__init__(
            tokenizer=tokenizer,
            variants=[_templated_variant(template_name, base_prompt + " ", "", "")],
            segment_count=1,
        )


class SRAFTablePromptBuilder(SegmentedPromptBuilder):
    SENTINELS = ["[[LLMFP_SRAF_0]]", "[[LLMFP_SRAF_1]]", "[[LLMFP_SRAF_2]]"]

    def __init__(self, tokenizer, base_prompt: str, target: str, templates: list[str]):
        self.base_prompt = base_prompt
        self.target = target
        variants = []
        user_prompt = self._user_prompt_with_sentinels(base_prompt)
        for template_name in templates:
            rendered = get_template(template_name).render(user_prompt, assistant_prefix="")
            variants.append(PromptVariantSpec(template_name, self._split_rendered(rendered), target))
        super().__init__(tokenizer=tokenizer, variants=variants, segment_count=3)

    @classmethod
    def _user_prompt_with_sentinels(cls, base_prompt: str) -> str:
        return (
            f"{base_prompt}\n\n"
            "| Field | Value |\n"
            "|---|---|\n"
            f"| Note | {cls.SENTINELS[0]} |\n"
            f"| Key | {cls.SENTINELS[1]} |\n"
            f"| Ref | {cls.SENTINELS[2]} |\n"
        )

    @classmethod
    def _split_rendered(cls, rendered: str) -> list[str]:
        pieces = []
        cursor = 0
        for sentinel in cls.SENTINELS:
            index = rendered.find(sentinel, cursor)
            if index < 0:
                raise ValueError("SRAF template did not preserve all mutable sentinels")
            pieces.append(rendered[cursor:index])
            cursor = index + len(sentinel)
        pieces.append(rendered[cursor:])
        return pieces
