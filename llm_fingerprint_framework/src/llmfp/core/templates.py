from __future__ import annotations

from dataclasses import dataclass


_MUTABLE_SENTINEL = "[[LLMFP_MUTABLE_SPAN]]"


@dataclass(frozen=True)
class ChatTemplate:
    name: str

    def render(self, user_prompt: str, assistant_prefix: str = "") -> str:
        if self.name == "raw":
            return user_prompt + assistant_prefix
        if self.name == "fastchat_zero_shot":
            return f"### Human: {user_prompt}\n### Assistant: {assistant_prefix}"
        if self.name == "alpaca":
            return (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{user_prompt}\n\n### Response:\n{assistant_prefix}"
            )
        if self.name == "llama2_chat":
            return f"<s>[INST] {user_prompt} [/INST] {assistant_prefix}"
        if self.name == "mistral_instruct":
            return f"<s>[INST] {user_prompt} [/INST] {assistant_prefix}"
        if self.name == "gemma_it":
            return f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n{assistant_prefix}"
        if self.name == "phi3_chat":
            return f"<|user|>\n{user_prompt}<|end|>\n<|assistant|>\n{assistant_prefix}"
        if self.name == "chatml":
            return (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n{assistant_prefix}"
            )
        if self.name == "vicuna_chat":
            return (
                "A chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the user's questions. "
                f"USER: {user_prompt} ASSISTANT: {assistant_prefix}"
            )
        if self.name == "chatglm_like":
            return f"[Round 1]\n\n问：{user_prompt}\n\n答：{assistant_prefix}"
        if self.name == "zero_shot":
            return f"{user_prompt}\n\nAnswer: {assistant_prefix}"
        if self.name == "sraf_default":
            return f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n{assistant_prefix}"
        if self.name == "sraf_alpaca":
            return f"### Instruction:\n{user_prompt}\n\n### Response:\n{assistant_prefix}"
        if self.name == "sraf_chatglm":
            return f"[Round 1]\n问：{user_prompt}\n答：{assistant_prefix}"
        if self.name == "sraf_llama2":
            return f"<s>[INST] {user_prompt} [/INST] </s>{assistant_prefix}"
        if self.name == "sraf_zero_shot":
            return user_prompt + assistant_prefix
        raise ValueError(f"Unknown template: {self.name}")

    def split_around_mutable(self, user_before: str, user_after: str) -> tuple[str, str]:
        rendered = self.render(f"{user_before}{_MUTABLE_SENTINEL}{user_after}", assistant_prefix="")
        if rendered.count(_MUTABLE_SENTINEL) != 1:
            raise ValueError(f"Template {self.name} did not preserve mutable sentinel exactly once")
        before, after = rendered.split(_MUTABLE_SENTINEL)
        return before, after


def get_template(name: str) -> ChatTemplate:
    supported = {
        "raw",
        "fastchat_zero_shot",
        "alpaca",
        "llama2_chat",
        "mistral_instruct",
        "gemma_it",
        "phi3_chat",
        "chatml",
        "vicuna_chat",
        "chatglm_like",
        "zero_shot",
        "sraf_default",
        "sraf_alpaca",
        "sraf_chatglm",
        "sraf_llama2",
        "sraf_zero_shot",
    }
    if name not in supported:
        raise ValueError(f"Unsupported template {name!r}. Supported: {sorted(supported)}")
    return ChatTemplate(name=name)
