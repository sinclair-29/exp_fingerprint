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
        if self.name == "chatglm_like":
            return f"[Round 1]\n\n问：{user_prompt}\n\n答：{assistant_prefix}"
        if self.name == "zero_shot":
            return f"{user_prompt}\n\nAnswer: {assistant_prefix}"
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
        "chatglm_like",
        "zero_shot",
    }
    if name not in supported:
        raise ValueError(f"Unsupported template {name!r}. Supported: {sorted(supported)}")
    return ChatTemplate(name=name)
