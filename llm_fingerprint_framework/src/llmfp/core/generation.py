from __future__ import annotations


def generate_text(model_backend, prompt_text: str, cfg: dict) -> str:
    return model_backend.generate(
        prompt_text,
        max_new_tokens=int(cfg.get("max_new_tokens", 32)),
        temperature=float(cfg.get("temperature", 1.0)),
        top_p=float(cfg.get("top_p", 1.0)),
        do_sample=bool(cfg.get("do_sample", False)),
    )
