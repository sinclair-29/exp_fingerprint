from __future__ import annotations

import importlib


METHODS = {
    "trap": "llmfp.methods.trap:TRAPMethod",
    "proflingo": "llmfp.methods.proflingo:ProFLingoMethod",
    "llmprint": "llmfp.methods.llmprint:LLMPrintMethod",
    "sraf": "llmfp.methods.sraf:SRAFMethod",
    "plugae": "llmfp.methods.plugae:PlugAEMethod",
}


def get_method(name: str):
    try:
        ref = METHODS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown method {name!r}. Available: {sorted(METHODS)}") from exc
    module_name, class_name = ref.split(":", 1)
    module = importlib.import_module(module_name)
    method_cls = getattr(module, class_name)
    return method_cls()
