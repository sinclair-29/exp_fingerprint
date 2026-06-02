import pytest

torch = pytest.importorskip("torch")

from llmfp.core.model_backend import ModelBackend


def test_model_backend_unload_clears_model_and_tokenizer():
    backend = ModelBackend(
        name="fake",
        model_name_or_path="fake",
        device=torch.device("cpu"),
        dtype=None,
        model=object(),
        tokenizer=object(),
    )
    backend.unload()
    assert backend.model is None
    assert backend.tokenizer is None
