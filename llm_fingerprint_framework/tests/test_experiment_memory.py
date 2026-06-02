import pytest

pytest.importorskip("torch")

from llmfp.runners import experiment


class FakeBackend:
    def __init__(self, name):
        self.name = name
        self.unloaded = False

    def unload(self):
        self.unloaded = True


def test_load_backend_for_spec_reuses_identical_model_configs(monkeypatch):
    created = []

    def fake_from_config(cfg):
        backend = FakeBackend(cfg["name"])
        created.append(backend)
        return backend

    monkeypatch.setattr(experiment.ModelBackend, "from_config", staticmethod(fake_from_config))
    cache = {}
    spec = {
        "model": {
            "name": "same-model",
            "model_name_or_path": "/models/same",
            "tokenizer_name_or_path": "/models/same",
            "device": "cuda:0",
            "dtype": "float16",
            "template": "raw",
        }
    }
    first = experiment._load_backend_for_spec(spec, cache)
    second = experiment._load_backend_for_spec(spec, cache)
    assert first is second
    assert len(created) == 1


def test_unload_backend_cache_unloads_each_backend_once():
    backend = FakeBackend("same-model")
    cache = {"a": backend, "b": backend}
    experiment._unload_backend_cache(cache)
    assert backend.unloaded
    assert cache == {}
