from llmfp.methods.plugae import PlugAEMethod, tokenizer_contains_tokens
from llmfp.schemas import FingerprintArtifact


class FakeTokenizer:
    unk_token_id = 0

    def __init__(self, tokens):
        self.added_tokens_encoder = {token: index + 1 for index, token in enumerate(tokens)}

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, list):
            return [self.convert_tokens_to_ids(token) for token in tokens]
        return self.added_tokens_encoder.get(tokens, self.unk_token_id)


class FakeBackend:
    name = "fake"
    template_name = "raw"

    def __init__(self, tokens, output):
        self.tokenizer = FakeTokenizer(tokens)
        self.output = output

    def load(self):
        return self

    def generate(self, prompt_text, **kwargs):
        self.prompt_text = prompt_text
        return self.output


def test_tokenizer_contains_tokens():
    tokenizer = FakeTokenizer(["<COPYRIGHT_TOKEN_0>"])
    assert tokenizer_contains_tokens(tokenizer, ["<COPYRIGHT_TOKEN_0>"])
    assert not tokenizer_contains_tokens(tokenizer, ["<MISSING>"])


def test_plugae_build_tasks_uses_query_targets(tmp_path):
    path = tmp_path / "queries.jsonl"
    path.write_text('{"id":"q1","question":"Question?","target":"answer","target_keywords":["answer"]}\n', encoding="utf-8")
    tasks = PlugAEMethod().build_tasks({"query_set_path": str(path), "num_fingerprints": 1})
    assert len(tasks) == 1
    assert tasks[0].input_text == "Question?"
    assert tasks[0].target == "answer"


def test_plugae_verify_normalized_target_match():
    artifact = FingerprintArtifact(
        fingerprint_id="plugae-q1",
        method="plugae",
        base_model="base",
        task_id="q1",
        prompt_text="",
        optimized_text="<COPYRIGHT_TOKEN_0>",
        target="tomato sauce",
        metadata={
            "query": "What do bees make?",
            "copyright_tokens": ["<COPYRIGHT_TOKEN_0>"],
            "target_keywords": ["tomato sauce"],
            "insertion_position": "prefix",
        },
    )
    result = PlugAEMethod().verify(artifact, FakeBackend(["<COPYRIGHT_TOKEN_0>"], "Tomato sauce."), {"max_new_tokens": 8})
    assert result.success
    assert result.score == 1.0
    assert result.metadata["suspect_contains_copyright_tokens"]


def test_plugae_verify_records_missing_token():
    artifact = {
        "fingerprint_id": "plugae-q1",
        "base_model": "base",
        "target": "answer",
        "metadata": {
            "query": "Question?",
            "copyright_tokens": ["<COPYRIGHT_TOKEN_0>"],
            "target_keywords": ["answer"],
        },
    }
    result = PlugAEMethod().verify(artifact, FakeBackend([], "nope"), {"max_new_tokens": 8})
    assert not result.success
    assert not result.metadata["suspect_contains_copyright_tokens"]
