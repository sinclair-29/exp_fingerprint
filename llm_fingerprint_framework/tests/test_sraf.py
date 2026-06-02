from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from llmfp.core.losses import MultiModelMultiTemplateTargetLoss
from llmfp.core.prompt_builders import BuiltPrompt, SRAFTablePromptBuilder


class TinyTokenizer:
    all_special_ids = []
    vocab_size = 256

    def __len__(self):
        return self.vocab_size

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = [ord(ch) for ch in text]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(int(token_id)) for token_id in ids)

    def batch_decode(self, rows, skip_special_tokens=False):
        return [self.decode(row.tolist(), skip_special_tokens=skip_special_tokens) for row in rows]


def test_sraf_table_builder_hides_mutable_tokens_in_configured_cells():
    tokenizer = TinyTokenizer()
    builder = SRAFTablePromptBuilder(
        tokenizer,
        base_prompt="Read the reference table.",
        target="cerulean harbor",
        templates=["sraf_zero_shot"],
        table_segments=4,
    )

    mutable_ids = torch.tensor([65, 66, 67, 68, 69, 70, 71, 72], dtype=torch.long)
    prompt = builder.build_prompt_text(mutable_ids)

    assert "[[LLMFP_SRAF_" not in prompt
    assert "| Field | Value |" in prompt
    assert "|---|---|" in prompt
    assert "| Segment 1 | AB |" in prompt
    assert "| Segment 2 | CD |" in prompt
    assert "| Segment 3 | EF |" in prompt
    assert "| Segment 4 | GH |" in prompt
    assert builder.mutable_segments(mutable_ids) == ["AB", "CD", "EF", "GH"]


class FakeEmbedding:
    def __init__(self):
        self.weight = torch.ones(2, 1)

    def __call__(self, ids):
        return torch.ones(*ids.shape, 1)


class FakeModel:
    def __init__(self, logits_by_template):
        self.logits_by_template = [torch.tensor(logits, dtype=torch.float32) for logits in logits_by_template]
        self.embedding = FakeEmbedding()

    def get_input_embeddings(self):
        return self.embedding

    def __call__(self, inputs_embeds):
        template_index = int(inputs_embeds[0, 0, 0].item())
        batch_size = inputs_embeds.shape[0]
        logits = self.logits_by_template[template_index].view(1, 1, -1).repeat(batch_size, 1, 1)
        return SimpleNamespace(logits=logits)


class FakeBackend:
    def __init__(self, name, logits_by_template):
        self.name = name
        self.device = torch.device("cpu")
        self.model = FakeModel(logits_by_template)


class FakePromptBuilder:
    def build_inputs(self, backend, mutable_ids=None, mutable_embeds=None):
        batch_size = mutable_ids.shape[0] if mutable_ids is not None else mutable_embeds.shape[0]
        prompts = []
        for template_index in range(len(backend.model.logits_by_template)):
            prompts.append(
                BuiltPrompt(
                    input_embeds=torch.full((batch_size, 1, 1), float(template_index)),
                    target_ids=torch.tensor([0], dtype=torch.long),
                    loss_slice=slice(0, 1),
                )
            )
        return prompts


def _template_loss(logits):
    return F.cross_entropy(torch.tensor(logits, dtype=torch.float32).view(1, -1), torch.tensor([0])).item()


def test_sraf_multimodel_loss_uses_weighted_backend_and_template_average():
    primary = FakeBackend("primary", logits_by_template=[[2.0, 0.0], [0.0, 0.0]])
    extra = FakeBackend("extra", logits_by_template=[[0.0, 2.0], [1.0, 0.0]])
    prompt_builder = FakePromptBuilder()
    loss_fn = MultiModelMultiTemplateTargetLoss(backend_specs=[{"backend": extra, "weight": 2.0}])

    candidate_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    losses = loss_fn.compute_for_candidates(primary, prompt_builder, candidate_ids)

    primary_loss = (_template_loss([2.0, 0.0]) + _template_loss([0.0, 0.0])) / 2
    extra_loss = (_template_loss([0.0, 2.0]) + _template_loss([1.0, 0.0])) / 2
    expected = (primary_loss + 2.0 * extra_loss) / 3.0
    assert torch.allclose(losses, torch.full((2,), expected))

    onehot = F.one_hot(candidate_ids[:1], num_classes=2).float().requires_grad_(True)
    onehot_loss = loss_fn.compute_with_onehot(primary, prompt_builder, onehot)
    assert torch.allclose(onehot_loss, torch.tensor([expected]))
