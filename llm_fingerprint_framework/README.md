# Unified LLM Fingerprinting Framework

This repository is a conference-paper research artifact for reproducing and comparing adversarial-example/adversarial-prompt LLM fingerprinting methods in one small Python framework.

Implemented methods:

- **TRAP**: Targeted Random Adversarial Prompt Honeypot
- **ProFLingo**: Fingerprinting-based IP Protection Scheme for LLMs
- **LLMPrint**: Fingerprinting LLMs via Prompt Injection
- **SRAF**: Stealthy and Robust Adversarial Fingerprint

This is not production software. It intentionally avoids Docker, databases, web UI, Ray, DeepSpeed, WandB, MLflow, and distributed execution.

## Design

The framework provides shared code for:

- HuggingFace causal-LM loading and generation
- prompt-template rendering
- GCG-style adversarial token optimization
- candidate filters
- losses
- rule-based verification
- JSONL fingerprint artifacts
- JSONL verification outputs
- CSV summaries

The GCG optimizer in `src/llmfp/optimizers/gcg.py` is refactored from the existing `legacy/nanojailbreak` implementation. It preserves the main loop idea: one-hot gradient over mutable tokens, top-k replacement sampling from the negative gradient direction, exact candidate loss evaluation, best-candidate updates, retokenization filtering, ASCII filtering, and adaptive batch-size evaluation. Jailbreak-specific assumptions have been removed.

## Install

From this directory:

```bash
python -m pip install -e .
```

Or run directly with:

```bash
PYTHONPATH=src python -m llmfp.cli --help
```

Dependencies are intentionally common research packages only:

```bash
python -m pip install -r requirements.txt
```

## Tiny Smoke Examples

The default configs use `distilgpt2` and very small step counts. They are for checking that the framework runs, not for paper-quality fingerprints.

```bash
PYTHONPATH=src python -m llmfp.cli construct \
  --method trap \
  --method-config configs/methods/trap.yaml \
  --model-config configs/models/example_model.yaml \
  --out results/fingerprints/trap.jsonl
```

```bash
PYTHONPATH=src python -m llmfp.cli verify \
  --method trap \
  --method-config configs/methods/trap.yaml \
  --suspect-model-config configs/models/example_model.yaml \
  --fingerprints results/fingerprints/trap.jsonl \
  --out results/runs/trap_verify.jsonl
```

```bash
PYTHONPATH=src python -m llmfp.cli benchmark \
  --config configs/experiments/example_benchmark.yaml
```

```bash
PYTHONPATH=src python -m llmfp.cli summarize \
  --results-dir results/runs \
  --out results/tables/summary.csv
```

## Model Configuration

Edit `configs/models/example_model.yaml`:

```yaml
name: my-model
model_name_or_path: /path/to/local/model
device: auto
dtype: float16
template: raw
```

Supported templates are `raw`, `fastchat_zero_shot`, `alpaca`, `llama2_chat`, `chatglm_like`, and `zero_shot`.

## Method Code

- `src/llmfp/methods/trap.py`
- `src/llmfp/methods/proflingo.py`
- `src/llmfp/methods/llmprint.py`
- `src/llmfp/methods/sraf.py`

Shared infrastructure lives under `src/llmfp/core/` and `src/llmfp/optimizers/`.

## Simplifications Compared With Papers

- Defaults are tiny smoke-test settings, not paper-scale settings.
- Verification uses transparent rule-based matching, not LLM judges.
- SRAF supports multi-template optimization and a one-model default; multi-model extensions use the same shared abstractions but are kept simple.
- Perplexity-style SRAF reporting is represented as `null` unless explicitly extended.
- LLMPrint token-pair data is a tiny JSONL file; invalid non-single-token pairs are skipped.

## Tests

The tests avoid loading large models or requiring a GPU:

```bash
PYTHONPATH=src python -m pytest tests/test_imports.py
PYTHONPATH=src python -m pytest tests/test_prompt_templates.py
PYTHONPATH=src python -m pytest tests/test_matching.py
PYTHONPATH=src python -m pytest tests/test_io.py
```
