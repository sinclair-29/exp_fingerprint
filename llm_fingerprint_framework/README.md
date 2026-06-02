# Unified LLM Fingerprinting Framework

This repository is a conference-paper research artifact for reproducing and comparing adversarial-example/adversarial-prompt LLM fingerprinting methods in one small Python framework.

Implemented methods:

- **TRAP**: Targeted Random Adversarial Prompt Honeypot
- **ProFLingo**: Fingerprinting-based IP Protection Scheme for LLMs
- **LLMPrint**: Fingerprinting LLMs via Prompt Injection
- **SRAF**: Stealthy and Robust Adversarial Fingerprint
- **PlugAE**: proactive copyright-token embedding protection

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
- unified raw JSONL experiment records
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

## Unified Five-Method Experiments

The unified runner constructs one fingerprint at a time and immediately writes a complete raw record with source, positive-suspect, negative-suspect, and deployment-variation verification results.

```bash
PYTHONPATH=src python -m llmfp.cli experiment \
  --config configs/experiments/smoke_unified_five_methods.yaml
```

Build the six derived tables from raw JSONL only:

```bash
PYTHONPATH=src python -m llmfp.cli build-tables \
  --raw-dir results/raw \
  --out-dir results/derived_tables
```

The table builder never reruns model inference.

## Model Configuration

Edit `configs/models/example_model.yaml`:

```yaml
name: my-model
model_name_or_path: /path/to/local/model
device: auto
dtype: float16
template: raw
```

Supported templates are `raw`, `fastchat_zero_shot`, `alpaca`, `llama2_chat`, `mistral_instruct`, `gemma_it`, `phi3_chat`, `chatml`, `vicuna_chat`, `chatglm_like`, `zero_shot`, and the SRAF paper-style templates `sraf_default`, `sraf_alpaca`, `sraf_chatglm`, `sraf_llama2`, and `sraf_zero_shot`.

## Method Code

- `src/llmfp/methods/trap.py`
- `src/llmfp/methods/proflingo.py`
- `src/llmfp/methods/llmprint.py`
- `src/llmfp/methods/sraf.py`
- `src/llmfp/methods/plugae.py`

Shared infrastructure lives under `src/llmfp/core/` and `src/llmfp/optimizers/`.

## PlugAE

PlugAE is proactive. Its `construct` step adds configured copyright tokens to the tokenizer, resizes the embedding layer when needed, optimizes those token embeddings in continuous space against query-target pairs, writes the optimized vectors into the embedding matrix, and saves a protected HuggingFace model/tokenizer artifact.

Example:

```bash
PYTHONPATH=src python -m llmfp.cli construct \
  --method plugae \
  --method-config configs/methods/smoke_plugae.yaml \
  --model-config configs/models/example_model.yaml \
  --out results/fingerprints/plugae.jsonl
```

```bash
PYTHONPATH=src python -m llmfp.cli verify \
  --method plugae \
  --method-config configs/methods/smoke_plugae.yaml \
  --suspect-model-config configs/models/example_model.yaml \
  --fingerprints results/fingerprints/plugae.jsonl \
  --out results/runs/plugae_verify.jsonl
```

Important config fields:

```yaml
num_adv_tokens: 1
copyright_tokens: ["<COPYRIGHT_TOKEN_0>"]
insertion_position: prefix
lr: 0.1
epochs: 30
templates: [default]
query_set_path: data/plugae_queries.jsonl
protected_model_output_dir: artifacts/plugae_protected/<source_model>
```

PlugAE positive suspects should be derivatives of the PlugAE-protected model. The unified experiment config has `plugae.protected_derivatives` slots for externally prepared SFT, LoRA/PEFT, and quantized derivatives. This repository does not add SFT, LoRA, or quantization utilities.

If a configured PlugAE positive suspect does not contain the copyright token(s), the raw record marks that verification row with `valid_for_method: false`.

## Raw Record Schema

Unified experiments write one JSONL line per fingerprint under `results/raw/<method>/`. Each line contains:

- `run_id`, `method`, `source_model`, `seed`, `fingerprint_id`
- `fingerprint_spec`: base prompt, target, full query, adversarial text, method metadata
- `generation`: source success, loss, best step, step count, loss curve or path
- `verification`: source/positive/negative rows with condition, sampling, output, score, validity, and method metadata
- `stealthiness`: optional PPL fields, `null` when unavailable
- `efficiency`: generation time, best-effort GPU memory, optimization steps, query counts

Use `null` for unavailable values; required sections are always present.

## Derived Tables

`build-tables` creates:

- `exp1_main_verification.csv`
- `exp2_model_modification_robustness.csv`
- `exp3_deployment_robustness.csv`
- `exp4_false_positive_specificity.csv`
- `exp5_stealthiness.csv`
- `exp6_efficiency.csv`

The tables are computed from raw records only. AUC and TPR@5%FPR are reported when both positive and negative valid rows are available.

## Simplifications Compared With Papers

- Defaults are tiny smoke-test settings, not paper-scale settings.
- Verification uses transparent rule-based matching, not LLM judges.
- ProFLingo and PlugAE use normalized string matching; semantic equivalence is not implemented.
- SRAF supports Markdown-table hidden segments, multi-template optimization, optional homologous co-model optimization, exact normalized matching, and optional GPT-2-style PPL metadata.
- LLMPrint token-pair data is a tiny JSONL file; invalid non-single-token pairs are skipped.
- PPL and peak GPU memory are best-effort and may be `null`.

## Tests

The tests avoid loading large models or requiring a GPU:

```bash
PYTHONPATH=src python -m pytest tests/test_imports.py
PYTHONPATH=src python -m pytest tests/test_prompt_templates.py
PYTHONPATH=src python -m pytest tests/test_matching.py
PYTHONPATH=src python -m pytest tests/test_io.py
```
