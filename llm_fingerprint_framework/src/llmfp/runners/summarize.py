from __future__ import annotations

from pathlib import Path

from llmfp.core.io import load_jsonl, save_csv
from llmfp.core.metrics import summarize_by_model


def run_summarize(results_dir: str | Path, out_path: str | Path):
    rows = []
    for path in sorted(Path(results_dir).glob("*.jsonl")):
        rows.extend(load_jsonl(path))
    summary = summarize_by_model(rows)
    save_csv(out_path, summary)
    return summary
