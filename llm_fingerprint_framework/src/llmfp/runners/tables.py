from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from llmfp.core.io import load_jsonl, save_csv
from llmfp.core.metrics import safe_mean


def _load_raw_records(raw_dir: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(raw_dir).glob("**/*.jsonl")):
        rows.extend(load_jsonl(path))
    return rows


def _flatten_verification(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        for verification in record.get("verification", []):
            rows.append(
                {
                    "run_id": record.get("run_id"),
                    "method": record.get("method"),
                    "source_model": record.get("source_model"),
                    "seed": record.get("seed"),
                    "fingerprint_id": record.get("fingerprint_id"),
                    "model": verification.get("model"),
                    "model_role": verification.get("model_role"),
                    "modification_type": verification.get("modification_type"),
                    "negative_type": verification.get("negative_type"),
                    "condition": verification.get("condition"),
                    "system_prompt": verification.get("system_prompt"),
                    "sampling": verification.get("sampling") or {},
                    "score": verification.get("score"),
                    "valid_for_method": verification.get("valid_for_method", True),
                }
            )
    return rows


def _mean_score(rows: Iterable[dict[str, Any]]) -> float | None:
    return safe_mean(row.get("score") for row in rows)


def _auc(pos_scores: list[float], neg_scores: list[float]) -> float | None:
    if not pos_scores or not neg_scores:
        return None
    wins = 0.0
    total = 0
    for pos in pos_scores:
        for neg in neg_scores:
            total += 1
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / total if total else None


def _tpr_at_fpr(pos_scores: list[float], neg_scores: list[float], max_fpr: float = 0.05) -> float | None:
    if not pos_scores or not neg_scores:
        return None
    thresholds = sorted(set(pos_scores + neg_scores), reverse=True)
    best = 0.0
    for threshold in thresholds:
        tpr = sum(score >= threshold for score in pos_scores) / len(pos_scores)
        fpr = sum(score >= threshold for score in neg_scores) / len(neg_scores)
        if fpr <= max_fpr:
            best = max(best, tpr)
    return best


def _group(rows: Iterable[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    return grouped


def _valid(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("valid_for_method", True)]


def _exp1(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base_rows = _valid(row for row in rows if row.get("condition") == "default" and row.get("model_role") in {"positive", "negative"})
    output = []
    for key, group_rows in _group(base_rows, ("method", "source_model")).items():
        positives = [row for row in group_rows if row.get("model_role") == "positive"]
        negatives = [row for row in group_rows if row.get("model_role") == "negative"]
        pos_scores = [float(row.get("score") or 0.0) for row in positives]
        neg_scores = [float(row.get("score") or 0.0) for row in negatives]
        output.append(
            {
                "method": key[0],
                "source_model": key[1],
                "num_positive": len(positives),
                "num_negative": len(negatives),
                "positive_mean_score": _mean_score(positives),
                "negative_mean_score": _mean_score(negatives),
                "fpr": _mean_score(negatives),
                "auc": _auc(pos_scores, neg_scores),
                "tpr_at_5pct_fpr": _tpr_at_fpr(pos_scores, neg_scores),
            }
        )
    return output


def _exp2(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base_rows = _valid(row for row in rows if row.get("condition") == "default" and row.get("model_role") == "positive")
    output = []
    for key, group_rows in _group(base_rows, ("method", "source_model", "modification_type")).items():
        output.append(
            {
                "method": key[0],
                "source_model": key[1],
                "modification_type": key[2],
                "n": len(group_rows),
                "mean_score": _mean_score(group_rows),
            }
        )
    return output


def _exp3(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base_rows = _valid(row for row in rows if row.get("model_role") in {"source", "positive"})
    output = []
    for key, group_rows in _group(base_rows, ("method", "source_model", "model_role", "condition")).items():
        output.append(
            {
                "method": key[0],
                "source_model": key[1],
                "model_role": key[2],
                "condition": key[3],
                "n": len(group_rows),
                "mean_score": _mean_score(group_rows),
            }
        )
    return output


def _exp4(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base_rows = _valid(row for row in rows if row.get("condition") == "default" and row.get("model_role") == "negative")
    output = []
    for key, group_rows in _group(base_rows, ("method", "source_model")).items():
        same_family = [row for row in group_rows if row.get("negative_type") == "same_family_hard_negative"]
        output.append(
            {
                "method": key[0],
                "source_model": key[1],
                "n": len(group_rows),
                "average_fpr": _mean_score(group_rows),
                "max_fpr": max((float(row.get("score") or 0.0) for row in group_rows), default=None),
                "same_family_hard_negative_fpr": _mean_score(same_family),
            }
        )
    return output


def _exp5(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group(records, ("method", "source_model"))
    output = []
    for key, rows in grouped.items():
        output.append(
            {
                "method": key[0],
                "source_model": key[1],
                "n": len(rows),
                "full_prompt_log_ppl": safe_mean((row.get("stealthiness") or {}).get("full_prompt_log_ppl") for row in rows),
                "adv_part_log_ppl": safe_mean((row.get("stealthiness") or {}).get("adv_part_log_ppl") for row in rows),
                "ppl_filter_pass_rate": safe_mean((row.get("stealthiness") or {}).get("ppl_filter_pass") for row in rows),
            }
        )
    return output


def _exp6(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group(records, ("method", "source_model"))
    output = []
    for key, rows in grouped.items():
        output.append(
            {
                "method": key[0],
                "source_model": key[1],
                "n": len(rows),
                "generation_time_sec": safe_mean((row.get("efficiency") or {}).get("generation_time_sec") for row in rows),
                "peak_gpu_memory_gb": safe_mean((row.get("efficiency") or {}).get("peak_gpu_memory_gb") for row in rows),
                "optimization_steps": safe_mean((row.get("efficiency") or {}).get("num_optimization_steps") for row in rows),
                "verification_query_count": safe_mean((row.get("efficiency") or {}).get("verification_queries_per_model") for row in rows),
            }
        )
    return output


def build_derived_tables(raw_dir: str | Path, out_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    records = _load_raw_records(raw_dir)
    rows = _flatten_verification(records)
    tables = {
        "exp1_main_verification.csv": _exp1(rows),
        "exp2_model_modification_robustness.csv": _exp2(rows),
        "exp3_deployment_robustness.csv": _exp3(rows),
        "exp4_false_positive_specificity.csv": _exp4(rows),
        "exp5_stealthiness.csv": _exp5(records),
        "exp6_efficiency.csv": _exp6(records),
    }
    out_path = Path(out_dir)
    for filename, table_rows in tables.items():
        save_csv(out_path / filename, table_rows)
    return tables
