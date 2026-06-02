from llmfp.core.io import save_jsonl
from llmfp.runners.tables import build_derived_tables


def _record(method, fingerprint_id, pos_score, neg_score):
    return {
        "run_id": "run",
        "method": method,
        "source_model": "source",
        "seed": 0,
        "fingerprint_id": fingerprint_id,
        "fingerprint_spec": {
            "base_prompt": "q",
            "target": "t",
            "full_query": "q",
            "adversarial_text": "a",
            "method_specific": {},
        },
        "generation": {
            "success_on_source": True,
            "final_loss": 0.1,
            "best_step": 1,
            "num_steps": 2,
            "loss_curve": [1, 0.1],
            "loss_curve_path": None,
            "method_specific": {},
        },
        "verification": [
            {
                "model": "source",
                "model_role": "source",
                "modification_type": "none",
                "negative_type": "none",
                "condition": "default",
                "system_prompt": None,
                "sampling": {},
                "output": "t",
                "score": 1.0,
                "valid_for_method": True,
                "method_specific": {},
            },
            {
                "model": "positive",
                "model_role": "positive",
                "modification_type": "sft",
                "negative_type": "none",
                "condition": "default",
                "system_prompt": None,
                "sampling": {},
                "output": "t",
                "score": pos_score,
                "valid_for_method": True,
                "method_specific": {},
            },
            {
                "model": "negative",
                "model_role": "negative",
                "modification_type": "none",
                "negative_type": "same_family_hard_negative",
                "condition": "default",
                "system_prompt": None,
                "sampling": {},
                "output": "",
                "score": neg_score,
                "valid_for_method": True,
                "method_specific": {},
            },
        ],
        "stealthiness": {
            "ppl_model": None,
            "full_prompt_log_ppl": None,
            "adv_part_log_ppl": None,
            "ppl_filter_pass": None,
        },
        "efficiency": {
            "generation_time_sec": 1.0,
            "peak_gpu_memory_gb": None,
            "num_optimization_steps": 2,
            "num_forward": None,
            "num_backward": None,
            "verification_queries_per_model": 1,
        },
    }


def test_build_derived_tables_from_raw_jsonl(tmp_path):
    raw_dir = tmp_path / "raw" / "trap"
    save_jsonl(raw_dir / "run.jsonl", [_record("trap", 0, 1.0, 0.0), _record("trap", 1, 1.0, 1.0)])
    tables = build_derived_tables(tmp_path / "raw", tmp_path / "tables")
    assert (tmp_path / "tables" / "exp1_main_verification.csv").exists()
    assert tables["exp1_main_verification.csv"][0]["positive_mean_score"] == 1.0
    assert tables["exp4_false_positive_specificity.csv"][0]["average_fpr"] == 0.5
