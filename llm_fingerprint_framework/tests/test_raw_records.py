from llmfp.core.raw_records import default_raw_record, validate_raw_record, verification_entry
from llmfp.schemas import FingerprintArtifact, VerificationResult


def test_default_raw_record_has_required_keys():
    artifact = FingerprintArtifact(
        fingerprint_id="trap-1",
        method="trap",
        base_model="base",
        task_id="1",
        prompt_text="prompt",
        optimized_text="adv",
        target="target",
        best_loss=1.2,
        best_step=3,
        metadata={"instruction": "base prompt", "loss_history": [2.0, 1.2]},
    )
    record = default_raw_record("run", "trap", "base", 0, 0, artifact)
    validate_raw_record(record)
    assert record["fingerprint_spec"]["base_prompt"] == "base prompt"
    assert record["generation"]["loss_curve"] == [2.0, 1.2]


def test_verification_entry_shape():
    result = VerificationResult(
        method="trap",
        suspect_model="suspect",
        fingerprint_id="trap-1",
        success=True,
        score=1.0,
        raw_output="target",
        metadata={"parsed": "target"},
    )
    row = verification_entry(result, "suspect", "positive", modification_type="sft")
    assert row["model_role"] == "positive"
    assert row["modification_type"] == "sft"
    assert row["score"] == 1.0
    assert row["method_specific"]["parsed"] == "target"
