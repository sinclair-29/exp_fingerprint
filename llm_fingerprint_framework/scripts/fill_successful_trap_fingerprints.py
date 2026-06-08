from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

from llmfp.core.io import append_jsonl, load_jsonl, load_yaml, save_jsonl
from llmfp.registry import get_method


def _result_success(row: dict[str, Any]) -> bool:
    return bool(row.get("success"))


def _load_existing_successes(fingerprint_path: Path, verification_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fingerprints = load_jsonl(fingerprint_path)
    verification = load_jsonl(verification_path)
    success_by_id = {row.get("fingerprint_id"): row for row in verification if _result_success(row)}
    successful_fingerprints = [row for row in fingerprints if row.get("fingerprint_id") in success_by_id]
    successful_results = [success_by_id[row.get("fingerprint_id")] for row in successful_fingerprints]
    return successful_fingerprints, successful_results


def _existing_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("fingerprint_id")) for row in rows if row.get("fingerprint_id") is not None}


def _existing_targets(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("target")) for row in rows if row.get("target") is not None}


def _with_unique_fingerprint_id(artifact, seed: int, original_index: int, used_ids: set[str]):
    base_id = f"trap-seed{seed}-candidate{original_index:04d}"
    fingerprint_id = base_id
    suffix = 1
    while fingerprint_id in used_ids:
        fingerprint_id = f"{base_id}-{suffix}"
        suffix += 1
    metadata = dict(getattr(artifact, "metadata", {}) or {})
    metadata["original_fingerprint_id"] = artifact.fingerprint_id
    metadata["fill_seed"] = seed
    metadata["fill_candidate_index"] = original_index
    return replace(
        artifact,
        fingerprint_id=fingerprint_id,
        task_id=f"seed{seed}-candidate{original_index:04d}",
        metadata=metadata,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fill a TRAP fingerprint pool until it contains N successful fingerprints.")
    parser.add_argument("--method-config", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--existing-fingerprints", required=True)
    parser.add_argument("--existing-verification", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target-successes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--candidates-per-seed", type=int, default=100)
    parser.add_argument("--max-new-candidates", type=int, default=100)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    out_dir = Path(args.out_dir)
    final_fingerprints_path = out_dir / "fingerprints" / "trap.jsonl"
    final_verification_path = out_dir / "runs" / "trap_verify.jsonl"
    rejected_fingerprints_path = out_dir / "fingerprints" / "trap_rejected.jsonl"
    rejected_verification_path = out_dir / "runs" / "trap_verify_rejected.jsonl"

    if final_fingerprints_path.exists() and final_verification_path.exists():
        final_fingerprints = load_jsonl(final_fingerprints_path)
        final_results = load_jsonl(final_verification_path)
    else:
        final_fingerprints, final_results = _load_existing_successes(
            Path(args.existing_fingerprints),
            Path(args.existing_verification),
        )
        final_fingerprints = final_fingerprints[: args.target_successes]
        final_results = final_results[: args.target_successes]
        save_jsonl(final_fingerprints_path, final_fingerprints)
        save_jsonl(final_verification_path, final_results)

    current_successes = sum(_result_success(row) for row in final_results)
    print(f"starting_successes={current_successes}")
    if current_successes >= args.target_successes:
        print(f"done final_fingerprints={final_fingerprints_path}")
        print(f"done final_verification={final_verification_path}")
        return 0

    method_cfg = load_yaml(args.method_config)
    model_cfg = load_yaml(args.model_config)
    method = get_method("trap")

    from llmfp.core.model_backend import ModelBackend

    backend = ModelBackend.from_config(model_cfg)
    rejected_fingerprints = load_jsonl(rejected_fingerprints_path)
    used_ids = _existing_ids(final_fingerprints) | _existing_ids(rejected_fingerprints)
    used_targets = _existing_targets(final_fingerprints) | _existing_targets(rejected_fingerprints)

    tried = 0
    seed = args.seed_start
    while current_successes < args.target_successes and tried < args.max_new_candidates:
        candidate_cfg = dict(method_cfg)
        candidate_cfg["seed"] = seed
        candidate_cfg["num_fingerprints"] = args.candidates_per_seed
        tasks = method.build_tasks(candidate_cfg)

        for index, task in enumerate(tasks):
            if current_successes >= args.target_successes or tried >= args.max_new_candidates:
                break
            if task.target in used_targets:
                continue
            tried += 1

            artifact = method.construct(task, backend, candidate_cfg)
            if artifact is None:
                continue
            artifact = _with_unique_fingerprint_id(artifact, seed, index, used_ids)
            used_ids.add(artifact.fingerprint_id)
            result = method.verify(artifact, backend, method_cfg)

            if result.success:
                append_jsonl(final_fingerprints_path, artifact)
                append_jsonl(final_verification_path, result)
                used_targets.add(artifact.target)
                current_successes += 1
                print(f"accepted {artifact.fingerprint_id} successes={current_successes}/{args.target_successes}")
            else:
                append_jsonl(rejected_fingerprints_path, artifact)
                append_jsonl(rejected_verification_path, result)
                print(f"rejected {artifact.fingerprint_id} successes={current_successes}/{args.target_successes}")

        seed += 1

    print(f"final_successes={current_successes}")
    print(f"new_candidates_tried={tried}")
    print(f"final_fingerprints={final_fingerprints_path}")
    print(f"final_verification={final_verification_path}")
    if current_successes < args.target_successes:
        print("warning: target not reached; increase --max-new-candidates or --seed-start and rerun.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
