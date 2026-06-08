from __future__ import annotations

import argparse
import time
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


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _progress_line(
    *,
    tried: int,
    new_successes: int,
    current_successes: int,
    target_successes: int,
    elapsed: float,
    fallback_success_rate: float,
) -> str:
    successes_needed = max(0, target_successes - current_successes)
    avg_seconds = elapsed / tried if tried else None
    observed_rate = new_successes / tried if tried and new_successes else fallback_success_rate
    eta = None
    estimated_candidates = None
    if avg_seconds is not None and observed_rate > 0:
        estimated_candidates = successes_needed / observed_rate
        eta = estimated_candidates * avg_seconds
    rate_text = f"{observed_rate:.3f}" if observed_rate else "unknown"
    candidates_text = f"{estimated_candidates:.1f}" if estimated_candidates is not None else "unknown"
    return (
        f"progress tried={tried} new_successes={new_successes} "
        f"total_successes={current_successes}/{target_successes} "
        f"success_rate_est={rate_text} avg_candidate_time={_format_duration(avg_seconds)} "
        f"estimated_candidates_left={candidates_text} eta={_format_duration(eta)} "
        f"elapsed={_format_duration(elapsed)}"
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
    parser.add_argument("--progress-every", type=int, default=1)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    out_dir = Path(args.out_dir)
    final_fingerprints_path = out_dir / "fingerprints" / "trap.jsonl"
    final_verification_path = out_dir / "runs" / "trap_verify.jsonl"
    rejected_fingerprints_path = out_dir / "fingerprints" / "trap_rejected.jsonl"
    rejected_verification_path = out_dir / "runs" / "trap_verify_rejected.jsonl"

    existing_verification = load_jsonl(Path(args.existing_verification))
    existing_successes = sum(_result_success(row) for row in existing_verification)
    fallback_success_rate = existing_successes / len(existing_verification) if existing_verification else 0.5

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
    print(f"fallback_success_rate_from_existing_run={fallback_success_rate:.3f}")
    if current_successes >= args.target_successes:
        print(f"done final_fingerprints={final_fingerprints_path}")
        print(f"done final_verification={final_verification_path}")
        return 0

    method_cfg = load_yaml(args.method_config)
    model_cfg = load_yaml(args.model_config)
    method = get_method("trap")

    from llmfp.core.model_backend import ModelBackend

    print("loading_model=true")
    load_start = time.monotonic()
    backend = ModelBackend.from_config(model_cfg)
    print(f"loading_model=false load_time={_format_duration(time.monotonic() - load_start)}")
    rejected_fingerprints = load_jsonl(rejected_fingerprints_path)
    used_ids = _existing_ids(final_fingerprints) | _existing_ids(rejected_fingerprints)
    used_targets = _existing_targets(final_fingerprints) | _existing_targets(rejected_fingerprints)

    tried = 0
    new_successes = 0
    seed = args.seed_start
    run_start = time.monotonic()
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

            candidate_start = time.monotonic()
            print(
                f"candidate_start seed={seed} index={index} tried={tried}/{args.max_new_candidates} "
                f"successes={current_successes}/{args.target_successes}"
            )
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
                new_successes += 1
                print(f"accepted {artifact.fingerprint_id} successes={current_successes}/{args.target_successes}")
            else:
                append_jsonl(rejected_fingerprints_path, artifact)
                append_jsonl(rejected_verification_path, result)
                print(f"rejected {artifact.fingerprint_id} successes={current_successes}/{args.target_successes}")

            candidate_elapsed = time.monotonic() - candidate_start
            elapsed = time.monotonic() - run_start
            print(f"candidate_time={_format_duration(candidate_elapsed)}")
            if args.progress_every > 0 and tried % args.progress_every == 0:
                print(
                    _progress_line(
                        tried=tried,
                        new_successes=new_successes,
                        current_successes=current_successes,
                        target_successes=args.target_successes,
                        elapsed=elapsed,
                        fallback_success_rate=fallback_success_rate,
                    )
                )

        seed += 1

    print(f"final_successes={current_successes}")
    print(f"new_candidates_tried={tried}")
    print(f"new_successes={new_successes}")
    print(f"elapsed={_format_duration(time.monotonic() - run_start)}")
    print(f"final_fingerprints={final_fingerprints_path}")
    print(f"final_verification={final_verification_path}")
    if current_successes < args.target_successes:
        print("warning: target not reached; increase --max-new-candidates or --seed-start and rerun.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
