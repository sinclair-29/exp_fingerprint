from __future__ import annotations

import argparse


METHOD_CHOICES = ["trap", "proflingo", "llmprint", "sraf", "plugae"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified adversarial-prompt LLM fingerprinting framework.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    construct = subparsers.add_parser("construct", help="Construct fingerprint artifacts.")
    construct.add_argument("--method", required=True, choices=METHOD_CHOICES)
    construct.add_argument("--method-config", required=True)
    construct.add_argument("--model-config", required=True)
    construct.add_argument("--out", required=True)

    verify = subparsers.add_parser("verify", help="Verify fingerprint artifacts against a suspect model.")
    verify.add_argument("--method", required=True, choices=METHOD_CHOICES)
    verify.add_argument("--method-config", required=True)
    verify.add_argument("--suspect-model-config", required=True)
    verify.add_argument("--fingerprints", required=True)
    verify.add_argument("--out", required=True)

    benchmark = subparsers.add_parser("benchmark", help="Run construct and verify for an experiment config.")
    benchmark.add_argument("--config", required=True)

    summarize = subparsers.add_parser("summarize", help="Collect JSONL verification runs into a CSV summary.")
    summarize.add_argument("--results-dir", required=True)
    summarize.add_argument("--out", required=True)

    experiment = subparsers.add_parser("experiment", help="Run the unified raw-record experiment pipeline.")
    experiment.add_argument("--config", required=True)

    build_tables = subparsers.add_parser("build-tables", help="Build derived experiment tables from raw JSONL records.")
    build_tables.add_argument("--raw-dir", default="results/raw")
    build_tables.add_argument("--out-dir", default="results/derived_tables")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "construct":
        from llmfp.core.io import load_yaml
        from llmfp.runners.construct import run_construct

        run_construct(args.method, load_yaml(args.method_config), load_yaml(args.model_config), args.out)
        return 0

    if args.command == "verify":
        from llmfp.core.io import load_yaml
        from llmfp.runners.verify import run_verify

        run_verify(args.method, load_yaml(args.method_config), load_yaml(args.suspect_model_config), args.fingerprints, args.out)
        return 0

    if args.command == "benchmark":
        from llmfp.core.io import load_yaml
        from llmfp.runners.benchmark import run_benchmark

        run_benchmark(load_yaml(args.config))
        return 0

    if args.command == "summarize":
        from llmfp.runners.summarize import run_summarize

        run_summarize(args.results_dir, args.out)
        return 0

    if args.command == "experiment":
        from llmfp.core.io import load_yaml
        from llmfp.runners.experiment import run_experiment

        run_experiment(load_yaml(args.config))
        return 0

    if args.command == "build-tables":
        from llmfp.runners.tables import build_derived_tables

        build_derived_tables(args.raw_dir, args.out_dir)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
