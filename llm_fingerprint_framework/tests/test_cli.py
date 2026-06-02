from llmfp.cli import build_parser


def test_cli_parser_has_experiment_commands():
    parser = build_parser()
    args = parser.parse_args(["experiment", "--config", "config.yaml"])
    assert args.command == "experiment"
    args = parser.parse_args(["build-tables", "--raw-dir", "raw", "--out-dir", "tables"])
    assert args.command == "build-tables"


def test_cli_parser_accepts_plugae_method():
    parser = build_parser()
    args = parser.parse_args(
        [
            "construct",
            "--method",
            "plugae",
            "--method-config",
            "configs/methods/plugae.yaml",
            "--model-config",
            "configs/models/example_model.yaml",
            "--out",
            "results/fingerprints/plugae.jsonl",
        ]
    )
    assert args.method == "plugae"
