from llmfp.core.io import load_jsonl, load_yaml, save_csv, save_jsonl


def test_jsonl_roundtrip(tmp_path):
    path = tmp_path / "rows.jsonl"
    save_jsonl(path, [{"a": 1}, {"b": "two"}])
    assert load_jsonl(path) == [{"a": 1}, {"b": "two"}]


def test_yaml_load(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("a: 1\nb: two\n", encoding="utf-8")
    assert load_yaml(path) == {"a": 1, "b": "two"}


def test_csv_write(tmp_path):
    path = tmp_path / "table.csv"
    save_csv(path, [{"a": 1}])
    assert path.exists()
    assert "a" in path.read_text(encoding="utf-8")
