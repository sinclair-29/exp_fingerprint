from llmfp.core.matching import (
    contains_normalized_target,
    match_keyword_target,
    match_trap_output,
    normalize_text,
    parse_digit_sequence,
)


def test_normalize_text():
    assert normalize_text("  North-East!! ") == "north east"


def test_contains_normalized_target():
    assert contains_normalized_target("The answer is NORTH.", "north")
    assert not contains_normalized_target("northern", "north")


def test_digit_parser():
    assert parse_digit_sequence("answer: 042", 3) == "042"
    assert parse_digit_sequence("12 3456", 3) is None


def test_trap_match():
    success, invalid, parsed = match_trap_output("042", "042")
    assert success is True
    assert invalid is False
    assert parsed == "042"


def test_keyword_match():
    assert match_keyword_target("It should be green.", "blue", ["green"])
