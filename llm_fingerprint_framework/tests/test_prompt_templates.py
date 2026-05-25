from llmfp.core.templates import get_template


def test_raw_template_rendering():
    assert get_template("raw").render("hello", " world") == "hello world"


def test_alpaca_template_contains_user_and_assistant_prefix():
    rendered = get_template("alpaca").render("Say hi", "Hi")
    assert "Say hi" in rendered
    assert "Hi" in rendered
    assert "### Response:" in rendered


def test_split_around_mutable():
    before, after = get_template("fastchat_zero_shot").split_around_mutable("a ", " b")
    assert "### Human:" in before
    assert "a " in before
    assert " b" in after
    assert "### Assistant:" in after
