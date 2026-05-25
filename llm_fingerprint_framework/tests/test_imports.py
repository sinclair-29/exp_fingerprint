def test_imports():
    import llmfp
    from llmfp.registry import get_method
    from llmfp.core.templates import get_template

    assert llmfp.__version__
    assert get_method("trap").name == "trap"
    assert get_template("raw").name == "raw"
