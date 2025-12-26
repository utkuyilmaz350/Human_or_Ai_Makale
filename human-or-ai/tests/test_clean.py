from human_or_ai.data_clean.clean import normalize_text


def test_normalize_text_collapses_whitespace() -> None:
    assert normalize_text("a\n\n b\t  c") == "a b c"


def test_normalize_text_strips() -> None:
    assert normalize_text("  hello  ") == "hello"
