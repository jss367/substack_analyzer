from substack_analyzer.ui import format_currency


def test_format_currency():
    assert format_currency(1234.56) == "$1,235"
