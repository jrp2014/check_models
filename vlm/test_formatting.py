import pytest

from vlm.check_models import format_field_value, _escape_markdown_in_text


def test_memory_format_bytes_to_gb_rounding():
    # 8,589,934,592 bytes (~8 GiB) should show as ~8.6 GB (decimal GB normalization)
    val_bytes = 8_589_934_592
    s = format_field_value("peak_memory", val_bytes)
    assert s == "8.6"


def test_memory_format_already_gb():
    s = format_field_value("peak_memory", 6.3)
    assert s == "6.3"


def test_memory_format_small_fractional_gb():
    s = format_field_value("peak_memory", 0.25)
    assert s == "0.25"


def test_escape_preserves_br_and_escapes_tag_like_tokens():
    text = "<s><br>ok"
    out = _escape_markdown_in_text(text)
    assert out == "&lt;s&gt;<br>ok"


def test_escape_pipes_and_backslashes():
    text = r"a|b\\c <unk>"
    out = _escape_markdown_in_text(text)
    assert out.startswith("a\\|b\\\\c ")
    assert "&lt;unk&gt;" in out
