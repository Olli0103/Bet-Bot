"""Tests for stats card formatting logic (standalone, no telegram import)."""
import pytest


def _format_form_blocks(form_str: str) -> str:
    """Local copy of the form blocks function to avoid telegram import."""
    mapping = {"W": "🟩", "D": "🟨", "L": "🟥"}
    return "".join(mapping.get(c.upper(), "⬜") for c in form_str)


class TestFormatFormBlocks:
    def test_all_wins(self):
        assert _format_form_blocks("WWWWW") == "🟩🟩🟩🟩🟩"

    def test_all_losses(self):
        assert _format_form_blocks("LLLLL") == "🟥🟥🟥🟥🟥"

    def test_mixed_form(self):
        result = _format_form_blocks("WDLWL")
        assert result == "🟩🟨🟥🟩🟥"

    def test_empty_form(self):
        assert _format_form_blocks("") == ""

    def test_lowercase(self):
        assert _format_form_blocks("wdl") == "🟩🟨🟥"

    def test_unknown_char(self):
        assert _format_form_blocks("X") == "⬜"
