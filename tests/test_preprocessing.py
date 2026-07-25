import pytest
from puzzle_joiner.preprocessing import parse_page_range


class TestParsePageRange:
    def test_simple_range(self):
        assert parse_page_range("1-3") == [1, 2, 3]

    def test_comma_separated(self):
        assert parse_page_range("1,3,5") == [1, 3, 5]

    def test_mixed(self):
        assert parse_page_range("1-3,5,7-8") == [1, 2, 3, 5, 7, 8]

    def test_deduplication(self):
        assert parse_page_range("1-3,2-4") == [1, 2, 3, 4]

    def test_single_page(self):
        assert parse_page_range("5") == [5]

    def test_empty_string(self):
        assert parse_page_range("") is None

    def test_all(self):
        assert parse_page_range("all") is None
        assert parse_page_range("ALL") is None

    def test_whitespace(self):
        assert parse_page_range("  1 - 3 , 5 ") == [1, 2, 3, 5]

    def test_sorted_output(self):
        assert parse_page_range("5,1,3") == [1, 3, 5]
