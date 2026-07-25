import os

from puzzle_joiner.cache import _cached_page_path, _cached_autocrop_path


class TestCachePaths:
    def test_page_path(self):
        path = _cached_page_path("/tmp/cache", 1)
        assert path == "/tmp/cache/page_0001.png"

    def test_page_path_large(self):
        path = _cached_page_path("/tmp/cache", 42)
        assert path == "/tmp/cache/page_0042.png"

    def test_autocrop_path(self):
        path = _cached_autocrop_path("/tmp/cache", 3)
        assert path == "/tmp/cache/page_0003_autocrop.json"
