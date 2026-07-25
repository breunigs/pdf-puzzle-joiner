import os
import time

from puzzle_joiner.cache import (
    _cached_page_path, _cached_autocrop_path,
    cleanup_old_cache_entries, _CACHE_MAX_AGE_SECONDS,
)


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


class TestCacheCleanup:
    def test_removes_old_entries(self, tmp_path, monkeypatch):
        monkeypatch.setattr("puzzle_joiner.cache.CACHE_BASE", str(tmp_path))
        old_dir = tmp_path / "old_hash"
        old_dir.mkdir()
        (old_dir / "page_0001.png").write_text("x")
        old_time = time.time() - _CACHE_MAX_AGE_SECONDS - 3600
        os.utime(old_dir, (old_time, old_time))

        cleanup_old_cache_entries()
        assert not old_dir.exists()

    def test_keeps_recent_entries(self, tmp_path, monkeypatch):
        monkeypatch.setattr("puzzle_joiner.cache.CACHE_BASE", str(tmp_path))
        recent_dir = tmp_path / "recent_hash"
        recent_dir.mkdir()
        (recent_dir / "page_0001.png").write_text("x")

        cleanup_old_cache_entries()
        assert recent_dir.exists()

    def test_ignores_files_in_cache_base(self, tmp_path, monkeypatch):
        monkeypatch.setattr("puzzle_joiner.cache.CACHE_BASE", str(tmp_path))
        stray_file = tmp_path / "stray.txt"
        stray_file.write_text("x")
        old_time = time.time() - _CACHE_MAX_AGE_SECONDS - 3600
        os.utime(stray_file, (old_time, old_time))

        cleanup_old_cache_entries()
        assert stray_file.exists()

    def test_nonexistent_cache_base(self, monkeypatch):
        monkeypatch.setattr("puzzle_joiner.cache.CACHE_BASE", "/nonexistent/path")
        cleanup_old_cache_entries()  # should not raise
