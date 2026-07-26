import json
import os
import time

from puzzle_joiner.cache import (
    _cached_page_path, _cached_autocrop_path,
    cleanup_old_cache_entries, _CACHE_MAX_AGE_SECONDS,
    save_layout, load_layout, _images_cache_dir,
)
from puzzle_joiner.model import PuzzlePiece


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


def _make_piece(source_path, x=0.0, y=0.0, rotation=0.0, scale=1.0,
                matched=False, locked=False, crop_rect=None):
    import numpy as np
    p = PuzzlePiece()
    p.source_path = source_path
    p.image = np.zeros((10, 10, 4), dtype=np.uint8)
    p.x = x
    p.y = y
    p.rotation_deg = rotation
    p.scale = scale
    p.is_matched = matched
    p.is_locked = locked
    p.crop_rect = crop_rect
    return p


class TestSaveLoadLayout:
    def test_round_trip(self, tmp_path):
        pieces = [
            _make_piece("/cache/page_0001.png", x=100, y=200, rotation=45,
                        scale=1.5, matched=True, locked=True, crop_rect=(10, 20, 30, 40)),
            _make_piece("/cache/page_0002.png", x=300, y=400),
        ]
        save_layout(str(tmp_path), pieces)

        # Create fresh pieces with default state
        restored = [
            _make_piece("/cache/page_0001.png"),
            _make_piece("/cache/page_0002.png"),
        ]
        applied, snap_pairs = load_layout(str(tmp_path), restored)
        assert applied

        assert restored[0].x == 100
        assert restored[0].y == 200
        assert restored[0].rotation_deg == 45
        assert restored[0].scale == 1.5
        assert restored[0].is_matched is True
        assert restored[0].is_locked is True
        assert restored[0].crop_rect == (10, 20, 30, 40)
        assert restored[1].x == 300
        assert restored[1].y == 400

    def test_no_layout_file(self, tmp_path):
        pieces = [_make_piece("/cache/page_0001.png")]
        applied, snap_pairs = load_layout(str(tmp_path), pieces)
        assert applied is False
        assert snap_pairs == []

    def test_no_cache_dir(self):
        pieces = [_make_piece("/cache/page_0001.png")]
        applied, snap_pairs = load_layout(None, pieces)
        assert applied is False
        assert snap_pairs == []
        save_layout(None, pieces)  # should not raise

    def test_partial_match(self, tmp_path):
        """Only pieces with matching source basenames get restored."""
        pieces = [_make_piece("/cache/page_0001.png", x=100, y=200)]
        save_layout(str(tmp_path), pieces)

        restored = [
            _make_piece("/other/page_0001.png"),  # same basename
            _make_piece("/cache/page_0099.png"),   # no match
        ]
        applied, _ = load_layout(str(tmp_path), restored)
        assert applied
        assert restored[0].x == 100
        assert restored[1].x == 0  # unchanged

    def test_json_format(self, tmp_path):
        pieces = [_make_piece("/cache/page_0001.png", x=50, locked=True)]
        save_layout(str(tmp_path), pieces)
        with open(os.path.join(str(tmp_path), "layout.json")) as f:
            data = json.load(f)
        assert "pieces" in data
        assert len(data["pieces"]) == 1
        assert data["pieces"][0]["source"] == "page_0001.png"
        assert data["pieces"][0]["x"] == 50
        assert data["pieces"][0]["is_locked"] is True

    def test_null_crop_rect(self, tmp_path):
        pieces = [_make_piece("/cache/page_0001.png", crop_rect=None)]
        save_layout(str(tmp_path), pieces)
        restored = [_make_piece("/cache/page_0001.png", crop_rect=(1, 2, 3, 4))]
        load_layout(str(tmp_path), restored)
        # null in JSON means "no crop" — don't overwrite with None
        assert restored[0].crop_rect == (1, 2, 3, 4)

    def test_snap_pairs_round_trip(self, tmp_path):
        p1 = _make_piece("/cache/page_0001.png", x=100)
        p2 = _make_piece("/cache/page_0002.png", x=200)
        save_layout(str(tmp_path), [p1, p2],
                    snap_pairs=[(p1, p2)])

        restored = [
            _make_piece("/cache/page_0001.png"),
            _make_piece("/cache/page_0002.png"),
        ]
        applied, snap_pairs = load_layout(str(tmp_path), restored)
        assert applied
        assert snap_pairs == [["page_0001.png", "page_0002.png"]]

    def test_old_format_compat(self, tmp_path):
        """Old flat-list format still loads correctly."""
        old_data = [{"source": "page_0001.png", "x": 42, "y": 0,
                     "rotation_deg": 0, "scale": 1}]
        with open(os.path.join(str(tmp_path), "layout.json"), "w") as f:
            json.dump(old_data, f)
        restored = [_make_piece("/cache/page_0001.png")]
        applied, snap_pairs = load_layout(str(tmp_path), restored)
        assert applied
        assert snap_pairs == []
        assert restored[0].x == 42


class TestImagesCacheDir:
    def test_deterministic(self, tmp_path, monkeypatch):
        monkeypatch.setattr("puzzle_joiner.cache.CACHE_BASE", str(tmp_path))
        d1 = _images_cache_dir(["/a.png", "/b.png"])
        d2 = _images_cache_dir(["/b.png", "/a.png"])  # order shouldn't matter
        assert d1 == d2
        assert os.path.isdir(d1)
        assert "images_" in os.path.basename(d1)

    def test_different_sets(self, tmp_path, monkeypatch):
        monkeypatch.setattr("puzzle_joiner.cache.CACHE_BASE", str(tmp_path))
        d1 = _images_cache_dir(["/a.png"])
        d2 = _images_cache_dir(["/a.png", "/b.png"])
        assert d1 != d2
