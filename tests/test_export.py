import math

import cv2
import numpy as np
import pytest
import tifffile

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from puzzle_joiner.model import PuzzlePiece
from puzzle_joiner import export as exp
from puzzle_joiner.export import (
    _compute_optimal_rotation, _export_affines, _layer_name, _union_bbox,
    export_flattened_tiff, export_layered_tiff,
)


def _make_piece(w=100, h=80):
    piece = PuzzlePiece()
    piece.image = np.zeros((h, w, 4), dtype=np.uint8)
    piece.image[:, :, 3] = 255
    return piece


def _make_noise_piece(w, h, seed):
    piece = PuzzlePiece()
    rng = np.random.default_rng(seed)
    piece.image = rng.integers(0, 256, (h, w, 4), dtype=np.uint8)
    piece.image[:, :, 3] = 255
    return piece


def _assert_matches(got, expected):
    """Compare against a full-canvas render.

    Warping into a band folds a different translation into the affine matrix, so
    OpenCV's fixed-point subpixel rounding can differ by one step on interpolated
    edge pixels. Everything else must be identical.
    """
    assert got.shape == expected.shape
    diff = np.abs(got.astype(np.int16) - expected.astype(np.int16))
    assert diff.max() <= 2, f"max deviation {diff.max()}"
    assert (diff > 0).mean() < 0.001, f"{(diff > 0).mean():.3%} of samples deviate"
    # Areas no piece covers must come out exactly transparent.
    empty = expected[:, :, 3] == 0
    assert not diff[empty].any()


def _reference_warp(piece, M, origin_x, origin_y, canvas_w, canvas_h):
    """Warp a piece onto the whole canvas in one go."""
    M_local = M.copy()
    M_local[0, 2] -= origin_x
    M_local[1, 2] -= origin_y
    return cv2.warpAffine(
        piece.get_cropped_image(), M_local, (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def _reference_layer(piece, M, origin_x, origin_y, canvas_w, canvas_h):
    """Naive single full-canvas layer — the implementation the streaming one replaced."""
    warped = _reference_warp(piece, M, origin_x, origin_y, canvas_w, canvas_h)
    return cv2.cvtColor(warped, cv2.COLOR_BGRA2RGBA)


def _reference_canvas(pieces, affines, origin_x, origin_y, canvas_w, canvas_h):
    """Naive full-canvas composite — the implementation the streaming one replaced."""
    canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
    for piece, M in zip(pieces, affines):
        warped = _reference_warp(piece, M, origin_x, origin_y, canvas_w, canvas_h)
        alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
        canvas[:] = np.clip(
            warped.astype(np.float32) * alpha + canvas.astype(np.float32) * (1.0 - alpha),
            0, 255,
        ).astype(np.uint8)
    return cv2.cvtColor(canvas, cv2.COLOR_BGRA2RGBA)


class TestOptimalRotation:
    def test_axis_aligned_no_rotation(self):
        """Two pieces side by side — already optimal, no rotation needed."""
        p1 = _make_piece(100, 80)
        p1.x, p1.y = 50, 40
        p2 = _make_piece(100, 80)
        p2.x, p2.y = 160, 40
        angle, _, _ = _compute_optimal_rotation([p1, p2])
        assert abs(angle) < 0.1

    def test_rotated_pieces_get_straightened(self):
        """Two pieces placed diagonally — rotation should reduce bounding box."""
        p1 = _make_piece(200, 50)
        p1.x, p1.y = 100, 25
        p1.rotation_deg = 30
        p2 = _make_piece(200, 50)
        cos30 = math.cos(math.radians(30))
        sin30 = math.sin(math.radians(30))
        p2.x = p1.x + 210 * cos30
        p2.y = p1.y + 210 * sin30
        p2.rotation_deg = 30

        angle, _, _ = _compute_optimal_rotation([p1, p2])
        # Should find ~-30 degree rotation to straighten them
        assert abs(angle - (-30)) < 2

    def test_rotation_reduces_area(self):
        """Verify the rotation actually produces a smaller bounding box."""
        p1 = _make_piece(200, 50)
        p1.x, p1.y = 100, 25
        p1.rotation_deg = 20
        p2 = _make_piece(200, 50)
        cos20 = math.cos(math.radians(20))
        sin20 = math.sin(math.radians(20))
        p2.x = p1.x + 210 * cos20
        p2.y = p1.y + 210 * sin20
        p2.rotation_deg = 20

        # Without rotation
        base_affines = [p.get_affine_matrix() for p in [p1, p2]]
        _, _, w_before, h_before = _union_bbox([p1, p2], base_affines)
        area_before = w_before * h_before

        # With optimal rotation
        opt_affines = _export_affines([p1, p2])
        _, _, w_after, h_after = _union_bbox([p1, p2], opt_affines)
        area_after = w_after * h_after

        assert area_after < area_before

    def test_single_piece_no_rotation(self):
        """Single piece — no rotation applied."""
        p = _make_piece(100, 80)
        p.x, p.y = 50, 40
        affines = _export_affines([p])
        np.testing.assert_array_almost_equal(affines[0], p.get_affine_matrix())

    def test_angle_bounded(self):
        """Rotation angle should be in [-45, 45] range."""
        p1 = _make_piece(100, 100)
        p1.x, p1.y = 50, 50
        p1.rotation_deg = 70
        p2 = _make_piece(100, 100)
        p2.x, p2.y = 200, 50
        p2.rotation_deg = 70
        angle, _, _ = _compute_optimal_rotation([p1, p2])
        assert -45 <= angle <= 45


class TestStreamedOutput:
    """The band-streamed writers must match a full-canvas render pixel for pixel."""

    def _pieces(self):
        p1 = _make_noise_piece(700, 400, seed=1)
        p1.x, p1.y = 360, 220
        p1.rotation_deg = 12.5
        p2 = _make_noise_piece(600, 500, seed=2)
        p2.x, p2.y = 980, 470
        p2.rotation_deg = -3.25
        p2.crop_rect = (40, 30, 500, 400)
        return [p1, p2]

    @pytest.fixture(autouse=True)
    def small_bands(self, monkeypatch):
        """Force many tiles, several bands, and several composite chunks per band."""
        monkeypatch.setattr(exp, "TILE", 64)
        monkeypatch.setattr(exp, "_BAND_BYTES", 32 * 1024)
        monkeypatch.setattr(exp, "_CHUNK_BYTES", 8 * 1024)

    def test_flattened_matches_reference(self, tmp_path):
        pieces = self._pieces()
        affines = _export_affines(pieces)
        ox, oy, w, h = _union_bbox(pieces, affines)
        assert exp._band_height(w) < h, "test should exercise more than one band"

        out = tmp_path / "flat.tif"
        export_flattened_tiff(pieces, str(out))
        got = tifffile.imread(str(out))
        _assert_matches(got, _reference_canvas(pieces, affines, ox, oy, w, h))

    def test_layered_matches_reference(self, tmp_path):
        pieces = self._pieces()
        affines = _export_affines(pieces)
        ox, oy, w, h = _union_bbox(pieces, affines)

        out = tmp_path / "layered.tif"
        export_layered_tiff(pieces, str(out))
        with tifffile.TiffFile(str(out)) as tif:
            assert len(tif.pages) == len(pieces)
            for page, piece, M in zip(tif.pages, pieces, affines):
                _assert_matches(page.asarray(),
                                _reference_layer(piece, M, ox, oy, w, h))

    def test_empty_piece_list_writes_nothing(self, tmp_path):
        out = tmp_path / "none.tif"
        export_flattened_tiff([], str(out))
        export_layered_tiff([], str(out))
        assert not out.exists()


class TestMemoryCheck:
    def test_available_memory_is_plausible(self):
        avail = exp._available_memory()
        assert avail > 64 * 1024**2

    def test_band_height_is_whole_tile_rows(self):
        for width in (1, 1000, 172224, 10**6):
            assert exp._band_height(width) % exp.TILE == 0
            assert exp._band_height(width) >= exp.TILE

    def test_check_memory_rejects_impossible_strip(self, monkeypatch):
        monkeypatch.setattr(exp, "_available_memory", lambda: 1 << 30)
        exp._check_memory(1000, exp.TILE)
        with pytest.raises(MemoryError, match="px wide strip"):
            exp._check_memory(10**7, exp.TILE)


class TestLayerName:
    def test_ascii_name_unchanged(self):
        p = _make_piece()
        p.display_name = "page_01"
        assert _layer_name(p) == "page_01"

    def test_non_ascii_replaced(self):
        p = _make_piece()
        p.display_name = "Übersicht Süd"
        result = _layer_name(p)
        assert result.isascii()
        assert result == "?bersicht S?d"

    def test_falls_back_to_source_path(self):
        p = _make_piece()
        p.display_name = ""
        p.source_path = "/tmp/Ärger.png"
        result = _layer_name(p)
        assert result.isascii()
        assert result == "?rger.png"
