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
    export_canvas, export_flattened_tiff, export_layered_tiff,
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

    A fully transparent pixel's colour is not meaningful and is not compared: the
    streamed writer leaves the canvas outside a piece's render box zeroed, while
    a full-canvas warp of a lassoed piece carries the masked-out colour there.
    Alpha itself is compared everywhere.
    """
    assert got.shape == expected.shape
    got, expected = got.copy(), expected.copy()
    got[got[:, :, 3] == 0, :3] = 0
    expected[expected[:, :, 3] == 0, :3] = 0
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

    @pytest.fixture(autouse=True, params=[32 * 1024, 2 * 1024 * 1024],
                    ids=["one-tile-row-band", "multi-tile-row-band"])
    def small_bands(self, request, monkeypatch):
        """Force many tiles, several bands, and several composite chunks per band.

        Parametrised over two band budgets: the smaller one makes a band exactly
        one tile row tall, the larger several. Both must still hand the writer
        one tile per yield.
        """
        monkeypatch.setattr(exp, "TILE", 64)
        monkeypatch.setattr(exp, "_BAND_BYTES", request.param)
        monkeypatch.setattr(exp, "_CHUNK_BYTES", 8 * 1024)

    def test_band_taller_than_a_tile_is_split_into_tile_rows(self):
        """A band spans whole tile rows; the writer still wants one tile per yield."""
        band = np.zeros((exp.TILE * 3, 150, 4), dtype=np.uint8)
        canvas_w = 300
        tiles = list(exp._band_tiles(band, 0, canvas_w, band.shape[0]))
        assert len(tiles) == 3 * math.ceil(canvas_w / exp.TILE)
        assert all(t.shape == (exp.TILE, exp.TILE, 4) for t in tiles)
        assert len(list(exp._band_tiles(None, 0, canvas_w, exp.TILE * 3))) == len(tiles)

    def test_flattened_matches_reference(self, tmp_path):
        pieces = self._pieces()
        affines = _export_affines(pieces)
        ox, oy, w, h = export_canvas(pieces, affines)
        assert exp._band_height(w) < h, "test should exercise more than one band"

        out = tmp_path / "flat.tif"
        export_flattened_tiff(pieces, str(out))
        got = tifffile.imread(str(out))
        _assert_matches(got, _reference_canvas(pieces, affines, ox, oy, w, h))

    def test_layered_matches_reference(self, tmp_path):
        pieces = self._pieces()
        affines = _export_affines(pieces)
        ox, oy, w, h = export_canvas(pieces, affines)

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


class TestLassoExport:
    def test_render_box_shrinks_to_the_lasso(self):
        """A lasso must shrink the area rendered for a piece to its hull."""
        p = _make_piece(200, 200)
        p.x, p.y = 100, 100
        rect_bbox = _union_bbox([p], [p.get_affine_matrix()])
        # Triangle covering the left half, inside the same crop rect.
        p.apply_crop((0, 0, 200, 200), [(0, 0), (100, 0), (0, 200)])
        lasso_bbox = _union_bbox([p], [p.get_affine_matrix()])
        # Width halves; height is unchanged bar the fringe pad on each side,
        # which is 3 px at scale 1.
        assert lasso_bbox[2] == rect_bbox[2] // 2 + 6
        assert lasso_bbox[3] == rect_bbox[3] + 6

    def test_the_canvas_itself_keeps_the_crop_rect(self):
        """The canvas does not shrink — it must stay shared with the uncropped file."""
        p = _make_piece(200, 200)
        p.x, p.y = 100, 100
        p.apply_crop((0, 0, 200, 200), [(0, 0), (100, 0), (0, 200)])
        # The rect plus the fringe pad, so no antialiased edge is clipped.
        assert export_canvas([p], [p.get_affine_matrix()])[2:] == (206, 206)

    def test_outside_the_lasso_exports_transparent(self, tmp_path):
        p = _make_noise_piece(200, 200, seed=7)
        p.x, p.y = 100, 100
        p.apply_crop((0, 0, 200, 200), [(0, 0), (190, 0), (0, 190)])

        out = tmp_path / "lasso.tif"
        export_flattened_tiff([p], str(out))
        got = tifffile.imread(str(out))
        ox, oy, _, _ = export_canvas([p], _export_affines([p]))
        # Bottom-right corner is outside the triangle; top-left is inside.
        assert got[int(185 - oy), int(185 - ox), 3] == 0
        assert got[int(5 - oy), int(5 - ox), 3] == 255

    def test_layered_export_matches_reference(self, tmp_path):
        p = _make_noise_piece(300, 240, seed=8)
        p.x, p.y = 150, 120
        p.rotation_deg = 15
        p.apply_crop((20, 20, 260, 200), [(20, 20), (280, 40), (200, 220), (40, 180)])

        affines = _export_affines([p])
        ox, oy, w, h = export_canvas([p], affines)
        out = tmp_path / "layered_lasso.tif"
        export_layered_tiff([p], str(out))
        with tifffile.TiffFile(str(out)) as tif:
            assert len(tif.pages) == 1
            _assert_matches(tif.pages[0].asarray(),
                            _reference_layer(p, affines[0], ox, oy, w, h))


class TestUncroppedExport:
    """`uncropped=True` keeps the lasso's bounding rect, pixels and all."""

    def _lassoed(self):
        p = _make_noise_piece(200, 200, seed=11)
        p.x, p.y = 100, 100
        p.apply_crop((0, 0, 200, 200), [(0, 0), (190, 0), (0, 190)])
        return p

    def _pair(self, pieces, tmp_path):
        """Write the lassoed and uncropped flat files, return both as arrays."""
        cropped, uncropped = tmp_path / "flat.tif", tmp_path / "unc.tif"
        export_flattened_tiff(pieces, str(cropped))
        export_flattened_tiff(pieces, str(uncropped), uncropped=True)
        return tifffile.imread(str(cropped)), tifffile.imread(str(uncropped))

    def test_both_files_have_the_same_dimensions(self, tmp_path):
        """The georeference worked out on one file has to apply to the other."""
        got, unc = self._pair([self._lassoed()], tmp_path)
        assert got.shape == unc.shape

    def test_the_lassoed_file_is_a_subset_of_the_uncropped_one(self, tmp_path):
        """Same canvas, same placement: the lasso only takes pixels away."""
        p = _make_noise_piece(300, 240, seed=14)
        p.x, p.y = 150, 120
        p.rotation_deg = 15
        p.apply_crop((20, 20, 260, 200), [(20, 20), (280, 40), (200, 220), (40, 180)])
        got, unc = self._pair([p], tmp_path)

        assert got.shape == unc.shape
        # Nothing shows in the lassoed file that the uncropped file does not have.
        assert not (got[:, :, 3][unc[:, :, 3] == 0]).any()
        # Where the lasso kept a pixel outright, both files agree on its colour.
        # The two renders feed warpAffine different box offsets, so interpolated
        # pixels can land one step apart in OpenCV's fixed-point arithmetic.
        kept = (got[:, :, 3] == 255) & (unc[:, :, 3] == 255)
        assert kept.sum() > 10_000, "test should keep a decent area"
        delta = np.abs(got[:, :, :3][kept].astype(np.int16)
                       - unc[:, :, :3][kept].astype(np.int16))
        assert delta.max() <= 1
        assert (delta > 0).mean() < 0.001, f"{(delta > 0).mean():.3%} deviate"

    def test_layers_share_the_canvas_too(self, tmp_path):
        p = self._lassoed()
        layered = tmp_path / "layered.tif"
        export_layered_tiff([p], str(layered))
        _, unc = self._pair([p], tmp_path)
        with tifffile.TiffFile(str(layered)) as tif:
            assert tif.pages[0].asarray().shape == unc.shape

    def test_pixels_outside_the_lasso_come_back(self, tmp_path):
        p = self._lassoed()
        out = tmp_path / "uncropped.tif"
        export_flattened_tiff([p], str(out), uncropped=True)
        got = tifffile.imread(str(out))
        ox, oy, _, _ = export_canvas([p], _export_affines([p]))
        # Bottom-right is outside the triangle, and now opaque.
        assert got[int(185 - oy), int(185 - ox), 3] == 255
        assert got[int(5 - oy), int(5 - ox), 3] == 255

    def test_matches_a_reference_render_of_the_unmasked_crop(self, tmp_path):
        p = _make_noise_piece(300, 240, seed=12)
        p.x, p.y = 150, 120
        p.rotation_deg = 15
        p.apply_crop((20, 20, 260, 200), [(20, 20), (280, 40), (200, 220), (40, 180)])

        affines = _export_affines([p])
        ox, oy, w, h = export_canvas([p], affines)
        out = tmp_path / "uncropped_rot.tif"
        export_flattened_tiff([p], str(out), uncropped=True)

        unmasked = _make_piece(300, 240)
        unmasked.image = p.image.copy()
        unmasked.image[:, :, 3] = p._base_alpha
        unmasked.crop_rect = p.crop_rect
        expected = _reference_canvas([unmasked], affines, ox, oy, w, h)
        _assert_matches(tifffile.imread(str(out)), expected)

    def test_the_mask_is_put_back_afterwards(self, tmp_path):
        p = self._lassoed()
        before = p.image[:, :, 3].copy()
        export_flattened_tiff([p], str(tmp_path / "u.tif"), uncropped=True)
        np.testing.assert_array_equal(p.image[:, :, 3], before)

    def test_layout_is_unchanged_by_the_flag(self):
        """Both files must place the pieces identically, lasso or no lasso."""
        pieces = [self._lassoed(), _make_piece(200, 200)]
        pieces[1].x, pieces[1].y = 400, 150
        pieces[1].rotation_deg = 20
        for a, b in zip(_export_affines(pieces), _export_affines(pieces)):
            np.testing.assert_array_almost_equal(a, b)

    def test_without_a_lasso_it_is_the_plain_export(self, tmp_path):
        p = _make_noise_piece(120, 90, seed=13)
        p.x, p.y = 60, 45
        plain, unc = tmp_path / "a.tif", tmp_path / "b.tif"
        export_flattened_tiff([p], str(plain))
        export_flattened_tiff([p], str(unc), uncropped=True)
        np.testing.assert_array_equal(tifffile.imread(str(plain)),
                                      tifffile.imread(str(unc)))


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


class TestChunkedWarp:
    """`_warp_region` fills the destination in columns from cropped sources.

    Both halves of that matter on a large plan: OpenCV's Lanczos path refuses a
    source at or past SHRT_MAX on either axis, and handing it the whole page
    for every band is wasteful even when it is allowed.
    """

    def _piece(self, rotation=0.0, scale=1.0, w=1700, h=900):
        rng = np.random.default_rng(7)
        piece = PuzzlePiece()
        piece.source_path = "/tmp/chunk.png"
        piece.image = rng.integers(0, 255, (h, w, 4), dtype=np.uint8)
        piece.image[:, :, 3] = 255
        piece.rotation_deg = rotation
        piece.scale = scale
        return piece

    def _direct(self, piece, M, origin_x, origin_y, x0, y0, w, h, flags):
        local = M.copy()
        local[0, 2] -= origin_x + x0
        local[1, 2] -= origin_y + y0
        return cv2.warpAffine(piece.get_cropped_view(), local, (w, h), flags=flags,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    @pytest.mark.parametrize("rotation,scale",
                             [(0.0, 1.0), (17.3, 1.0), (-33.0, 0.8), (5.0, 1.4)])
    def test_chunking_does_not_change_the_picture(self, rotation, scale, monkeypatch):
        piece = self._piece(rotation, scale)
        M = piece.get_affine_matrix()
        args = (-900.0, -500.0, 10, 20, 2600, 700)
        direct = self._direct(piece, M, *args, cv2.INTER_LINEAR)
        monkeypatch.setattr(exp, "_WARP_DST_CHUNK", 512)
        chunked = exp._warp_region(piece, M, *args, cv2.INTER_LINEAR)
        # Fixed-point coordinate maps round differently once the source is
        # re-origined, so a single level of difference is expected; more is not.
        assert np.abs(direct.astype(int) - chunked.astype(int)).max() <= 1

    @pytest.mark.parametrize("rotation,scale", [(0.0, 1.0), (17.3, 1.0), (-33.0, 0.8)])
    def test_lanczos_chunking_stays_within_a_few_levels(self, rotation, scale,
                                                        monkeypatch):
        """The interpolation the georeferenced export actually uses.

        Lanczos has a sharper kernel than bilinear, so the sub-1/32-pixel shift
        from re-origining the source shows up more: a dozen levels on the odd
        pixel rather than one. Bound it, and bound how many pixels move at all,
        so a genuine seam or a dropped column cannot hide inside the tolerance.
        """
        piece = self._piece(rotation, scale)
        M = piece.get_affine_matrix()
        args = (-900.0, -500.0, 10, 20, 2600, 700)
        direct = self._direct(piece, M, *args, cv2.INTER_LANCZOS4)
        monkeypatch.setattr(exp, "_WARP_DST_CHUNK", 512)
        chunked = exp._warp_region(piece, M, *args, cv2.INTER_LANCZOS4)
        delta = np.abs(direct.astype(int) - chunked.astype(int))
        assert delta.max() <= 24
        assert (delta > 2).mean() < 0.005

    def test_rows_are_tiled_so_a_tall_band_keeps_lanczos(self, monkeypatch):
        """A narrow canvas gives bands tens of thousands of rows tall.

        Without row tiling the source window is that tall too, which trips the
        size guard and silently drops to cubic even though Lanczos would have
        coped.
        """
        calls = []
        real = cv2.warpAffine
        monkeypatch.setattr(cv2, "warpAffine",
                            lambda *a, **k: (calls.append(k.get("flags")), real(*a, **k))[1])
        piece = self._piece(rotation=45.0, scale=1.0, w=4000, h=4000)
        M = piece.get_affine_matrix()
        exp._warp_region(piece, M, -2000.0, -16000.0, 0, 0, 1200, 30000,
                         cv2.INTER_LANCZOS4)
        assert calls, "nothing was warped"
        assert all(f == cv2.INTER_LANCZOS4 for f in calls)

    def test_the_source_slice_is_bounded_by_the_destination(self):
        """The whole page must not be handed over for one band of rows."""
        piece = self._piece(rotation=10.0)
        M = piece.get_affine_matrix()
        local = M.copy()
        local[0, 2] -= -900.0
        local[1, 2] -= -500.0
        sx0, sy0, sx1, sy1 = exp._source_window(piece.image.shape, local, 300, 100)
        assert sx1 - sx0 < 600
        assert sy1 - sy0 < 400

    def test_a_destination_the_source_cannot_reach_stays_empty(self):
        piece = self._piece()
        M = piece.get_affine_matrix()
        out = exp._warp_region(piece, M, 0.0, 0.0, 90000, 90000, 256, 256)
        assert out.shape == (256, 256, 4)
        assert not out.any()

    def test_lanczos_survives_a_source_wider_than_the_short_limit(self, monkeypatch):
        """The exact failure the georeferenced export hit on a real plan."""
        monkeypatch.setattr(exp, "_WARP_SRC_LIMIT", 24000)
        piece = self._piece(rotation=4.0, w=34000, h=700)
        M = piece.get_affine_matrix()
        out = exp._warp_region(piece, M, -17000.0, -350.0, 0, 0, 9000, 512,
                               cv2.INTER_LANCZOS4)
        assert out.shape == (512, 9000, 4)
        assert out[:, :, 3].any()
