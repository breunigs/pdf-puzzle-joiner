import math

import numpy as np
import pytest

# QRectF needed by PuzzlePiece — set offscreen platform before importing
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from puzzle_joiner.model import PuzzlePiece


def _make_piece(w=100, h=80):
    piece = PuzzlePiece()
    piece.image = np.zeros((h, w, 4), dtype=np.uint8)
    piece.image[:, :, 3] = 255  # opaque
    return piece


class TestGetCroppedImage:
    def test_no_crop(self):
        piece = _make_piece(100, 80)
        img = piece.get_cropped_image()
        assert img.shape == (80, 100, 4)

    def test_with_crop(self):
        piece = _make_piece(100, 80)
        piece.crop_rect = (10, 20, 50, 30)
        img = piece.get_cropped_image()
        assert img.shape == (30, 50, 4)

    def test_crop_is_copy(self):
        piece = _make_piece(100, 80)
        piece.crop_rect = (10, 20, 50, 30)
        img = piece.get_cropped_image()
        img[:] = 128
        assert piece.image[20, 10, 0] == 0  # original unmodified


class TestGetAffineMatrix:
    def test_identity_at_center(self):
        piece = _make_piece(100, 80)
        piece.x = 50.0
        piece.y = 40.0
        M = piece.get_affine_matrix()
        assert M.shape == (2, 3)
        # Should map center (50, 40) to (50, 40)
        assert M[0, 2] == pytest.approx(0)
        assert M[1, 2] == pytest.approx(0)
        assert M[0, 0] == pytest.approx(1)
        assert M[1, 1] == pytest.approx(1)

    def test_translation(self):
        piece = _make_piece(100, 80)
        piece.x = 200.0
        piece.y = 300.0
        M = piece.get_affine_matrix()
        # Top-left corner (0,0) should map to (200-50, 300-40) = (150, 260)
        result = M @ np.array([0, 0, 1])
        assert result[0] == pytest.approx(150)
        assert result[1] == pytest.approx(260)
        # Center should map to (200, 300)
        center = M @ np.array([50, 40, 1])
        assert center[0] == pytest.approx(200)
        assert center[1] == pytest.approx(300)

    def test_scale(self):
        piece = _make_piece(100, 80)
        piece.x = 50.0
        piece.y = 40.0
        piece.scale = 2.0
        M = piece.get_affine_matrix()
        # Center stays fixed
        center = M @ np.array([50, 40, 1])
        assert center[0] == pytest.approx(50)
        assert center[1] == pytest.approx(40)
        # Top-left moves away from center
        tl = M @ np.array([0, 0, 1])
        assert tl[0] == pytest.approx(-50)
        assert tl[1] == pytest.approx(-40)

    def test_rotation(self):
        piece = _make_piece(100, 100)
        piece.x = 50.0
        piece.y = 50.0
        piece.rotation_deg = 90.0
        M = piece.get_affine_matrix()
        # Center stays fixed
        center = M @ np.array([50, 50, 1])
        assert center[0] == pytest.approx(50, abs=1e-10)
        assert center[1] == pytest.approx(50, abs=1e-10)

    def test_with_crop(self):
        piece = _make_piece(200, 160)
        piece.crop_rect = (50, 40, 100, 80)
        piece.x = 50.0
        piece.y = 40.0
        M = piece.get_affine_matrix()
        # Cropped image is 100x80, center is (50, 40)
        center = M @ np.array([50, 40, 1])
        assert center[0] == pytest.approx(50)
        assert center[1] == pytest.approx(40)


class TestGetBoundingBox:
    def test_no_transform(self):
        piece = _make_piece(100, 80)
        piece.x = 50.0
        piece.y = 40.0
        bb = piece.get_bounding_box()
        assert bb.width() == pytest.approx(100)
        assert bb.height() == pytest.approx(80)
        assert bb.x() == pytest.approx(0)
        assert bb.y() == pytest.approx(0)

    def test_translated(self):
        piece = _make_piece(100, 80)
        piece.x = 200.0
        piece.y = 300.0
        bb = piece.get_bounding_box()
        assert bb.width() == pytest.approx(100)
        assert bb.height() == pytest.approx(80)
        assert bb.x() == pytest.approx(150)
        assert bb.y() == pytest.approx(260)

    def test_scaled(self):
        piece = _make_piece(100, 80)
        piece.x = 50.0
        piece.y = 40.0
        piece.scale = 2.0
        bb = piece.get_bounding_box()
        assert bb.width() == pytest.approx(200)
        assert bb.height() == pytest.approx(160)
