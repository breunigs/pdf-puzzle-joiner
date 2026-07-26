import math

import numpy as np
import pytest

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from puzzle_joiner.model import PuzzlePiece
from puzzle_joiner.export import _compute_optimal_rotation, _export_affines, _union_bbox


def _make_piece(w=100, h=80):
    piece = PuzzlePiece()
    piece.image = np.zeros((h, w, 4), dtype=np.uint8)
    piece.image[:, :, 3] = 255
    return piece


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
