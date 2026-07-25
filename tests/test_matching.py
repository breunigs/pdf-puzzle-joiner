import math

import numpy as np
import pytest

from puzzle_joiner.matching import decompose_affine


class TestDecomposeAffine:
    def test_identity(self):
        M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
        tx, ty, rot, scale = decompose_affine(M)
        assert tx == pytest.approx(0)
        assert ty == pytest.approx(0)
        assert rot == pytest.approx(0)
        assert scale == pytest.approx(1)

    def test_translation(self):
        M = np.array([[1, 0, 100], [0, 1, -50]], dtype=np.float64)
        tx, ty, rot, scale = decompose_affine(M)
        assert tx == pytest.approx(100)
        assert ty == pytest.approx(-50)
        assert rot == pytest.approx(0)
        assert scale == pytest.approx(1)

    def test_rotation_90(self):
        angle = math.radians(90)
        M = np.array([
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle),  math.cos(angle), 0],
        ], dtype=np.float64)
        tx, ty, rot, scale = decompose_affine(M)
        assert rot == pytest.approx(90)
        assert scale == pytest.approx(1)

    def test_scale(self):
        s = 2.5
        M = np.array([[s, 0, 0], [0, s, 0]], dtype=np.float64)
        tx, ty, rot, scale = decompose_affine(M)
        assert scale == pytest.approx(2.5)
        assert rot == pytest.approx(0)

    def test_combined(self):
        s = 1.5
        angle = math.radians(30)
        tx_in, ty_in = 10.0, 20.0
        M = np.array([
            [s * math.cos(angle), -s * math.sin(angle), tx_in],
            [s * math.sin(angle),  s * math.cos(angle), ty_in],
        ], dtype=np.float64)
        tx, ty, rot, scale = decompose_affine(M)
        assert tx == pytest.approx(tx_in)
        assert ty == pytest.approx(ty_in)
        assert rot == pytest.approx(30)
        assert scale == pytest.approx(1.5)

    def test_negative_rotation(self):
        angle = math.radians(-45)
        M = np.array([
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle),  math.cos(angle), 0],
        ], dtype=np.float64)
        _, _, rot, _ = decompose_affine(M)
        assert rot == pytest.approx(-45)
