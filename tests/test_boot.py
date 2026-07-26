"""Smoke test: verify the app can construct its main window without crashing."""

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import sys
import numpy as np
from PySide6.QtWidgets import QApplication
from puzzle_joiner.main_window import MainWindow
from puzzle_joiner.model import PuzzlePiece
from puzzle_joiner.__main__ import _check_external_dependencies


def test_main_window_boots():
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    assert window.windowTitle() == "Puzzle Joiner"
    assert window.pieces == []
    window.close()


def test_check_deps_passes_when_all_present():
    # On a dev machine with all deps installed, should return None
    result = _check_external_dependencies()
    assert result is None


def test_recenter_syncs_scene_when_already_centered():
    """Layout restore must sync scene items even when centroid is near zero."""
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()

    # Create two pieces whose positions average to (0, 0)
    for i, (x, y, rot) in enumerate([(-100, -50, 30), (100, 50, -30)]):
        p = PuzzlePiece()
        p.source_path = f"/tmp/page_{i}.png"
        p.image = np.zeros((100, 100, 4), dtype=np.uint8)
        p.x = x
        p.y = y
        p.rotation_deg = rot
        p.rebuild_display_pixmap()
        p.rebuild_thumbnail()
        window.pieces.append(p)
        window.scene.add_piece(p)

    # Simulate load_layout updating model to new (still centered) positions
    window.pieces[0].x = -500.0
    window.pieces[0].y = -300.0
    window.pieces[0].rotation_deg = 45.0
    window.pieces[1].x = 500.0
    window.pieces[1].y = 300.0
    window.pieces[1].rotation_deg = -45.0

    # Centroid is (0, 0), so the early-return bug would skip syncing
    window._recenter_pieces()

    item0 = window.scene.piece_items[window.pieces[0]]
    item1 = window.scene.piece_items[window.pieces[1]]
    assert item0.rotation() == 45.0
    assert item1.rotation() == -45.0
    window.close()


def test_check_deps_reports_missing(monkeypatch):
    original_which = os.path.exists.__class__  # unused, just for clarity
    import shutil
    real_which = shutil.which

    def fake_which(name):
        if name in ("pdfseparate", "convert"):
            return None
        return real_which(name)

    monkeypatch.setattr("shutil.which", fake_which)
    result = _check_external_dependencies()
    assert result is not None
    assert "poppler-utils" in result
    assert "imagemagick" in result
    assert "sudo apt-get install" in result
    assert "pdfseparate" in result
    assert "convert" in result
