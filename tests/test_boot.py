"""Smoke test: verify the app can construct its main window without crashing."""

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import sys
from PySide6.QtWidgets import QApplication
from puzzle_joiner.main_window import MainWindow
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
