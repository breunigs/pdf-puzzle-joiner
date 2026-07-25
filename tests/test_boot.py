"""Smoke test: verify the app can construct its main window without crashing."""

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import sys
from PySide6.QtWidgets import QApplication
from puzzle_joiner.main_window import MainWindow


def test_main_window_boots():
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    assert window.windowTitle() == "Puzzle Joiner"
    assert window.pieces == []
    window.close()
