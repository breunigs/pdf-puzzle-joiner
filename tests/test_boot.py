"""Smoke test: verify the app can construct its main window without crashing."""

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import sys
import numpy as np
from PySide6.QtCore import QThreadPool
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


def _window_with_page(tmp_path):
    """A window holding one page, with layout autosave pointed at tmp_path."""
    QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    page = PuzzlePiece()
    page.source_path = "/tmp/page_0001.png"
    page.display_name = "Page 1"
    rng = np.random.default_rng(3)
    page.image = rng.integers(0, 256, (300, 400, 4), dtype=np.uint8)
    page.image[:, :, 3] = 255
    page.crop_rect = (0, 0, 200, 300)
    page.x, page.y = 100, 150
    page.rebuild_display_pixmap()
    page.rebuild_thumbnail()
    window.pieces.append(page)
    window.scene.add_piece(page)
    window.thumbnail_panel.add_piece(page)
    window._layout_dir = str(tmp_path)
    return window, page


def _stub_legend_dialog(monkeypatch, rect, lasso=None):
    """Make the cut dialog accept immediately with a fixed region."""
    from PySide6.QtWidgets import QDialog

    class FakeDialog:
        def __init__(self, piece, parent=None, mode="crop"):
            assert mode == "legend"
            self.result_crop_rect = rect
            self.result_lasso_poly = lasso

        def exec(self):
            return QDialog.DialogCode.Accepted

    monkeypatch.setattr("puzzle_joiner.main_window.CutToolDialog", FakeDialog)


def test_adding_a_legend_is_undoable(tmp_path, monkeypatch):
    window, page = _window_with_page(tmp_path)
    _stub_legend_dialog(monkeypatch, (300, 20, 80, 60))

    window.add_legend_from_page(page)
    assert len(window.pieces) == 2
    legend = window.pieces[1]
    assert legend.is_legend
    assert legend.layout_key() == "page_0001.png#legend1"
    assert legend in window.scene.piece_items
    assert legend in window.thumbnail_panel.thumb_widgets

    window.undo_stack.undo()
    assert window.pieces == [page]
    assert legend not in window.scene.piece_items
    assert legend not in window.thumbnail_panel.thumb_widgets

    window.undo_stack.redo()
    assert window.pieces[1] is legend
    window.close()


def test_duplicating_a_legend_gets_a_fresh_instance(tmp_path, monkeypatch):
    window, page = _window_with_page(tmp_path)
    _stub_legend_dialog(monkeypatch, (300, 20, 80, 60))
    window.add_legend_from_page(page)
    window.duplicate_legend(window.pieces[1])

    first, second = window.pieces[1], window.pieces[2]
    assert second.is_legend
    assert first.layout_key() != second.layout_key()
    assert second.legend_rect == first.legend_rect
    window.close()


def test_legends_survive_a_layout_round_trip(tmp_path, monkeypatch):
    window, page = _window_with_page(tmp_path)
    _stub_legend_dialog(monkeypatch, (300, 20, 80, 60))
    window.add_legend_from_page(page)
    legend = window.pieces[1]
    legend.x, legend.y = 777.0, -888.0
    legend.rotation_deg = 31.0
    window._auto_save_layout()
    window.close()

    # A fresh session loading the same page must rebuild the legend.
    reopened, _ = _window_with_page(tmp_path)
    assert reopened._restore_layout()
    assert len(reopened.pieces) == 2
    restored = reopened.pieces[1]
    assert restored.is_legend
    assert restored.legend_rect == (300, 20, 80, 60)
    # Positions come back relative to the centroid recentring.
    assert restored.rotation_deg == 31.0
    assert restored.image.shape == (60, 80, 4)
    reopened.close()


def test_restoring_twice_does_not_duplicate_legends(tmp_path, monkeypatch):
    window, page = _window_with_page(tmp_path)
    _stub_legend_dialog(monkeypatch, (300, 20, 80, 60))
    window.add_legend_from_page(page)
    window._auto_save_layout()
    window._restore_layout()
    window._restore_layout()
    assert sum(1 for p in window.pieces if p.is_legend) == 1
    window.close()


def test_auto_align_leaves_legends_alone(tmp_path, monkeypatch):
    window, page = _window_with_page(tmp_path)
    _stub_legend_dialog(monkeypatch, (300, 20, 80, 60))
    window.add_legend_from_page(page)
    legend = window.pieces[1]
    legend.x, legend.y = 900.0, 50.0

    seen = {}

    def fake_detect(pieces):
        seen["pieces"] = list(pieces)

    monkeypatch.setattr("puzzle_joiner.main_window.auto_detect_placements",
                        fake_detect)
    window.on_auto_detect()
    # The alignment runs on the thread pool; wait for it before reading back.
    assert QThreadPool.globalInstance().waitForDone(10000)
    QApplication.processEvents()
    assert seen.get("pieces") == [page], "legends must be kept out of matching"
    assert (legend.x, legend.y) == (900.0, 50.0)
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


def _entry_script_path():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "puzzle-joiner")


def test_entry_points_guard_their_main_call():
    """Neither entry point may call main() while merely being imported.

    Python 3.14 starts pool workers with forkserver, which rebuilds the main
    module by re-running it — the launcher by path as "__mp_main__", the
    package module by name. An unguarded call there starts a second copy of
    the whole app inside every worker.
    """
    import ast

    for path in (_entry_script_path(),
                 os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "puzzle_joiner", "__main__.py")):
        tree = ast.parse(open(path).read())
        # Other top-level calls are fine — the launcher has to fix up sys.path
        # before anything can be imported, re-run or not. Only main() must wait.
        unguarded = [ast.unparse(node) for node in tree.body
                     if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                     and "main" in ast.unparse(node.value.func)]
        assert unguarded == [], f"{os.path.basename(path)} calls {unguarded} at import"

        guarded = [node for node in tree.body
                   if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test)
                   and "main()" in ast.unparse(node)]
        assert guarded, f"{os.path.basename(path)} never calls main() under a guard"
