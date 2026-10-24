"""Lasso crop: alpha masking in the model, and the polygon editor's bookkeeping."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import (
    QApplication, QDialog, QGraphicsScene, QMessageBox,
)

from puzzle_joiner.model import PuzzlePiece
from puzzle_joiner.undo import CropCommand
from puzzle_joiner.widgets import CutToolDialog, LassoItem, LassoVertexItem


# A right triangle covering the top-left half of a 100x80 image.
TRIANGLE = [(0, 0), (100, 0), (0, 80)]


def _make_piece(w=100, h=80, alpha=255):
    piece = PuzzlePiece()
    piece.image = np.full((h, w, 4), 200, dtype=np.uint8)
    piece.image[:, :, 3] = alpha
    return piece


class TestLassoMask:
    def test_outside_polygon_becomes_transparent(self):
        piece = _make_piece()
        piece.apply_crop((0, 0, 100, 80), TRIANGLE)
        alpha = piece.image[:, :, 3]
        assert alpha[2, 2] == 255          # well inside the triangle
        assert alpha[70, 90] == 0          # bottom-right, outside
        assert alpha[79, 99] == 0

    def test_colour_channels_are_untouched(self):
        piece = _make_piece()
        piece.apply_crop((0, 0, 100, 80), TRIANGLE)
        assert (piece.image[:, :, :3] == 200).all()

    def test_edge_is_antialiased(self):
        piece = _make_piece()
        piece.apply_crop((0, 0, 100, 80), TRIANGLE)
        alpha = piece.image[:, :, 3]
        assert ((alpha > 0) & (alpha < 255)).any(), "expected graded edge pixels"

    def test_clearing_the_lasso_restores_original_alpha(self):
        piece = _make_piece()
        piece.apply_crop((0, 0, 100, 80), TRIANGLE)
        piece.apply_crop(None, None)
        assert (piece.image[:, :, 3] == 255).all()
        assert piece._base_alpha is None

    def test_source_transparency_is_preserved(self):
        """An imported PNG's own alpha must survive lasso masking and removal."""
        piece = _make_piece()
        piece.image[0:10, :, 3] = 0
        piece.apply_crop((0, 0, 100, 80), TRIANGLE)
        assert (piece.image[0:10, :, 3] == 0).all()
        piece.apply_crop(None, None)
        assert (piece.image[0:10, :, 3] == 0).all()
        assert (piece.image[10:, :, 3] == 255).all()

    def test_repeated_edits_do_not_degrade_pixels(self):
        piece = _make_piece()
        for _ in range(3):
            piece.apply_crop((0, 0, 100, 80), [(0, 0), (20, 0), (20, 20), (0, 20)])
            piece.apply_crop((0, 0, 100, 80), TRIANGLE)
        alpha = piece.image[:, :, 3]
        assert alpha[2, 2] == 255
        assert alpha[70, 90] == 0

    def test_cropped_view_is_masked_and_still_a_view(self):
        piece = _make_piece()
        piece.apply_crop((10, 10, 60, 50), [(10, 10), (70, 10), (10, 60)])
        view = piece.get_cropped_view()
        assert view.base is piece.image
        assert view.shape[:2] == (50, 60)
        assert view[45, 55, 3] == 0        # outside the triangle


class TestMatchImage:
    def test_lasso_is_ignored_for_matching(self):
        piece = _make_piece()
        piece.apply_crop((0, 0, 100, 80), TRIANGLE)
        img = piece.get_match_image()
        assert (img[:, :, 3] == 255).all()
        assert (img[:, :, :3] == 200).all()

    def test_match_image_uses_the_crop_rect(self):
        piece = _make_piece()
        piece.apply_crop((10, 20, 40, 30), [(10, 20), (50, 20), (50, 50), (10, 50)])
        img = piece.get_match_image()
        assert img.shape[:2] == (30, 40)

    def test_match_image_keeps_source_transparency(self):
        piece = _make_piece()
        piece.image[0:10, :, 3] = 0
        piece.apply_crop((0, 0, 100, 80), TRIANGLE)
        img = piece.get_match_image()
        assert (img[0:10, :, 3] == 0).all()
        assert (img[10:, :, 3] == 255).all()

    def test_match_image_without_lasso_is_the_plain_crop(self):
        piece = _make_piece()
        piece.crop_rect = (5, 5, 20, 20)
        np.testing.assert_array_equal(piece.get_match_image(), piece.get_cropped_image())


class TestOutlinePoints:
    def test_rect_crop_yields_corners(self):
        piece = _make_piece()
        piece.crop_rect = (10, 20, 40, 30)
        np.testing.assert_array_equal(
            piece.get_outline_points(),
            np.array([[0, 0], [40, 0], [40, 30], [0, 30]], dtype=np.float32),
        )

    def test_lasso_points_are_relative_to_the_crop_origin(self):
        piece = _make_piece()
        piece.apply_crop((10, 20, 40, 30), [(10, 20), (50, 20), (30, 50)])
        np.testing.assert_allclose(
            piece.get_outline_points(),
            np.array([[0, 0], [40, 0], [20, 30]], dtype=np.float32),
        )

    def test_points_outside_the_crop_are_clipped(self):
        piece = _make_piece()
        piece.apply_crop((10, 10, 20, 20), [(0, 0), (100, 0), (100, 80)])
        pts = piece.get_outline_points()
        assert pts.min() >= 0
        assert (pts[:, 0] <= 20).all() and (pts[:, 1] <= 20).all()


class TestCropCommand:
    def test_undo_restores_crop_lasso_and_position(self):
        piece = _make_piece()
        piece.x, piece.y = 50.0, 40.0
        synced = []
        cmd = CropCommand(
            piece, None, None, 50.0, 40.0,
            (0, 0, 100, 80), TRIANGLE, 20.0, 10.0,
            sync_fn=synced.append,
        )
        cmd.redo()
        assert piece.crop_rect == (0, 0, 100, 80)
        assert piece.lasso_poly == [(0.0, 0.0), (100.0, 0.0), (0.0, 80.0)]
        assert (piece.x, piece.y) == (20.0, 10.0)
        assert piece.image[70, 90, 3] == 0

        cmd.undo()
        assert piece.crop_rect is None
        assert piece.lasso_poly is None
        assert (piece.x, piece.y) == (50.0, 40.0)
        assert (piece.image[:, :, 3] == 255).all()
        assert synced == [piece, piece]


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def lasso(qapp):
    scene = QGraphicsScene()
    item = LassoItem(QRectF(0, 0, 100, 80), scene,
                     points=[(10, 10), (90, 10), (90, 70)], closed=True)
    yield item
    item.remove_from_scene()


class TestLassoItem:
    def test_closed_shape_has_a_midpoint_per_edge(self, lasso):
        vertices = [h for h in lasso.handles if not h.midpoint]
        midpoints = [h for h in lasso.handles if h.midpoint]
        assert len(vertices) == 3
        assert len(midpoints) == 3
        # Midpoint i sits between vertex i and i+1, wrapping at the closing edge.
        assert midpoints[0].pos() == QPointF(50, 10)
        assert midpoints[2].pos() == QPointF(50, 40)

    def test_while_open_there_are_no_midpoints(self, qapp):
        scene = QGraphicsScene()
        item = LassoItem(QRectF(0, 0, 100, 80), scene)
        item.add_point(QPointF(10, 10))
        item.add_point(QPointF(90, 10))
        assert all(not h.midpoint for h in item.handles)
        assert len(item.handles) == 2
        item.remove_from_scene()

    def test_insert_point_returns_the_new_index(self, lasso):
        index = lasso.insert_point(0, QPointF(50, 5))
        assert index == 1
        assert lasso.polygon() == [(10, 10), (50, 5), (90, 10), (90, 70)]
        assert len([h for h in lasso.handles if not h.midpoint]) == 4

    def test_set_point_moves_only_that_vertex(self, lasso):
        lasso.set_point(1, QPointF(80, 20))
        assert lasso.polygon() == [(10, 10), (80, 20), (90, 70)]

    def test_points_are_clamped_to_the_image(self, lasso):
        lasso.set_point(0, QPointF(-50, 500))
        assert lasso.polygon()[0] == (0, 80)

    def test_remove_point_keeps_a_minimum_of_three(self, lasso):
        lasso.remove_point(1)
        assert len(lasso.points) == 3, "a triangle cannot shrink further"
        lasso.insert_point(0, QPointF(50, 5))
        lasso.remove_point(1)
        assert lasso.polygon() == [(10, 10), (90, 10), (90, 70)]

    def test_close_needs_three_points(self, qapp):
        scene = QGraphicsScene()
        item = LassoItem(QRectF(0, 0, 100, 80), scene)
        item.add_point(QPointF(10, 10))
        item.add_point(QPointF(90, 10))
        assert not item.close_shape()
        assert not item.closed
        item.add_point(QPointF(90, 70))
        assert item.close_shape()
        assert item.closed
        item.remove_from_scene()

    def test_duplicate_click_is_dropped(self, qapp):
        """The press preceding a closing double-click must not leave a dead point."""
        scene = QGraphicsScene()
        item = LassoItem(QRectF(0, 0, 100, 80), scene)
        item.add_point(QPointF(10, 10))
        item.add_point(QPointF(10.2, 10.2))
        assert len(item.points) == 1
        item.remove_from_scene()

    def test_pop_point_undoes_the_last_click(self, qapp):
        scene = QGraphicsScene()
        item = LassoItem(QRectF(0, 0, 100, 80), scene)
        for p in [(10, 10), (90, 10), (90, 70)]:
            item.add_point(QPointF(*p))
        item.pop_point()
        assert item.polygon() == [(10, 10), (90, 10)]
        item.remove_from_scene()

    def test_bounding_rect_spans_all_points(self, lasso):
        r = lasso.bounding_rect()
        assert (r.left(), r.top(), r.right(), r.bottom()) == (10, 10, 90, 70)

    def test_overlay_dims_outside_only(self, lasso):
        path = lasso.path()
        assert path.contains(QPointF(2, 2)), "outside the lasso should be dimmed"
        assert not path.contains(QPointF(60, 20)), "inside must stay clear"

    def test_open_shape_dims_nothing(self, qapp):
        scene = QGraphicsScene()
        item = LassoItem(QRectF(0, 0, 100, 80), scene)
        for p in [(10, 10), (90, 10), (90, 70)]:
            item.add_point(QPointF(*p))
        assert item.path().isEmpty(), "the full image stays visible while drawing"
        item.remove_from_scene()

    def test_rubber_band_follows_the_cursor(self, qapp):
        scene = QGraphicsScene()
        item = LassoItem(QRectF(0, 0, 100, 80), scene)
        for p in [(10, 10), (90, 10), (90, 70)]:
            item.add_point(QPointF(*p))
        item.set_cursor_point(QPointF(60, 40))
        assert item._outline.path().currentPosition() == QPointF(60, 40)
        # Once closed the outline follows the polygon, not the cursor.
        item.close_shape()
        item.set_cursor_point(QPointF(20, 20))
        assert item._outline.path().currentPosition() == QPointF(10, 10)
        item.remove_from_scene()

    def test_remove_from_scene_takes_the_handles(self, qapp):
        scene = QGraphicsScene()
        item = LassoItem(QRectF(0, 0, 100, 80), scene,
                         points=[(10, 10), (90, 10), (90, 70)], closed=True)
        item.remove_from_scene()
        assert not [i for i in scene.items() if isinstance(i, LassoVertexItem)]


def _press(view, x, y, button="left"):
    buttons = {"left": Qt.MouseButton.LeftButton, "right": Qt.MouseButton.RightButton}
    b = buttons[button]
    view.mousePressEvent(QMouseEvent(
        QEvent.Type.MouseButtonPress, QPointF(x, y), b, b, Qt.KeyboardModifier.NoModifier))


def _move(view, x, y, button="left"):
    b = {"left": Qt.MouseButton.LeftButton}.get(button, Qt.MouseButton.NoButton)
    view.mouseMoveEvent(QMouseEvent(
        QEvent.Type.MouseMove, QPointF(x, y), Qt.MouseButton.NoButton, b,
        Qt.KeyboardModifier.NoModifier))


def _release(view, x, y):
    view.mouseReleaseEvent(QMouseEvent(
        QEvent.Type.MouseButtonRelease, QPointF(x, y), Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier))


def _key(view, key):
    view.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier))


@pytest.fixture
def dialog(qapp):
    piece = _make_piece(200, 160)
    piece.x, piece.y = 100, 80
    dlg = CutToolDialog(piece)
    # 1:1 scene-to-viewport mapping keeps the synthetic coordinates readable.
    dlg.view.resetTransform()
    dlg.view._zoom = 1.0
    dlg.view.centerOn(0, 0)
    dlg.view.horizontalScrollBar().setValue(0)
    dlg.view.verticalScrollBar().setValue(0)
    yield dlg
    dlg.close()


class TestCropViewLassoInteraction:
    def _scene_pts(self, view):
        return [(round(p.x()), round(p.y())) for p in view.lasso_item.points]

    def test_clicking_builds_and_closes_a_polygon(self, dialog):
        view = dialog.view
        dialog._set_tool("lasso")
        for x, y in [(10, 10), (80, 10), (80, 60)]:
            vp = view.mapFromScene(QPointF(x, y))
            _press(view, vp.x(), vp.y())
            _release(view, vp.x(), vp.y())
        assert self._scene_pts(view) == [(10, 10), (80, 10), (80, 60)]
        assert not view.lasso_item.closed

        # Clicking the first vertex again closes the shape.
        vp = view.mapFromScene(QPointF(10, 10))
        _press(view, vp.x(), vp.y())
        _release(view, vp.x(), vp.y())
        assert view.lasso_item.closed
        assert len(view.lasso_item.points) == 3

    def test_dragging_a_vertex_corrects_a_misclick(self, dialog):
        view = dialog.view
        dialog._set_tool("lasso")
        view.lasso_item = LassoItem(QRectF(0, 0, 200, 160), view.scene(),
                                    points=[(10, 10), (80, 10), (80, 60)], closed=True)
        start = view.mapFromScene(QPointF(80, 10))
        _press(view, start.x(), start.y())
        assert view._mode == "lasso_vertex"
        end = view.mapFromScene(QPointF(120, 30))
        _move(view, end.x(), end.y())
        _release(view, end.x(), end.y())
        assert self._scene_pts(view) == [(10, 10), (120, 30), (80, 60)]
        assert view._mode is None

    def test_dragging_a_midpoint_inserts_a_vertex(self, dialog):
        view = dialog.view
        dialog._set_tool("lasso")
        view.lasso_item = LassoItem(QRectF(0, 0, 200, 160), view.scene(),
                                    points=[(10, 10), (90, 10), (90, 70)], closed=True)
        mid = view.mapFromScene(QPointF(50, 10))   # between vertex 0 and 1
        _press(view, mid.x(), mid.y())
        end = view.mapFromScene(QPointF(50, 40))
        _move(view, end.x(), end.y())
        _release(view, end.x(), end.y())
        assert self._scene_pts(view) == [(10, 10), (50, 40), (90, 10), (90, 70)]

    def test_right_click_deletes_a_vertex(self, dialog):
        view = dialog.view
        dialog._set_tool("lasso")
        view.lasso_item = LassoItem(
            QRectF(0, 0, 200, 160), view.scene(),
            points=[(10, 10), (90, 10), (90, 70), (10, 70)], closed=True)
        vp = view.mapFromScene(QPointF(90, 10))
        _press(view, vp.x(), vp.y(), button="right")
        assert self._scene_pts(view) == [(10, 10), (90, 70), (10, 70)]

    def test_backspace_undoes_the_last_click(self, dialog):
        view = dialog.view
        dialog._set_tool("lasso")
        for x, y in [(10, 10), (80, 10)]:
            vp = view.mapFromScene(QPointF(x, y))
            _press(view, vp.x(), vp.y())
            _release(view, vp.x(), vp.y())
        _key(view, Qt.Key.Key_Backspace)
        assert self._scene_pts(view) == [(10, 10)]
        _key(view, Qt.Key.Key_Backspace)
        assert view.lasso_item is None, "removing the last point clears the lasso"

    def test_enter_closes_the_shape(self, dialog):
        view = dialog.view
        dialog._set_tool("lasso")
        for x, y in [(10, 10), (80, 10), (80, 60)]:
            vp = view.mapFromScene(QPointF(x, y))
            _press(view, vp.x(), vp.y())
            _release(view, vp.x(), vp.y())
        _key(view, Qt.Key.Key_Return)
        assert view.lasso_item.closed

    def test_switching_tools_discards_the_other_region(self, dialog):
        view = dialog.view
        assert view.crop_rect_item is not None
        dialog._set_tool("lasso")
        assert view.crop_rect_item is None
        view.lasso_item = LassoItem(QRectF(0, 0, 200, 160), view.scene(),
                                    points=[(10, 10), (90, 10), (90, 70)], closed=True)
        dialog._set_tool("rect")
        assert view.lasso_item is None

    def test_apply_reports_lasso_and_bounding_rect(self, dialog):
        view = dialog.view
        dialog._set_tool("lasso")
        view.lasso_item = LassoItem(QRectF(0, 0, 200, 160), view.scene(),
                                    points=[(10, 20), (90, 20), (90, 70)], closed=True)
        dialog._apply()
        assert dialog.result_crop_rect == (10, 20, 80, 50)
        assert dialog.result_lasso_poly == [(10, 20), (90, 20), (90, 70)]
        assert dialog.result() == QDialog.DialogCode.Accepted
        # The piece itself is untouched — the caller applies it via the undo stack.
        assert dialog.piece.lasso_poly is None

    def test_apply_rejects_an_unfinished_lasso(self, dialog, monkeypatch):
        warned = []
        monkeypatch.setattr(QMessageBox, "warning",
                            lambda *a, **k: warned.append(a[1]))
        view = dialog.view
        dialog._set_tool("lasso")
        vp = view.mapFromScene(QPointF(10, 10))
        _press(view, vp.x(), vp.y())
        _release(view, vp.x(), vp.y())
        dialog._apply()
        assert warned == ["Incomplete lasso"]
        assert dialog.result_crop_rect is None

    def test_apply_after_reset_clears_the_crop(self, dialog):
        dialog.piece.apply_crop((10, 10, 50, 50), None)
        dialog._reset()
        dialog._apply()
        assert dialog.result_crop_rect is None
        assert dialog.result_lasso_poly is None

    def test_existing_lasso_is_restored_for_editing(self, qapp):
        piece = _make_piece(200, 160)
        piece.apply_crop((10, 10, 80, 60), [(10, 10), (90, 10), (90, 70)])
        dlg = CutToolDialog(piece)
        assert dlg.view.tool == "lasso"
        assert dlg.lasso_btn.isChecked()
        assert dlg.view.lasso_item.closed
        assert dlg.view.lasso_item.polygon() == [(10, 10), (90, 10), (90, 70)]
        dlg.close()


class TestOnCutIntegration:
    """The cut tool must go through the undo stack and refresh every cached pixmap."""

    def _window(self):
        from puzzle_joiner.main_window import MainWindow
        win = MainWindow()
        img = np.full((160, 200, 4), 200, dtype=np.uint8)
        img[:, :, 3] = 255
        win._add_piece_from_array(img, "/tmp/page_0.png", 0)
        return win

    def _patch_dialog(self, monkeypatch, crop, poly):
        import puzzle_joiner.main_window as mw

        class FakeDialog:
            def __init__(self, piece, parent=None):
                self.piece = piece
                self.result_crop_rect = crop
                self.result_lasso_poly = poly

            def exec(self):
                return QDialog.DialogCode.Accepted

        monkeypatch.setattr(mw, "CutToolDialog", FakeDialog)

    def test_applying_a_lasso_is_undoable(self, qapp, monkeypatch):
        win = self._window()
        piece = win.pieces[0]
        win.scene.clearSelection()
        win.scene.piece_items[piece].setSelected(True)
        before_x, before_y = piece.x, piece.y

        self._patch_dialog(monkeypatch, (0, 0, 200, 160),
                           [(0, 0), (200, 0), (0, 160)])
        win.on_cut()

        assert win.undo_stack.count() == 1
        assert piece.lasso_poly == [(0.0, 0.0), (200.0, 0.0), (0.0, 160.0)]
        assert piece.image[150, 190, 3] == 0
        assert piece.display_pixmap.hasAlphaChannel()

        win.undo_stack.undo()
        assert piece.lasso_poly is None
        assert piece.crop_rect is None
        assert (piece.image[:, :, 3] == 255).all()
        assert (piece.x, piece.y) == (before_x, before_y)

        win.undo_stack.redo()
        assert piece.lasso_poly is not None
        assert piece.image[150, 190, 3] == 0
        win.close()

    def test_crop_shift_keeps_content_in_place(self, qapp, monkeypatch):
        """Cropping must not move the visible image, or snapped neighbours drift."""
        win = self._window()
        piece = win.pieces[0]
        win.scene.piece_items[piece].setSelected(True)
        # Lasso over the right half; its bbox origin moves by (100, 0).
        self._patch_dialog(monkeypatch, (100, 0, 100, 160),
                           [(100, 0), (200, 0), (200, 160), (100, 160)])
        win.on_cut()
        assert piece.x == pytest.approx(100.0 + 50.0)
        win.close()

    def test_cancelled_dialog_changes_nothing(self, qapp, monkeypatch):
        import puzzle_joiner.main_window as mw

        class Rejected:
            def __init__(self, piece, parent=None):
                self.result_crop_rect = (0, 0, 10, 10)
                self.result_lasso_poly = None

            def exec(self):
                return QDialog.DialogCode.Rejected

        win = self._window()
        win.scene.piece_items[win.pieces[0]].setSelected(True)
        monkeypatch.setattr(mw, "CutToolDialog", Rejected)
        win.on_cut()
        assert win.undo_stack.count() == 0
        assert win.pieces[0].crop_rect is None
        win.close()
