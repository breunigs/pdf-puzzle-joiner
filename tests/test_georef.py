"""Georeferencing: GCP bookkeeping, canvas mapping, persistence, dialog defaults.

The parts that need QGIS are marked and skipped where it is absent, so the
suite still passes on a machine without it. Everything that can be checked
without QGIS — the mapping from page pixels to canvas pixels above all — is
left unconditional, because that is the piece this project owns.
"""

import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from puzzle_joiner import georef
from puzzle_joiner.cache import save_layout, load_layout, load_georef_method
from puzzle_joiner.model import PuzzlePiece

HAS_QGIS = georef.available()
needs_qgis = unittest.skipUnless(HAS_QGIS, "QGIS bindings not installed")


def make_piece(w=200, h=100, name="page_0001.png"):
    piece = PuzzlePiece()
    piece.source_path = f"/tmp/{name}"
    piece.image = np.zeros((h, w, 4), dtype=np.uint8)
    piece.image[:, :, 3] = 255
    return piece


class TestPersistence(unittest.TestCase):
    """Control points ride along with the layout; the method is per document."""

    def test_round_trip(self):
        piece = make_piece()
        piece.gcps = [(1.5, 2.5, 100.0, 200.0), (3.0, 4.0, 300.0, 400.0)]
        with tempfile.TemporaryDirectory() as cache_dir:
            save_layout(cache_dir, [piece], None, "Linear")
            restored = make_piece()
            load_layout(cache_dir, [restored])
            self.assertEqual(restored.gcps, piece.gcps)
            self.assertEqual(load_georef_method(cache_dir), "Linear")

    def test_absent_points_stay_out_of_the_file(self):
        piece = make_piece()
        with tempfile.TemporaryDirectory() as cache_dir:
            save_layout(cache_dir, [piece])
            with open(os.path.join(cache_dir, "layout.json")) as f:
                data = json.load(f)
            self.assertNotIn("gcps", data["pieces"][0])
            self.assertIsNone(load_georef_method(cache_dir))

    def test_old_layout_without_points_still_loads(self):
        piece = make_piece()
        piece.gcps = [(1.0, 2.0, 3.0, 4.0)]
        with tempfile.TemporaryDirectory() as cache_dir:
            with open(os.path.join(cache_dir, "layout.json"), "w") as f:
                json.dump([{"source": piece.layout_key(), "x": 0, "y": 0,
                            "rotation_deg": 0, "scale": 1}], f)
            load_layout(cache_dir, [piece])
            self.assertEqual(piece.gcps, [])


@needs_qgis
class TestFit(unittest.TestCase):
    """The QGIS fit, and the reflection it needs taking out of its input."""

    # Pixel (x, y) -> map (x, y) at 10 units per pixel, north up.
    NORTH_UP = [(0.0, 0.0, 1000.0, 5000.0),
                (400.0, 0.0, 5000.0, 5000.0),
                (0.0, 300.0, 1000.0, 2000.0)]

    def expected(self, x, y):
        return (1000.0 + 10.0 * x, 5000.0 - 10.0 * y)

    def test_helmert_reproduces_a_north_up_mapping(self):
        """The reflection check: without the row flip this is 45 degrees out."""
        transform = georef.fit(self.NORTH_UP, "Helmert")
        for x, y in [(0, 0), (400, 0), (0, 300), (200, 150), (400, 300)]:
            got = transform(x, y)
            want = self.expected(x, y)
            self.assertAlmostEqual(got[0], want[0], places=3)
            self.assertAlmostEqual(got[1], want[1], places=3)

    def test_linear_reproduces_it_too(self):
        transform = georef.fit(self.NORTH_UP, "Linear")
        self.assertAlmostEqual(transform(200, 150)[1], self.expected(200, 150)[1],
                               places=3)

    def test_exact_fit_has_no_residuals(self):
        for method in georef.METHODS:
            with self.subTest(method=method):
                self.assertEqual(
                    [round(r, 6) for r in georef.residuals(self.NORTH_UP, method)],
                    [0.0, 0.0, 0.0])

    def test_rotation_is_reported_as_error_by_linear_only(self):
        """Linear cannot rotate, so a rotated page is where the two differ."""
        rotated = [(0.0, 0.0, 0.0, 0.0), (100.0, 0.0, 70.7, 70.7),
                   (0.0, 100.0, 70.7, -70.7)]
        self.assertLess(max(georef.residuals(rotated, "Helmert")), 1.0)
        self.assertGreater(max(georef.residuals(rotated, "Linear")), 1.0)

    def test_too_few_points_do_not_fit(self):
        self.assertIsNone(georef.fit(self.NORTH_UP[:1], "Helmert"))
        self.assertIsNone(georef.residuals([], "Helmert"))

    def test_unknown_method_falls_back_to_the_default(self):
        self.assertIsNotNone(georef.fit(self.NORTH_UP, "Nonsense"))

    def test_minimum_counts_are_reported(self):
        for method in georef.METHODS:
            self.assertGreaterEqual(georef.minimum_gcp_count(method), 2)


class TestDialogDefaults(unittest.TestCase):
    """The split follows the page's long axis."""

    def setUp(self):
        from PySide6.QtWidgets import QApplication
        self.app = QApplication.instance() or QApplication([])

    def _orientation(self, w, h):
        from PySide6.QtCore import Qt
        from puzzle_joiner.widgets import GeorefDialog
        piece = make_piece(w, h)
        orientation = GeorefDialog._default_orientation(
            type("Stub", (), {"piece": piece})())
        return orientation, Qt.Orientation

    def test_tall_page_puts_the_panes_side_by_side(self):
        orientation, Qt_ = self._orientation(400, 900)
        self.assertEqual(orientation, Qt_.Horizontal)

    def test_wide_page_stacks_the_panes(self):
        orientation, Qt_ = self._orientation(900, 400)
        self.assertEqual(orientation, Qt_.Vertical)

    def test_square_page_puts_the_panes_side_by_side(self):
        orientation, Qt_ = self._orientation(500, 500)
        self.assertEqual(orientation, Qt_.Horizontal)


class TestInitialMapView(unittest.TestCase):
    """Where the map pane opens: Hamburg, or round whatever points exist."""

    def setUp(self):
        from PySide6.QtWidgets import QApplication
        self.app = QApplication.instance() or QApplication([])

    def bounds(self, piece, pieces=None):
        from puzzle_joiner.widgets import GeorefDialog
        dlg = GeorefDialog(piece, pieces)
        try:
            return dlg._initial_map_bounds(pieces or [piece])
        finally:
            dlg.deleteLater()

    def test_no_points_anywhere_opens_on_hamburg(self):
        bounds = self.bounds(make_piece())
        self.assertEqual(bounds, georef.DEFAULT_MAP_RECT)

    def test_hamburg_default_contains_the_city_centre(self):
        min_x, min_y, max_x, max_y = georef.DEFAULT_MAP_RECT
        centre_x, centre_y = 1112494.0, 7085594.0  # 9.9937 E, 53.5511 N
        self.assertLess(min_x, centre_x)
        self.assertGreater(max_x, centre_x)
        self.assertLess(min_y, centre_y)
        self.assertGreater(max_y, centre_y)

    def test_one_point_opens_a_box_of_the_margin(self):
        """A lone point has no extent, so the margin alone sets the zoom."""
        piece = make_piece()
        piece.gcps = [(0.0, 0.0, 1000.0, 2000.0)]
        min_x, min_y, max_x, max_y = self.bounds(piece)
        margin = georef.POINT_MARGIN
        self.assertEqual((min_x, max_x), (1000.0 - margin, 1000.0 + margin))
        self.assertEqual((min_y, max_y), (2000.0 - margin, 2000.0 + margin))

    def test_many_points_are_boxed_and_padded(self):
        piece = make_piece()
        piece.gcps = [(0.0, 0.0, 1000.0, 2000.0),
                      (1.0, 1.0, 4000.0, 2000.0),
                      (2.0, 2.0, 2000.0, 9000.0)]
        margin = georef.POINT_MARGIN
        self.assertEqual(self.bounds(piece),
                         (1000.0 - margin, 2000.0 - margin,
                          4000.0 + margin, 9000.0 + margin))

    def test_another_page_s_points_are_used_when_this_one_has_none(self):
        blank = make_piece(name="page_0001.png")
        other = make_piece(name="page_0002.png")
        other.gcps = [(0.0, 0.0, 5000.0, 6000.0)]
        margin = georef.POINT_MARGIN
        self.assertEqual(self.bounds(blank, [blank, other]),
                         (5000.0 - margin, 6000.0 - margin,
                          5000.0 + margin, 6000.0 + margin))

    def test_this_page_s_own_points_win(self):
        mine = make_piece(name="page_0001.png")
        mine.gcps = [(0.0, 0.0, 100.0, 200.0)]
        other = make_piece(name="page_0002.png")
        other.gcps = [(0.0, 0.0, 999999.0, 999999.0)]
        self.assertEqual(self.bounds(mine, [mine, other])[0],
                         100.0 - georef.POINT_MARGIN)


class TestQtEnvironment(unittest.TestCase):
    """QGIS' Qt5 must not be let near the Qt plugins opencv ships."""

    def setUp(self):
        self.saved = {k: os.environ.get(k) for k in
                      ("QT_QPA_PLATFORM", "QT_QPA_PLATFORM_PLUGIN_PATH",
                       "QT_QPA_FONTDIR")}

    def tearDown(self):
        for key, value in self.saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_opencv_plugin_path_is_cleared(self):
        """Left in place, QGIS loads cv2's libqxcb.so and aborts the process."""
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/somewhere/cv2/qt/plugins"
        os.environ["QT_QPA_FONTDIR"] = "/somewhere/cv2/qt/fonts"
        georef._clear_opencv_qt_env()
        self.assertNotIn("QT_QPA_PLATFORM_PLUGIN_PATH", os.environ)
        self.assertNotIn("QT_QPA_FONTDIR", os.environ)

    def test_qgis_is_sent_offscreen(self):
        os.environ["QT_QPA_PLATFORM"] = "xcb"
        georef._clear_opencv_qt_env()
        self.assertEqual(os.environ["QT_QPA_PLATFORM"], "offscreen")

    def test_clearing_is_safe_when_nothing_is_set(self):
        for key in self.saved:
            os.environ.pop(key, None)
        georef._clear_opencv_qt_env()
        self.assertEqual(os.environ["QT_QPA_PLATFORM"], "offscreen")


class TestDialogInteraction(unittest.TestCase):
    """Placing, deleting and abandoning point pairs."""

    def setUp(self):
        from PySide6.QtWidgets import QApplication
        from puzzle_joiner.widgets import GeorefDialog
        self.app = QApplication.instance() or QApplication([])
        self.piece = make_piece(400, 300)
        self.dlg = GeorefDialog(self.piece)

    def tearDown(self):
        self.dlg.deleteLater()

    def place(self, img_x, img_y, map_x, map_y):
        from PySide6.QtCore import QPointF
        from puzzle_joiner.widgets import OsmView
        self.dlg._on_page_clicked(QPointF(img_x, img_y))
        self.dlg._on_map_clicked(OsmView.map_to_scene(map_x, map_y))

    def test_a_pair_of_clicks_makes_one_point(self):
        self.place(10, 20, 1000.0, 6000.0)
        self.assertEqual(len(self.dlg._gcps), 1)
        self.assertEqual(self.dlg._gcps[0], (10.0, 20.0, 1000.0, 6000.0))
        self.assertEqual(self.dlg.table.rowCount(), 1)

    def test_northing_survives_the_scene_flip(self):
        """Scene y runs down and northings run up; the pair must round-trip."""
        from puzzle_joiner.widgets import OsmView
        scene_pos = OsmView.map_to_scene(1000.0, 6000.0)
        self.assertEqual(scene_pos.y(), -6000.0)
        self.assertEqual(OsmView.scene_to_map(scene_pos), (1000.0, 6000.0))

    def test_a_map_click_alone_adds_nothing(self):
        from puzzle_joiner.widgets import OsmView
        self.dlg._on_map_clicked(OsmView.map_to_scene(1.0, 2.0))
        self.assertEqual(self.dlg._gcps, [])

    def test_clicking_the_page_twice_moves_the_pending_point(self):
        from PySide6.QtCore import QPointF
        self.dlg._on_page_clicked(QPointF(10, 20))
        self.dlg._on_page_clicked(QPointF(30, 40))
        self.assertEqual(self.dlg._pending, (30.0, 40.0))
        self.place(50, 60, 1.0, 2.0)
        self.assertEqual(len(self.dlg._gcps), 1)
        self.assertEqual(self.dlg._gcps[0][:2], (50.0, 60.0))

    def test_escape_drops_a_half_made_pair(self):
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QKeyEvent
        from PySide6.QtCore import QEvent, Qt
        self.dlg._on_page_clicked(QPointF(10, 20))
        self.dlg.keyPressEvent(QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier))
        self.assertIsNone(self.dlg._pending)
        self.assertEqual(self.dlg._gcps, [])

    def test_delete_removes_the_selected_point_and_renumbers(self):
        self.place(1, 1, 10.0, 10.0)
        self.place(2, 2, 20.0, 20.0)
        self.place(3, 3, 30.0, 30.0)
        self.dlg.table.selectRow(1)
        self.dlg._delete_selected()
        self.assertEqual([g[0] for g in self.dlg._gcps], [1.0, 3.0])
        self.assertEqual(self.dlg.table.rowCount(), 2)
        self.assertEqual(self.dlg.table.item(1, 0).text(), "2")

    def test_deleting_the_last_point_empties_the_table(self):
        self.place(1, 1, 10.0, 10.0)
        self.dlg.table.selectRow(0)
        self.dlg._delete_selected()
        self.assertEqual(self.dlg._gcps, [])
        self.assertEqual(self.dlg.table.rowCount(), 0)

    def test_markers_stay_paired_while_one_is_pending(self):
        from PySide6.QtCore import QPointF
        self.place(1, 1, 10.0, 10.0)
        self.dlg._on_page_clicked(QPointF(5, 5))
        self.assertEqual(len(self.dlg._page_markers), len(self.dlg._osm_markers))

    def test_apply_hands_back_the_points_without_touching_the_piece(self):
        self.place(1, 1, 10.0, 10.0)
        self.assertEqual(self.piece.gcps, [])
        self.dlg._apply()
        self.assertEqual(self.dlg.result_gcps, [(1.0, 1.0, 10.0, 10.0)])

    def test_cancelling_leaves_no_result(self):
        self.place(1, 1, 10.0, 10.0)
        self.dlg.reject()
        self.assertIsNone(self.dlg.result_gcps)

    def test_existing_points_are_loaded_and_copied(self):
        from puzzle_joiner.widgets import GeorefDialog
        self.piece.gcps = [(1.0, 2.0, 3.0, 4.0)]
        dlg = GeorefDialog(self.piece)
        self.assertEqual(dlg._gcps, self.piece.gcps)
        dlg.table.selectRow(0)
        dlg._delete_selected()
        # Edited on a copy: the piece only changes when the caller applies.
        self.assertEqual(self.piece.gcps, [(1.0, 2.0, 3.0, 4.0)])
        dlg.deleteLater()

    def test_method_choice_is_reported_back(self):
        self.dlg.method_combo.setCurrentText("Linear")
        self.assertEqual(self.dlg.result_method, "Linear")

    def test_orientation_toggles(self):
        from PySide6.QtCore import Qt
        first = self.dlg.splitter.orientation()
        self.dlg._toggle_orientation()
        self.assertNotEqual(self.dlg.splitter.orientation(), first)
        self.dlg._toggle_orientation()
        self.assertEqual(self.dlg.splitter.orientation(), first)
        self.assertIn(first, (Qt.Orientation.Horizontal, Qt.Orientation.Vertical))


if __name__ == "__main__":
    unittest.main()
