"""Georeferencing: GCP bookkeeping, canvas mapping, persistence, dialog defaults.

The parts that need QGIS are marked and skipped where it is absent, so the
suite still passes on a machine without it. Everything that can be checked
without QGIS — the mapping from page pixels to canvas pixels above all — is
left unconditional, because that is the piece this project owns.
"""

import json
import math
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from puzzle_joiner import georef
from puzzle_joiner.cache import save_layout, load_layout, load_georef_method
from puzzle_joiner.export import (
    canvas_gcps, export_canvas, _export_affines, _union_bbox,
)
from puzzle_joiner.model import PuzzlePiece

HAS_QGIS = georef.available()
needs_qgis = unittest.skipUnless(HAS_QGIS, "QGIS bindings not installed")


def make_piece(w=200, h=100, name="page_0001.png"):
    piece = PuzzlePiece()
    piece.source_path = f"/tmp/{name}"
    piece.image = np.zeros((h, w, 4), dtype=np.uint8)
    piece.image[:, :, 3] = 255
    return piece


class TestCanvasGcps(unittest.TestCase):
    """Page control points, pooled and expressed in exported-canvas pixels."""

    def test_a_point_lands_where_its_pixel_lands(self):
        """Canvas coords are absolute: a piece sits centred on its position."""
        piece = make_piece(200, 100)
        piece.gcps = [(10.0, 20.0, 1000.0, 2000.0)]
        px, py, map_x, map_y = canvas_gcps([piece])[0]
        # The piece is centred on (x, y), so its top-left is at (-100, -50).
        self.assertAlmostEqual(px, 10.0 - 100.0, places=3)
        self.assertAlmostEqual(py, 20.0 - 50.0, places=3)
        self.assertEqual((map_x, map_y), (1000.0, 2000.0))

    def test_a_crop_that_keeps_content_still_keeps_the_point_still(self):
        """The cut tool compensates position so pixels stay put; so must points.

        A crop redefines the piece's local origin and its centre, which on its
        own would drag the image across the canvas. `on_cut` cancels that, and
        a control point has to ride along with the pixel it marks.
        """
        crop = (20, 10, 150, 80)
        plain = make_piece(200, 100)
        plain.gcps = [(60.0, 40.0, 0.0, 0.0)]
        cropped = make_piece(200, 100)
        cropped.gcps = [(60.0, 40.0, 0.0, 0.0)]
        cropped.apply_crop(crop)
        # The same compensation `MainWindow.on_cut` applies, with no rotation.
        cropped.x += (crop[0] + crop[2] / 2.0) - 200 / 2.0
        cropped.y += (crop[1] + crop[3] / 2.0) - 100 / 2.0
        self.assertAlmostEqual(canvas_gcps([plain])[0][0],
                               canvas_gcps([cropped])[0][0], places=3)
        self.assertAlmostEqual(canvas_gcps([plain])[0][1],
                               canvas_gcps([cropped])[0][1], places=3)

    def test_moving_the_page_moves_its_points_the_same_way(self):
        piece = make_piece()
        piece.gcps = [(10.0, 20.0, 0.0, 0.0)]
        before = canvas_gcps([piece])[0]
        piece.x += 500.0
        piece.y -= 300.0
        after = canvas_gcps([piece])[0]
        self.assertAlmostEqual(after[0] - before[0], 500.0, places=3)
        self.assertAlmostEqual(after[1] - before[1], -300.0, places=3)

    def test_other_pieces_do_not_move_a_point(self):
        """Absolute coords, so nothing else on the canvas can shift them.

        This is what lets a legend be added or removed without invalidating a
        georeference, and what lets the fit be made once for the whole canvas.
        """
        a = make_piece(name="page_0001.png")
        a.gcps = [(10.0, 20.0, 0.0, 0.0)]
        lone = canvas_gcps([a])[0]
        b = make_piece(name="page_0002.png")
        b.x = -1000.0
        legend = make_piece(name="page_0001.png")
        legend.is_legend = True
        legend.legend_instance = 1
        legend.x = -2000.0
        self.assertEqual(canvas_gcps([a, b, legend])[0], lone)

    def test_points_are_pooled_across_pages(self):
        a = make_piece(name="page_0001.png")
        a.gcps = [(1.0, 2.0, 10.0, 20.0)]
        b = make_piece(name="page_0002.png")
        b.gcps = [(3.0, 4.0, 30.0, 40.0), (5.0, 6.0, 50.0, 60.0)]
        self.assertEqual(len(canvas_gcps([a, b])), 3)

    def test_pages_without_points_contribute_none(self):
        a = make_piece(name="page_0001.png")
        b = make_piece(name="page_0002.png")
        b.gcps = [(3.0, 4.0, 30.0, 40.0)]
        self.assertEqual(len(canvas_gcps([a, b])), 1)

    def test_legends_are_skipped(self):
        """A legend is beside the map, not on it, so it carries no geography."""
        page = make_piece(name="page_0001.png")
        page.gcps = [(1.0, 2.0, 10.0, 20.0)]
        legend = make_piece(name="page_0001.png")
        legend.is_legend = True
        legend.legend_instance = 1
        legend.gcps = [(9.0, 9.0, 90.0, 90.0)]
        pooled = canvas_gcps([page, legend])
        self.assertEqual(len(pooled), 1)
        self.assertEqual(pooled[0][2:], (10.0, 20.0))

    def test_point_outside_the_crop_is_kept(self):
        """The dialog shows the uncropped page; such points must survive."""
        piece = make_piece()
        piece.apply_crop((50, 50, 100, 40))
        piece.gcps = [(0.0, 0.0, 10.0, 20.0)]
        pooled = canvas_gcps([piece])
        self.assertEqual(len(pooled), 1)
        self.assertLess(pooled[0][0], 0.0)

    def test_canvas_is_sized_from_the_crop_not_the_lasso(self):
        """Both exported files share this canvas, so one fit suits either.

        The uncropped file needs the pixels outside the hull, so the canvas has
        to be the rect-sized one — if it ever became the hull-sized one, these
        pixel coordinates would stop meaning the same thing in the two files.
        """
        piece = make_piece()
        piece.apply_crop((10, 10, 150, 70),
                         [(20, 20), (60, 20), (60, 50), (20, 50)])
        affines = _export_affines([piece])
        self.assertEqual(export_canvas([piece], affines),
                         _union_bbox([piece], affines, uncropped=True))
        self.assertNotEqual(export_canvas([piece], affines),
                            _union_bbox([piece], affines, uncropped=False))


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


@needs_qgis
class TestGeorefTransform(unittest.TestCase):
    """Canvas coords to north-up raster pixels, the heart of the geo export."""

    def transform(self, gcps, method="Helmert"):
        from puzzle_joiner.export import georef_transform
        return georef_transform(gcps, method)

    def test_pixel_size_is_the_fit_scale(self):
        """Which is what leaves the map a pure rotation and legends unscaled."""
        _, scale = self.transform(TestFit.NORTH_UP)
        self.assertAlmostEqual(scale, 10.0, places=6)

    def test_the_map_transform_is_a_rotation_not_a_scaling(self):
        G, _ = self.transform(TestFit.NORTH_UP)
        linear = G[:2, :2]
        self.assertAlmostEqual(abs(np.linalg.det(linear)), 1.0, places=6)
        # Orthonormal: columns of unit length and at right angles.
        self.assertAlmostEqual(float(linear[:, 0] @ linear[:, 1]), 0.0, places=6)
        self.assertAlmostEqual(float(np.linalg.norm(linear[:, 0])), 1.0, places=6)

    def test_a_north_up_fit_leaves_the_canvas_unrotated(self):
        G, _ = self.transform(TestFit.NORTH_UP)
        np.testing.assert_allclose(G[:2, :2], np.eye(2), atol=1e-6)

    def grid_to_world(self, G, scale, canvas_x, canvas_y):
        """Undo the grid, the way `_geotiff_tags` records it."""
        grid = G @ np.array([canvas_x, canvas_y, 1.0])
        return grid[0] * scale, -grid[1] * scale

    def test_the_grid_round_trips_back_to_the_map_coordinates(self):
        """The one assertion that pins every sign in the chain at once.

        A control point pushed through the grid and read back with the pixel
        size must land on the map coordinate it was given. Rotated fits are
        included because that is where a flipped sign hides — a reflection
        survives the north-up case untouched.
        """
        for name, gcps in (("north up", TestFit.NORTH_UP),
                           ("rotated", self.ROTATED),
                           ("rotated the other way", self.ROTATED_BACK)):
            with self.subTest(fit=name):
                G, scale = self.transform(gcps)
                for canvas_x, canvas_y, map_x, map_y in gcps:
                    got = self.grid_to_world(G, scale, canvas_x, canvas_y)
                    self.assertAlmostEqual(got[0], map_x, places=3)
                    self.assertAlmostEqual(got[1], map_y, places=3)

    @staticmethod
    def _rotated(degrees):
        """Canvas points and the map coords of a page turned by `degrees`.

        Canvas rows run down and northings run up, so the northing is negated
        — the same flip `georef.fit` takes out before fitting.
        """
        rad = math.radians(degrees)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        out = []
        for canvas_x, canvas_y in [(0.0, 0.0), (100.0, 0.0), (0.0, 100.0)]:
            out.append((canvas_x, canvas_y,
                        cos_a * canvas_x - sin_a * canvas_y,
                        -(sin_a * canvas_x + cos_a * canvas_y)))
        return out

    ROTATED = _rotated(30.0)
    ROTATED_BACK = _rotated(-55.0)

    def test_a_rotated_fit_keeps_its_scale(self):
        for gcps in (self.ROTATED, self.ROTATED_BACK):
            _, scale = self.transform(gcps)
            self.assertAlmostEqual(scale, 1.0, places=6)

    def test_a_rotated_fit_stays_a_rotation(self):
        """No reflection: a mirrored map would be worse than a crooked one."""
        for gcps in (self.ROTATED, self.ROTATED_BACK):
            G, _ = self.transform(gcps)
            self.assertAlmostEqual(float(np.linalg.det(G[:2, :2])), 1.0, places=6)

    def test_no_fit_yields_no_transform(self):
        self.assertIsNone(self.transform([]))

    def test_geotiff_tags_describe_a_north_up_grid(self):
        from puzzle_joiner.export import _geotiff_tags
        tags = {code: value for code, _dtype, _count, value, _w
                in _geotiff_tags(2.5, -100.0, -40.0)}
        self.assertEqual(tags[33550][:2], (2.5, 2.5))
        # Tie point: raster (0,0) is the canvas origin, converted to metres and
        # with the northing flipped back the right way up.
        self.assertEqual(tags[33922][3:5], (-250.0, 100.0))
        self.assertIn(3857, tags[34735])


@needs_qgis
class TestQgisProject(unittest.TestCase):
    """The project file veloroute's `mix velo.map.image.from_qgis` consumes."""

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        georef.initialize()
        from puzzle_joiner.export import write_qgis_project
        cls.dir = tempfile.mkdtemp()
        cls.tif = os.path.join(cls.dir, "demo_warped.tif")
        _write_tiny_geotiff(cls.tif)
        cls.qgs = os.path.join(cls.dir, "demo_warped.qgs")
        cls.written = write_qgis_project(cls.qgs, cls.tif)

    def find_tifs(self):
        """Mirror of the Mix task's `find_tifs`, step for step.

        The real task could not be executed here — veloroute's compiled deps
        are built against a different OTP than the Elixir on this machine — so
        the pipeline is reproduced instead: checked layer-tree-layers, sources
        ending in .tif, resolved against the project's own directory.
        """
        import xml.etree.ElementTree as ET

        root = ET.parse(self.qgs).getroot()
        self.assertEqual(root.tag, "qgis")
        sources = [node.get("source")
                   for node in root.iter("layer-tree-layer")
                   if node.get("checked") == "Qt::Checked"]
        tifs = [s for s in sources if s.lower().endswith(".tif")]
        return [os.path.normpath(os.path.join(os.path.dirname(self.qgs), t))
                for t in tifs]

    def test_the_project_is_written(self):
        self.assertTrue(self.written)
        self.assertTrue(os.path.exists(self.qgs))

    def test_it_yields_exactly_the_warped_raster(self):
        self.assertEqual(self.find_tifs(), [self.tif])

    def test_the_raster_source_is_relative_to_the_project(self):
        """The task resolves against the project dir, so the pair can move."""
        import xml.etree.ElementTree as ET

        root = ET.parse(self.qgs).getroot()
        sources = [n.get("source") for n in root.iter("layer-tree-layer")]
        tif = next(s for s in sources if s.lower().endswith(".tif"))
        self.assertFalse(os.path.isabs(tif))

    def test_the_basemap_is_not_mistaken_for_a_layer_to_import(self):
        import xml.etree.ElementTree as ET

        root = ET.parse(self.qgs).getroot()
        names = {n.get("name") for n in root.iter("layer-tree-layer")}
        self.assertIn("osm", names)
        self.assertEqual(len(self.find_tifs()), 1)

    def test_a_missing_raster_writes_no_project(self):
        from puzzle_joiner.export import write_qgis_project
        out = os.path.join(self.dir, "nope.qgs")
        self.assertFalse(write_qgis_project(out, os.path.join(self.dir, "nope.tif")))


def _write_tiny_geotiff(path):
    """A 4x4 north-up GeoTIFF, enough for GDAL to call the layer valid."""
    import subprocess
    import tifffile

    plain = path + ".plain.tif"
    tifffile.imwrite(plain, np.zeros((4, 4, 4), np.uint8), photometric="rgb")
    subprocess.run(["gdal_translate", "-q", "-a_srs", georef.MAP_CRS,
                    "-a_ullr", "0", "100", "100", "0", plain, path], check=True)
    os.remove(plain)


class TestExportOptionsDialog(unittest.TestCase):
    """Choosing which files an export writes."""

    def setUp(self):
        from PySide6.QtWidgets import QApplication
        self.app = QApplication.instance() or QApplication([])

    def dialog(self, can_warp=True, selected=None):
        from puzzle_joiner.widgets import ExportOptionsDialog
        return ExportOptionsDialog(can_warp, selected)

    def test_the_georeferenced_file_is_the_only_default(self):
        dlg = self.dialog()
        dlg._apply()
        self.assertEqual(dlg.result_options,
                         {"flat": False, "uncropped": False, "layered": False,
                          "warp": True})

    def test_the_default_without_control_points_is_the_flat_file_alone(self):
        dlg = self.dialog(can_warp=False)
        dlg._apply()
        self.assertEqual(dlg.result_options,
                         {"flat": True, "uncropped": False, "layered": False,
                          "warp": False})

    def test_warp_is_unavailable_without_control_points(self):
        dlg = self.dialog(can_warp=False)
        self.assertFalse(dlg.warp_box.isEnabled())
        self.assertFalse(dlg.warp_box.isChecked())
        dlg._apply()
        self.assertFalse(dlg.result_options["warp"])

    def test_the_geo_file_does_not_depend_on_the_flat_one(self):
        """It composites the pieces itself; it no longer warps that file."""
        dlg = self.dialog()
        dlg.boxes["flat"].setChecked(True)
        dlg.boxes["flat"].setChecked(False)
        self.assertTrue(dlg.warp_box.isEnabled())
        self.assertTrue(dlg.warp_box.isChecked())
        dlg._apply()
        self.assertEqual(dlg.result_options,
                         {"flat": False, "uncropped": False, "layered": False,
                          "warp": True})

    def test_the_geo_file_alone_is_a_valid_export(self):
        dlg = self.dialog()
        for box in dlg.boxes.values():
            box.setChecked(False)
        self.assertTrue(dlg.warp_box.isChecked())
        self.assertTrue(dlg.ok_button.isEnabled())

    def test_ok_is_refused_when_nothing_at_all_is_selected(self):
        dlg = self.dialog()
        for box in dlg.boxes.values():
            box.setChecked(False)
        dlg.warp_box.setChecked(False)
        self.assertFalse(dlg.ok_button.isEnabled())
        dlg.boxes["layered"].setChecked(True)
        self.assertTrue(dlg.ok_button.isEnabled())

    def test_a_previous_selection_is_offered_again(self):
        previous = {"flat": False, "uncropped": True, "layered": False,
                    "warp": True}
        dlg = self.dialog(selected=previous)
        self.assertFalse(dlg.boxes["flat"].isChecked())
        self.assertTrue(dlg.boxes["uncropped"].isChecked())
        # Independent of the flat file, so the remembered choice stands.
        self.assertTrue(dlg.warp_box.isChecked())

    def test_cancelling_yields_no_options(self):
        dlg = self.dialog()
        dlg.reject()
        self.assertIsNone(dlg.result_options)


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
