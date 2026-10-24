import math
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import cv2
import numpy as np
import pytest
import tifffile
from PySide6.QtWidgets import QApplication

from puzzle_joiner.model import PuzzlePiece
from puzzle_joiner.widgets import PuzzleScene
from puzzle_joiner.legend import (
    SEPARATION_MARGIN, _ray_overlap_interval, anchor_local, anchor_pair,
    clone_legend_piece, make_legend_piece, map_hull, nearest_hull_point,
    upright_affine,
)
from puzzle_joiner.export import (
    _compute_optimal_rotation, _export_affines, _layer_name, _rotation_about,
    export_canvas, export_flattened_tiff, export_layered_tiff, export_order,
)
from puzzle_joiner.cache import save_layout, load_layout, load_legends


def _page(w=400, h=300, name="page_0001.png"):
    piece = PuzzlePiece()
    rng = np.random.default_rng(len(name))
    piece.image = rng.integers(0, 256, (h, w, 4), dtype=np.uint8)
    piece.image[:, :, 3] = 255
    piece.source_path = f"/tmp/{name}"
    piece.source_file = "/tmp/doc.pdf"
    piece.display_name = "Page 1"
    piece.x, piece.y = w / 2.0, h / 2.0
    return piece


def _map_piece(w=200, h=200, x=100.0, y=100.0, rot=0.0):
    piece = PuzzlePiece()
    piece.image = np.zeros((h, w, 4), dtype=np.uint8)
    piece.image[:, :, 3] = 255
    piece.x, piece.y = x, y
    piece.rotation_deg = rot
    return piece


class TestMakeLegendPiece:
    def test_carries_its_own_pixels(self):
        """Editing the page must not reach into a legend cut from it."""
        page = _page()
        legend = make_legend_piece(page, (300, 20, 80, 60))
        before = legend.image.copy()
        page.image[:, :, :] = 7
        np.testing.assert_array_equal(legend.image, before)

    def test_takes_the_region_pixels(self):
        page = _page()
        legend = make_legend_piece(page, (300, 20, 80, 60))
        assert legend.image.shape == (60, 80, 4)
        np.testing.assert_array_equal(
            legend.image[:, :, :3], page.image[20:80, 300:380, :3])

    def test_page_lasso_does_not_make_the_legend_transparent(self):
        """A lasso drawn round the map zeroes alpha elsewhere on the page."""
        page = _page()
        page.apply_crop((0, 0, 200, 200), [(0, 0), (200, 0), (200, 200), (0, 200)])
        assert page.image[40, 340, 3] == 0, "region is outside the page lasso"
        legend = make_legend_piece(page, (300, 20, 80, 60))
        assert (legend.image[:, :, 3] == 255).all()

    def test_is_flagged_and_keyed(self):
        page = _page()
        legend = make_legend_piece(page, (300, 20, 80, 60), instance=2)
        assert legend.is_legend
        assert legend.legend_rect == (300, 20, 80, 60)
        assert legend.legend_source_key == "page_0001.png"
        assert legend.layout_key() == "page_0001.png#legend2"

    def test_starts_over_the_region_it_came_from(self):
        """An untransformed page puts the legend exactly where it was drawn."""
        page = _page()
        legend = make_legend_piece(page, (300, 20, 80, 60))
        assert legend.x == pytest.approx(340.0)
        assert legend.y == pytest.approx(50.0)

    def test_lasso_is_rebased_onto_the_legend(self):
        page = _page()
        legend = make_legend_piece(
            page, (300, 20, 80, 60),
            lasso_poly=[(300, 20), (380, 20), (300, 80)])
        assert legend.crop_rect == (0, 0, 80, 60)
        assert legend.lasso_poly[1] == (80.0, 0.0)
        # Bottom-right is outside the triangle, so the mask cleared it.
        assert legend.image[55, 75, 3] == 0


class TestCloneLegendPiece:
    def test_clone_has_an_independent_buffer(self):
        page = _page()
        legend = make_legend_piece(page, (300, 20, 80, 60), instance=1)
        clone = clone_legend_piece(legend, instance=2)
        assert clone.layout_key() != legend.layout_key()
        assert clone.image is not legend.image
        clone.apply_crop((0, 0, 80, 60), [(0, 0), (40, 0), (0, 30)])
        assert (legend.image[:, :, 3] == 255).all()

    def test_clone_keeps_the_region_and_lasso(self):
        page = _page()
        legend = make_legend_piece(
            page, (300, 20, 80, 60),
            lasso_poly=[(300, 20), (380, 20), (300, 80)], instance=1)
        clone = clone_legend_piece(legend, instance=2)
        assert clone.legend_rect == legend.legend_rect
        assert clone.lasso_poly == legend.lasso_poly
        assert clone.image[55, 75, 3] == 0


class TestLayoutKey:
    def test_page_key_is_its_basename(self):
        assert _page().layout_key() == "page_0001.png"

    def test_instances_of_one_page_stay_distinct(self):
        page = _page()
        keys = {make_legend_piece(page, (300, 20, 80, 60), instance=i).layout_key()
                for i in (1, 2, 3)}
        assert len(keys) == 3
        assert page.layout_key() not in keys


class TestMapHull:
    def test_legends_are_left_out(self):
        page = _page()
        m = _map_piece()
        legend = make_legend_piece(page, (300, 20, 80, 60))
        legend.x, legend.y = 5000, 5000
        hull = map_hull([m, legend])
        assert hull[:, 0].max() <= 200
        assert hull[:, 1].max() <= 200

    def test_none_without_a_map(self):
        page = _page()
        assert map_hull([make_legend_piece(page, (300, 20, 80, 60))]) is None


class TestAnchor:
    def _legend_at(self, x, y, rot=0.0):
        page = _page()
        legend = make_legend_piece(page, (300, 20, 80, 60))
        legend.x, legend.y = x, y
        legend.rotation_deg = rot
        return legend

    def test_legend_to_the_right_anchors_on_its_left_edge(self):
        m = _map_piece()  # covers world (0,0)-(200,200)
        legend = self._legend_at(400, 100)
        assert anchor_local(legend, map_hull([m]))[0] == pytest.approx(0.0)

    def test_legend_below_anchors_on_its_top_edge(self):
        m = _map_piece()
        legend = self._legend_at(100, 500)
        assert anchor_local(legend, map_hull([m]))[1] == pytest.approx(0.0)

    def test_falls_back_to_the_centre_without_a_map(self):
        legend = self._legend_at(400, 100)
        anchor = anchor_local(legend, None)
        assert anchor[0] == pytest.approx(40.0)
        assert anchor[1] == pytest.approx(30.0)

    def test_anchor_is_the_nearest_point_of_the_whole_outline(self):
        """Sampling edges, not just corners: a side-on legend anchors mid-edge."""
        m = _map_piece()
        legend = self._legend_at(400, 100)
        hull = map_hull([m])
        anchor = anchor_local(legend, hull)
        local = legend.get_outline_points()
        world = np.array([anchor[0] + 400 - 40, anchor[1] + 100 - 30])
        best = nearest_hull_point(world.astype(np.float32), hull)
        chosen = float(np.linalg.norm(world - best))
        # No outline vertex can be closer than the point the search picked.
        for px, py in local:
            vertex = np.array([px + 400 - 40, py + 100 - 30], dtype=np.float32)
            assert chosen <= float(np.linalg.norm(
                vertex - nearest_hull_point(vertex, hull))) + 1e-3

    def test_anchor_pair_spans_legend_to_map(self):
        m = _map_piece()
        legend = self._legend_at(400, 100)
        anchor, target = anchor_pair(legend, [m, legend])
        assert anchor[0] == pytest.approx(360.0)   # legend's left edge
        assert target[0] == pytest.approx(200.0)   # map's right edge

    def test_anchor_pair_is_none_without_a_map(self):
        legend = self._legend_at(400, 100)
        assert anchor_pair(legend, [legend]) is None


class TestUprightAffine:
    def test_removes_rotation_and_pins_the_anchor(self):
        legend = _page(80, 60)
        legend.is_legend = True
        legend.rotation_deg = 37.0
        legend.scale = 1.5
        M = legend.get_affine_matrix()
        anchor = np.array([0.0, 0.0], dtype=np.float32)

        before = M @ np.array([anchor[0], anchor[1], 1.0])
        up = upright_affine(M, anchor)
        after = up @ np.array([anchor[0], anchor[1], 1.0])

        np.testing.assert_allclose(after, before, atol=1e-9)
        assert up[0, 1] == pytest.approx(0.0)
        assert up[1, 0] == pytest.approx(0.0)
        assert up[0, 0] == pytest.approx(1.5)
        assert up[1, 1] == pytest.approx(1.5)


class TestExportAffines:
    def _rotated_map(self, rot=20.0):
        a = _map_piece(200, 50, 100, 25, rot)
        b = _map_piece(200, 50, rot=rot)
        b.x = a.x + 210 * math.cos(math.radians(rot))
        b.y = a.y + 210 * math.sin(math.radians(rot))
        return [a, b]

    def _legend(self, rot=0.0):
        page = _page()
        legend = make_legend_piece(page, (300, 20, 80, 60))
        legend.x, legend.y = 600, 200
        legend.rotation_deg = rot
        return legend

    def test_legend_comes_out_upright(self):
        pieces = self._rotated_map() + [self._legend(rot=35.0)]
        M = _export_affines(pieces)[-1]
        assert M[0, 1] == pytest.approx(0.0, abs=1e-9)
        assert M[1, 0] == pytest.approx(0.0, abs=1e-9)

    def test_legend_upright_even_with_no_map_rotation(self):
        """The piece's own canvas rotation goes too, not just the global one."""
        pieces = [_map_piece(), _map_piece(x=350.0), self._legend(rot=35.0)]
        assert abs(_compute_optimal_rotation(pieces[:2])[0]) < 0.1
        M = _export_affines(pieces)[-1]
        assert M[0, 1] == pytest.approx(0.0, abs=1e-9)
        assert M[1, 0] == pytest.approx(0.0, abs=1e-9)

    def test_legend_upright_as_the_only_other_piece(self):
        """One map page and one legend — still no rotation left on the legend."""
        pieces = [_map_piece(), self._legend(rot=35.0)]
        M = _export_affines(pieces)[-1]
        assert M[0, 1] == pytest.approx(0.0, abs=1e-9)

    def test_anchor_lands_where_the_plain_rotation_would_put_it(self):
        """Uprighting only spins the legend about the anchor — it never moves it."""
        map_pieces = self._rotated_map()
        legend = self._legend(rot=35.0)
        pieces = map_pieces + [legend]
        anchor = anchor_local(legend, map_hull(map_pieces))
        pt = np.array([anchor[0], anchor[1], 1.0])

        # What the legend's matrix would be if it just rode the global rotation.
        angle, cx, cy = _compute_optimal_rotation(map_pieces)
        R3 = _rotation_about(angle, cx, cy)
        rode_along = (R3 @ np.vstack([legend.get_affine_matrix(), [0, 0, 1]]))[:2]

        upright = _export_affines(pieces)[-1]
        np.testing.assert_allclose(upright @ pt, rode_along @ pt, atol=1e-6)
        # ...and it really did rotate: any other point moved.
        corner = np.array([80.0, 60.0, 1.0])
        assert np.linalg.norm(upright @ corner - rode_along @ corner) > 1.0

    def test_distance_to_the_map_is_preserved(self):
        pieces = self._rotated_map() + [self._legend(rot=35.0)]
        legend = pieces[-1]
        canvas_gap = float(np.linalg.norm(
            np.subtract(*anchor_pair(legend, pieces))))

        affines = _export_affines(pieces)
        hull = []
        for piece, M in zip(pieces[:-1], affines[:-1]):
            o = piece.get_outline_points()
            hull.append(o @ M[:, :2].T + M[:, 2])
        hull = cv2.convexHull(np.vstack(hull).astype(np.float32)).reshape(-1, 2)

        anchor = anchor_local(legend, map_hull(pieces[:-1]))
        world = affines[-1] @ np.array([anchor[0], anchor[1], 1.0])
        export_gap = float(np.linalg.norm(
            world - nearest_hull_point(world.astype(np.float32), hull)))
        assert export_gap == pytest.approx(canvas_gap, abs=0.5)

    def test_legends_do_not_sway_the_optimal_rotation(self):
        """A legend is not part of the shape being minimised."""
        map_pieces = self._rotated_map()
        legend = self._legend(rot=70.0)
        legend.x, legend.y = 4000, -4000
        without = _export_affines(map_pieces)
        with_legend = _export_affines(map_pieces + [legend])
        for a, b in zip(without, with_legend[:-1]):
            np.testing.assert_allclose(a, b, atol=1e-9)

    def test_map_pieces_still_get_straightened(self):
        pieces = self._rotated_map() + [self._legend(rot=35.0)]
        M = _export_affines(pieces)[0]
        # 20 degrees of map rotation undone leaves an axis-aligned page.
        assert math.degrees(math.atan2(M[1, 0], M[0, 0])) == pytest.approx(0, abs=2)

    def test_single_map_piece_is_untouched(self):
        p = _map_piece()
        np.testing.assert_allclose(_export_affines([p])[0], p.get_affine_matrix())


class TestExportOrder:
    def test_legends_composite_last(self):
        a, b = _map_piece(), _map_piece()
        legend = make_legend_piece(_page(), (300, 20, 80, 60))
        assert export_order([legend, a, b]) == [a, b, legend]

    def test_map_order_is_kept(self):
        pieces = [_map_piece() for _ in range(4)]
        assert export_order(pieces) == pieces

    def test_layer_name_names_the_source_page(self):
        legend = make_legend_piece(_page(), (300, 20, 80, 60), instance=1)
        assert _layer_name(legend) == "Legend 1 (page_0001.png)"


class TestLegendPersistence:
    def _saved(self, tmp_path, legend_kwargs=None):
        page = _page()
        legend = make_legend_piece(page, (300, 20, 80, 60),
                                   **(legend_kwargs or {"instance": 1}))
        legend.x, legend.y = 1234.5, -678.25
        legend.rotation_deg = 42.0
        legend.scale = 1.25
        save_layout(str(tmp_path), [page, legend], [])
        return page, legend

    def test_legend_definition_round_trips(self, tmp_path):
        self._saved(tmp_path)
        definitions = load_legends(str(tmp_path))
        assert len(definitions) == 1
        assert definitions[0]["rect"] == [300, 20, 80, 60]
        assert definitions[0]["source_key"] == "page_0001.png"
        assert definitions[0]["instance"] == 1

    def test_pages_are_not_reported_as_legends(self, tmp_path):
        page = _page()
        save_layout(str(tmp_path), [page], [])
        assert load_legends(str(tmp_path)) == []

    def test_transform_is_restored_onto_a_recut_legend(self, tmp_path):
        page, saved = self._saved(tmp_path)
        fresh_page = _page()
        definition = load_legends(str(tmp_path))[0]
        rebuilt = make_legend_piece(
            fresh_page, definition["rect"], None,
            instance=definition["instance"],
            display_name=definition["display_name"])

        applied, _ = load_layout(str(tmp_path), [fresh_page, rebuilt])
        assert applied
        assert rebuilt.x == pytest.approx(saved.x)
        assert rebuilt.y == pytest.approx(saved.y)
        assert rebuilt.rotation_deg == pytest.approx(42.0)
        assert rebuilt.scale == pytest.approx(1.25)

    def test_a_page_does_not_pick_up_its_legend_entry(self, tmp_path):
        """Both share a source path, so only the instance suffix tells them apart."""
        page, legend = self._saved(tmp_path)
        fresh_page = _page()
        load_layout(str(tmp_path), [fresh_page])
        assert fresh_page.x == pytest.approx(page.x)
        assert fresh_page.rotation_deg == pytest.approx(0.0)

    def test_two_instances_restore_separately(self, tmp_path):
        page = _page()
        first = make_legend_piece(page, (300, 20, 80, 60), instance=1)
        second = make_legend_piece(page, (300, 20, 80, 60), instance=2)
        first.x, second.x = 10.0, 20.0
        save_layout(str(tmp_path), [page, first, second], [])

        assert len(load_legends(str(tmp_path))) == 2
        a = make_legend_piece(page, (300, 20, 80, 60), instance=1)
        b = make_legend_piece(page, (300, 20, 80, 60), instance=2)
        load_layout(str(tmp_path), [page, a, b])
        assert a.x == pytest.approx(10.0)
        assert b.x == pytest.approx(20.0)

    def test_lasso_on_a_legend_round_trips(self, tmp_path):
        page = _page()
        legend = make_legend_piece(
            page, (300, 20, 80, 60),
            lasso_poly=[(300, 20), (380, 20), (300, 80)], instance=1)
        save_layout(str(tmp_path), [page, legend], [])

        rebuilt = make_legend_piece(page, (300, 20, 80, 60), None, instance=1)
        load_layout(str(tmp_path), [page, rebuilt])
        assert rebuilt.lasso_poly == legend.lasso_poly
        assert rebuilt.image[55, 75, 3] == 0

    def test_snap_pairs_use_layout_keys(self, tmp_path):
        a = _page(name="page_0001.png")
        b = _page(name="page_0002.png")
        save_layout(str(tmp_path), [a, b], [(a, b)])
        _, pairs = load_layout(str(tmp_path), [a, b])
        assert pairs == [["page_0001.png", "page_0002.png"]]


def _outline_world(piece, M):
    return piece.get_outline_points() @ M[:, :2].T + M[:, 2]


def _overlap_area(a, b):
    area, _ = cv2.intersectConvexConvex(a.astype(np.float32), b.astype(np.float32))
    return area


class TestSeparation:
    """Squaring a legend up changes its footprint, so it can hit a neighbour."""

    def _map(self):
        return [_map_piece(400, 100, 200, 50), _map_piece(400, 100, 610, 50)]

    def _legend(self, x, y, rot, instance):
        legend = make_legend_piece(_page(), (300, 20, 80, 60), instance=instance)
        legend.x, legend.y = x, y
        legend.rotation_deg = rot
        return legend

    def _shapes(self, pieces):
        affines = _export_affines(pieces)
        return {p.legend_instance: _outline_world(p, M)
                for p, M in zip(pieces, affines) if p.is_legend}

    def _push(self, pieces):
        """Per-instance distance each legend was moved by the separation pass."""
        affines = _export_affines(pieces)
        hull = map_hull([p for p in pieces if not p.is_legend])
        pushed = {}
        for piece, M in zip(pieces, affines):
            if not piece.is_legend:
                continue
            pure = upright_affine(piece.get_affine_matrix(),
                                  anchor_local(piece, hull))
            pushed[piece.legend_instance] = float(
                np.linalg.norm(M[:, 2] - pure[:, 2]))
        return pushed

    def test_clear_legends_are_not_touched(self):
        """Stacked well apart — the pass must be a no-op."""
        pieces = self._map() + [self._legend(300, 250, 40, 1),
                                self._legend(300, 340, 40, 2)]
        assert self._push(pieces) == {1: 0.0, 2: 0.0}

    def test_adjacent_legends_stop_overlapping(self):
        pieces = self._map() + [self._legend(300, 300, 30, 1),
                                self._legend(360, 255, 30, 2)]
        shapes = self._shapes(pieces)
        assert _overlap_area(shapes[1], shapes[2]) == 0

    def test_legends_at_different_angles_stop_overlapping(self):
        """Different rotations give different anchors, so the drift is worse."""
        pieces = self._map() + [self._legend(300, 300, 30, 1),
                                self._legend(395, 300, -25, 2)]
        shapes = self._shapes(pieces)
        assert _overlap_area(shapes[1], shapes[2]) == 0

    def test_the_legend_nearest_the_map_keeps_its_place(self):
        pieces = self._map() + [self._legend(300, 300, 30, 1),
                                self._legend(360, 255, 30, 2)]
        push = self._push(pieces)
        assert push[2] == 0.0, "nearest the map, so it should not yield"
        assert push[1] > 0.0

    def test_the_push_is_the_smallest_one_that_clears(self):
        """A 20x15 corner overlap clears with a 15 px nudge plus the margin."""
        pieces = self._map() + [self._legend(300, 300, 30, 1),
                                self._legend(360, 255, 30, 2)]
        assert self._push(pieces)[1] == pytest.approx(15.0 + SEPARATION_MARGIN,
                                                      abs=0.1)

    def test_sideways_beats_shoving_one_past_the_other(self):
        """Side by side, a small sideways nudge must win over a 60 px shove."""
        pieces = self._map() + [self._legend(300, 300, 30, 1),
                                self._legend(395, 300, -25, 2)]
        push = self._push(pieces)
        assert max(push.values()) < 60.0

    def test_a_legend_on_the_map_is_pushed_off_it(self):
        pieces = self._map() + [self._legend(300, 80, 20, 1)]
        affines = _export_affines(pieces)
        hull = map_hull(pieces[:2])
        legend = _outline_world(pieces[-1], affines[-1])
        assert _overlap_area(legend, hull) == 0

    def test_a_legend_on_the_map_leaves_by_the_nearest_edge(self):
        """Clearing a 100-tall map from the middle is ~51 px, not ~111."""
        pieces = self._map() + [self._legend(300, 80, 20, 1)]
        assert self._push(pieces)[1] < 60.0

    def test_three_legends_all_end_up_clear(self):
        pieces = self._map() + [self._legend(300, 300, 30, 1),
                                self._legend(360, 255, 30, 2),
                                self._legend(330, 285, 15, 3)]
        shapes = self._shapes(pieces)
        hull = map_hull(pieces[:2])
        for a in (1, 2, 3):
            assert _overlap_area(shapes[a], hull) == 0
            for b in (1, 2, 3):
                if a < b:
                    assert _overlap_area(shapes[a], shapes[b]) == 0, f"{a} vs {b}"

    def test_result_does_not_depend_on_piece_order(self):
        """Resolution order comes from distance and instance, not list position."""
        maps = self._map()
        first = self._legend(300, 300, 30, 1)
        second = self._legend(360, 255, 30, 2)
        forward = self._shapes(maps + [first, second])
        backward = self._shapes(maps + [second, first])
        for instance in (1, 2):
            np.testing.assert_allclose(forward[instance], backward[instance],
                                       atol=1e-9)

    def test_separation_still_leaves_them_upright(self):
        """A nudge is a translation, so it cannot reintroduce rotation."""
        pieces = self._map() + [self._legend(300, 300, 30, 1),
                                self._legend(360, 255, 30, 2)]
        for piece, M in zip(pieces, _export_affines(pieces)):
            if piece.is_legend:
                assert M[0, 1] == pytest.approx(0.0, abs=1e-9)
                assert M[1, 0] == pytest.approx(0.0, abs=1e-9)


class TestRayOverlapInterval:
    def _square(self, cx, cy, half=10.0):
        return np.array([[cx - half, cy - half], [cx + half, cy - half],
                         [cx + half, cy + half], [cx - half, cy + half]])

    def test_overlapping_squares_report_the_exit(self):
        a = self._square(0, 0)
        b = self._square(5, 0)
        lo, hi = _ray_overlap_interval(a, b, np.array([1.0, 0.0]))
        # Moving a to the right, it clears b once it has travelled 25.
        assert hi == pytest.approx(25.0)
        assert lo == pytest.approx(-15.0)

    def test_disjoint_squares_off_the_ray_report_nothing(self):
        a = self._square(0, 0)
        b = self._square(0, 500)
        assert _ray_overlap_interval(a, b, np.array([1.0, 0.0])) is None

    def test_touching_squares_meet_at_a_point(self):
        a = self._square(0, 0)
        b = self._square(20, 0)
        lo, hi = _ray_overlap_interval(a, b, np.array([1.0, 0.0]))
        assert lo <= 0.0 <= hi


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def scene_with_legend(qapp):
    """A two-page map with one legend parked to the right of it."""
    scene = PuzzleScene()
    a = _map_piece()
    b = _map_piece(x=350.0)
    legend = make_legend_piece(_page(), (300, 20, 80, 60), instance=1)
    legend.x, legend.y = 900, 100
    for piece in (a, b, legend):
        piece.rebuild_display_pixmap()
        scene.add_piece(piece)
    yield scene, a, b, legend


class TestSceneGating:
    def test_a_legend_is_never_a_neighbour(self, scene_with_legend):
        scene, a, b, legend = scene_with_legend
        legend.x, legend.y = b.x + 10, b.y  # right on top of a map page
        assert legend not in scene.get_neighbors(b)
        assert a in scene.get_neighbors(b)

    def test_a_legend_has_no_neighbours_of_its_own(self, scene_with_legend):
        scene, a, b, legend = scene_with_legend
        legend.x, legend.y = b.x + 10, b.y
        assert scene.get_neighbors(legend) == []

    def test_a_locked_legend_still_drags_alone(self, scene_with_legend):
        scene, a, b, legend = scene_with_legend
        for piece in (a, b, legend):
            piece.is_locked = True
        assert scene.get_drag_group(legend) == [legend]
        assert set(scene.get_drag_group(a)) == {a, b}

    def test_legends_sit_above_the_map(self, scene_with_legend):
        scene, a, b, legend = scene_with_legend
        assert scene.piece_items[legend].zValue() > scene.piece_items[a].zValue()

    def test_anchor_indicator_tracks_the_selection(self, scene_with_legend):
        scene, a, b, legend = scene_with_legend
        scene.piece_items[legend].setSelected(True)
        assert scene._anchor_item is not None
        scene.piece_items[legend].setSelected(False)
        scene.piece_items[a].setSelected(True)
        assert scene._anchor_item is None


class TestLegendInExportedFile:
    def _pieces(self):
        """A map 20 degrees off-axis, plus a legend at an unrelated angle.

        The angles differ on purpose: the global rotation alone would leave the
        legend at 35 degrees, so only the uprighting can square it up.
        """
        a = _map_piece(300, 80, 150, 40, rot=20.0)
        b = _map_piece(300, 80, rot=20.0)
        b.x = a.x + 310 * math.cos(math.radians(20))
        b.y = a.y + 310 * math.sin(math.radians(20))
        page = _page()
        legend = make_legend_piece(page, (300, 20, 80, 60), instance=1)
        legend.x, legend.y = 400, 500
        legend.rotation_deg = 55.0
        return [a, b, legend]

    def test_legend_layer_is_axis_aligned(self, tmp_path):
        """An upright 80x60 legend occupies exactly 80x60 opaque pixels."""
        pieces = self._pieces()
        out = tmp_path / "layered.tif"
        export_layered_tiff(pieces, str(out))
        with tifffile.TiffFile(str(out)) as tif:
            assert len(tif.pages) == 3
            # Legends are written last.
            legend_layer = tif.pages[2].asarray()
        ys, xs = np.nonzero(legend_layer[:, :, 3] > 128)
        assert xs.max() - xs.min() + 1 == pytest.approx(80, abs=1)
        assert ys.max() - ys.min() + 1 == pytest.approx(60, abs=1)

    def test_riding_the_global_rotation_alone_would_leave_it_skewed(self):
        """Guards the test above: without uprighting the legend stays at an angle."""
        pieces = self._pieces()
        legend = pieces[-1]
        angle, cx, cy = _compute_optimal_rotation(pieces[:-1])
        rode_along = (_rotation_about(angle, cx, cy)
                      @ np.vstack([legend.get_affine_matrix(), [0, 0, 1]]))[:2]
        corners = legend.get_outline_points() @ rode_along[:, :2].T + rode_along[:, 2]
        span = corners.max(axis=0) - corners.min(axis=0)
        assert span[0] > 90 and span[1] > 75, "should be a rotated, larger bbox"

    def test_map_layers_are_straightened_by_the_global_rotation(self, tmp_path):
        pieces = self._pieces()
        out = tmp_path / "layered.tif"
        export_layered_tiff(pieces, str(out))
        with tifffile.TiffFile(str(out)) as tif:
            page_layer = tif.pages[0].asarray()
        ys, xs = np.nonzero(page_layer[:, :, 3] > 128)
        assert xs.max() - xs.min() + 1 == pytest.approx(300, abs=3)
        assert ys.max() - ys.min() + 1 == pytest.approx(80, abs=3)

    def test_flattened_export_includes_the_legend(self, tmp_path):
        pieces = self._pieces()
        out = tmp_path / "flat.tif"
        export_flattened_tiff(pieces, str(out))
        got = tifffile.imread(str(out))
        affines = _export_affines(export_order(pieces))
        ox, oy, w, h = export_canvas(export_order(pieces), affines)
        assert got.shape[:2] == (h, w)
        # The legend's own centre must be opaque in the composite.
        M = affines[-1]
        cx, cy = M @ np.array([40.0, 30.0, 1.0])
        assert got[int(cy - oy), int(cx - ox), 3] == 255
