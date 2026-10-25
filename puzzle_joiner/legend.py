"""Legends: page regions placed beside the assembled map, kept upright on export.

A legend is an ordinary `PuzzlePiece` with `is_legend` set. It carries its own
copy of the pixels it was cut from, so editing the source page cannot punch holes
in it, and the same region can be included any number of times.

On the canvas a legend rotates freely, which is what lets a user line it up
against a map that sits at an arbitrary angle. The export throws that rotation
away: it pivots the legend about its *anchor* — the point of the legend nearest
the assembled map — and emits it axis-aligned. Pivoting on the anchor is what
keeps the legend as close to the map as the user placed it, whatever rotation the
export applies to the map itself.
"""

import math

import cv2
import numpy as np

from .model import PuzzlePiece

# Points sampled along each outline edge when looking for the anchor. Sampling
# matters because the nearest point of a legend to the map is usually in the
# middle of an edge, not on one of the four corners.
_ANCHOR_SAMPLES = 64

# Clearance left between two shapes the separation pass had to pull apart.
SEPARATION_MARGIN = 1.0


def make_legend_piece(source: PuzzlePiece, legend_rect: tuple,
                      lasso_poly: list = None, instance: int = 1,
                      display_name: str = "") -> PuzzlePiece:
    """Cut a legend instance out of `source`.

    `legend_rect` and `lasso_poly` are in the source page's full-image pixel
    coords — the space the cut dialog reports. The result owns its pixels and is
    positioned over the region it came from, so it starts where the user drew it.
    """
    x, y, w, h = (int(v) for v in legend_rect)
    piece = PuzzlePiece()
    piece.is_legend = True
    piece.source_path = source.source_path
    piece.source_file = source.source_file
    piece.legend_source_key = source.layout_key()
    piece.legend_rect = (x, y, w, h)
    piece.legend_instance = instance
    piece.display_name = display_name or f"Legend {instance}"

    # Take the alpha the page had before any lasso masking: a lasso drawn around
    # the map must not make the legend transparent.
    img = source.image[y:y + h, x:x + w].copy()
    if source._base_alpha is not None:
        img[:, :, 3] = source._base_alpha[y:y + h, x:x + w]
    piece.image = img

    if lasso_poly:
        piece.apply_crop((0, 0, w, h),
                         [(px - x, py - y) for px, py in lasso_poly])

    # Place it where it was cut from, in the source page's own frame.
    M = source.get_affine_matrix()
    ox, oy = source.crop_rect[:2] if source.crop_rect is not None else (0, 0)
    local = np.array([[[x + w / 2.0 - ox, y + h / 2.0 - oy]]], dtype=np.float32)
    world = cv2.transform(local, M.astype(np.float32)).reshape(2)
    piece.x, piece.y = float(world[0]), float(world[1])
    piece.rotation_deg = source.rotation_deg
    piece.scale = source.scale
    return piece


def clone_legend_piece(legend: PuzzlePiece, instance: int,
                       display_name: str = "") -> PuzzlePiece:
    """Duplicate a legend instance, offset enough to be visible underneath."""
    piece = PuzzlePiece()
    for attr in ("source_path", "source_file", "legend_source_key", "legend_rect",
                 "rotation_deg", "scale"):
        setattr(piece, attr, getattr(legend, attr))
    piece.is_legend = True
    piece.legend_instance = instance
    piece.display_name = display_name or f"Legend {instance}"
    # Its own buffer: `apply_crop` writes the alpha channel in place.
    piece.image = legend.image.copy()
    if legend._base_alpha is not None:
        piece.image[:, :, 3] = legend._base_alpha
    if legend.lasso_poly is not None:
        piece.apply_crop(legend.crop_rect, legend.lasso_poly)
    else:
        piece.crop_rect = legend.crop_rect
    w, h = piece._cropped_size()
    piece.x = legend.x + w * 0.1
    piece.y = legend.y + h * 0.1
    return piece


def transformed_hull(pairs) -> np.ndarray:
    """Convex hull of (piece, affine) pairs' opaque extents, or None if empty."""
    pts = []
    for piece, M in pairs:
        outline = piece.get_outline_points().reshape(-1, 1, 2)
        pts.append(cv2.transform(outline, M.astype(np.float32)).reshape(-1, 2))
    if not pts:
        return None
    all_pts = np.vstack(pts).astype(np.float32)
    if len(all_pts) < 3:
        return all_pts
    return cv2.convexHull(all_pts).reshape(-1, 2)


def transformed_outline(piece: PuzzlePiece, M: np.ndarray) -> np.ndarray:
    """A piece's opaque outline in world coords, concavities and all."""
    pts = piece.get_outline_points().reshape(-1, 1, 2)
    return cv2.transform(pts, M.astype(np.float32)).reshape(-1, 2)


def map_outlines(pieces) -> list:
    """Each non-legend piece's outline in world coords. [] without a map.

    Outlines, and not hulls. One hull over the whole map fills in every bend of
    the road it follows; a hull per page still came out two to three times the
    area the page covers. Legends pushed clear of a shape that size finished up
    to 120 m from the map they belong to, having been moved as much as 250 m to
    get there. A legend only has to clear the map, so the map is what it is
    measured against.
    """
    return [transformed_outline(p, p.get_affine_matrix())
            for p in pieces
            if not p.is_legend and len(p.get_outline_points()) >= 3]


def convex_parts(polygon: np.ndarray) -> list:
    """A simple polygon split into triangles, by ear clipping.

    `_ray_overlap_interval` clips a ray against a Minkowski difference and
    needs both shapes convex. A triangle always is, and together they cover the
    outline exactly, so nothing is fattened to buy that convexity.
    """
    pts = [(float(x), float(y)) for x, y in polygon]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts.pop()
    if len(pts) < 3:
        return []
    if _signed_area(pts) < 0.0:
        pts.reverse()

    remaining = list(range(len(pts)))
    triangles = []
    # Each pass clips at least one ear off a simple polygon, so the count is a
    # termination guard rather than a tuning knob.
    for _ in range(len(pts)):
        if len(remaining) <= 3:
            break
        for k in range(len(remaining)):
            prev, ear, nxt = (remaining[k - 1], remaining[k],
                              remaining[(k + 1) % len(remaining)])
            a, b, c = pts[prev], pts[ear], pts[nxt]
            if _cross2(a, b, c) <= 0.0:
                continue  # reflex corner, not an ear
            if any(_in_triangle(pts[j], a, b, c)
                   for j in remaining if j not in (prev, ear, nxt)):
                continue  # another vertex is inside, so it is not an ear
            triangles.append(np.array([a, b, c], dtype=np.float32))
            remaining.pop(k)
            break
        else:
            break  # nothing clippable left; keep what the fan gave us
    if len(remaining) >= 3:
        triangles.append(np.array([pts[i] for i in remaining[:3]], dtype=np.float32))
        for k in range(3, len(remaining)):
            triangles.append(np.array(
                [pts[remaining[0]], pts[remaining[k - 1]], pts[remaining[k]]],
                dtype=np.float32))
    return triangles


def _signed_area(pts) -> float:
    return sum((pts[i][0] * pts[(i + 1) % len(pts)][1]
                - pts[(i + 1) % len(pts)][0] * pts[i][1])
               for i in range(len(pts))) / 2.0


def _cross2(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _in_triangle(p, a, b, c) -> bool:
    d1, d2, d3 = _cross2(a, b, p), _cross2(b, c, p), _cross2(c, a, p)
    return (d1 >= 0 and d2 >= 0 and d3 >= 0) or (d1 <= 0 and d2 <= 0 and d3 <= 0)


def _usable_hulls(hulls) -> list:
    """Drop the shapes with too few points to enclose anything."""
    return [h for h in hulls if h is not None and len(h) >= 3]


def _contours(hulls):
    return [h.reshape(-1, 1, 2).astype(np.float32) for h in hulls]


def _depth_in_map(point, contours) -> float:
    """Signed distance into the map: positive inside any piece, else negative.

    Taking the maximum is what makes a list of pieces behave as one map — a
    point inside any of them is inside, and outside them all the largest value
    is the one nearest a boundary.
    """
    return max(cv2.pointPolygonTest(c, point, True) for c in contours)


def _densify(pts: np.ndarray, per_edge: int = _ANCHOR_SAMPLES) -> np.ndarray:
    """Sample a closed polygon's edges so a nearest point can land mid-edge."""
    if len(pts) < 2:
        return pts.astype(np.float32)
    steps = np.linspace(0.0, 1.0, per_edge, endpoint=False).reshape(-1, 1)
    rolled = np.roll(pts, -1, axis=0)
    edges = [a + (b - a) * steps for a, b in zip(pts, rolled)]
    return np.vstack(edges).astype(np.float32)


def anchor_local(legend: PuzzlePiece, hulls) -> np.ndarray:
    """The legend-local point to pivot on: the one nearest the assembled map.

    Falls back to the centre of the legend's outline when there is no map.
    Distances are measured in canvas space; the export's global rotation is
    rigid, so the winning point is the same one in export space.
    """
    local = legend.get_outline_points()
    hulls = _usable_hulls(hulls or [])
    if not hulls:
        return local.mean(axis=0)
    dense = _densify(local)
    M = legend.get_affine_matrix().astype(np.float32)
    world = cv2.transform(dense.reshape(-1, 1, 2), M).reshape(-1, 2)
    contours = _contours(hulls)
    # Depth is positive inside the map, so negating orders the candidates as:
    # deepest inside, then nearest outside.
    dists = [-_depth_in_map((float(px), float(py)), contours)
             for px, py in world]
    return dense[int(np.argmin(dists))]


def _nearest_on_segment(p, a, b):
    """Closest point to p on segment ab."""
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-12:
        return a
    t = min(max(float((p - a) @ ab) / denom, 0.0), 1.0)
    return a + ab * t


def nearest_hull_point(point: np.ndarray, hull: np.ndarray) -> np.ndarray:
    """Closest point to `point` on a hull polygon's boundary."""
    best = None
    best_d = float("inf")
    rolled = np.roll(hull, -1, axis=0)
    for a, b in zip(hull, rolled):
        c = _nearest_on_segment(point, a, b)
        d = float(np.sum((c - point) ** 2))
        if d < best_d:
            best_d = d
            best = c
    return best


def nearest_map_point(point: np.ndarray, hulls) -> np.ndarray:
    """Closest point to `point` on the boundary of any of the map's pieces."""
    best = None
    best_d = float("inf")
    for hull in hulls:
        candidate = nearest_hull_point(point, hull)
        d = float(np.sum((candidate - point) ** 2))
        if d < best_d:
            best_d = d
            best = candidate
    return best


def anchor_pair(legend: PuzzlePiece, pieces) -> tuple:
    """World-space (legend anchor, nearest map point) for a legend, or None.

    Used by the canvas to show which point the export will pivot on.
    """
    hulls = map_outlines(pieces)
    if not hulls:
        return None
    local = anchor_local(legend, hulls)
    M = legend.get_affine_matrix().astype(np.float32)
    world = cv2.transform(local.reshape(1, 1, 2).astype(np.float32), M).reshape(2)
    return world, nearest_map_point(world, hulls)


def _outward_direction(anchor_world, hulls, legend_pts) -> tuple:
    """(unit vector away from the map, whether the legend starts on top of it).

    Normally that is the direction from the map's nearest point out to the
    anchor. When the anchor is *inside* the map the legend already overlaps it,
    and the shortest way clear is towards the nearest edge — the opposite sign.
    """
    hulls = _usable_hulls(hulls or [])
    if hulls:
        point = (float(anchor_world[0]), float(anchor_world[1]))
        inside = _depth_in_map(point, _contours(hulls)) >= 0
        near = nearest_map_point(anchor_world.astype(np.float32), hulls)
        away = (near - anchor_world) if inside else (anchor_world - near)
        length = float(np.linalg.norm(away))
        if length > 1e-6:
            return away / length, inside
        away = legend_pts.mean(axis=0) - np.vstack(hulls).mean(axis=0)
        length = float(np.linalg.norm(away))
        if length > 1e-6:
            return away / length, inside
        return np.array([1.0, 0.0]), inside
    return np.array([1.0, 0.0]), False


def _ray_overlap_interval(moving, static, direction):
    """Range of t for which `moving + t*direction` meets `static`, or None.

    `moving + t*d` meets `static` exactly when `t*d` lies in the Minkowski
    difference `static (-) moving`. That set is convex, so the overlapping t form
    a single interval, which falls out of clipping the ray against its edges.
    """
    diff = (static[:, None, :] - moving[None, :, :]).reshape(-1, 2)
    if len(diff) < 3:
        return None
    poly = cv2.convexHull(diff.astype(np.float32)).reshape(-1, 2).astype(np.float64)
    if len(poly) < 3:
        return None
    centre = poly.mean(axis=0)
    t_lo, t_hi = -math.inf, math.inf
    for p, q in zip(poly, np.roll(poly, -1, axis=0)):
        edge = q - p
        normal = np.array([edge[1], -edge[0]])
        if normal @ (centre - p) > 0:
            normal = -normal              # point it out of the polygon
        # Inside this edge means normal . x <= offset, with x = t * direction.
        offset = float(normal @ p)
        denom = float(normal @ direction)
        if abs(denom) < 1e-12:
            if offset < 0:
                return None               # ray runs parallel and outside
            continue
        t = offset / denom
        if denom > 0:
            t_hi = min(t_hi, t)
        else:
            t_lo = max(t_lo, t)
    if t_lo > t_hi:
        return None
    return t_lo, t_hi


def _clear_distance(moving, obstacles, direction) -> float:
    """Smallest t >= 0 putting `moving + t*direction` clear of every obstacle."""
    spans = []
    for obstacle in obstacles:
        span = _ray_overlap_interval(moving, obstacle, direction)
        if span is not None and span[1] >= 0:
            spans.append(span)
    t = 0.0
    moved = True
    while moved:
        moved = False
        for lo, hi in spans:
            if lo <= t <= hi:
                # Clearing one obstacle can slide into another, so re-check all.
                t = hi + SEPARATION_MARGIN
                moved = True
    return t


def _separation_shift(moving, obstacles, outward, allow_reverse=False) -> np.ndarray:
    """Shortest translation clearing every obstacle, over a few directions.

    Straight outward is tried first, then sideways. Two legends side by side
    overlap across their outward axis, where shoving one of them past the other's
    whole depth is a far bigger move than nudging it a little to the side.

    Going back towards the map is only on the table once the legend is already
    sitting on it: for a legend that starts clear, crossing to the far side would
    be a wild move even where it measures shorter.
    """
    perp = np.array([-outward[1], outward[0]])
    candidates = [outward, perp, -perp]
    if allow_reverse:
        candidates.append(-outward)
    best_t, best_dir = math.inf, outward
    for direction in candidates:
        t = _clear_distance(moving, obstacles, direction)
        if t < best_t - 1e-9:
            best_t, best_dir = t, direction
    return best_t * best_dir


def separate_legends(pieces, affines, anchors) -> list:
    """Nudge uprighted legends until none overlaps the map or another legend.

    Uprighting pivots each legend on its own anchor, which changes its footprint
    and can drive it into a neighbour that was clear on the canvas — the pivots
    are per legend, so nothing about the rotation keeps two of them apart.

    Legends are resolved nearest-the-map first, so the closest one keeps its
    place and the farther one yields; ties go to the lower instance number,
    which makes the result independent of the order pieces arrive in.
    """
    indices = [i for i, p in enumerate(pieces) if p.is_legend]
    if not indices:
        return affines
    result = list(affines)
    outlines = _usable_hulls(transformed_outline(p, M)
                             for p, M in zip(pieces, affines) if not p.is_legend)
    # Outlines answer "how far from the map"; their triangles are what the ray
    # clipping can actually work against.
    hulls = outlines
    shapes = {i: transformed_hull([(pieces[i], affines[i])]).astype(np.float64)
              for i in indices}

    def rank(i):
        pts = shapes[i]
        if not hulls:
            return (0.0, pieces[i].legend_instance)
        gap = min(float(np.linalg.norm(p - nearest_map_point(p, hulls)))
                  for p in pts.astype(np.float32))
        return (gap, pieces[i].legend_instance)

    obstacles = [t.astype(np.float64) for o in outlines for t in convex_parts(o)]
    for i in sorted(indices, key=rank):
        moving = shapes[i]
        anchor_world = result[i] @ np.array([anchors[i][0], anchors[i][1], 1.0])
        outward, on_the_map = _outward_direction(anchor_world, hulls, moving)
        shift = _separation_shift(moving, obstacles, outward,
                                  allow_reverse=on_the_map)
        if np.any(shift):
            M = result[i].copy()
            M[:, 2] += shift
            result[i] = M
            moving = moving + shift
        obstacles.append(moving)
    return result


def upright_affine(M: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Strip rotation from affine M, keeping local point `anchor` where M puts it.

    Scale is read back off M rather than taken from the piece, so a scale folded
    in by the caller survives.
    """
    ax, ay = float(anchor[0]), float(anchor[1])
    wx = M[0, 0] * ax + M[0, 1] * ay + M[0, 2]
    wy = M[1, 0] * ax + M[1, 1] * ay + M[1, 2]
    s = math.hypot(M[0, 0], M[1, 0])
    return np.array([[s, 0.0, wx - s * ax],
                     [0.0, s, wy - s * ay]], dtype=np.float64)
