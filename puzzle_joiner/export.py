import contextlib
import math
import os

import cv2
import numpy as np
import tifffile

from . import georef
from .model import PuzzlePiece
from .legend import anchor_local, map_hull, separate_legends, upright_affine

TILE = 1024
# Byte budget for one horizontal band of the canvas. Bands are rendered and
# streamed out one at a time so the full canvas never has to fit in RAM.
_BAND_BYTES = 512 << 20
# Byte budget for the float32 temporaries of a single composite step.
_CHUNK_BYTES = 64 << 20
# Threads tifffile may use to deflate tiles. Three quarters of the cores leaves
# room for the band rendering (OpenCV runs its own pool) and keeps the machine
# usable.
_ENCODE_WORKERS = max(1, (os.cpu_count() or 2) * 3 // 4)
# Destination columns a single warp call fills, and the source size beyond
# which OpenCV's Lanczos path refuses the job (its coordinate maps are short
# typed, so the real ceiling is SHRT_MAX; this leaves room for the margins).
_WARP_DST_CHUNK = 8192
_WARP_SRC_LIMIT = 24000


def _layer_name(piece: PuzzlePiece) -> str:
    """Derive a human-readable layer name (7-bit ASCII for TIFF compat)."""
    name = piece.display_name or os.path.basename(piece.source_path)
    if piece.is_legend and piece.legend_source_key:
        name = f"{name} ({piece.legend_source_key})"
    return name.encode("ascii", errors="replace").decode("ascii")


def export_order(pieces: list) -> list:
    """Pieces in compositing order — legends last, so they sit on top."""
    return sorted(pieces, key=lambda p: p.is_legend)


def _compute_optimal_rotation(pieces):
    """Find rotation angle (degrees) that minimizes the axis-aligned bounding box."""
    all_corners = []
    for piece in pieces:
        corners = piece.get_outline_points().reshape(-1, 1, 2)
        M = piece.get_affine_matrix()
        transformed = cv2.transform(corners, M.astype(np.float32)).reshape(-1, 2)
        all_corners.append(transformed)
    all_pts = np.vstack(all_corners).astype(np.float32)
    rect = cv2.minAreaRect(all_pts)
    angle = -rect[2]
    # Normalize to [-45, 45] — pick the smallest rotation
    while angle > 45:
        angle -= 90
    while angle < -45:
        angle += 90
    return angle, rect[0][0], rect[0][1]


def _rotation_about(angle, cx, cy):
    """3x3 rotation of `angle` degrees about the point (cx, cy)."""
    rad = math.radians(angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    return np.array([
        [cos_a, -sin_a, cx - cos_a * cx + sin_a * cy],
        [sin_a,  cos_a, cy - sin_a * cx - cos_a * cy],
        [0, 0, 1],
    ], dtype=np.float64)


def _export_affines(pieces, global_transform=None):
    """Piece affine matrices as the export lays them out.

    The map is rotated to the angle that minimises the canvas. Legends ride along
    with that rotation only as far as their anchor point: each is then re-emitted
    axis-aligned about that point, so it comes out upright without drifting away
    from the map. Legends are left out of the angle search — they are not part of
    the shape being minimised, and they do not keep the angle they are measured at.

    Squaring a legend up changes its footprint, so a last pass nudges legends off
    the map and off each other; see `separate_legends`.

    `global_transform` replaces that angle search with a transform of the
    caller's choosing — the georeferenced export, where the orientation is
    dictated by the geography and not by the smallest canvas. Everything about
    legends then applies unchanged, which is the point: a legend rides the
    georeferencing as far as its anchor and is re-emitted upright there, so it
    follows the map round without being turned with it.
    """
    base = [p.get_affine_matrix() for p in pieces]
    map_pieces = [p for p in pieces if not p.is_legend]
    legends = [p for p in pieces if p.is_legend]

    R3 = global_transform
    if R3 is None and len(map_pieces) >= 2:
        angle, cx, cy = _compute_optimal_rotation(map_pieces)
        if abs(angle) >= 0.1:
            R3 = _rotation_about(angle, cx, cy)
    if R3 is None and not legends:
        return base

    hull = map_hull(map_pieces) if legends else None
    result = []
    anchors = {}
    for idx, (piece, M) in enumerate(zip(pieces, base)):
        if R3 is not None:
            M = (R3 @ np.vstack([M, [0, 0, 1]]))[:2]
        if piece.is_legend:
            anchors[idx] = anchor_local(piece, hull)
            M = upright_affine(M, anchors[idx])
        result.append(M)
    if legends:
        result = separate_legends(pieces, result, anchors)
    return result


def _piece_bbox(piece, M, uncropped=False):
    """Axis-aligned bounding box (x, y, w, h) of a piece's opaque area under affine M."""
    corners = piece.get_outline_points(use_lasso=not uncropped).reshape(-1, 1, 2)
    # float64: the georeferenced export's affines translate by absolute web
    # mercator metres over the pixel size, where float32 steps in units of
    # whole pixels. Rounding the box there loses a strip off the edge of every
    # piece, because `_canvas_bbox` floors and ceils whatever it is given.
    transformed = cv2.transform(corners.astype(np.float64), M).reshape(-1, 2)
    min_xy = transformed.min(axis=0)
    max_xy = transformed.max(axis=0)
    if piece.lasso_poly is not None:
        # `fillPoly` antialiases the mask about a pixel outside the polygon, and
        # the bilinear warp spreads that another pixel in canvas space — so the
        # opaque area reaches a little past the transformed vertices. Padding by
        # the piece's scale plus two covers both, whichever way round they are.
        # The rect box gets the same pad, so it stays a superset of the lasso box
        # and the shared canvas cannot clip the fringe off either file.
        pad = math.sqrt(abs(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])) + 2.0
        min_xy = min_xy - pad
        max_xy = max_xy + pad
    return min_xy[0], min_xy[1], max_xy[0] - min_xy[0], max_xy[1] - min_xy[1]


def export_canvas(pieces, affines):
    """The canvas every file of one export shares: (origin_x, origin_y, w, h).

    Sized from the crop rects, not the lasso hulls. The uncropped file needs the
    pixels outside the hulls, so sizing to the hulls would clip it; sizing both
    to the rects instead makes every file come out the same shape, on the same
    origin, overlapping pixel for pixel. A georeference worked out on the
    uncropped file therefore carries over to the lassoed one unchanged — the
    lassoed one just has transparent pixels where the uncropped one has map.
    """
    return _union_bbox(pieces, affines, uncropped=True)


def _union_bbox(pieces, affines, uncropped=False):
    """Compute (origin_x, origin_y, canvas_w, canvas_h) union of all pieces."""
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    for piece, M in zip(pieces, affines):
        x, y, w, h = _piece_bbox(piece, M, uncropped)
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)
    return min_x, min_y, int(math.ceil(max_x - min_x)), int(math.ceil(max_y - min_y))


def _canvas_bbox(piece, M, origin_x, origin_y, canvas_w, canvas_h, uncropped=False):
    """Integer canvas-space bbox (x0, y0, x1, y1) of a piece, clipped to the canvas."""
    bx_f, by_f, bw_f, bh_f = _piece_bbox(piece, M, uncropped)
    x0 = max(0, int(math.floor(bx_f - origin_x)))
    y0 = max(0, int(math.floor(by_f - origin_y)))
    x1 = min(canvas_w, int(math.ceil(bx_f + bw_f - origin_x)))
    y1 = min(canvas_h, int(math.ceil(by_f + bh_f - origin_y)))
    return x0, y0, x1, y1


def _band_height(canvas_w: int) -> int:
    """Rows per band: a whole number of tile rows that fits the band budget."""
    tile_rows = max(1, _BAND_BYTES // max(1, canvas_w * 4 * TILE))
    return tile_rows * TILE


def _source_window(src_shape, M_local, w, h):
    """The box of source pixels that can reach a (w, h) destination.

    Padded by a few pixels so the interpolation kernel has the neighbours it
    reads, and clipped to the image.
    """
    inverse = cv2.invertAffineTransform(M_local)
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]],
                       dtype=np.float32).reshape(-1, 1, 2)
    back = cv2.transform(corners, inverse).reshape(-1, 2)
    pad = 4
    return (max(0, int(math.floor(back[:, 0].min())) - pad),
            max(0, int(math.floor(back[:, 1].min())) - pad),
            min(src_shape[1], int(math.ceil(back[:, 0].max())) + pad),
            min(src_shape[0], int(math.ceil(back[:, 1].max())) + pad))


def _warp_region(piece, M, origin_x, origin_y, x0, y0, w, h,
                 interpolation=cv2.INTER_LINEAR):
    """Warp a piece into a (h, w) BGRA buffer whose top-left is canvas (x0, y0).

    Only the slice of the source that can actually reach the destination is
    handed to OpenCV, and the destination is filled a column at a time. Both
    matter once a plan runs to a few kilometres at 300 dpi: the whole page was
    otherwise passed for every band of every row it touches, and OpenCV's
    Lanczos path flatly refuses a source at or past SHRT_MAX on either axis —
    which a page of this kind reaches on its own.
    """
    src = piece.get_cropped_view()
    M_local = M.copy()
    M_local[0, 2] -= origin_x + x0
    M_local[1, 2] -= origin_y + y0
    out = np.zeros((h, w, 4), dtype=np.uint8)
    # A piece drawn smaller than life needs correspondingly more source per
    # column, so the column shrinks to keep that slice inside the limit.
    scale = math.sqrt(abs(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]))
    chunk = max(1024, min(_WARP_DST_CHUNK,
                          int(_WARP_DST_CHUNK * min(1.0, scale))))
    for cx in range(0, w, chunk):
        cw = min(chunk, w - cx)
        M_chunk = M_local.copy()
        M_chunk[0, 2] -= cx
        sx0, sy0, sx1, sy1 = _source_window(src.shape, M_chunk, cw, h)
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        # Re-aim the affine at the cropped source's own origin.
        M_chunk[0, 2] += M_chunk[0, 0] * sx0 + M_chunk[0, 1] * sy0
        M_chunk[1, 2] += M_chunk[1, 0] * sx0 + M_chunk[1, 1] * sy0
        flags = interpolation
        if (interpolation == cv2.INTER_LANCZOS4
                and max(sx1 - sx0, sy1 - sy0) >= _WARP_SRC_LIMIT):
            # Nothing reasonable reaches here, but a silently wrong picture is
            # worse than a slightly softer one: cubic takes any size.
            flags = cv2.INTER_CUBIC
        out[:, cx:cx + cw] = cv2.warpAffine(
            src[sy0:sy1, sx0:sx1], M_chunk, (cw, h),
            flags=flags,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
    return out


def _alpha_composite(dst, src):
    """Alpha-composite src over dst in place, in row chunks to bound temporaries."""
    h, w = dst.shape[:2]
    rows = max(1, _CHUNK_BYTES // max(1, w * 4 * 4))
    for y in range(0, h, rows):
        s = src[y:y + rows]
        if not s[:, :, 3].any():
            continue
        d = dst[y:y + rows]
        alpha = s[:, :, 3:4].astype(np.float32) / 255.0
        d[:] = np.clip(
            s.astype(np.float32) * alpha + d.astype(np.float32) * (1.0 - alpha),
            0, 255,
        ).astype(np.uint8)


def _band_tiles(band, band_x, canvas_w, band_h):
    """Yield a band's tiles as RGBA, row-major; `None` means the band is empty.

    A band spans a whole number of tile rows, so it is sliced back into TILE-high
    rows here — the writer consumes exactly one tile per yield.
    """
    for y in range(0, band_h, TILE):
        yield from _row_tiles(None if band is None else band[y:y + TILE],
                              band_x, canvas_w)


def _row_tiles(row, row_x, canvas_w):
    """Yield one tile row as RGBA; row covers canvas columns [row_x, row_x + width).

    Converting at tile granularity keeps the colour swap off the band-sized buffer.
    Tile positions the row does not reach are emitted as fully transparent tiles.
    """
    row_to = row_x + row.shape[1] if row is not None else 0
    for x in range(0, canvas_w, TILE):
        if row is None or x >= row_to or x + TILE <= row_x:
            yield np.zeros((TILE, TILE, 4), dtype=np.uint8)
        elif x >= row_x and x + TILE <= row_to:
            yield cv2.cvtColor(row[:, x - row_x:x - row_x + TILE], cv2.COLOR_BGRA2RGBA)
        else:
            # Tile straddles the edge of the row — pad the uncovered columns.
            tile = np.zeros((row.shape[0], TILE, 4), dtype=np.uint8)
            lo, hi = max(x, row_x), min(x + TILE, row_to)
            tile[:, lo - x:hi - x] = row[:, lo - row_x:hi - row_x]
            yield cv2.cvtColor(tile, cv2.COLOR_BGRA2RGBA)


def _flattened_tiles(pieces, affines, origin_x, origin_y, canvas_w, canvas_h,
                     uncropped=False, interpolation=cv2.INTER_LINEAR):
    """Stream the composited canvas as RGBA tiles, one band of rows at a time."""
    boxes = [_canvas_bbox(p, M, origin_x, origin_y, canvas_w, canvas_h, uncropped)
             for p, M in zip(pieces, affines)]
    band_h = _band_height(canvas_w)
    for band_y in range(0, canvas_h, band_h):
        bh = min(band_h, canvas_h - band_y)
        # Only the columns some piece reaches in these rows need a buffer.
        rows = [(p, M, b) for p, M, b in zip(pieces, affines, boxes)
                if b[0] < b[2] and max(b[1], band_y) < min(b[3], band_y + bh)]
        if not rows:
            yield from _band_tiles(None, 0, canvas_w, bh)
            continue
        band_x = min(b[0] for _, _, b in rows)
        band_w = max(b[2] for _, _, b in rows) - band_x
        band = np.zeros((bh, band_w, 4), dtype=np.uint8)
        for piece, M, (x0, y0, x1, y1) in rows:
            ry0 = max(y0, band_y)
            ry1 = min(y1, band_y + bh)
            warped = _warp_region(piece, M, origin_x, origin_y,
                                  x0, ry0, x1 - x0, ry1 - ry0, interpolation)
            _alpha_composite(
                band[ry0 - band_y:ry1 - band_y, x0 - band_x:x1 - band_x], warped)
            del warped
        yield from _band_tiles(band, band_x, canvas_w, bh)
        del band


def _layer_tiles(piece, M, origin_x, origin_y, canvas_w, canvas_h):
    """Stream one piece as RGBA tiles over the full canvas, rendering only its bbox."""
    x0, y0, x1, y1 = _canvas_bbox(piece, M, origin_x, origin_y, canvas_w, canvas_h)
    for band_y in range(0, canvas_h, TILE):
        bh = min(TILE, canvas_h - band_y)
        ry0 = max(y0, band_y)
        ry1 = min(y1, band_y + bh)
        if ry1 <= ry0 or x1 <= x0:
            yield from _band_tiles(None, 0, canvas_w, bh)
            continue
        band = np.zeros((bh, x1 - x0, 4), dtype=np.uint8)
        band[ry0 - band_y:ry1 - band_y] = _warp_region(
            piece, M, origin_x, origin_y, x0, ry0, x1 - x0, ry1 - ry0)
        yield from _band_tiles(band, x0, canvas_w, bh)
        del band


def export_layered_tiff(pieces: list, output_path: str):
    if not pieces:
        return
    pieces = export_order(pieces)
    affines = _export_affines(pieces)
    origin_x, origin_y, canvas_w, canvas_h = export_canvas(pieces, affines)
    _check_memory(canvas_w, TILE)

    with tifffile.TiffWriter(output_path, bigtiff=True) as tiff:
        for piece, M in zip(pieces, affines):
            tiff.write(
                _layer_tiles(piece, M, origin_x, origin_y, canvas_w, canvas_h),
                shape=(canvas_h, canvas_w, 4),
                dtype=np.uint8,
                photometric="rgb",
                tile=(TILE, TILE),
                compression="deflate",
                maxworkers=_ENCODE_WORKERS,
                extratags=[(285, tifffile.DATATYPE.ASCII, 0, _layer_name(piece), False)],
            )


def _available_memory() -> int:
    """Bytes of RAM usable without swapping, counting reclaimable page cache.

    Prefers /proc/meminfo MemAvailable; SC_AVPHYS_PAGES is only MemFree and
    badly underreports on a machine with a warm page cache.
    """
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        pass
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")


def _check_memory(canvas_w: int, band_h: int):
    """Guard against a canvas so wide that even a single band will not fit."""
    # One band, plus the warp buffer for a piece spanning it, plus composite chunks.
    needed = 2 * canvas_w * band_h * 4 + 6 * _CHUNK_BYTES
    avail = _available_memory()
    if needed > avail * 0.8:
        raise MemoryError(
            f"Export needs {needed / 1024**3:.1f} GB to render a "
            f"{canvas_w} px wide strip but only {avail / 1024**3:.1f} GB "
            f"of RAM is available."
        )


@contextlib.contextmanager
def _lassos_lifted(pieces):
    """Put every piece's unmasked alpha back for the duration of the block.

    A lasso mask lives in `image`'s own alpha channel, which is what lets the
    export warp a zero-copy view of the crop. The uncropped export wants those
    pixels, so the mask is lifted here instead of the pixels being copied, and
    recomputed from `_base_alpha` afterwards.
    """
    lifted = [p for p in pieces
              if p.lasso_poly is not None and p._base_alpha is not None]
    for piece in lifted:
        piece.image[:, :, 3] = piece._base_alpha
    try:
        yield
    finally:
        for piece in lifted:
            piece.refresh_lasso_mask()


def canvas_gcps(pieces: list) -> list:
    """Every page's control points in canvas coordinates, pooled.

    Returns [(canvas_x, canvas_y, map_x, map_y), ...] across all pages, because
    one georeference covers the whole assembled map however many pages were
    marked up. No origin is subtracted — the georeferenced export derives its
    own origin after transforming, so subtracting one here would only have to
    be undone.

    The space is each piece's own `get_affine_matrix()`: the scene the user
    arranged, and nothing else. Deliberately **not** `_export_affines`, which
    first rotates the whole assembly to whatever angle minimises the canvas.
    That angle depends on every other piece, so points taken through it would
    shift whenever an unrelated page moved, and — worse — the georeferenced
    export passes its own transform in place of that rotation, so a fit made
    in the rotated space was being applied to the unrotated one. On a real
    five-page plan sitting at 16.6 degrees that put the output up to 578 m out.
    Canvas minimising is a packing trick for the other exports; geography must
    not be measured through it.

    A point may land outside a piece's crop: the dialog shows the uncropped
    page, so a feature the crop later throws away can still be marked. Those
    are kept — they constrain the fit perfectly well, and dropping them would
    quietly weaken a fit the user believed they had made.
    """
    out = []
    for piece in pieces:
        if piece.is_legend or not piece.gcps:
            continue
        M = piece.get_affine_matrix()
        ox, oy = piece.crop_rect[:2] if piece.crop_rect is not None else (0, 0)
        local = np.array([[g[0] - ox, g[1] - oy] for g in piece.gcps],
                         dtype=np.float64).reshape(-1, 1, 2)
        world = cv2.transform(local, M).reshape(-1, 2)
        for (wx, wy), gcp in zip(world, piece.gcps):
            out.append((float(wx), float(wy), float(gcp[2]), float(gcp[3])))
    return out


def _fit_matrix(gcps, method):
    """The fit as a 2x3 affine from canvas coords to map units, or None.

    QGIS hands back a callable, but the export needs the matrix. Both offered
    methods are linear, so evaluating three points recovers it exactly.
    """
    mapped = georef.transform_points(gcps, method, [(0, 0), (1, 0), (0, 1)])
    if mapped is None:
        return None
    (x0, y0), (x1, y1), (x2, y2) = mapped
    return np.array([[x1 - x0, x2 - x0, x0],
                     [y1 - y0, y2 - y0, y0]], dtype=np.float64)


def georef_transform(gcps, method=georef.DEFAULT_METHOD):
    """Canvas coords to north-up raster pixels, as (3x3 matrix, metres/pixel).

    None when the points cannot be fitted.

    The raster's pixel is deliberately the fit's own scale, which is what keeps
    this cheap and keeps legends legible. At that pixel size the transform the
    map takes on is a pure rotation: no resampling beyond the one every export
    already does, and a legend re-emitted upright about its anchor keeps the
    size the user gave it rather than being rescaled to suit the geography.

    Map northings climb while raster rows descend, so the y axis is negated
    here. The fit already carries one such flip (see `georef.fit`); the two
    cancel, which is why the result is a rotation and not a reflection.
    """
    F = _fit_matrix(gcps, method)
    if F is None:
        return None
    scale = math.sqrt(abs(F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]))
    if scale <= 0:
        return None
    to_pixels = np.array([[1.0 / scale, 0.0, 0.0],
                          [0.0, -1.0 / scale, 0.0],
                          [0.0, 0.0, 1.0]], dtype=np.float64)
    G = to_pixels @ np.vstack([F, [0.0, 0.0, 1.0]])
    return G, scale


def _geotiff_tags(scale: float, origin_x: float, origin_y: float):
    """GeoTIFF tags for a north-up raster: pixel size, tie point, CRS.

    Written straight into the TIFF rather than bolted on afterwards with
    `gdal_translate`, which would re-encode every tile to record six numbers.
    The grid is axis-aligned by construction, so a pixel scale and one tie
    point say everything; `origin_*` is the raster's top-left in raster-pixel
    units, converted back to map units here.
    """
    code = int(georef.MAP_CRS.split(":")[1])
    geo_keys = [
        1, 1, 0, 3,            # directory version, key revision, minor, count
        1024, 0, 1, 1,         # GTModelTypeGeoKey = projected
        1025, 0, 1, 1,         # GTRasterTypeGeoKey = pixel is area
        3072, 0, 1, code,      # ProjectedCSTypeGeoKey
    ]
    return [
        (33550, tifffile.DATATYPE.DOUBLE, 3, (scale, scale, 0.0), True),
        (33922, tifffile.DATATYPE.DOUBLE, 6,
         (0.0, 0.0, 0.0, origin_x * scale, -origin_y * scale, 0.0), True),
        (34735, tifffile.DATATYPE.SHORT, len(geo_keys), tuple(geo_keys), True),
    ]


def export_geotiff(pieces: list, output_path: str, gcps: list,
                   method: str = georef.DEFAULT_METHOD) -> bool:
    """Composite the pieces straight into a georeferenced TIFF. False if unfitted.

    There is no separate warping step and no intermediate raster. The
    georeferencing is folded into the affines the compositor already applies,
    so the pixels are read once and resampled once, and the result streams out
    band by band like every other export — which is what makes it survive a
    canvas far larger than memory.

    This is also what keeps legends legible. They are not warped: each rides
    the georeferencing as far as its anchor and is re-emitted upright there
    (`_export_affines`), so it follows the map round to its new orientation
    while staying square to the page and at its own scale. `export_order`
    composites them last, so they sit on top of the map as before.
    """
    if not pieces or not gcps:
        return False
    transform = georef_transform(gcps, method)
    if transform is None:
        return False
    G, scale = transform

    pieces = export_order(pieces)
    affines = _export_affines(pieces, global_transform=G)
    origin_x, origin_y, canvas_w, canvas_h = export_canvas(pieces, affines)
    _check_memory(canvas_w, _band_height(canvas_w))

    with tifffile.TiffWriter(output_path, bigtiff=True) as tiff:
        tiff.write(
            _flattened_tiles(pieces, affines, origin_x, origin_y,
                             canvas_w, canvas_h,
                             interpolation=cv2.INTER_LANCZOS4),
            shape=(canvas_h, canvas_w, 4),
            dtype=np.uint8,
            photometric="rgb",
            tile=(TILE, TILE),
            compression="deflate",
            maxworkers=_ENCODE_WORKERS,
            extratags=_geotiff_tags(scale, origin_x, origin_y),
        )
    return True


def write_qgis_project(project_path: str, raster_path: str) -> bool:
    """Write a QGIS project listing `raster_path`, over an OSM basemap.

    Written by QGIS itself rather than assembled as XML here, so it is a real
    project a person can open, not just something a parser will tolerate.

    It exists for veloroute's `mix velo.map.image.from_qgis`, which takes a
    project rather than a raster: it reads every `layer-tree-layer` that is
    checked, keeps the sources ending in `.tif`, and resolves them against the
    project's own directory. Hence relative path storage, and hence writing the
    project beside the raster. The basemap is a tile layer, so that filter
    drops it and only the warped raster is handed on.
    """
    core, _ = georef._qgis()
    project = core.QgsProject()
    # Relative, so moving the pair of files together keeps the project working.
    project.setFilePathStorage(core.Qgis.FilePathType.Relative)
    project.setCrs(core.QgsCoordinateReferenceSystem(georef.MAP_CRS))

    basemap = georef.osm_layer_for_project()
    raster = core.QgsRasterLayer(
        raster_path, os.path.splitext(os.path.basename(raster_path))[0], "gdal")
    if not raster.isValid():
        return False
    # Basemap first so it lands underneath; the map is what sits on top.
    project.addMapLayers([layer for layer in (basemap, raster) if layer.isValid()])
    return project.write(project_path)


def export_flattened_tiff(pieces: list, output_path: str, uncropped: bool = False):
    """Composite every piece onto one canvas and write it as a tiled BigTIFF.

    With `uncropped`, a lassoed piece keeps the pixels outside its polygon: the
    mask comes off and the piece contributes its whole crop rect. Canvas and
    layout are the same either way (see `export_canvas`), so the two files
    overlay exactly — one carries everything that was scanned, the other only
    the pixels inside the lassos.
    """
    if not pieces:
        return
    pieces = export_order(pieces)
    affines = _export_affines(pieces)
    origin_x, origin_y, canvas_w, canvas_h = export_canvas(pieces, affines)
    _check_memory(canvas_w, _band_height(canvas_w))

    with contextlib.ExitStack() as stack:
        if uncropped:
            stack.enter_context(_lassos_lifted(pieces))
        tiff = stack.enter_context(tifffile.TiffWriter(output_path, bigtiff=True))
        tiff.write(
            _flattened_tiles(pieces, affines, origin_x, origin_y,
                             canvas_w, canvas_h, uncropped),
            shape=(canvas_h, canvas_w, 4),
            dtype=np.uint8,
            photometric="rgb",
            tile=(TILE, TILE),
            compression="deflate",
            maxworkers=_ENCODE_WORKERS,
        )
