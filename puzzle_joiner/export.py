import math
import os

import cv2
import numpy as np
import tifffile

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


def _export_affines(pieces):
    """Piece affine matrices as the export lays them out.

    The map is rotated to the angle that minimises the canvas. Legends ride along
    with that rotation only as far as their anchor point: each is then re-emitted
    axis-aligned about that point, so it comes out upright without drifting away
    from the map. Legends are left out of the angle search — they are not part of
    the shape being minimised, and they do not keep the angle they are measured at.

    Squaring a legend up changes its footprint, so a last pass nudges legends off
    the map and off each other; see `separate_legends`.
    """
    base = [p.get_affine_matrix() for p in pieces]
    map_pieces = [p for p in pieces if not p.is_legend]
    legends = [p for p in pieces if p.is_legend]

    R3 = None
    if len(map_pieces) >= 2:
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


def _piece_bbox(piece, M):
    """Axis-aligned bounding box (x, y, w, h) of a piece's opaque area under affine M."""
    corners = piece.get_outline_points().reshape(-1, 1, 2)
    transformed = cv2.transform(corners, M.astype(np.float32)).reshape(-1, 2)
    min_xy = transformed.min(axis=0)
    max_xy = transformed.max(axis=0)
    if piece.lasso_poly is not None:
        # The lasso's antialiased edge and the bilinear warp both reach about a
        # pixel past the polygon vertices — keep that fringe inside the canvas.
        min_xy = min_xy - 1.0
        max_xy = max_xy + 1.0
    return min_xy[0], min_xy[1], max_xy[0] - min_xy[0], max_xy[1] - min_xy[1]


def _union_bbox(pieces, affines):
    """Compute (origin_x, origin_y, canvas_w, canvas_h) union of all pieces."""
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    for piece, M in zip(pieces, affines):
        x, y, w, h = _piece_bbox(piece, M)
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)
    return min_x, min_y, int(math.ceil(max_x - min_x)), int(math.ceil(max_y - min_y))


def _canvas_bbox(piece, M, origin_x, origin_y, canvas_w, canvas_h):
    """Integer canvas-space bbox (x0, y0, x1, y1) of a piece, clipped to the canvas."""
    bx_f, by_f, bw_f, bh_f = _piece_bbox(piece, M)
    x0 = max(0, int(math.floor(bx_f - origin_x)))
    y0 = max(0, int(math.floor(by_f - origin_y)))
    x1 = min(canvas_w, int(math.ceil(bx_f + bw_f - origin_x)))
    y1 = min(canvas_h, int(math.ceil(by_f + bh_f - origin_y)))
    return x0, y0, x1, y1


def _band_height(canvas_w: int) -> int:
    """Rows per band: a whole number of tile rows that fits the band budget."""
    tile_rows = max(1, _BAND_BYTES // max(1, canvas_w * 4 * TILE))
    return tile_rows * TILE


def _warp_region(piece, M, origin_x, origin_y, x0, y0, w, h):
    """Warp a piece into a (h, w) BGRA buffer whose top-left is canvas (x0, y0)."""
    M_local = M.copy()
    M_local[0, 2] -= origin_x + x0
    M_local[1, 2] -= origin_y + y0
    return cv2.warpAffine(
        piece.get_cropped_view(), M_local, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


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


def _flattened_tiles(pieces, affines, origin_x, origin_y, canvas_w, canvas_h):
    """Stream the composited canvas as RGBA tiles, one band of rows at a time."""
    boxes = [_canvas_bbox(p, M, origin_x, origin_y, canvas_w, canvas_h)
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
                                  x0, ry0, x1 - x0, ry1 - ry0)
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
    origin_x, origin_y, canvas_w, canvas_h = _union_bbox(pieces, affines)
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


def export_flattened_tiff(pieces: list, output_path: str):
    if not pieces:
        return
    pieces = export_order(pieces)
    affines = _export_affines(pieces)
    origin_x, origin_y, canvas_w, canvas_h = _union_bbox(pieces, affines)
    _check_memory(canvas_w, _band_height(canvas_w))

    with tifffile.TiffWriter(output_path, bigtiff=True) as tiff:
        tiff.write(
            _flattened_tiles(pieces, affines, origin_x, origin_y, canvas_w, canvas_h),
            shape=(canvas_h, canvas_w, 4),
            dtype=np.uint8,
            photometric="rgb",
            tile=(TILE, TILE),
            compression="deflate",
            maxworkers=_ENCODE_WORKERS,
        )
