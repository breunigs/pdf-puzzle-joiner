import math
import os

import cv2
import numpy as np
import tifffile

from .model import PuzzlePiece


def _layer_name(piece: PuzzlePiece) -> str:
    """Derive a human-readable layer name."""
    return piece.display_name or os.path.basename(piece.source_path)


def _compute_optimal_rotation(pieces):
    """Find rotation angle (degrees) that minimizes the axis-aligned bounding box."""
    all_corners = []
    for piece in pieces:
        w, h = piece._cropped_size()
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
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


def _export_affines(pieces):
    """Piece affine matrices with optimal rotation applied for minimal export size."""
    base = [p.get_affine_matrix() for p in pieces]
    if len(pieces) < 2:
        return base
    angle, cx, cy = _compute_optimal_rotation(pieces)
    if abs(angle) < 0.1:
        return base
    rad = math.radians(angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    R = np.array([
        [cos_a, -sin_a, cx - cos_a * cx + sin_a * cy],
        [sin_a,  cos_a, cy - sin_a * cx - cos_a * cy],
    ], dtype=np.float64)
    R3 = np.vstack([R, [0, 0, 1]])
    result = []
    for M in base:
        M3 = np.vstack([M, [0, 0, 1]])
        result.append((R3 @ M3)[:2])
    return result


def _piece_bbox(piece, M):
    """Axis-aligned bounding box (x, y, w, h) for piece under affine M."""
    w, h = piece._cropped_size()
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.transform(corners, M.astype(np.float32)).reshape(-1, 2)
    min_xy = transformed.min(axis=0)
    max_xy = transformed.max(axis=0)
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


def export_layered_tiff(pieces: list, output_path: str):
    if not pieces:
        return
    affines = _export_affines(pieces)
    origin_x, origin_y, canvas_w, canvas_h = _union_bbox(pieces, affines)

    with tifffile.TiffWriter(output_path, bigtiff=True) as tiff:
        for piece, M in zip(pieces, affines):
            M = M.copy()
            M[0, 2] -= origin_x
            M[1, 2] -= origin_y
            img = piece.get_cropped_image()
            warped = cv2.warpAffine(
                img, M, (canvas_w, canvas_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )
            warped_rgba = cv2.cvtColor(warped, cv2.COLOR_BGRA2RGBA)
            tiff.write(
                warped_rgba,
                photometric="rgb",
                tile=(1024, 1024),
                compression="deflate",
                extratags=[(285, tifffile.DATATYPE.ASCII, 0, _layer_name(piece), False)],
            )


def export_flattened_tiff(pieces: list, output_path: str):
    if not pieces:
        return
    affines = _export_affines(pieces)
    origin_x, origin_y, canvas_w, canvas_h = _union_bbox(pieces, affines)

    canvas_bytes = canvas_h * canvas_w * 4
    try:
        import psutil
        avail = psutil.virtual_memory().available
    except ImportError:
        avail = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    if canvas_bytes > avail * 0.8:
        raise MemoryError(
            f"Flattened export canvas would be {canvas_w} x {canvas_h} "
            f"({canvas_bytes / 1024**3:.1f} GB) but only "
            f"{avail / 1024**3:.1f} GB of RAM is available."
        )

    canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)

    for piece, M in zip(pieces, affines):
        bx_f, by_f, bw_f, bh_f = _piece_bbox(piece, M)
        bx = max(0, int(math.floor(bx_f - origin_x)))
        by = max(0, int(math.floor(by_f - origin_y)))
        bx2 = min(canvas_w, int(math.ceil(bx_f + bw_f - origin_x)))
        by2 = min(canvas_h, int(math.ceil(by_f + bh_f - origin_y)))
        bw = bx2 - bx
        bh = by2 - by
        if bw <= 0 or bh <= 0:
            continue

        M_local = M.copy()
        M_local[0, 2] -= origin_x + bx
        M_local[1, 2] -= origin_y + by
        img = piece.get_cropped_image()
        warped = cv2.warpAffine(
            img, M_local, (bw, bh),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        # Alpha composite: warped over canvas region only
        region = canvas[by:by2, bx:bx2]
        alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
        region[:] = np.clip(
            warped.astype(np.float32) * alpha + region.astype(np.float32) * (1.0 - alpha),
            0, 255,
        ).astype(np.uint8)

    canvas_rgba = cv2.cvtColor(canvas, cv2.COLOR_BGRA2RGBA)
    tifffile.imwrite(
        output_path,
        canvas_rgba,
        photometric="rgb",
        tile=(1024, 1024),
        compression="deflate",
    )
