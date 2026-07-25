import math
import os
import re

import cv2
import numpy as np
import tifffile

from .model import PuzzlePiece


def _layer_name(piece: PuzzlePiece) -> str:
    """Derive a human-readable layer name from source_path."""
    name = os.path.basename(piece.source_path)
    m = re.match(r"page_0*(\d+)\.\w+$", name)
    if m:
        return f"Page {m.group(1)}"
    return name


def export_layered_tiff(pieces: list, output_path: str):
    if not pieces:
        return
    boxes = [p.get_bounding_box() for p in pieces]
    union = boxes[0]
    for b in boxes[1:]:
        union = union.united(b)
    canvas_w = int(math.ceil(union.width()))
    canvas_h = int(math.ceil(union.height()))
    origin_x = union.x()
    origin_y = union.y()

    with tifffile.TiffWriter(output_path, bigtiff=True) as tiff:
        for piece in pieces:
            M = piece.get_affine_matrix().copy()
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
    boxes = [p.get_bounding_box() for p in pieces]
    union = boxes[0]
    for b in boxes[1:]:
        union = union.united(b)
    canvas_w = int(math.ceil(union.width()))
    canvas_h = int(math.ceil(union.height()))
    origin_x = union.x()
    origin_y = union.y()

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

    for piece, box in zip(pieces, boxes):
        M = piece.get_affine_matrix().copy()
        M[0, 2] -= origin_x
        M[1, 2] -= origin_y

        # Warp into just this piece's bounding box region, not the full canvas
        bx = max(0, int(math.floor(box.x() - origin_x)))
        by = max(0, int(math.floor(box.y() - origin_y)))
        bx2 = min(canvas_w, int(math.ceil(box.x() + box.width() - origin_x)))
        by2 = min(canvas_h, int(math.ceil(box.y() + box.height() - origin_y)))
        bw = bx2 - bx
        bh = by2 - by
        if bw <= 0 or bh <= 0:
            continue

        M_local = M.copy()
        M_local[0, 2] -= bx
        M_local[1, 2] -= by
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
