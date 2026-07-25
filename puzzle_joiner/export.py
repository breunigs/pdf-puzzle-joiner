import math

import cv2
import numpy as np
import tifffile

from .model import PuzzlePiece


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
                extratags=[(285, tifffile.DATATYPE.ASCII, 0, piece.source_path, False)],
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

    canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)

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
        # Alpha composite: warped over canvas
        alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
        canvas_f = canvas.astype(np.float32)
        warped_f = warped.astype(np.float32)
        canvas_f = warped_f * alpha + canvas_f * (1.0 - alpha)
        canvas = np.clip(canvas_f, 0, 255).astype(np.uint8)

    canvas_rgba = cv2.cvtColor(canvas, cv2.COLOR_BGRA2RGBA)
    tifffile.imwrite(
        output_path,
        canvas_rgba,
        photometric="rgb",
        tile=(1024, 1024),
        compression="deflate",
    )
