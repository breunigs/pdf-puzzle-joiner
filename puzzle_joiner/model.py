import math
import os
from typing import Optional

import cv2
import numpy as np

from PySide6.QtCore import QRectF
from PySide6.QtGui import QPixmap, QImage


class PuzzlePiece:
    def __init__(self):
        self.source_path: str = ""
        self.source_file: str = ""   # original input file (PDF or image)
        self.display_name: str = ""
        self.image: np.ndarray = None          # Full-res BGRA
        self.thumbnail: QPixmap = None         # 256px preview
        self.display_pixmap: QPixmap = None    # Reduced-res for canvas
        self.display_scale: float = 1.0
        self.x: float = 0.0
        self.y: float = 0.0
        self.rotation_deg: float = 0.0
        self.scale: float = 1.0
        self.crop_rect: Optional[tuple] = None  # (x, y, w, h) local unrotated coords
        self.lasso_poly: Optional[list] = None  # [(x, y), ...] same space as crop_rect
        self._base_alpha: Optional[np.ndarray] = None  # alpha before lasso masking
        self.is_matched: bool = False
        self.is_locked: bool = False
        self.layer_index: int = 0
        # Legend state. A legend is a region lifted off a page and placed beside
        # the assembled map: it takes no part in matching, and the export undoes
        # its rotation so it comes out upright. `legend_rect` records where it was
        # cut from so the instance can be rebuilt from a saved layout.
        self.is_legend: bool = False
        self.legend_source_key: str = ""
        self.legend_rect: Optional[tuple] = None
        self.legend_instance: int = 0

    def layout_key(self) -> str:
        """Stable identity for layout persistence.

        Legend instances share their page's source path, so they need the
        instance number to be told apart from the page and from each other.
        """
        base = os.path.basename(self.source_path)
        if self.is_legend:
            return f"{base}#legend{self.legend_instance}"
        return base

    def apply_crop(self, crop_rect: Optional[tuple], lasso_poly: Optional[list] = None):
        """Set the crop rect and optional lasso polygon, refreshing the alpha mask.

        Polygon points are in full-image (uncropped, unrotated) pixel coords — the
        same space as crop_rect, which should hold the polygon's bounding box so
        that all the rect-based geometry stays valid.
        """
        self.crop_rect = tuple(crop_rect) if crop_rect is not None else None
        self.lasso_poly = ([(float(x), float(y)) for x, y in lasso_poly]
                           if lasso_poly else None)
        self.refresh_lasso_mask()

    def refresh_lasso_mask(self):
        """Rewrite the image's alpha so only pixels inside the lasso are opaque.

        The mask is folded into `image` itself rather than a masked copy, so
        `get_cropped_view()` stays zero-copy. `_base_alpha` keeps the original
        channel — an imported PNG may carry real transparency — which also makes
        the mask recomputable, so repeated vertex edits never degrade pixels.
        """
        if self.image is None:
            return
        if self.lasso_poly is None:
            if self._base_alpha is not None:
                self.image[:, :, 3] = self._base_alpha
                self._base_alpha = None
            return
        if self._base_alpha is None:
            self._base_alpha = self.image[:, :, 3].copy()
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        pts = np.round(np.array(self.lasso_poly, dtype=np.float64)).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255, lineType=cv2.LINE_AA)
        self.image[:, :, 3] = np.minimum(self._base_alpha, mask)

    def get_cropped_image(self) -> np.ndarray:
        if self.crop_rect is not None:
            return self.get_cropped_view().copy()
        return self.image

    def get_cropped_view(self) -> np.ndarray:
        """Cropped image as a view on the source buffer — no copy, do not mutate.

        Lets readers (export warps) avoid a full-size copy of the crop.
        """
        if self.crop_rect is not None:
            cx, cy, cw, ch = self.crop_rect
            return self.image[cy:cy+ch, cx:cx+cw]
        return self.image

    def get_match_image(self) -> np.ndarray:
        """Rect-cropped image with the lasso mask lifted off the alpha channel.

        Matching deliberately ignores the lasso and works on the polygon's
        bounding rect, so a roughly drawn outline cannot starve a pair of the
        overlap it needs to match.
        """
        if self._base_alpha is None:
            return self.get_cropped_image()
        img = self.get_cropped_view().copy()
        if self.crop_rect is not None:
            cx, cy, cw, ch = self.crop_rect
            img[:, :, 3] = self._base_alpha[cy:cy+ch, cx:cx+cw]
        else:
            img[:, :, 3] = self._base_alpha
        return img

    def get_outline_points(self) -> np.ndarray:
        """Opaque extent in piece-local (cropped) coords, as an Nx2 float32 array.

        The lasso polygon when one is set, else the four crop-rect corners. Lets
        the export size its canvas to the pixels that actually show.
        """
        w, h = self._cropped_size()
        if self.lasso_poly is None:
            return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        ox, oy = self.crop_rect[:2] if self.crop_rect is not None else (0, 0)
        pts = np.array(self.lasso_poly, dtype=np.float32) - np.array([ox, oy], np.float32)
        return np.clip(pts, [0, 0], [w, h])

    def _cropped_size(self):
        """Return (w, h) of the cropped image without copying pixel data."""
        if self.crop_rect is not None:
            return self.crop_rect[2], self.crop_rect[3]
        return self.image.shape[1], self.image.shape[0]

    def get_affine_matrix(self) -> np.ndarray:
        """2x3 affine matrix mapping piece-local coords to world coords."""
        w, h = self._cropped_size()
        cx, cy = w / 2.0, h / 2.0
        cos_a = math.cos(math.radians(self.rotation_deg))
        sin_a = math.sin(math.radians(self.rotation_deg))
        s = self.scale
        a = s * cos_a
        b = -s * sin_a
        c = s * sin_a
        d = s * cos_a
        tx = self.x - a * cx - b * cy
        ty = self.y - c * cx - d * cy
        return np.array([[a, b, tx], [c, d, ty]], dtype=np.float64)

    def get_bounding_box(self) -> QRectF:
        w, h = self._cropped_size()
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
        M = self.get_affine_matrix()
        transformed = cv2.transform(corners, M).reshape(-1, 2)
        min_x, min_y = transformed.min(axis=0)
        max_x, max_y = transformed.max(axis=0)
        return QRectF(min_x, min_y, max_x - min_x, max_y - min_y)

    def rebuild_display_pixmap(self, max_dim=2048):
        img = self.get_cropped_image()
        h, w = img.shape[:2]
        self.display_scale = min(max_dim / max(w, h, 1), 1.0)
        if self.display_scale < 1.0:
            new_w = int(w * self.display_scale)
            new_h = int(h * self.display_scale)
            disp = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            disp = img.copy()
        # Convert BGRA -> RGBA for Qt
        rgba = cv2.cvtColor(disp, cv2.COLOR_BGRA2RGBA)
        qimg = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0],
                      QImage.Format.Format_RGBA8888)
        self.display_pixmap = QPixmap.fromImage(qimg.copy())

    def rebuild_thumbnail(self, max_dim=256):
        img = self.get_cropped_image()
        h, w = img.shape[:2]
        scale = min(max_dim / max(w, h, 1), 1.0)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        thumb = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgba = cv2.cvtColor(thumb, cv2.COLOR_BGRA2RGBA)
        qimg = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0],
                      QImage.Format.Format_RGBA8888)
        self.thumbnail = QPixmap.fromImage(qimg.copy())
