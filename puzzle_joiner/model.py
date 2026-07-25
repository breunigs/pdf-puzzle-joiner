import math
from typing import Optional

import cv2
import numpy as np

from PySide6.QtCore import QRectF
from PySide6.QtGui import QPixmap, QImage


class PuzzlePiece:
    def __init__(self):
        self.source_path: str = ""
        self.image: np.ndarray = None          # Full-res BGRA
        self.thumbnail: QPixmap = None         # 256px preview
        self.display_pixmap: QPixmap = None    # Reduced-res for canvas
        self.display_scale: float = 1.0
        self.x: float = 0.0
        self.y: float = 0.0
        self.rotation_deg: float = 0.0
        self.scale: float = 1.0
        self.crop_rect: Optional[tuple] = None  # (x, y, w, h) local unrotated coords
        self.is_matched: bool = False
        self.is_locked: bool = False
        self.layer_index: int = 0

    def get_cropped_image(self) -> np.ndarray:
        if self.crop_rect is not None:
            cx, cy, cw, ch = self.crop_rect
            return self.image[cy:cy+ch, cx:cx+cw].copy()
        return self.image

    def get_affine_matrix(self) -> np.ndarray:
        """2x3 affine matrix mapping piece-local coords to world coords."""
        img = self.get_cropped_image()
        h, w = img.shape[:2]
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
        img = self.get_cropped_image()
        h, w = img.shape[:2]
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
