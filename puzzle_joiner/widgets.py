import os
import math
import subprocess
from typing import Optional

import cv2
import numpy as np

from PySide6.QtWidgets import (
    QGraphicsScene, QGraphicsView,
    QGraphicsPixmapItem, QGraphicsItem, QGraphicsRectItem,
    QGraphicsPathItem, QGraphicsEllipseItem,
    QWidget, QHBoxLayout, QVBoxLayout, QScrollArea, QLabel, QPushButton,
    QButtonGroup, QDialog, QDialogButtonBox, QLineEdit, QFormLayout,
    QMenu, QFrame, QGridLayout,
    QAbstractScrollArea, QMessageBox,
    QSplitter, QComboBox, QTableWidget, QTableWidgetItem, QCheckBox,
    QHeaderView, QAbstractItemView,
)
from PySide6.QtCore import (
    Qt, QRectF, QPointF, QSizeF, QObject, Signal,
    QRunnable, QThreadPool, QEvent, QEasingCurve,
    QVariantAnimation, QTimer,
)
from PySide6.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor, QBrush, QCursor,
    QFont, QTransform, QPainterPath,
)

from . import georef
from .model import PuzzlePiece
from .legend import anchor_pair
from .priority import LOW_PRIO
from .worker import Worker
from .preprocessing import parse_page_range


HANDLE_SIZE = 8   # pixels on screen
ROTATE_HANDLE_DIST = 30  # pixels above bounding rect top


def _make_rotate_cursor():
    """Create a circular-arrow cursor for rotation."""
    size = 24
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(0, 0, 0), 2)
    p.setPen(pen)
    # Draw a 270° arc
    margin = 3
    rect = QRectF(margin, margin, size - 2 * margin, size - 2 * margin)
    p.drawArc(rect, 30 * 16, 300 * 16)  # start 30°, span 300°
    # Arrowhead at the end of the arc (≈330° = 30+300, which is near top-right)
    # End point of arc at 330° (Qt angles: 0°=3-o'clock, counter-clockwise)
    cx, cy = size / 2, size / 2
    r = (size - 2 * margin) / 2
    end_angle = math.radians(330)
    ex = cx + r * math.cos(end_angle)
    ey = cy - r * math.sin(end_angle)
    # Arrow pointing along the tangent (clockwise direction)
    tangent = end_angle - math.pi / 2
    arrow_len = 6
    a1x = ex + arrow_len * math.cos(tangent + 0.5)
    a1y = ey - arrow_len * math.sin(tangent + 0.5)
    a2x = ex + arrow_len * math.cos(tangent - 0.5)
    a2y = ey - arrow_len * math.sin(tangent - 0.5)
    path = QPainterPath()
    path.moveTo(a1x, a1y)
    path.lineTo(ex, ey)
    path.lineTo(a2x, a2y)
    p.drawPath(path)
    p.end()
    return QCursor(pm, size // 2, size // 2)


_rotate_cursor = None


def _get_rotate_cursor():
    global _rotate_cursor
    if _rotate_cursor is None:
        _rotate_cursor = _make_rotate_cursor()
    return _rotate_cursor


def _resize_cursor_for_handle(handle, piece_item):
    """Pick the correct resize cursor based on handle index and piece rotation."""
    # Base angles from piece center for each handle (unrotated, y-down coords)
    if handle.handle_type == HandleItem.CORNER:
        # TL=0→225°, TR=1→315°, BR=2→45°, BL=3→135°
        base = [225, 315, 45, 135][handle.index]
    else:
        # T=0→270°, R=1→0°, B=2→90°, L=3→180°
        base = [270, 0, 90, 180][handle.index]
    angle = (base + piece_item.piece.rotation_deg) % 360
    sector = int((angle + 22.5) / 45) % 8
    # 0=E, 1=SE, 2=S, 3=SW, 4=W, 5=NW, 6=N, 7=NE
    cursors = [
        Qt.CursorShape.SizeHorCursor,     # 0 E
        Qt.CursorShape.SizeFDiagCursor,   # 1 SE  (\)
        Qt.CursorShape.SizeVerCursor,     # 2 S
        Qt.CursorShape.SizeBDiagCursor,   # 3 SW  (/)
        Qt.CursorShape.SizeHorCursor,     # 4 W
        Qt.CursorShape.SizeFDiagCursor,   # 5 NW  (\)
        Qt.CursorShape.SizeVerCursor,     # 6 N
        Qt.CursorShape.SizeBDiagCursor,   # 7 NE  (/)
    ]
    return QCursor(cursors[sector])


class HandleItem(QGraphicsRectItem):
    CORNER = "corner"
    EDGE = "edge"
    ROTATE = "rotate"

    def __init__(self, handle_type, index, parent):
        super().__init__(-HANDLE_SIZE / 2, -HANDLE_SIZE / 2, HANDLE_SIZE, HANDLE_SIZE, parent)
        self.handle_type = handle_type
        self.index = index
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setAcceptHoverEvents(True)
        self.setBrush(QBrush(QColor(255, 255, 255)))
        self.setPen(QPen(QColor(0, 120, 255), 1))
        self.setZValue(10)
        self._dragging = False
        self._drag_start_pos = None
        self._drag_start_piece_x = 0.0
        self._drag_start_piece_y = 0.0
        self._drag_start_scale = 1.0
        self._drag_start_rot = 0.0
        self._drag_group_starts = {}  # {PuzzlePiece: (x, y, rot, scale)}

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            scene = self.scene()
            if scene and not getattr(scene, 'editing_enabled', True):
                event.accept()
                return
            self._dragging = True
            self._drag_start_pos = event.scenePos()
            piece_item = self.parentItem()
            piece = piece_item.piece
            self._drag_start_piece_x = piece.x
            self._drag_start_piece_y = piece.y
            self._drag_start_scale = piece.scale
            self._drag_start_rot = piece.rotation_deg
            # Capture drag group start states
            if isinstance(scene, PuzzleScene):
                group = scene.get_drag_group(piece)
                self._drag_group_starts = {p: (p.x, p.y, p.rotation_deg, p.scale) for p in group}
            else:
                self._drag_group_starts = {piece: (piece.x, piece.y, piece.rotation_deg, piece.scale)}
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self._dragging:
            super().mouseMoveEvent(event)
            return
        event.accept()
        piece_item = self.parentItem()
        piece = piece_item.piece
        scene = self.scene()
        cx_world = self._drag_start_piece_x
        cy_world = self._drag_start_piece_y

        current_pos = event.scenePos()

        if self.handle_type == self.ROTATE:
            start_angle = math.degrees(math.atan2(
                self._drag_start_pos.y() - cy_world,
                self._drag_start_pos.x() - cx_world,
            ))
            current_angle = math.degrees(math.atan2(
                current_pos.y() - cy_world,
                current_pos.x() - cx_world,
            ))
            delta_rot = current_angle - start_angle
            rad = math.radians(delta_rot)
            cos_d = math.cos(rad)
            sin_d = math.sin(rad)
            for p, (sx, sy, sr, ss) in self._drag_group_starts.items():
                dx = sx - cx_world
                dy = sy - cy_world
                p.x = cx_world + cos_d * dx - sin_d * dy
                p.y = cy_world + sin_d * dx + cos_d * dy
                p.rotation_deg = sr + delta_rot
                item = scene.piece_items.get(p) if isinstance(scene, PuzzleScene) else None
                if item:
                    item.sync_from_piece()
        else:
            # Scale from center
            start_dist = math.sqrt(
                (self._drag_start_pos.x() - cx_world) ** 2 +
                (self._drag_start_pos.y() - cy_world) ** 2
            )
            current_dist = math.sqrt(
                (current_pos.x() - cx_world) ** 2 +
                (current_pos.y() - cy_world) ** 2
            )
            if start_dist > 1e-6:
                ratio = current_dist / start_dist
                for p, (sx, sy, sr, ss) in self._drag_group_starts.items():
                    dx = sx - cx_world
                    dy = sy - cy_world
                    p.x = cx_world + dx * ratio
                    p.y = cy_world + dy * ratio
                    p.scale = max(0.01, ss * ratio)
                    item = scene.piece_items.get(p) if isinstance(scene, PuzzleScene) else None
                    if item:
                        item.sync_from_piece()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            event.accept()
            piece_item = self.parentItem()
            piece = piece_item.piece
            changed = (piece.x != self._drag_start_piece_x or
                       piece.y != self._drag_start_piece_y or
                       piece.rotation_deg != self._drag_start_rot or
                       piece.scale != self._drag_start_scale)
            if changed:
                desc = "Rotate" if self.handle_type == self.ROTATE else "Scale"
                scene = self.scene()
                if isinstance(scene, PuzzleScene) and scene.undo_stack and len(self._drag_group_starts) > 1:
                    from .undo import TransformCommand, GroupTransformCommand
                    cmds = []
                    for p, (ox, oy, orot, oscale) in self._drag_group_starts.items():
                        if p.x != ox or p.y != oy or p.rotation_deg != orot or p.scale != oscale:
                            item = scene.piece_items.get(p)
                            sync_fn = item._undo_sync if item else None
                            cmds.append(TransformCommand(
                                p, ox, oy, orot, oscale,
                                p.x, p.y, p.rotation_deg, p.scale,
                                sync_fn=sync_fn, description=desc,
                            ))
                    if cmds:
                        scene.undo_stack.push(GroupTransformCommand(cmds, desc))
                else:
                    piece_item.push_transform_command(
                        self._drag_start_piece_x, self._drag_start_piece_y,
                        self._drag_start_rot, self._drag_start_scale, desc,
                    )
            scene = self.scene()
            if scene and hasattr(scene, 'piece_moved_signal'):
                scene.piece_moved_signal.emit()
            self._drag_group_starts = {}
        else:
            super().mouseReleaseEvent(event)


class TransformHandles:
    """Manages 9 handle items attached to a PuzzlePieceItem."""

    def __init__(self, piece_item):
        self.piece_item = piece_item
        self.handles = []
        # Corners: TL, TR, BR, BL
        for i in range(4):
            h = HandleItem(HandleItem.CORNER, i, piece_item)
            h.setVisible(False)
            self.handles.append(h)
        # Edge midpoints: T, R, B, L
        for i in range(4):
            h = HandleItem(HandleItem.EDGE, i, piece_item)
            h.setVisible(False)
            self.handles.append(h)
        # Rotation handle
        rot = HandleItem(HandleItem.ROTATE, 0, piece_item)
        rot.setVisible(False)
        rot.setRect(-6, -6, 12, 12)  # circle-ish
        rot.setBrush(QBrush(QColor(255, 200, 0)))
        self.handles.append(rot)

    def update_positions(self):
        piece = self.piece_item.piece
        w, h = piece._cropped_size()
        ds = piece.display_scale
        dw, dh = w * ds, h * ds  # display-pixmap dimensions

        # In display pixmap space (before piece transform):
        # corners: TL, TR, BR, BL
        corners = [(0, 0), (dw, 0), (dw, dh), (0, dh)]
        for i, (cx, cy) in enumerate(corners):
            self.handles[i].setPos(cx, cy)

        # Edge midpoints
        mids = [
            (dw / 2, 0),
            (dw, dh / 2),
            (dw / 2, dh),
            (0, dh / 2),
        ]
        for i, (cx, cy) in enumerate(mids):
            self.handles[4 + i].setPos(cx, cy)

        # Rotation handle: above top center
        self.handles[8].setPos(dw / 2, -ROTATE_HANDLE_DIST)

    def set_visible(self, visible: bool):
        for h in self.handles:
            h.setVisible(visible)


class PuzzlePieceItem(QGraphicsPixmapItem):
    def __init__(self, piece: PuzzlePiece, scene_ref):
        super().__init__(piece.display_pixmap)
        self.piece = piece
        self.scene_ref = scene_ref
        self._syncing = False

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        )
        # Selection is managed via thumbnail panel, not canvas clicks
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setTransformOriginPoint(self.boundingRect().center())
        self.transform_handles = TransformHandles(self)
        self.setZValue(self._base_z())
        self.sync_from_piece()

    def _base_z(self) -> float:
        """Legends sit above the map, which is where they are read."""
        return 1.0 if self.piece.is_legend else 0.0

    def sync_from_piece(self):
        self._syncing = True
        piece = self.piece

        if self.pixmap() is not piece.display_pixmap:
            self.setPixmap(piece.display_pixmap)

        # Use actual pixmap center for both pos offset and transformOriginPoint
        # so they are consistent. Using float w*ds would mismatch with the
        # int-truncated pixmap dimensions, causing a multi-pixel visual offset.
        pm = self.pixmap()
        dcx = pm.width() / 2.0
        dcy = pm.height() / 2.0

        # Qt applies setScale/setRotation around transformOriginPoint (dcx, dcy).
        # The scene position of the origin is always pos + origin, regardless of
        # rotation/scale. So to place the center at (piece.x, piece.y):
        self.setPos(piece.x - dcx, piece.y - dcy)
        self.setRotation(piece.rotation_deg)
        self.setScale(piece.scale / piece.display_scale)

        self.setTransformOriginPoint(dcx, dcy)
        if self.transform_handles.handles[0].isVisible():
            self.transform_handles.update_positions()
        self._syncing = False

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            is_sel = bool(value)
            self.setZValue(5 if is_sel else self._base_z())
            self.transform_handles.set_visible(is_sel)
            if self.scene_ref and hasattr(self.scene_ref, 'piece_selection_changed'):
                self.scene_ref.piece_selection_changed(self.piece, is_sel)
        return super().itemChange(change, value)

    def push_transform_command(self, old_x, old_y, old_rot, old_scale, desc):
        if not self.scene_ref or not hasattr(self.scene_ref, 'undo_stack'):
            return
        undo_stack = self.scene_ref.undo_stack
        if undo_stack is None:
            return
        from .undo import TransformCommand
        p = self.piece
        cmd = TransformCommand(
            p, old_x, old_y, old_rot, old_scale,
            p.x, p.y, p.rotation_deg, p.scale,
            sync_fn=self._undo_sync, description=desc,
        )
        undo_stack.push(cmd)

    def _undo_sync(self, piece):
        """Called by undo/redo to sync visuals."""
        item = self.scene_ref.piece_items.get(piece) if self.scene_ref else None
        if item:
            item.sync_from_piece()

    def paint(self, painter, option, widget=None):
        if self.isSelected():
            painter.setOpacity(0.5)
        else:
            painter.setOpacity(1.0)
        super().paint(painter, option, widget)
        if self.isSelected():
            painter.setOpacity(1.0)
            pen = QPen(QColor(220, 30, 30), 2)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawRect(self.boundingRect())

    def rebuild_pixmap(self):
        self.piece.rebuild_display_pixmap()
        self.setPixmap(self.piece.display_pixmap)
        self.setTransformOriginPoint(self.boundingRect().center())
        self.transform_handles.update_positions()
        self.sync_from_piece()


class PuzzleScene(QGraphicsScene):
    selection_changed_signal = Signal(object)  # piece or None
    piece_moved_signal = Signal()  # emitted after any piece drag

    def __init__(self, undo_stack=None):
        super().__init__()
        self.piece_items: dict = {}  # PuzzlePiece -> PuzzlePieceItem
        self.snap_pairs: list = []   # list of (PuzzlePiece, PuzzlePiece) tuples
        self.snap_transforms: dict = {}  # frozenset({a,b}) -> (tx,ty,rot,scale)
        self.undo_stack = undo_stack
        self.editing_enabled = True
        self._anchor_item = None  # export-pivot indicator for a selected legend
        self.setSceneRect(-50000, -50000, 100000, 100000)
        self.piece_moved_signal.connect(self.update_legend_anchor)

    def add_piece(self, piece: PuzzlePiece) -> PuzzlePieceItem:
        item = PuzzlePieceItem(piece, self)
        self.addItem(item)
        self.piece_items[piece] = item
        self.update_scene_rect()
        return item

    def remove_piece(self, piece: PuzzlePiece):
        item = self.piece_items.pop(piece, None)
        if item:
            self.removeItem(item)
            self.snap_pairs = [(a, b) for a, b in self.snap_pairs
                               if a is not piece and b is not piece]
            pid = id(piece)
            self.snap_transforms = {k: v for k, v in self.snap_transforms.items()
                                    if pid not in k}
            self.update_scene_rect()

    def update_scene_rect(self):
        MARGIN = 20000
        items_rect = self.itemsBoundingRect()
        if items_rect.isEmpty():
            self.setSceneRect(-50000, -50000, 100000, 100000)
        else:
            self.setSceneRect(items_rect.adjusted(-MARGIN, -MARGIN, MARGIN, MARGIN))

    def get_selected_piece(self) -> Optional[PuzzlePiece]:
        selected = self.selectedItems()
        if selected:
            item = selected[0]
            if isinstance(item, PuzzlePieceItem):
                return item.piece
        return None

    def get_neighbors(self, piece: PuzzlePiece, margin=500) -> list:
        """Pieces close enough to `piece` to be worth matching against.

        Legends never qualify: they carry no map content to align to, and a
        legend must keep the position the user chose for it.
        """
        if piece.is_legend:
            return []
        bb = piece.get_bounding_box()
        expanded = bb.adjusted(-margin, -margin, margin, margin)
        neighbors = []
        for p, item in self.piece_items.items():
            if p is piece or p.is_legend:
                continue
            if p.get_bounding_box().intersects(expanded):
                neighbors.append(p)
        return neighbors

    # --- Snap group management ---

    def add_snap_pair(self, a: PuzzlePiece, b: PuzzlePiece):
        """Record that pieces a and b are snapped together."""
        for pa, pb in self.snap_pairs:
            if (pa is a and pb is b) or (pa is b and pb is a):
                self.record_snap_transform(a, b)
                return  # already exists
        self.snap_pairs.append((a, b))
        self.record_snap_transform(a, b)

    def remove_snap_pair(self, a: PuzzlePiece, b: PuzzlePiece):
        """Remove a snap pair."""
        self.snap_pairs = [(pa, pb) for pa, pb in self.snap_pairs
                           if not ((pa is a and pb is b) or (pa is b and pb is a))]
        self.snap_transforms.pop(frozenset((id(a), id(b))), None)

    @staticmethod
    def _relative_transform(a: PuzzlePiece, b: PuzzlePiece):
        """Compute M_A^{-1} * M_B decomposed as (tx, ty, rot_deg, scale)."""
        ma = np.vstack([a.get_affine_matrix(), [0, 0, 1]])
        mb = np.vstack([b.get_affine_matrix(), [0, 0, 1]])
        rel = np.linalg.inv(ma) @ mb
        s = math.sqrt(rel[0, 0] ** 2 + rel[1, 0] ** 2)
        r = math.degrees(math.atan2(rel[1, 0], rel[0, 0]))
        return rel[0, 2], rel[1, 2], r, s

    def record_snap_transform(self, a: PuzzlePiece, b: PuzzlePiece):
        """Store the current relative transform for a snap pair."""
        self.snap_transforms[frozenset((id(a), id(b)))] = self._relative_transform(a, b)

    def is_snap_valid(self, a: PuzzlePiece, b: PuzzlePiece) -> bool:
        """Check if a snap pair's relative transform is still valid."""
        key = frozenset((id(a), id(b)))
        stored = self.snap_transforms.get(key)
        if stored is None:
            return False
        tx0, ty0, r0, s0 = stored
        tx1, ty1, r1, s1 = self._relative_transform(a, b)
        if abs(tx1 - tx0) > 1.0 or abs(ty1 - ty0) > 1.0:
            return False
        dr = (r1 - r0 + 180) % 360 - 180
        if abs(dr) > 0.05:
            return False
        if abs(s1 - s0) > 0.001:
            return False
        return True

    def get_snap_group(self, piece: PuzzlePiece) -> list:
        """Return all pieces connected to piece via snap pairs (including piece itself)."""
        if not self.snap_pairs:
            return [piece]
        group = set()
        queue = [piece]
        while queue:
            current = queue.pop()
            if current in group:
                continue
            group.add(current)
            for a, b in self.snap_pairs:
                if a is current and b not in group:
                    queue.append(b)
                elif b is current and a not in group:
                    queue.append(a)
        return list(group)

    def get_drag_group(self, piece: PuzzlePiece) -> list:
        """Return all pieces that should move together when dragging piece.

        Unlocked pieces move individually. Locked pieces move all locked pieces.
        Legends always move alone — they are positioned against the map by hand.
        """
        if not piece.is_locked or piece.is_legend:
            return [piece]
        return [p for p in self.piece_items if p.is_locked and not p.is_legend]

    def piece_selection_changed(self, piece: PuzzlePiece, selected: bool):
        if selected:
            self.selection_changed_signal.emit(piece)
        else:
            if not self.selectedItems():
                self.selection_changed_signal.emit(None)
        self.update_legend_anchor()

    def update_legend_anchor(self):
        """Mark the point a selected legend will be pivoted on at export time.

        The export rotates a legend upright about the point of it nearest the
        map, so that is the one spot guaranteed not to move. Showing it makes the
        swing the rest of the legend takes predictable.
        """
        if self._anchor_item is not None:
            self.removeItem(self._anchor_item)
            self._anchor_item = None
        piece = self.get_selected_piece()
        if piece is None or not piece.is_legend:
            return
        pair = anchor_pair(piece, list(self.piece_items))
        if pair is None:
            return
        anchor, target = pair
        ax, ay = float(anchor[0]), float(anchor[1])
        bb = piece.get_bounding_box()
        r = max(4.0, 0.03 * max(bb.width(), bb.height()))
        path = QPainterPath()
        path.moveTo(ax, ay)
        path.lineTo(float(target[0]), float(target[1]))
        path.addEllipse(QRectF(ax - r, ay - r, 2 * r, 2 * r))
        item = QGraphicsPathItem(path)
        pen = QPen(QColor(60, 190, 110), 2)
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        item.setPen(pen)
        item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        item.setZValue(6)
        item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.addItem(item)
        self._anchor_item = item

    def sync_all_from_pieces(self):
        for piece, item in self.piece_items.items():
            item.sync_from_piece()
        self.update_scene_rect()
        self.update_legend_anchor()


class PuzzleView(QGraphicsView):
    select_next = Signal()
    select_prev = Signal()

    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMouseTracking(True)
        self._panning = False
        self._pan_start = None
        self._space_held = False
        self._zoom = 1.0
        # Drag state for move/rotate
        self._drag_mode = None  # 'move' or 'rotate'
        self._drag_start_scene = None
        self._drag_start_x = 0.0
        self._drag_start_y = 0.0
        self._drag_start_rot = 0.0
        self._drag_start_scale = 1.0
        self._drag_piece_item = None
        self._drag_group_starts = {}  # {PuzzlePiece: (x, y, rot, scale)}
        self._scroll_anim = None

    def smoothCenterOn(self, x, y, duration_ms=500):
        """Animate scrolling to center on (x, y) in scene coordinates."""
        if self._scroll_anim:
            self._scroll_anim.stop()
        target = self.mapFromScene(QPointF(x, y))
        viewport_center = QPointF(self.viewport().width() / 2.0,
                                  self.viewport().height() / 2.0)
        dx = target.x() - viewport_center.x()
        dy = target.y() - viewport_center.y()
        start_h = self.horizontalScrollBar().value()
        start_v = self.verticalScrollBar().value()
        anim = QVariantAnimation(self)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setDuration(duration_ms)
        anim.setEasingCurve(QEasingCurve.Type.InOutCubic)

        def _step(value):
            self.horizontalScrollBar().setValue(int(start_h + dx * value))
            self.verticalScrollBar().setValue(int(start_v + dy * value))

        anim.valueChanged.connect(_step)
        anim.finished.connect(lambda: setattr(self, '_scroll_anim', None))
        self._scroll_anim = anim
        anim.start()

    def _get_selected_item(self):
        scene = self.scene()
        if not isinstance(scene, PuzzleScene):
            return None
        piece = scene.get_selected_piece()
        if piece:
            return scene.piece_items.get(piece)
        return None

    def _hit_test_piece(self, view_pos):
        """Return 'inside' if view_pos is over the selected piece, else 'outside'."""
        item = self._get_selected_item()
        if item is None:
            return None
        scene_pos = self.mapToScene(view_pos)
        item_pos = item.mapFromScene(scene_pos)
        if item.boundingRect().contains(item_pos):
            return 'inside'
        return 'outside'

    def _update_hover_cursor(self, view_pos):
        if self._space_held:
            return
        # Check if hovering over a handle
        item_under = self.itemAt(view_pos)
        if isinstance(item_under, HandleItem):
            if item_under.handle_type == HandleItem.ROTATE:
                self.setCursor(_get_rotate_cursor())
            else:
                piece_item = item_under.parentItem()
                self.setCursor(_resize_cursor_for_handle(item_under, piece_item))
            return
        hit = self._hit_test_piece(view_pos)
        if hit == 'inside':
            self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
        elif hit == 'outside':
            self.setCursor(_get_rotate_cursor())
        else:
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    def event(self, ev):
        if ev.type() == QEvent.Type.KeyPress:
            if ev.key() == Qt.Key.Key_Tab:
                self.select_next.emit()
                return True
            if ev.key() == Qt.Key.Key_Backtab:
                self.select_prev.emit()
                return True
        return super().event(ev)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 1.25 if delta > 0 else 0.8
        new_zoom = self._zoom * factor
        new_zoom = max(0.01, min(10.0, new_zoom))
        factor = new_zoom / self._zoom
        self._zoom = new_zoom
        self.scale(factor, factor)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self._space_held = True
            self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self._space_held = False
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton or \
                (event.button() == Qt.MouseButton.LeftButton and self._space_held):
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
            event.accept()
            return

        # Let handles handle themselves
        item_under = self.itemAt(event.pos())
        if isinstance(item_under, HandleItem):
            super().mousePressEvent(event)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            item = self._get_selected_item()
            if item is not None and self.scene().editing_enabled:
                scene_pos = self.mapToScene(event.pos())
                hit = self._hit_test_piece(event.pos())
                piece = item.piece
                self._drag_piece_item = item
                self._drag_start_scene = scene_pos
                self._drag_start_x = piece.x
                self._drag_start_y = piece.y
                self._drag_start_rot = piece.rotation_deg
                self._drag_start_scale = piece.scale
                # Capture start state of all drag group members
                scene = self.scene()
                group = scene.get_drag_group(piece) if isinstance(scene, PuzzleScene) else [piece]
                self._drag_group_starts = {p: (p.x, p.y, p.rotation_deg, p.scale) for p in group}
                if hit == 'inside':
                    self._drag_mode = 'move'
                    self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
                else:
                    self._drag_mode = 'rotate'
                    self.setCursor(_get_rotate_cursor())
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning and self._pan_start is not None:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            event.accept()
            return

        if self._drag_mode and self._drag_piece_item:
            scene_pos = self.mapToScene(event.pos())
            piece = self._drag_piece_item.piece
            scene = self.scene()

            if self._drag_mode == 'move':
                dx = scene_pos.x() - self._drag_start_scene.x()
                dy = scene_pos.y() - self._drag_start_scene.y()
                for p, (sx, sy, sr, ss) in self._drag_group_starts.items():
                    p.x = sx + dx
                    p.y = sy + dy
                    item = scene.piece_items.get(p)
                    if item:
                        item.sync_from_piece()
            elif self._drag_mode == 'rotate':
                # Rotate around selected piece's (fixed) center
                cx, cy = self._drag_start_x, self._drag_start_y
                start_angle = math.degrees(math.atan2(
                    self._drag_start_scene.y() - cy,
                    self._drag_start_scene.x() - cx,
                ))
                current_angle = math.degrees(math.atan2(
                    scene_pos.y() - cy,
                    scene_pos.x() - cx,
                ))
                delta_rot = current_angle - start_angle
                rad = math.radians(delta_rot)
                cos_d = math.cos(rad)
                sin_d = math.sin(rad)
                for p, (sx, sy, sr, ss) in self._drag_group_starts.items():
                    dx = sx - self._drag_start_x
                    dy = sy - self._drag_start_y
                    p.x = self._drag_start_x + cos_d * dx - sin_d * dy
                    p.y = self._drag_start_y + sin_d * dx + cos_d * dy
                    p.rotation_deg = sr + delta_rot
                    item = scene.piece_items.get(p)
                    if item:
                        item.sync_from_piece()
            event.accept()
            return

        # Hover cursor update
        self._update_hover_cursor(event.pos())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._panning:
            self._panning = False
            if self._space_held:
                self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
            else:
                self._update_hover_cursor(event.pos())
            event.accept()
            return

        if self._drag_mode and self._drag_piece_item:
            piece = self._drag_piece_item.piece
            changed = (piece.x != self._drag_start_x or
                       piece.y != self._drag_start_y or
                       piece.rotation_deg != self._drag_start_rot or
                       piece.scale != self._drag_start_scale)
            if changed:
                desc = "Move" if self._drag_mode == 'move' else "Rotate"
                self._push_group_transform_command(desc)
            scene = self.scene()
            if hasattr(scene, 'piece_moved_signal'):
                scene.piece_moved_signal.emit()
            self._drag_mode = None
            self._drag_piece_item = None
            self._drag_group_starts = {}
            self._update_hover_cursor(event.pos())
            event.accept()
            return

        super().mouseReleaseEvent(event)
        self._update_hover_cursor(event.pos())

    def _push_group_transform_command(self, desc):
        """Push undo command(s) for all pieces in the drag group that changed."""
        scene = self.scene()
        if not isinstance(scene, PuzzleScene) or not scene.undo_stack:
            return
        from .undo import TransformCommand, GroupTransformCommand
        cmds = []
        for p, (ox, oy, orot, oscale) in self._drag_group_starts.items():
            if p.x != ox or p.y != oy or p.rotation_deg != orot or p.scale != oscale:
                item = scene.piece_items.get(p)
                sync_fn = item._undo_sync if item else None
                cmds.append(TransformCommand(
                    p, ox, oy, orot, oscale,
                    p.x, p.y, p.rotation_deg, p.scale,
                    sync_fn=sync_fn, description=desc,
                ))
        if not cmds:
            return
        if len(cmds) == 1:
            scene.undo_stack.push(cmds[0])
        else:
            scene.undo_stack.push(GroupTransformCommand(cmds, desc))

    def fit_all(self):
        rect = self.scene().itemsBoundingRect()
        if not rect.isEmpty():
            self.fitInView(rect.adjusted(-50, -50, 50, 50), Qt.AspectRatioMode.KeepAspectRatio)
            # Update zoom tracking
            t = self.transform()
            self._zoom = t.m11()


class ThumbnailWidget(QWidget):
    clicked = Signal(object)  # PuzzlePiece
    right_clicked = Signal(object, object)  # PuzzlePiece, QPoint

    def __init__(self, piece: PuzzlePiece):
        super().__init__()
        self.piece = piece
        self._selected = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(1)

        self.thumb_label = QLabel()
        self.thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb_label.setScaledContents(False)
        self.thumb_label.setFixedHeight(64)
        layout.addWidget(self.thumb_label)

        self.name_label = QLabel()
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.name_label.setWordWrap(False)
        font = QFont()
        font.setPointSize(9)
        self.name_label.setFont(font)
        layout.addWidget(self.name_label)

        self.update_display()
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_right_click)

    def update_display(self):
        piece = self.piece
        if piece.thumbnail:
            scaled = piece.thumbnail.scaled(
                94, 64,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            if piece.is_legend:
                # Legends are never matched, so the unmatched tint below would
                # mark every one of them as a problem.
                self.thumb_label.setPixmap(scaled)
            elif piece.is_locked:
                # Draw lock overlay
                pm = QPixmap(scaled.size())
                pm.fill(Qt.GlobalColor.transparent)
                painter = QPainter(pm)
                painter.drawPixmap(0, 0, scaled)
                painter.setPen(QPen(QColor(255, 200, 0), 2))
                painter.setFont(QFont("monospace", 14, QFont.Weight.Bold))
                painter.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, "L")
                painter.end()
                self.thumb_label.setPixmap(pm)
            elif not piece.is_matched:
                # Red tint
                pm = QPixmap(scaled.size())
                pm.fill(Qt.GlobalColor.transparent)
                painter = QPainter(pm)
                painter.drawPixmap(0, 0, scaled)
                painter.fillRect(pm.rect(), QColor(255, 0, 0, 60))
                painter.end()
                self.thumb_label.setPixmap(pm)
            else:
                self.thumb_label.setPixmap(scaled)

        name = piece.display_name or os.path.basename(piece.source_path)
        extra = ""
        if not piece.is_legend:
            if piece.is_locked:
                extra = " [L]"
            elif not piece.is_matched:
                extra = " (!)"
        self.name_label.setText(name + extra)

        # Update border, preserving selection highlight
        if not self._selected:
            self.setStyleSheet(f"QWidget {{ border: 2px solid {self._border_colour()}; }}")

    def _border_colour(self) -> str:
        if self.piece.is_legend:
            return "#44aa66"
        if self.piece.is_locked:
            return "orange"
        return "transparent"

    def set_selected(self, selected: bool):
        self._selected = selected
        if selected:
            self.setStyleSheet("QWidget { border: 2px solid #4488ff; background: #ddeeff; }")
        else:
            self.setStyleSheet(f"QWidget {{ border: 2px solid {self._border_colour()}; }}")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.piece)
        super().mousePressEvent(event)

    def _on_right_click(self, pos):
        self.right_clicked.emit(self.piece, self.mapToGlobal(pos))


class ThumbnailPanel(QWidget):
    piece_selected = Signal(object)        # PuzzlePiece
    add_legend_requested = Signal(object)  # source page PuzzlePiece
    duplicate_requested = Signal(object)   # legend PuzzlePiece
    delete_requested = Signal(object)      # PuzzlePiece

    def __init__(self):
        super().__init__()
        self.setFixedHeight(105)
        self.thumb_widgets: dict = {}  # PuzzlePiece -> ThumbnailWidget

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        outer_layout.addWidget(self.scroll_area)

        self.container = QWidget()
        self.inner_layout = QHBoxLayout(self.container)
        self.inner_layout.setContentsMargins(4, 2, 4, 2)
        self.inner_layout.setSpacing(4)
        self.inner_layout.addStretch()
        self.scroll_area.setWidget(self.container)

    def add_piece(self, piece: PuzzlePiece):
        w = ThumbnailWidget(piece)
        w.clicked.connect(self._on_thumb_clicked)
        w.right_clicked.connect(self._on_thumb_right_clicked)
        self.thumb_widgets[piece] = w
        # Insert before the stretch
        idx = self.inner_layout.count() - 1
        self.inner_layout.insertWidget(idx, w)

    def remove_piece(self, piece: PuzzlePiece):
        w = self.thumb_widgets.pop(piece, None)
        if w:
            self.inner_layout.removeWidget(w)
            w.deleteLater()

    def update_piece(self, piece: PuzzlePiece):
        w = self.thumb_widgets.get(piece)
        if w:
            w.update_display()

    def update_all(self):
        for w in self.thumb_widgets.values():
            w.update_display()

    def select_piece(self, piece: Optional[PuzzlePiece]):
        for p, w in self.thumb_widgets.items():
            w.set_selected(p is piece)
        if piece:
            w = self.thumb_widgets.get(piece)
            if w:
                self.scroll_area.ensureWidgetVisible(w)

    def _on_thumb_clicked(self, piece: PuzzlePiece):
        self.piece_selected.emit(piece)

    def _on_thumb_right_clicked(self, piece: PuzzlePiece, global_pos):
        menu = QMenu(self)
        lock_action = None
        if piece.is_legend:
            dup_action = menu.addAction("Duplicate Legend")
            del_action = menu.addAction("Delete Legend")
            add_action = None
        else:
            lock_action = menu.addAction("Unlock" if piece.is_locked else "Lock")
            menu.addSeparator()
            add_action = menu.addAction("Add Legend From This Page...")
            dup_action = del_action = None

        chosen = menu.exec(global_pos)
        if chosen is None:
            return
        if chosen is lock_action:
            piece.is_locked = not piece.is_locked
            self.update_piece(piece)
            self.piece_selected.emit(piece)  # trigger scene update
        elif chosen is add_action:
            self.add_legend_requested.emit(piece)
        elif chosen is dup_action:
            self.duplicate_requested.emit(piece)
        elif chosen is del_action:
            self.delete_requested.emit(piece)


class ThumbnailWorkerSignals(QObject):
    """Signals for ThumbnailWorker: emits (page_num, QPixmap) when done."""
    ready = Signal(int, object)  # page_num (1-indexed), QPixmap


class ThumbnailWorker(QRunnable):
    """Renders one PDF page to a QPixmap thumbnail using pdftoppm subprocess."""

    def __init__(self, pdf_path: str, page_num: int):
        super().__init__()
        self.pdf_path = pdf_path
        self.page_num = page_num
        self.signals = ThumbnailWorkerSignals()

    def run(self):
        try:
            # pdftocairo supports writing to stdout with -singlefile and -
            result = subprocess.run(
                [
                    *LOW_PRIO, "pdftocairo", "-png", "-singlefile",
                    "-f", str(self.page_num),
                    "-l", str(self.page_num),
                    "-scale-to", "200",
                    self.pdf_path, "-",
                ],
                capture_output=True,
            )
            png_bytes = result.stdout
            if not png_bytes:
                # Fallback to ImageMagick convert
                page_idx = self.page_num - 1
                result = subprocess.run(
                    [
                        *LOW_PRIO, "convert", "-density", "72",
                        f"{self.pdf_path}[{page_idx}]",
                        "-resize", "200x200",
                        "png:-",
                    ],
                    capture_output=True,
                )
                png_bytes = result.stdout
            if not png_bytes:
                return
            pixmap = QPixmap()
            pixmap.loadFromData(png_bytes, "PNG")
            if not pixmap.isNull():
                self.signals.ready.emit(self.page_num, pixmap)
        except Exception:
            pass


class PdfImportDialog(QDialog):
    THUMB_COLS = 4

    def __init__(self, pdf_path: str, page_count: int, page_sizes: dict = None, parent=None):
        super().__init__(parent)
        self.pdf_path = pdf_path
        self.page_count = page_count if page_count else 0
        self.page_sizes = page_sizes or {}  # {page_num: (width_pts, height_pts)}
        self._selected_pages: set = set()  # 1-indexed page numbers
        self._thumb_labels: dict = {}      # page_num -> QLabel (pixmap container)
        self._thumb_frames: dict = {}      # page_num -> QFrame (border highlight)
        self._updating_text = False        # guard re-entrant text/selection sync

        self.setWindowTitle("Import PDF")
        self.setMinimumSize(620, 560)

        root_layout = QVBoxLayout(self)
        root_layout.setSpacing(8)

        # --- Top info row ---
        info_layout = QFormLayout()
        info_layout.addRow("File:", QLabel(os.path.basename(pdf_path)))
        info_layout.addRow(
            "Total pages:",
            QLabel(str(page_count) if page_count else "Unknown"),
        )
        root_layout.addLayout(info_layout)

        # --- Help text ---
        help_label = QLabel(
            "Enter page numbers like: <b>1-3,5,7-8</b> or <b>all</b> for all pages"
        )
        help_label.setTextFormat(Qt.TextFormat.RichText)
        root_layout.addWidget(help_label)

        # --- Page range field ---
        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Pages to import:"))
        self.page_range_edit = QLineEdit("all")
        self.page_range_edit.setPlaceholderText("e.g. 1-3,5,7-8 or all")
        self.page_range_edit.textChanged.connect(self._on_text_changed)
        range_row.addWidget(self.page_range_edit)
        root_layout.addLayout(range_row)

        # --- Select All / Deselect All buttons ---
        sel_row = QHBoxLayout()
        sel_all_btn = QPushButton("Select All")
        sel_all_btn.clicked.connect(self._select_all)
        desel_all_btn = QPushButton("Deselect All")
        desel_all_btn.clicked.connect(self._deselect_all)
        sel_row.addWidget(sel_all_btn)
        sel_row.addWidget(desel_all_btn)
        sel_row.addStretch()
        root_layout.addLayout(sel_row)

        # --- Thumbnail grid inside a scroll area ---
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        self._grid_widget = QWidget()
        self._grid_layout = QGridLayout(self._grid_widget)
        self._grid_layout.setSpacing(10)
        self._scroll_area.setWidget(self._grid_widget)
        root_layout.addWidget(self._scroll_area, stretch=1)

        # --- Dialog buttons ---
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root_layout.addWidget(buttons)

        # Build placeholder cells and start async thumbnail loading
        if self.page_count > 0:
            self._build_placeholder_grid()
            # "all" means all pages selected — initialize highlights
            self._selected_pages = set(range(1, self.page_count + 1))
            self._refresh_all_highlights()
            self._start_thumbnail_workers()

    # ------------------------------------------------------------------
    # Grid construction
    # ------------------------------------------------------------------

    def _thumb_size_for_page(self, page_num: int) -> tuple:
        """Return (thumb_w, thumb_h) scaled to fit within max 118x140, preserving aspect ratio."""
        max_w, max_h = 118, 140
        size = self.page_sizes.get(page_num)
        if size:
            pw, ph = size
            if pw > 0 and ph > 0:
                scale = min(max_w / pw, max_h / ph)
                return max(1, int(pw * scale)), max(1, int(ph * scale))
        return max_w, max_h

    def _build_placeholder_grid(self):
        for page_num in range(1, self.page_count + 1):
            row, col = divmod(page_num - 1, self.THUMB_COLS)

            tw, th = self._thumb_size_for_page(page_num)

            # Outer frame provides the selection highlight border
            frame = QFrame()
            frame.setFixedSize(tw + 12, th + 30)
            frame.setStyleSheet(
                "QFrame { border: 2px solid transparent; border-radius: 4px; }"
            )
            frame_layout = QVBoxLayout(frame)
            frame_layout.setContentsMargins(4, 4, 4, 4)
            frame_layout.setSpacing(4)

            # Pixmap label (placeholder grey box with correct aspect ratio)
            thumb_label = QLabel()
            thumb_label.setFixedSize(tw, th)
            thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumb_label.setStyleSheet(
                "background: #cccccc; border: none;"
            )
            thumb_label.setText("…")

            # Page number label
            page_label = QLabel(str(page_num))
            page_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            font = QFont()
            font.setPointSize(9)
            page_label.setFont(font)

            frame_layout.addWidget(thumb_label)
            frame_layout.addWidget(page_label)

            self._thumb_labels[page_num] = thumb_label
            self._thumb_frames[page_num] = frame

            # Make the frame clickable by installing an event filter
            frame.setProperty("page_num", page_num)
            frame.installEventFilter(self)
            thumb_label.setProperty("page_num", page_num)
            thumb_label.installEventFilter(self)

            self._grid_layout.addWidget(frame, row, col)

    # ------------------------------------------------------------------
    # Async thumbnail loading
    # ------------------------------------------------------------------

    def _start_thumbnail_workers(self):
        pool = QThreadPool.globalInstance()
        for page_num in range(1, self.page_count + 1):
            worker = ThumbnailWorker(self.pdf_path, page_num)
            worker.signals.ready.connect(self._on_thumbnail_ready)
            pool.start(worker)

    def _on_thumbnail_ready(self, page_num: int, pixmap):
        label = self._thumb_labels.get(page_num)
        if label is None:
            return
        tw, th = self._thumb_size_for_page(page_num)
        scaled = pixmap.scaled(
            tw, th,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(scaled)
        label.setText("")

    # ------------------------------------------------------------------
    # Selection logic
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent
        if event.type() == QEvent.Type.MouseButtonPress:
            page_num = obj.property("page_num")
            if page_num is not None:
                self._toggle_page(int(page_num))
                return True
        return super().eventFilter(obj, event)

    def _toggle_page(self, page_num: int):
        if page_num in self._selected_pages:
            self._selected_pages.discard(page_num)
        else:
            self._selected_pages.add(page_num)
        self._refresh_highlight(page_num)
        self._sync_text_from_selection()

    def _refresh_highlight(self, page_num: int):
        frame = self._thumb_frames.get(page_num)
        if frame is None:
            return
        if page_num in self._selected_pages:
            frame.setStyleSheet(
                "QFrame { border: 2px solid #0078d4; border-radius: 4px;"
                " background: #ddeeff; }"
            )
        else:
            frame.setStyleSheet(
                "QFrame { border: 2px solid transparent; border-radius: 4px; }"
            )

    def _refresh_all_highlights(self):
        for page_num in self._thumb_frames:
            self._refresh_highlight(page_num)

    def _select_all(self):
        self._selected_pages = set(range(1, self.page_count + 1))
        self._refresh_all_highlights()
        self._sync_text_from_selection()

    def _deselect_all(self):
        self._selected_pages = set()
        self._refresh_all_highlights()
        self._sync_text_from_selection()

    # ------------------------------------------------------------------
    # Text <-> selection synchronisation
    # ------------------------------------------------------------------

    def _sync_text_from_selection(self):
        """Update the text field to reflect the current thumbnail selection."""
        if self._updating_text:
            return
        self._updating_text = True
        try:
            if not self._selected_pages:
                self.page_range_edit.setText("")
            elif self._selected_pages == set(range(1, self.page_count + 1)):
                self.page_range_edit.setText("all")
            else:
                pages = sorted(self._selected_pages)
                parts = []
                start = pages[0]
                end = pages[0]
                for p in pages[1:]:
                    if p == end + 1:
                        end = p
                    else:
                        parts.append(str(start) if start == end else f"{start}-{end}")
                        start = end = p
                parts.append(str(start) if start == end else f"{start}-{end}")
                self.page_range_edit.setText(",".join(parts))
        finally:
            self._updating_text = False

    def _on_text_changed(self, text: str):
        """Update thumbnail highlights when the user edits the text field."""
        if self._updating_text:
            return
        self._updating_text = True
        try:
            parsed = parse_page_range(text)
            if parsed is None:
                # "all" or empty meaning all
                self._selected_pages = set(range(1, self.page_count + 1))
            else:
                self._selected_pages = {
                    p for p in parsed if 1 <= p <= self.page_count
                }
            self._refresh_all_highlights()
        except Exception:
            pass
        finally:
            self._updating_text = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_pages(self):
        """Return sorted list of 1-indexed page numbers, or None for all pages."""
        return parse_page_range(self.page_range_edit.text())


_CROP_HANDLE_SIZE = 10


class CropHandleItem(QGraphicsRectItem):
    """Small square knob on the crop rectangle edge/corner."""

    # index: 0-3 corners (TL, TR, BR, BL), 4-7 edges (T, R, B, L)
    _CURSORS = [
        Qt.CursorShape.SizeFDiagCursor,   # TL
        Qt.CursorShape.SizeBDiagCursor,   # TR
        Qt.CursorShape.SizeFDiagCursor,   # BR
        Qt.CursorShape.SizeBDiagCursor,   # BL
        Qt.CursorShape.SizeVerCursor,     # T
        Qt.CursorShape.SizeHorCursor,     # R
        Qt.CursorShape.SizeVerCursor,     # B
        Qt.CursorShape.SizeHorCursor,     # L
    ]

    def __init__(self, index, crop_rect_item):
        hs = _CROP_HANDLE_SIZE
        super().__init__(-hs / 2, -hs / 2, hs, hs)
        self.index = index
        self.crop_rect_item = crop_rect_item
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setBrush(QBrush(QColor(255, 255, 255)))
        self.setPen(QPen(QColor(255, 80, 80), 1))
        self.setZValue(20)
        self.setCursor(QCursor(self._CURSORS[index]))
        self.setAcceptHoverEvents(True)
        self._dragging = False

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_start = event.scenePos()
            self._start_rect = QRectF(self.crop_rect_item.crop_rect)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self._dragging:
            return super().mouseMoveEvent(event)
        event.accept()
        pos = event.scenePos()
        r = QRectF(self._start_rect)
        idx = self.index
        img_rect = self.crop_rect_item.image_rect

        if idx == 0:    # TL corner
            r.setTopLeft(pos)
        elif idx == 1:  # TR corner
            r.setTopRight(pos)
        elif idx == 2:  # BR corner
            r.setBottomRight(pos)
        elif idx == 3:  # BL corner
            r.setBottomLeft(pos)
        elif idx == 4:  # T edge
            r.setTop(pos.y())
        elif idx == 5:  # R edge
            r.setRight(pos.x())
        elif idx == 6:  # B edge
            r.setBottom(pos.y())
        elif idx == 7:  # L edge
            r.setLeft(pos.x())

        # Normalize and clamp to image bounds
        r = r.normalized()
        r = r.intersected(img_rect)
        if r.width() >= 2 and r.height() >= 2:
            self.crop_rect_item.set_crop_rect(r)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class CropRectItem(QGraphicsPathItem):
    """Crop rectangle with dimmed overlay outside and 8 drag handles."""

    def __init__(self, rect: QRectF, image_rect: QRectF, scene: QGraphicsScene):
        super().__init__()
        self.crop_rect = QRectF(rect)
        self.image_rect = QRectF(image_rect)
        self._scene = scene
        scene.addItem(self)
        self.setZValue(5)
        self.setPen(QPen(Qt.PenStyle.NoPen))
        self.setBrush(QBrush(QColor(0, 0, 0, 120)))
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

        # Border pen (drawn as a child rect)
        self._border = QGraphicsRectItem(self)
        pen = QPen(QColor(255, 80, 80), 2)
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        self._border.setPen(pen)
        self._border.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self._border.setZValue(6)

        # 8 handles: 4 corners + 4 edges
        self.handles = []
        for i in range(8):
            h = CropHandleItem(i, self)
            scene.addItem(h)
            self.handles.append(h)

        self._update_path()

    def set_crop_rect(self, r: QRectF):
        self.crop_rect = QRectF(r)
        self._update_path()

    def _update_path(self):
        # Overlay: image rect minus crop rect (dimmed area outside crop)
        outer = QPainterPath()
        outer.addRect(self.image_rect)
        inner = QPainterPath()
        inner.addRect(self.crop_rect)
        self.setPath(outer.subtracted(inner))

        # Border
        self._border.setRect(self.crop_rect)

        # Handle positions
        r = self.crop_rect
        positions = [
            r.topLeft(), r.topRight(), r.bottomRight(), r.bottomLeft(),  # corners
            QPointF(r.center().x(), r.top()),     # T
            QPointF(r.right(), r.center().y()),    # R
            QPointF(r.center().x(), r.bottom()),   # B
            QPointF(r.left(), r.center().y()),     # L
        ]
        for h, p in zip(self.handles, positions):
            h.setPos(p)

    def remove_from_scene(self):
        for h in self.handles:
            self._scene.removeItem(h)
        self._scene.removeItem(self)


_LASSO_HANDLE_SIZE = 9
_LASSO_CLOSE_DIST = 14   # viewport px within which a click closes the polygon


class LassoVertexItem(QGraphicsEllipseItem):
    """Round knob on a lasso polygon — a vertex, or a midpoint that inserts one.

    Purely visual: `CropView` does the hit-testing and dragging, because moving a
    point can rebuild the whole handle list and an item holding the mouse grab
    would be destroyed under itself.
    """

    def __init__(self, index: int, midpoint: bool = False):
        hs = _LASSO_HANDLE_SIZE - (2 if midpoint else 0)
        super().__init__(-hs / 2, -hs / 2, hs, hs)
        self.index = index
        self.midpoint = midpoint
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        fill = QColor(255, 255, 255, 110) if midpoint else QColor(255, 255, 255)
        self.setBrush(QBrush(fill))
        self.setPen(QPen(QColor(255, 80, 80), 1))
        self.setZValue(21)
        self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)


class LassoItem(QGraphicsPathItem):
    """Piecewise-linear lasso: dimmed overlay outside, one draggable knob per point.

    Points live in scene coords, which are image pixels in the crop dialog.
    """

    def __init__(self, image_rect: QRectF, scene: QGraphicsScene,
                 points=None, closed=False):
        super().__init__()
        self.image_rect = QRectF(image_rect)
        self.points = [self._clamp(QPointF(*p) if isinstance(p, tuple) else QPointF(p))
                       for p in (points or [])]
        self.closed = closed
        self._cursor_point = None   # rubber-band target while still drawing
        self._scene = scene
        scene.addItem(self)
        self.setZValue(5)
        self.setPen(QPen(Qt.PenStyle.NoPen))
        self.setBrush(QBrush(QColor(0, 0, 0, 120)))
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

        self._outline = QGraphicsPathItem(self)
        pen = QPen(QColor(255, 80, 80), 2)
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        self._outline.setPen(pen)
        self._outline.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self._outline.setZValue(6)

        self.handles = []
        self._rebuild_handles()

    # --- geometry ---

    def _clamp(self, p: QPointF) -> QPointF:
        r = self.image_rect
        return QPointF(min(max(p.x(), r.left()), r.right()),
                       min(max(p.y(), r.top()), r.bottom()))

    def polygon(self) -> list:
        return [(p.x(), p.y()) for p in self.points]

    def bounding_rect(self) -> QRectF:
        xs = [p.x() for p in self.points]
        ys = [p.y() for p in self.points]
        return QRectF(QPointF(min(xs), min(ys)), QPointF(max(xs), max(ys)))

    def first_point(self) -> Optional[QPointF]:
        return self.points[0] if self.points else None

    # --- editing ---

    def add_point(self, pos: QPointF):
        p = self._clamp(pos)
        # A double-click to close arrives after a plain press, so drop the
        # duplicate point it would otherwise leave behind.
        if self.points and (self.points[-1] - p).manhattanLength() < 1.0:
            return
        self.points.append(p)
        self._rebuild_handles()

    def insert_point(self, after_index: int, pos: QPointF) -> int:
        idx = after_index + 1
        self.points.insert(idx, self._clamp(pos))
        self._rebuild_handles()
        return idx

    def set_point(self, index: int, pos: QPointF):
        if 0 <= index < len(self.points):
            self.points[index] = self._clamp(pos)
            self._update_shape()

    def remove_point(self, index: int):
        if len(self.points) > 3 and 0 <= index < len(self.points):
            del self.points[index]
            self._rebuild_handles()

    def pop_point(self):
        if self.points:
            self.points.pop()
            self._rebuild_handles()

    def close_shape(self) -> bool:
        if len(self.points) < 3:
            return False
        self.closed = True
        self._cursor_point = None
        self._rebuild_handles()
        return True

    def set_cursor_point(self, pos: Optional[QPointF]):
        if self.closed:
            return
        self._cursor_point = QPointF(pos) if pos is not None else None
        self._update_shape()

    # --- drawing ---

    def _rebuild_handles(self):
        """Match the handle items to the point list, then reposition everything."""
        wanted = [(i, False) for i in range(len(self.points))]
        if self.closed:
            wanted += [(i, True) for i in range(len(self.points))]
        for h in self.handles:
            self._scene.removeItem(h)
        self.handles = []
        for index, midpoint in wanted:
            h = LassoVertexItem(index, midpoint)
            self._scene.addItem(h)
            self.handles.append(h)
        self._update_shape()

    def _update_shape(self):
        # Dim everything outside the polygon — only meaningful once closed.
        overlay = QPainterPath()
        if self.closed and len(self.points) >= 3:
            outer = QPainterPath()
            outer.addRect(self.image_rect)
            overlay = outer.subtracted(self._polygon_path())
        self.setPath(overlay)

        outline = QPainterPath()
        if self.points:
            outline.moveTo(self.points[0])
            for p in self.points[1:]:
                outline.lineTo(p)
            if self.closed and len(self.points) >= 3:
                outline.closeSubpath()
            elif self._cursor_point is not None:
                outline.lineTo(self._cursor_point)
        self._outline.setPath(outline)

        n = len(self.points)
        for h in self.handles:
            if h.midpoint:
                a = self.points[h.index]
                b = self.points[(h.index + 1) % n]
                h.setPos((a + b) / 2.0)
            else:
                h.setPos(self.points[h.index])

    def _polygon_path(self) -> QPainterPath:
        path = QPainterPath()
        path.moveTo(self.points[0])
        for p in self.points[1:]:
            path.lineTo(p)
        path.closeSubpath()
        return path

    def remove_from_scene(self):
        for h in self.handles:
            self._scene.removeItem(h)
        self.handles = []
        self._scene.removeItem(self)


class CropView(QGraphicsView):
    """Zoomable/pannable view for the crop dialog."""

    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.viewport().setMouseTracking(True)
        self._panning = False
        self._pan_start = None
        self._zoom = 1.0
        self._mode = None  # "draw", "move", "lasso_vertex", or None
        self._drag_start = None
        self._move_start_rect = None
        self._lasso_index = None
        self.tool = "rect"  # "rect" or "lasso"
        self.crop_rect_item: CropRectItem | None = None
        self.lasso_item: LassoItem | None = None
        self._image_rect = QRectF()

    def set_image_rect(self, r: QRectF):
        self._image_rect = QRectF(r)

    def set_tool(self, tool: str):
        """Switch tools, discarding the other tool's region — only one can apply."""
        if tool == self.tool:
            return
        self.tool = tool
        if tool == "lasso" and self.crop_rect_item is not None:
            self.crop_rect_item.remove_from_scene()
            self.crop_rect_item = None
        elif tool == "rect" and self.lasso_item is not None:
            self.lasso_item.remove_from_scene()
            self.lasso_item = None

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 1.25 if delta > 0 else 0.8
        new_zoom = self._zoom * factor
        new_zoom = max(0.05, min(20.0, new_zoom))
        factor = new_zoom / self._zoom
        self._zoom = new_zoom
        self.scale(factor, factor)

    def _hit_handle(self, event_pos):
        """Check if viewport pos hits a crop handle."""
        items = self.items(event_pos)
        for item in items:
            if isinstance(item, CropHandleItem):
                return True
        return False

    def _hit_lasso_handle(self, event_pos):
        """Return the lasso knob under a viewport pos, or None."""
        for item in self.items(event_pos):
            if isinstance(item, LassoVertexItem):
                return item
        return None

    def _near_first_point(self, event_pos) -> bool:
        """True when a viewport pos is close enough to the first vertex to close."""
        first = self.lasso_item.first_point() if self.lasso_item else None
        if first is None:
            return False
        delta = self.mapFromScene(first) - event_pos
        return delta.manhattanLength() <= _LASSO_CLOSE_DIST

    def _lasso_press(self, event):
        scene_pos = self.mapToScene(event.pos())
        lasso = self.lasso_item
        if lasso is None:
            self.lasso_item = LassoItem(self._image_rect, self.scene())
            self.lasso_item.add_point(scene_pos)
        elif (not lasso.closed and len(lasso.points) >= 3
                and self._near_first_point(event.pos())):
            # Checked before the knobs: the first vertex has one sitting on it,
            # and closing the shape is what a click there means.
            lasso.close_shape()
        elif (handle := self._hit_lasso_handle(event.pos())) is not None:
            index = handle.index
            if handle.midpoint:
                index = lasso.insert_point(index, scene_pos)
            self._mode = "lasso_vertex"
            self._lasso_index = index
        elif not lasso.closed:
            lasso.add_point(scene_pos)
        event.accept()

    def mouseDoubleClickEvent(self, event):
        if (self.tool == "lasso" and event.button() == Qt.MouseButton.LeftButton
                and self.lasso_item is not None and not self.lasso_item.closed):
            self.lasso_item.close_shape()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        if self.tool == "lasso" and self.lasso_item is not None:
            key = event.key()
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if self.lasso_item.close_shape():
                    event.accept()
                    return
            elif key == Qt.Key.Key_Backspace and not self.lasso_item.closed:
                self.lasso_item.pop_point()
                if not self.lasso_item.points:
                    self.lasso_item.remove_from_scene()
                    self.lasso_item = None
                event.accept()
                return
        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
            event.accept()
        elif (self.tool == "lasso" and event.button() == Qt.MouseButton.RightButton
                and self.lasso_item is not None):
            handle = self._hit_lasso_handle(event.pos())
            if handle is not None and not handle.midpoint:
                self.lasso_item.remove_point(handle.index)
            event.accept()
        elif self.tool == "lasso" and event.button() == Qt.MouseButton.LeftButton:
            self._lasso_press(event)
        elif event.button() == Qt.MouseButton.LeftButton:
            # Let handles take priority
            if self._hit_handle(event.pos()):
                super().mousePressEvent(event)
                return
            scene_pos = self.mapToScene(event.pos())
            if self.crop_rect_item and self.crop_rect_item.crop_rect.contains(scene_pos):
                # Start moving the crop rect
                self._mode = "move"
                self._drag_start = scene_pos
                self._move_start_rect = QRectF(self.crop_rect_item.crop_rect)
                self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
                event.accept()
            else:
                # Start drawing a new crop rectangle
                self._mode = "draw"
                self._drag_start = scene_pos
                event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning and self._pan_start is not None:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            event.accept()
        elif self._mode == "lasso_vertex" and self.lasso_item is not None:
            self.lasso_item.set_point(self._lasso_index, self.mapToScene(event.pos()))
            event.accept()
        elif self.tool == "lasso":
            if self.lasso_item is not None:
                self.lasso_item.set_cursor_point(self.mapToScene(event.pos()))
            super().mouseMoveEvent(event)
        elif self._mode == "draw" and self._drag_start is not None:
            scene_pos = self.mapToScene(event.pos())
            r = QRectF(self._drag_start, scene_pos).normalized()
            r = r.intersected(self._image_rect)
            if r.width() >= 2 and r.height() >= 2:
                if self.crop_rect_item:
                    self.crop_rect_item.set_crop_rect(r)
                else:
                    self.crop_rect_item = CropRectItem(r, self._image_rect, self.scene())
            event.accept()
        elif self._mode == "move" and self._drag_start is not None:
            scene_pos = self.mapToScene(event.pos())
            delta = scene_pos - self._drag_start
            r = self._move_start_rect.translated(delta)
            img = self._image_rect
            if r.left() < img.left():
                r.moveLeft(img.left())
            if r.top() < img.top():
                r.moveTop(img.top())
            if r.right() > img.right():
                r.moveRight(img.right())
            if r.bottom() > img.bottom():
                r.moveBottom(img.bottom())
            self.crop_rect_item.set_crop_rect(r)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._panning:
            self._panning = False
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
            event.accept()
        elif self._mode is not None:
            self._mode = None
            self._drag_start = None
            self._move_start_rect = None
            self._lasso_index = None
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class CutToolDialog(QDialog):
    """Pick a region of a page, either as its crop or as a legend to lift off it.

    In "legend" mode nothing is preselected — the page's own crop is outlined as a
    reference instead, because a legend is cut from outside it.
    """

    def __init__(self, piece: PuzzlePiece, parent=None, mode: str = "crop"):
        super().__init__(parent)
        self.piece = piece
        self.mode = mode
        verb = "Select Legend Region" if mode == "legend" else "Cut"
        self.setWindowTitle(f"{verb} - {os.path.basename(piece.source_path)}")
        self.setMinimumSize(800, 600)

        layout = QVBoxLayout(self)

        # Show full UNROTATED image so user can see borders/legends
        self.scene = QGraphicsScene()
        self.view = CropView(self.scene)
        layout.addWidget(self.view)

        img = piece.image  # full uncropped image
        h, w = img.shape[:2]
        rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        if piece._base_alpha is not None:
            # Show the image unmasked so an existing lasso can be pushed outwards.
            rgba[:, :, 3] = piece._base_alpha
        qimg = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0],
                      QImage.Format.Format_RGBA8888)
        pm = QPixmap.fromImage(qimg.copy())
        self._pixmap_item = QGraphicsPixmapItem(pm)
        self.scene.addItem(self._pixmap_item)
        image_rect = QRectF(0, 0, w, h)
        self.scene.setSceneRect(image_rect)
        self.view.set_image_rect(image_rect)

        # Results the caller reads after Accepted; the dialog never touches the piece.
        self.result_crop_rect = None if mode == "legend" else piece.crop_rect
        self.result_lasso_poly = None if mode == "legend" else piece.lasso_poly

        if mode == "legend":
            self._outline_map_area()
        # Restore whichever region the piece already carries
        elif piece.lasso_poly:
            self.view.tool = "lasso"
            self.view.lasso_item = LassoItem(
                image_rect, self.scene, points=piece.lasso_poly, closed=True)
        else:
            if piece.crop_rect:
                cx, cy, cw, ch = piece.crop_rect
                crop_r = QRectF(cx, cy, cw, ch)
            else:
                crop_r = QRectF(image_rect)
            self.view.crop_rect_item = CropRectItem(crop_r, image_rect, self.scene)

        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        t = self.view.transform()
        self.view._zoom = t.m11()

        # Tool selector
        tools = QHBoxLayout()
        self.rect_btn = QPushButton("Rectangle")
        self.lasso_btn = QPushButton("Lasso")
        self._tool_group = QButtonGroup(self)
        for btn, name in ((self.rect_btn, "rect"), (self.lasso_btn, "lasso")):
            btn.setCheckable(True)
            self._tool_group.addButton(btn)
            btn.clicked.connect(lambda _checked, n=name: self._set_tool(n))
            tools.addWidget(btn)
        self.lasso_btn.setChecked(self.view.tool == "lasso")
        self.rect_btn.setChecked(self.view.tool == "rect")
        tools.addStretch(1)
        layout.addLayout(tools)

        # Buttons
        buttons = QHBoxLayout()
        self.apply_btn = QPushButton("Apply")
        self.reset_btn = QPushButton("Reset Crop")
        self.cancel_btn = QPushButton("Cancel")
        self.apply_btn.clicked.connect(self._apply)
        self.reset_btn.clicked.connect(self._reset)
        self.cancel_btn.clicked.connect(self.reject)
        if mode == "legend":
            # There is no crop to reset — an unwanted legend is just cancelled.
            self.reset_btn.hide()
        for btn in (self.apply_btn, self.reset_btn, self.cancel_btn):
            # Enter closes the lasso instead of activating a button.
            btn.setAutoDefault(False)
            buttons.addWidget(btn)
        layout.addLayout(buttons)

        self.label = QLabel()
        self.label.setWordWrap(True)
        layout.addWidget(self.label)
        self._update_hint()
        self.view.setFocus()

    _RECT_HINT = ("Drag handles to resize crop. Drag inside to move. "
                  "Draw outside to create new region.")
    _LASSO_HINT = ("Click to place points; click the first point, double-click or "
                   "press Enter to close. Backspace undoes the last point. Drag a "
                   "point to correct it, drag a small midpoint to add one, "
                   "right-click a point to delete it.")

    _LEGEND_HINT = "Select the legend outside the dashed map outline. "

    def _outline_map_area(self):
        """Mark the page's own crop, so it is clear which part is already in use."""
        if not self.piece.crop_rect:
            return
        cx, cy, cw, ch = self.piece.crop_rect
        ref = QGraphicsRectItem(QRectF(cx, cy, cw, ch))
        pen = QPen(QColor(80, 200, 120), 2)
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        ref.setPen(pen)
        ref.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        ref.setZValue(4)
        ref.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.scene.addItem(ref)

    def _update_hint(self):
        base = self._LASSO_HINT if self.view.tool == "lasso" else self._RECT_HINT
        prefix = self._LEGEND_HINT if self.mode == "legend" else ""
        self.label.setText(prefix + base + " Middle-click to pan. Scroll to zoom.")

    def _set_tool(self, name: str):
        self.view.set_tool(name)
        self.rect_btn.setChecked(name == "rect")
        self.lasso_btn.setChecked(name == "lasso")
        self._update_hint()
        self.view.setFocus()

    def _int_crop(self, r: QRectF):
        """Clip a scene rect to integer image pixel coords, or None if degenerate."""
        h, w = self.piece.image.shape[:2]
        cx = max(0, int(math.floor(r.left())))
        cy = max(0, int(math.floor(r.top())))
        cw = min(int(math.ceil(r.right())) - cx, w - cx)
        ch = min(int(math.ceil(r.bottom())) - cy, h - cy)
        if cw < 2 or ch < 2:
            return None
        return (cx, cy, cw, ch)

    def _apply(self):
        if self.view.tool == "lasso":
            self._apply_lasso()
        else:
            self._apply_rect()

    def _warn_missing_region(self) -> bool:
        """True (after warning) when legend mode was applied with nothing selected.

        Crop mode reads an empty selection as "clear the crop", which is a valid
        thing to apply; there is no such reading for a legend.
        """
        if self.mode != "legend":
            return False
        QMessageBox.warning(self, "No region",
                            "Select the legend region first, or cancel.")
        return True

    def _apply_rect(self):
        cri = self.view.crop_rect_item
        if cri is None:
            if self._warn_missing_region():
                return
            self.result_crop_rect = None
            self.result_lasso_poly = None
            self.accept()
            return
        crop = self._int_crop(cri.crop_rect)
        if crop is None:
            QMessageBox.warning(self, "Too small", "Crop region is too small.")
            return
        self.result_crop_rect = crop
        self.result_lasso_poly = None
        self.accept()

    def _apply_lasso(self):
        li = self.view.lasso_item
        if li is None or not li.points:
            if self._warn_missing_region():
                return
            self.result_crop_rect = None
            self.result_lasso_poly = None
            self.accept()
            return
        if len(li.points) < 3:
            QMessageBox.warning(self, "Incomplete lasso",
                                "A lasso needs at least 3 points.")
            return
        li.close_shape()
        crop = self._int_crop(li.bounding_rect())
        if crop is None:
            QMessageBox.warning(self, "Too small", "Lasso region is too small.")
            return
        self.result_crop_rect = crop
        self.result_lasso_poly = li.polygon()
        self.accept()

    def _reset(self):
        if self.view.crop_rect_item:
            self.view.crop_rect_item.remove_from_scene()
            self.view.crop_rect_item = None
        if self.view.lasso_item:
            self.view.lasso_item.remove_from_scene()
            self.view.lasso_item = None
        self.view.setFocus()
        self.accept()


# --- Georeferencing ---

# Markers keep a fixed size on screen, so this is in device pixels.
_GCP_MARKER_RADIUS = 6

# How long a pan or zoom must settle before the OSM pane asks for new tiles.
_OSM_SETTLE_MS = 180


class GcpMarkerItem(QGraphicsItem):
    """Numbered crosshair marking one end of a control point pair.

    Ignores the view transform so it stays legible at any zoom — the OSM pane
    ranges from the whole world to a single building.
    """

    def __init__(self, label: str, scene, pending: bool = False):
        super().__init__()
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setZValue(20)
        self._label = label
        self._pending = pending
        self._highlighted = False
        scene.addItem(self)

    def set_highlighted(self, on: bool):
        if on != self._highlighted:
            self._highlighted = on
            self.update()

    def boundingRect(self) -> QRectF:
        r = _GCP_MARKER_RADIUS + 3
        return QRectF(-r, -r - 14, 2 * r + 34, 2 * r + 16)

    def paint(self, painter, option, widget=None):
        if self._pending:
            color = QColor(255, 170, 0)
        elif self._highlighted:
            color = QColor(255, 60, 60)
        else:
            color = QColor(30, 110, 240)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        # White underlay, so the marker reads over dark map and dark scan alike.
        for width, pen_color in ((3.0, QColor(255, 255, 255, 200)), (1.5, color)):
            painter.setPen(QPen(pen_color, width))
            r = _GCP_MARKER_RADIUS
            painter.drawEllipse(QPointF(0, 0), r, r)
            painter.drawLine(QPointF(-r - 3, 0), QPointF(-2, 0))
            painter.drawLine(QPointF(2, 0), QPointF(r + 3, 0))
            painter.drawLine(QPointF(0, -r - 3), QPointF(0, -2))
            painter.drawLine(QPointF(0, 2), QPointF(0, r + 3))
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        pos = QPointF(_GCP_MARKER_RADIUS + 5, -_GCP_MARKER_RADIUS - 2)
        painter.setPen(QPen(QColor(255, 255, 255, 220), 3))
        path = QPainterPath()
        path.addText(pos, font, self._label)
        painter.drawPath(path)
        painter.fillPath(path, QBrush(color))


class GcpView(QGraphicsView):
    """Pan/zoom view that reports left clicks in scene coords.

    Shared by both georeference panes so the two halves of a pair are placed
    the same way: left-click marks, middle-drag pans, wheel zooms — the same
    bindings the cut tool uses.
    """

    clicked = Signal(QPointF)
    navigated = Signal()

    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        self._panning = False
        self._pan_start = None

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)
        self.navigated.emit()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
            event.accept()
        elif event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.mapToScene(event.pos()))
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            h = self.horizontalScrollBar()
            v = self.verticalScrollBar()
            h.setValue(h.value() - delta.x())
            v.setValue(v.value() - delta.y())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton and self._panning:
            self._panning = False
            self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
            self.navigated.emit()
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.navigated.emit()


class OsmView(GcpView):
    """The ground-truth pane: OpenStreetMap, rendered by QGIS on demand.

    Scene coordinates are map metres with the northing negated, so that scene
    y runs downwards like every other view here and like the rendered image
    rows. Everything the user clicks is therefore already in map coordinates,
    give or take that sign.

    Tiles are fetched for the visible rectangle only, after navigation settles,
    on a worker thread — a render reaches the network and must not block the
    dialog. Each request carries a token so a slow one landing late cannot
    overwrite the result of a newer one.
    """

    def __init__(self, parent=None):
        scene = QGraphicsScene(parent)
        world = georef.WORLD_EXTENT
        scene.setSceneRect(QRectF(-world, -world, 2 * world, 2 * world))
        super().__init__(scene)
        self.setBackgroundBrush(QBrush(QColor(235, 235, 232)))
        # The scene is the whole planet, so a scrollbar says nothing useful
        # about where the view is; middle-drag is the way around.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._tiles = QGraphicsPixmapItem()
        self._tiles.setZValue(-10)
        self._tiles.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        scene.addItem(self._tiles)
        self._token = 0
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(_OSM_SETTLE_MS)
        self._timer.timeout.connect(self._render)
        self.navigated.connect(self._timer.start)
        # Every scroll ends up here, however it was made — middle-drag, arrow
        # keys, a trackpad. Watching the bars rather than the gestures is what
        # stops the view sliding off the tiles it was given.
        for bar in (self.horizontalScrollBar(), self.verticalScrollBar()):
            # Discard the signal's value — passed on, it becomes the interval.
            bar.valueChanged.connect(lambda _value: self._timer.start())

    @staticmethod
    def scene_to_map(pos: QPointF):
        return pos.x(), -pos.y()

    @staticmethod
    def map_to_scene(map_x, map_y) -> QPointF:
        return QPointF(map_x, -map_y)

    def show_map_rect(self, min_x, min_y, max_x, max_y):
        """Frame a map-coordinate rectangle exactly as asked.

        No margin is added here: callers pad in metres, which is the unit they
        think in, and a proportional margin on top would make that padding mean
        something different at every zoom. Keeping the aspect ratio still shows
        more than asked on the roomier axis — the rectangle is a minimum.
        """
        rect = QRectF(QPointF(min_x, -max_y), QPointF(max_x, -min_y))
        if rect.width() <= 0 or rect.height() <= 0:
            rect.adjust(-georef.POINT_MARGIN, -georef.POINT_MARGIN,
                        georef.POINT_MARGIN, georef.POINT_MARGIN)
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        self._timer.start()

    def _render(self):
        viewport = self.viewport().rect()
        if viewport.isEmpty() or not georef.ready():
            return
        rect = self.mapToScene(viewport).boundingRect()
        self._token += 1
        token = self._token
        extent = (rect.left(), -rect.bottom(), rect.right(), -rect.top())
        width, height = viewport.width(), viewport.height()

        def task(progress):
            return georef.render_osm(extent, width, height)

        worker = Worker(task)
        worker.signals.finished.connect(
            lambda arr: self._apply_tiles(token, arr, rect))
        # A failed render leaves the previous tiles up; the alternative is a
        # blank pane every time the network hiccups mid-pan.
        worker.signals.error.connect(lambda _msg: None)
        QThreadPool.globalInstance().start(worker)

    def _apply_tiles(self, token, arr, rect: QRectF):
        if token != self._token or arr is None or arr.size == 0:
            return
        arr = np.ascontiguousarray(arr)
        img = QImage(arr.data, arr.shape[1], arr.shape[0], arr.strides[0],
                     QImage.Format.Format_RGBA8888)
        self._tiles.setPixmap(QPixmap.fromImage(img.copy()))
        self._tiles.setPos(rect.topLeft())
        # The render is one device pixel per viewport pixel; scale it back into
        # scene units so it lands exactly on the rectangle it was asked for.
        self._tiles.setScale(rect.width() / max(1, arr.shape[1]))


class GeorefDialog(QDialog):
    """Tie one page to the world by pairing points on it with points on OSM.

    Georeferencing is per page because that is where features are recognisable;
    the export pools every page's points into one warp of the assembled canvas.

    The page is shown uncropped, like the cut tool, so a crisp landmark just
    outside the crop can still be used — the export keeps such points (see
    `export.canvas_gcps`). The page's crop is outlined as a reminder of which
    part of what is on screen actually survives to the output.
    """

    _HINT = ("Click a point on the page, then the same place on the map. "
             "Middle-drag to pan, wheel to zoom.")
    _HINT_PENDING = "Now click the same place on the map (Esc to cancel)."

    def __init__(self, piece: PuzzlePiece, pieces: list = None,
                 method: str = georef.DEFAULT_METHOD, parent=None):
        super().__init__(parent)
        self.piece = piece
        self.setWindowTitle(f"Georeference - {os.path.basename(piece.source_path)}")
        self.setMinimumSize(1000, 700)

        # Edited here and read back by the caller; the piece is never touched.
        self._gcps = [tuple(g) for g in piece.gcps]
        self._pending = None  # page point waiting for its map point
        self.result_gcps = None
        self.result_method = method if method in georef.METHODS else georef.DEFAULT_METHOD

        self._page_markers = []
        self._osm_markers = []

        layout = QVBoxLayout(self)
        self.splitter = QSplitter(self._default_orientation())
        self.page_view = self._build_page_view()
        self.osm_view = OsmView(self)
        self.splitter.addWidget(self.page_view)
        self.splitter.addWidget(self.osm_view)
        self.splitter.setSizes([500, 500])
        layout.addWidget(self.splitter, 1)

        self.page_view.clicked.connect(self._on_page_clicked)
        self.osm_view.clicked.connect(self._on_map_clicked)

        layout.addLayout(self._build_controls())
        layout.addWidget(self._build_table())
        layout.addLayout(self._build_buttons())

        self.hint = QLabel()
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)

        self._initial_map_rect = self._initial_map_bounds(pieces or [piece])
        self._fitted = False
        self._refresh()
        self.page_view.setFocus()

    def showEvent(self, event):
        """Frame both panes once the dialog has the size it will actually have.

        Fitting in the constructor fits to the widgets' provisional geometry,
        which leaves the page floating in a half-empty pane and the map on the
        wrong extent. Deferring past layout is what makes both land right.
        """
        super().showEvent(event)
        if not self._fitted:
            self._fitted = True
            QTimer.singleShot(0, self._fit_panes)

    def _fit_panes(self):
        self.page_view.fitInView(self.page_view.scene().sceneRect(),
                                 Qt.AspectRatioMode.KeepAspectRatio)
        self.osm_view.show_map_rect(*self._initial_map_rect)

    # --- construction ---

    def _default_orientation(self):
        """Split across the page's long axis, so both panes stay roomy.

        A tall page leaves width to spare, so the panes go side by side; a wide
        one leaves height, so they stack.
        """
        h, w = self.piece.image.shape[:2]
        return (Qt.Orientation.Horizontal if h >= w
                else Qt.Orientation.Vertical)

    def _build_page_view(self) -> GcpView:
        scene = QGraphicsScene(self)
        img = self.piece.image
        h, w = img.shape[:2]
        rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        if self.piece._base_alpha is not None:
            # Unmasked, so a point can be put on ground a lasso cut away.
            rgba[:, :, 3] = self.piece._base_alpha
        qimg = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0],
                      QImage.Format.Format_RGBA8888)
        scene.addItem(QGraphicsPixmapItem(QPixmap.fromImage(qimg.copy())))
        scene.setSceneRect(QRectF(0, 0, w, h))
        if self.piece.crop_rect is not None:
            cx, cy, cw, ch = self.piece.crop_rect
            outline = QGraphicsRectItem(QRectF(cx, cy, cw, ch))
            pen = QPen(QColor(80, 200, 120), 2)
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            outline.setPen(pen)
            outline.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            outline.setZValue(4)
            scene.addItem(outline)
        view = GcpView(scene)
        view.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        return view

    def _build_controls(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.orientation_btn = QPushButton()
        self.orientation_btn.setAutoDefault(False)
        self.orientation_btn.clicked.connect(self._toggle_orientation)
        row.addWidget(self.orientation_btn)
        row.addSpacing(16)
        row.addWidget(QLabel("Transform:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(georef.METHODS)
        self.method_combo.setCurrentText(self.result_method)
        self.method_combo.currentTextChanged.connect(self._on_method_changed)
        row.addWidget(self.method_combo)
        row.addStretch(1)
        self.fit_label = QLabel()
        row.addWidget(self.fit_label)
        return row

    def _build_table(self) -> QTableWidget:
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["#", "Page x, y", "Map x, y", "Residual (m)"])
        self.table.verticalHeader().hide()
        self.table.setMaximumHeight(170)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        for col in (1, 2, 3):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        return self.table

    def _build_buttons(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.delete_btn = QPushButton("Delete Point")
        self.apply_btn = QPushButton("Apply")
        self.cancel_btn = QPushButton("Cancel")
        self.delete_btn.clicked.connect(self._delete_selected)
        self.apply_btn.clicked.connect(self._apply)
        self.cancel_btn.clicked.connect(self.reject)
        for btn in (self.delete_btn, self.apply_btn, self.cancel_btn):
            btn.setAutoDefault(False)
            row.addWidget(btn)
        return row

    def _initial_map_bounds(self, pieces: list):
        """Open the map where there is already work, rather than on the world.

        This page's own points first; failing that any other page's, since the
        pages of one document are usually neighbours. A single point is enough:
        its box has no size, so `POINT_MARGIN` is what sets the zoom, and the
        same margin keeps a box of many points clear of the pane edges.
        """
        for candidates in (self._gcps,
                           [g for p in pieces for g in p.gcps]):
            if candidates:
                xs = [g[2] for g in candidates]
                ys = [g[3] for g in candidates]
                margin = georef.POINT_MARGIN
                return (min(xs) - margin, min(ys) - margin,
                        max(xs) + margin, max(ys) + margin)
        return georef.DEFAULT_MAP_RECT

    # --- interaction ---

    def _toggle_orientation(self):
        self.splitter.setOrientation(
            Qt.Orientation.Vertical
            if self.splitter.orientation() == Qt.Orientation.Horizontal
            else Qt.Orientation.Horizontal)
        self.splitter.setSizes([500, 500])
        self._update_orientation_button()

    def _update_orientation_button(self):
        side_by_side = self.splitter.orientation() == Qt.Orientation.Horizontal
        self.orientation_btn.setText(
            "Stack top/bottom" if side_by_side else "Place left/right")

    def _on_method_changed(self, method: str):
        self.result_method = method
        self._refresh()

    def _on_page_clicked(self, pos: QPointF):
        self._pending = (pos.x(), pos.y())
        self._refresh()

    def _on_map_clicked(self, pos: QPointF):
        if self._pending is None:
            self._set_hint(self._HINT)
            return
        map_x, map_y = OsmView.scene_to_map(pos)
        self._gcps.append((self._pending[0], self._pending[1], map_x, map_y))
        self._pending = None
        self._refresh()
        self.table.selectRow(len(self._gcps) - 1)

    def _delete_selected(self):
        row = self.table.currentRow()
        if 0 <= row < len(self._gcps):
            del self._gcps[row]
            self._refresh()

    def _on_row_selected(self):
        row = self.table.currentRow()
        for index, markers in enumerate(zip(self._page_markers, self._osm_markers)):
            for marker in markers:
                marker.set_highlighted(index == row)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape and self._pending is not None:
            # Esc drops a half-made pair before it drops the dialog.
            self._pending = None
            self._refresh()
            event.accept()
            return
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self._delete_selected()
            event.accept()
            return
        super().keyPressEvent(event)

    # --- refresh ---

    def _refresh(self):
        self._rebuild_markers()
        self._rebuild_table()
        self._update_orientation_button()
        self._set_hint(self._HINT_PENDING if self._pending else self._HINT)

    def _rebuild_markers(self):
        """Rebuild every marker: deleting a point renumbers the ones after it."""
        for marker in self._page_markers + self._osm_markers:
            marker.scene().removeItem(marker)
        self._page_markers = []
        self._osm_markers = []
        page_scene = self.page_view.scene()
        osm_scene = self.osm_view.scene()
        for index, (img_x, img_y, map_x, map_y) in enumerate(self._gcps, start=1):
            page_marker = GcpMarkerItem(str(index), page_scene)
            page_marker.setPos(QPointF(img_x, img_y))
            osm_marker = GcpMarkerItem(str(index), osm_scene)
            osm_marker.setPos(OsmView.map_to_scene(map_x, map_y))
            self._page_markers.append(page_marker)
            self._osm_markers.append(osm_marker)
        if self._pending is not None:
            pending = GcpMarkerItem(str(len(self._gcps) + 1), page_scene, pending=True)
            pending.setPos(QPointF(*self._pending))
            self._page_markers.append(pending)
            # Keeps the two lists the same length, so `_on_row_selected` can zip
            # them without the pending marker shifting the pairing.
            self._osm_markers.append(GcpMarkerItem("", osm_scene, pending=True))
            self._osm_markers[-1].setVisible(False)

    def _rebuild_table(self):
        errors = self._residuals()
        self.table.setRowCount(len(self._gcps))
        for row, (img_x, img_y, map_x, map_y) in enumerate(self._gcps):
            cells = [
                str(row + 1),
                f"{img_x:.0f}, {img_y:.0f}",
                f"{map_x:.1f}, {map_y:.1f}",
                "-" if errors is None else f"{errors[row]:.1f}",
            ]
            for col, text in enumerate(cells):
                self.table.setItem(row, col, QTableWidgetItem(text))
        self._update_fit_label(errors)

    def _residuals(self):
        try:
            return georef.residuals(self._gcps, self.result_method)
        except Exception:
            return None

    def _update_fit_label(self, errors):
        needed = self._minimum_points()
        if errors is None:
            missing = max(0, needed - len(self._gcps))
            self.fit_label.setText(
                f"{len(self._gcps)} points - {missing} more needed"
                if missing else f"{len(self._gcps)} points - no fit")
            return
        rms = float(np.sqrt(np.mean(np.square(errors)))) if errors else 0.0
        self.fit_label.setText(f"{len(self._gcps)} points - RMS {rms:.1f} m")

    def _minimum_points(self) -> int:
        try:
            return georef.minimum_gcp_count(self.result_method)
        except Exception:
            return 2

    def _set_hint(self, text: str):
        self.hint.setText(text)

    def _apply(self):
        self.result_gcps = list(self._gcps)
        self.accept()


class ExportOptionsDialog(QDialog):
    """Pick which files an export writes.

    Every entry stands alone. The georeferenced file used to be a warp of the
    flattened one and so had to be asked for alongside it; it now composites
    the pieces itself (`export.export_geotiff`) and shares nothing with the
    others, so the only thing it still depends on is having control points.
    """

    _VARIANTS = (
        ("flat", "Flattened TIFF", "{name}_flat.tif",
         "Every piece composited onto one canvas."),
        ("uncropped", "Uncropped flattened TIFF", "{name}_uncropped_flat.tif",
         "The same composite with each lasso reduced to its bounding rect."),
        ("layered", "Layered TIFF", "{name}_layered.tif",
         "Each piece as its own layer."),
    )

    def __init__(self, can_warp: bool, selected: dict = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export")
        self._can_warp = can_warp
        selected = selected or {}
        # The georeferenced file is the point of an export once the pages are
        # tied to the world, so on its own it is the default. Without control
        # points it cannot be written at all, and the flat file takes over —
        # otherwise the dialog would open with nothing to agree to.
        defaults = {"warp"} if can_warp else {"flat"}
        self.result_options = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Write which files?"))

        self.boxes = {}
        for key, label, suffix, hint in self._VARIANTS:
            box = QCheckBox(f"{label}   ({suffix})")
            box.setChecked(selected.get(key, key in defaults))
            box.setToolTip(hint)
            layout.addWidget(box)
            self.boxes[key] = box

        self.warp_box = QCheckBox(
            "Georeferenced TIFF   ({name}_warped.tif, {name}_warped.qgs)")
        self.warp_box.setToolTip(
            "The map placed in the world, legends left upright, plus a QGIS "
            "project pointing at it."
            if can_warp else
            "No control points yet — georeference a page first (Ctrl+G).")
        self.warp_box.setEnabled(can_warp)
        self.warp_box.setChecked(can_warp and selected.get("warp", "warp" in defaults))
        layout.addWidget(self.warp_box)

        self.note = QLabel()
        self.note.setWordWrap(True)
        layout.addWidget(self.note)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._apply)
        buttons.rejected.connect(self.reject)
        self.ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        layout.addWidget(buttons)

        if not any(box.isChecked() for box in self._all_boxes()):
            # A selection remembered from a georeferenced document leaves every
            # ordinary box explicitly off, so reopening it on one without
            # control points would tick nothing and grey out OK. Fall back.
            for key in defaults:
                if key != "warp":
                    self.boxes[key].setChecked(True)

        for box in self._all_boxes():
            box.toggled.connect(self._sync_ok)
        if not can_warp:
            self.note.setText("Georeference a page to enable the geo-referenced file.")
        self._sync_ok()

    def _all_boxes(self):
        return list(self.boxes.values()) + [self.warp_box]

    def _sync_ok(self):
        """Any one file is a valid export, the georeferenced one included."""
        self.ok_button.setEnabled(
            any(box.isChecked() for box in self._all_boxes()))

    def _apply(self):
        self.result_options = {key: box.isChecked()
                               for key, box in self.boxes.items()}
        self.result_options["warp"] = self.warp_box.isChecked()
        self.accept()
