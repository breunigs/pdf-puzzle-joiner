import os
import math
import re
import subprocess
from typing import Optional

import cv2
import numpy as np

from PySide6.QtWidgets import (
    QGraphicsScene, QGraphicsView,
    QGraphicsPixmapItem, QGraphicsItem, QGraphicsRectItem,
    QGraphicsPathItem,
    QWidget, QHBoxLayout, QVBoxLayout, QScrollArea, QLabel, QPushButton,
    QDialog, QDialogButtonBox, QLineEdit, QFormLayout,
    QMenu, QFrame, QGridLayout,
    QAbstractScrollArea, QMessageBox,
)
from PySide6.QtCore import (
    Qt, QRectF, QPointF, QSizeF, QObject, Signal,
    QRunnable, QThreadPool, QEvent,
)
from PySide6.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor, QBrush, QCursor,
    QFont, QTransform, QPainterPath,
)

from .model import PuzzlePiece
from .priority import LOW_PRIO
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
        img = piece.get_cropped_image()
        h, w = img.shape[:2]
        cx_world = piece.x
        cy_world = piece.y

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
            piece.rotation_deg = self._drag_start_rot + (current_angle - start_angle)
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
                piece.scale = max(0.01, self._drag_start_scale * ratio)

        piece_item.sync_from_piece()

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
                piece_item.push_transform_command(
                    self._drag_start_piece_x, self._drag_start_piece_y,
                    self._drag_start_rot, self._drag_start_scale, desc,
                )
            scene = self.scene()
            if scene and hasattr(scene, 'piece_moved_signal'):
                scene.piece_moved_signal.emit()
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
        img = piece.get_cropped_image()
        h, w = img.shape[:2]
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
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        # Selection is managed via thumbnail panel, not canvas clicks
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setTransformOriginPoint(self.boundingRect().center())
        self.transform_handles = TransformHandles(self)
        self.sync_from_piece()

    def sync_from_piece(self):
        self._syncing = True
        piece = self.piece
        img = piece.get_cropped_image()
        h, w = img.shape[:2]
        ds = piece.display_scale

        # Display pixmap center in item-local coords
        dcx = (w * ds) / 2.0
        dcy = (h * ds) / 2.0

        # Qt applies setScale/setRotation around transformOriginPoint (dcx, dcy).
        # The scene position of the origin is always pos + origin, regardless of
        # rotation/scale. So to place the center at (piece.x, piece.y):
        self.setPos(piece.x - dcx, piece.y - dcy)
        self.setRotation(piece.rotation_deg)
        self.setScale(piece.scale / ds)

        self.transform_handles.update_positions()
        self._syncing = False

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            is_sel = bool(value)
            self.transform_handles.set_visible(is_sel and not self.piece.is_locked)
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
        self.undo_stack = undo_stack
        self.editing_enabled = True
        self.setSceneRect(-50000, -50000, 100000, 100000)

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
        bb = piece.get_bounding_box()
        expanded = bb.adjusted(-margin, -margin, margin, margin)
        neighbors = []
        for p, item in self.piece_items.items():
            if p is piece:
                continue
            if p.get_bounding_box().intersects(expanded):
                neighbors.append(p)
        return neighbors

    def piece_selection_changed(self, piece: PuzzlePiece, selected: bool):
        if selected:
            self.selection_changed_signal.emit(piece)
        else:
            if not self.selectedItems():
                self.selection_changed_signal.emit(None)

    def sync_all_from_pieces(self):
        for piece, item in self.piece_items.items():
            item.sync_from_piece()
        self.update_scene_rect()


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

    def _get_selected_item(self):
        scene = self.scene()
        if not isinstance(scene, PuzzleScene):
            return None
        piece = scene.get_selected_piece()
        if piece and not piece.is_locked:
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

            if self._drag_mode == 'move':
                dx = scene_pos.x() - self._drag_start_scene.x()
                dy = scene_pos.y() - self._drag_start_scene.y()
                piece.x = self._drag_start_x + dx
                piece.y = self._drag_start_y + dy
                self._drag_piece_item.sync_from_piece()
            elif self._drag_mode == 'rotate':
                cx, cy = piece.x, piece.y
                start_angle = math.degrees(math.atan2(
                    self._drag_start_scene.y() - cy,
                    self._drag_start_scene.x() - cx,
                ))
                current_angle = math.degrees(math.atan2(
                    scene_pos.y() - cy,
                    scene_pos.x() - cx,
                ))
                piece.rotation_deg = self._drag_start_rot + (current_angle - start_angle)
                self._drag_piece_item.sync_from_piece()
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
                self._drag_piece_item.push_transform_command(
                    self._drag_start_x, self._drag_start_y,
                    self._drag_start_rot, self._drag_start_scale, desc,
                )
            scene = self.scene()
            if hasattr(scene, 'piece_moved_signal'):
                scene.piece_moved_signal.emit()
            self._drag_mode = None
            self._drag_piece_item = None
            self._update_hover_cursor(event.pos())
            event.accept()
            return

        super().mouseReleaseEvent(event)
        self._update_hover_cursor(event.pos())

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
        self.setFixedWidth(120)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        self.thumb_label = QLabel()
        self.thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb_label.setFixedHeight(90)
        self.thumb_label.setScaledContents(False)
        layout.addWidget(self.thumb_label)

        self.name_label = QLabel()
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.name_label.setWordWrap(True)
        font = QFont()
        font.setPointSize(7)
        self.name_label.setFont(font)
        layout.addWidget(self.name_label)

        self.update_display()
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_right_click)

    def update_display(self):
        piece = self.piece
        if piece.thumbnail:
            scaled = piece.thumbnail.scaled(
                110, 80,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            if piece.is_locked:
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

        name = os.path.basename(piece.source_path)
        # Parse "page_0003.png" → "Page 3"
        m = re.match(r"page_0*(\d+)\.\w+$", name)
        if m:
            name = f"Page {m.group(1)}"
        extra = ""
        if piece.is_locked:
            extra = " [L]"
        elif not piece.is_matched:
            extra = " (!)"
        self.name_label.setText(name + extra)

        # Border for locked
        if piece.is_locked:
            self.setStyleSheet("QWidget { border: 2px solid orange; }")
        else:
            self.setStyleSheet("")

    def set_selected(self, selected: bool):
        if selected:
            self.setStyleSheet("QWidget { border: 2px solid #4488ff; background: #ddeeff; }")
        elif self.piece.is_locked:
            self.setStyleSheet("QWidget { border: 2px solid orange; }")
        else:
            self.setStyleSheet("")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.piece)
        super().mousePressEvent(event)

    def _on_right_click(self, pos):
        self.right_clicked.emit(self.piece, self.mapToGlobal(pos))


class ThumbnailPanel(QWidget):
    piece_selected = Signal(object)  # PuzzlePiece

    def __init__(self):
        super().__init__()
        self.setFixedHeight(130)
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
        if piece.is_locked:
            action = menu.addAction("Unlock")
        else:
            action = menu.addAction("Lock")
        chosen = menu.exec(global_pos)
        if chosen:
            piece.is_locked = not piece.is_locked
            self.update_piece(piece)
            self.piece_selected.emit(piece)  # trigger scene update


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


class CropView(QGraphicsView):
    """Zoomable/pannable view for the crop dialog."""

    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._panning = False
        self._pan_start = None
        self._zoom = 1.0
        self._mode = None  # "draw", "move", or None
        self._drag_start = None
        self._move_start_rect = None
        self.crop_rect_item: CropRectItem | None = None
        self._image_rect = QRectF()

    def set_image_rect(self, r: QRectF):
        self._image_rect = QRectF(r)

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

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
            event.accept()
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
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class CutToolDialog(QDialog):
    def __init__(self, piece: PuzzlePiece, parent=None):
        super().__init__(parent)
        self.piece = piece
        self.setWindowTitle(f"Cut - {os.path.basename(piece.source_path)}")
        self.setMinimumSize(800, 600)

        layout = QVBoxLayout(self)

        # Show full UNROTATED image so user can see borders/legends
        self.scene = QGraphicsScene()
        self.view = CropView(self.scene)
        layout.addWidget(self.view)

        img = piece.image  # full uncropped image
        h, w = img.shape[:2]
        rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        qimg = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0],
                      QImage.Format.Format_RGBA8888)
        pm = QPixmap.fromImage(qimg.copy())
        self._pixmap_item = QGraphicsPixmapItem(pm)
        self.scene.addItem(self._pixmap_item)
        image_rect = QRectF(0, 0, w, h)
        self.scene.setSceneRect(image_rect)
        self.view.set_image_rect(image_rect)

        # Initialize crop rect from piece (or full image)
        if piece.crop_rect:
            cx, cy, cw, ch = piece.crop_rect
            crop_r = QRectF(cx, cy, cw, ch)
        else:
            crop_r = QRectF(image_rect)

        self.view.crop_rect_item = CropRectItem(crop_r, image_rect, self.scene)

        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        t = self.view.transform()
        self.view._zoom = t.m11()

        # Buttons
        buttons = QHBoxLayout()
        self.apply_btn = QPushButton("Apply")
        self.reset_btn = QPushButton("Reset Crop")
        self.cancel_btn = QPushButton("Cancel")
        self.apply_btn.clicked.connect(self._apply)
        self.reset_btn.clicked.connect(self._reset)
        self.cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(self.apply_btn)
        buttons.addWidget(self.reset_btn)
        buttons.addWidget(self.cancel_btn)
        layout.addLayout(buttons)

        self.label = QLabel(
            "Drag handles to resize crop. Drag inside to move. "
            "Draw outside to create new region. Middle-click to pan. Scroll to zoom."
        )
        layout.addWidget(self.label)

    def _apply(self):
        cri = self.view.crop_rect_item
        if cri is None:
            QMessageBox.warning(self, "No selection", "Please define a crop region first.")
            return
        r = cri.crop_rect
        if r.width() < 2 or r.height() < 2:
            QMessageBox.warning(self, "Too small", "Crop region is too small.")
            return
        h, w = self.piece.image.shape[:2]
        cx = max(0, int(r.left()))
        cy = max(0, int(r.top()))
        cw = min(int(r.width()), w - cx)
        ch = min(int(r.height()), h - cy)
        if cw > 0 and ch > 0:
            self.piece.crop_rect = (cx, cy, cw, ch)
            self.accept()

    def _reset(self):
        self.piece.crop_rect = None
        if self.view.crop_rect_item:
            self.view.crop_rect_item.remove_from_scene()
            self.view.crop_rect_item = None
        self.accept()
