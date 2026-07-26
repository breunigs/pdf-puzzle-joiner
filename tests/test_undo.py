import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
import numpy as np
import pytest

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QUndoStack

from puzzle_joiner.model import PuzzlePiece
from puzzle_joiner.undo import TransformCommand, SnapCommand, SnapAllCommand, GroupTransformCommand, LockCommand

app = QApplication.instance() or QApplication(sys.argv)


def _make_piece():
    piece = PuzzlePiece()
    piece.image = np.zeros((100, 100, 4), dtype=np.uint8)
    piece.x = 50.0
    piece.y = 50.0
    piece.rotation_deg = 0.0
    piece.scale = 1.0
    return piece


class TestTransformCommand:
    def test_undo_redo(self):
        stack = QUndoStack()
        p = _make_piece()
        cmd = TransformCommand(p, 50, 50, 0, 1, 100, 200, 45, 2.0)
        stack.push(cmd)
        assert p.x == 100
        assert p.y == 200
        assert p.rotation_deg == 45
        assert p.scale == 2.0

        stack.undo()
        assert p.x == 50
        assert p.y == 50
        assert p.rotation_deg == 0
        assert p.scale == 1.0

        stack.redo()
        assert p.x == 100
        assert p.y == 200

    def test_no_op_push(self):
        stack = QUndoStack()
        p = _make_piece()
        cmd = TransformCommand(p, 50, 50, 0, 1, 50, 50, 0, 1)
        stack.push(cmd)
        assert stack.count() == 1  # still pushed, values unchanged
        assert p.x == 50


class TestSnapCommand:
    def test_undo_restores_matched(self):
        stack = QUndoStack()
        p = _make_piece()
        p.is_matched = False
        cmd = SnapCommand(p, 50, 50, 0, 1, False, False,
                          80, 90, 10, 1.5, True, True)
        stack.push(cmd)
        assert p.is_matched is True
        assert p.is_locked is True
        assert p.x == 80
        assert p.rotation_deg == 10

        stack.undo()
        assert p.is_matched is False
        assert p.is_locked is False
        assert p.x == 50
        assert p.rotation_deg == 0


class TestSnapAllCommand:
    def test_undo_all(self):
        stack = QUndoStack()
        p1 = _make_piece()
        p2 = _make_piece()
        p2.x = 200

        cmds = [
            SnapCommand(p1, 50, 50, 0, 1, False, False,
                        60, 60, 5, 1, True, True),
            SnapCommand(p2, 200, 50, 0, 1, False, False,
                        210, 60, -5, 1, True, True),
        ]
        stack.push(SnapAllCommand(cmds))
        assert p1.x == 60
        assert p2.x == 210

        stack.undo()
        assert p1.x == 50
        assert p2.x == 200
        assert p1.is_matched is False
        assert p2.is_matched is False


class TestGroupTransformCommand:
    def test_undo_redo(self):
        stack = QUndoStack()
        p1 = _make_piece()
        p2 = _make_piece()
        p2.x = 200
        cmds = [
            TransformCommand(p1, 50, 50, 0, 1, 60, 70, 10, 1.2),
            TransformCommand(p2, 200, 50, 0, 1, 210, 70, 10, 1.2),
        ]
        stack.push(GroupTransformCommand(cmds))
        assert p1.x == 60
        assert p2.x == 210

        stack.undo()
        assert p1.x == 50
        assert p2.x == 200

        stack.redo()
        assert p1.x == 60
        assert p2.x == 210


class TestLockCommand:
    def test_toggle(self):
        stack = QUndoStack()
        p = _make_piece()
        assert p.is_locked is False

        stack.push(LockCommand(p))
        assert p.is_locked is True

        stack.undo()
        assert p.is_locked is False

        stack.redo()
        assert p.is_locked is True

    def test_unlock(self):
        stack = QUndoStack()
        p = _make_piece()
        p.is_locked = True

        stack.push(LockCommand(p))
        assert p.is_locked is False

        stack.undo()
        assert p.is_locked is True


class TestSyncFnCalled:
    def test_transform_calls_sync(self):
        calls = []
        stack = QUndoStack()
        p = _make_piece()
        cmd = TransformCommand(p, 50, 50, 0, 1, 100, 100, 0, 1,
                               sync_fn=lambda piece: calls.append(("sync", piece)))
        stack.push(cmd)
        assert len(calls) == 1
        stack.undo()
        assert len(calls) == 2
        stack.redo()
        assert len(calls) == 3
