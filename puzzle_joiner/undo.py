from PySide6.QtGui import QUndoCommand


class TransformCommand(QUndoCommand):
    """Undoable move/rotate/scale/snap operation on a single piece."""

    def __init__(self, piece, old_x, old_y, old_rot, old_scale,
                 new_x, new_y, new_rot, new_scale,
                 sync_fn=None, description="Transform"):
        super().__init__(description)
        self.piece = piece
        self.old_x = old_x
        self.old_y = old_y
        self.old_rot = old_rot
        self.old_scale = old_scale
        self.new_x = new_x
        self.new_y = new_y
        self.new_rot = new_rot
        self.new_scale = new_scale
        self.sync_fn = sync_fn

    def redo(self):
        self.piece.x = self.new_x
        self.piece.y = self.new_y
        self.piece.rotation_deg = self.new_rot
        self.piece.scale = self.new_scale
        if self.sync_fn:
            self.sync_fn(self.piece)

    def undo(self):
        self.piece.x = self.old_x
        self.piece.y = self.old_y
        self.piece.rotation_deg = self.old_rot
        self.piece.scale = self.old_scale
        if self.sync_fn:
            self.sync_fn(self.piece)


class SnapCommand(QUndoCommand):
    """Undoable snap operation — may change position/rotation/scale, is_matched, is_locked,
    and snap pair relationships."""

    def __init__(self, piece, old_x, old_y, old_rot, old_scale, old_matched, old_locked,
                 new_x, new_y, new_rot, new_scale, new_matched, new_locked,
                 snap_pairs=None, scene=None, sync_fn=None):
        super().__init__("Snap")
        self.piece = piece
        self.old_x = old_x
        self.old_y = old_y
        self.old_rot = old_rot
        self.old_scale = old_scale
        self.old_matched = old_matched
        self.old_locked = old_locked
        self.new_x = new_x
        self.new_y = new_y
        self.new_rot = new_rot
        self.new_scale = new_scale
        self.new_matched = new_matched
        self.new_locked = new_locked
        self.snap_pairs = snap_pairs or []  # list of (piece, neighbor) tuples
        self.scene = scene
        self.sync_fn = sync_fn

    def redo(self):
        self.piece.x = self.new_x
        self.piece.y = self.new_y
        self.piece.rotation_deg = self.new_rot
        self.piece.scale = self.new_scale
        self.piece.is_matched = self.new_matched
        self.piece.is_locked = self.new_locked
        if self.scene:
            for pair in self.snap_pairs:
                self.scene.add_snap_pair(*pair)
        if self.sync_fn:
            self.sync_fn(self.piece)

    def undo(self):
        if self.scene:
            for pair in self.snap_pairs:
                self.scene.remove_snap_pair(*pair)
        self.piece.x = self.old_x
        self.piece.y = self.old_y
        self.piece.rotation_deg = self.old_rot
        self.piece.scale = self.old_scale
        self.piece.is_matched = self.old_matched
        self.piece.is_locked = self.old_locked
        if self.sync_fn:
            self.sync_fn(self.piece)


class SnapAllCommand(QUndoCommand):
    """Undoable snap-all — wraps multiple SnapCommands as children."""

    def __init__(self, snap_commands):
        super().__init__("Snap All")
        self._commands = snap_commands

    def redo(self):
        for cmd in self._commands:
            cmd.redo()

    def undo(self):
        for cmd in reversed(self._commands):
            cmd.undo()


class GroupTransformCommand(QUndoCommand):
    """Undoable transform of multiple pieces (snap group movement)."""

    def __init__(self, commands, description="Group Transform"):
        super().__init__(description)
        self._commands = commands

    def redo(self):
        for cmd in self._commands:
            cmd.redo()

    def undo(self):
        for cmd in reversed(self._commands):
            cmd.undo()


class LockCommand(QUndoCommand):
    """Undoable lock/unlock toggle."""

    def __init__(self, piece, sync_fn=None):
        state = "Unlock" if piece.is_locked else "Lock"
        super().__init__(state)
        self.piece = piece
        self.sync_fn = sync_fn

    def redo(self):
        self.piece.is_locked = not self.piece.is_locked
        if self.sync_fn:
            self.sync_fn(self.piece)

    def undo(self):
        self.piece.is_locked = not self.piece.is_locked
        if self.sync_fn:
            self.sync_fn(self.piece)
