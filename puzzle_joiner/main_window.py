import os
import sys
import tempfile
import concurrent.futures
from typing import Optional

import cv2
import numpy as np

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QToolBar, QStatusBar,
    QFileDialog, QDialog, QMessageBox, QProgressBar, QLabel,
)
from PySide6.QtCore import Qt, QTimer, QThreadPool
from PySide6.QtGui import QAction, QUndoStack, QKeySequence

from .model import PuzzlePiece
from .preprocessing import (
    parse_page_range, split_pdf_pages, pdf_page_info,
    _convert_and_process_page,
)
from .cache import (
    _cache_dir_for_pdf, _cached_page_path, _cached_autocrop_path,
    _load_cached_page, save_layout, load_layout, _images_cache_dir,
)
from .matching import auto_detect_placements, snap_piece_to_neighbors
from .export import export_layered_tiff, export_flattened_tiff
from .worker import Worker
from .priority import _set_low_priority
from .widgets import (
    PuzzleScene, PuzzleView, ThumbnailPanel,
    PdfImportDialog, CutToolDialog,
)
from .undo import TransformCommand, SnapCommand, SnapAllCommand, LockCommand


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Puzzle Joiner")
        self.resize(1400, 900)
        self.pieces: list = []
        self._work_dir = None
        self._layout_dir: Optional[str] = None
        self._thread_pool = QThreadPool.globalInstance()
        self.undo_stack = QUndoStack(self)
        self.undo_stack.indexChanged.connect(self._auto_save_layout)

        self._build_ui()

    def _build_ui(self):
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Toolbar
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        U = "\u0332"  # combining low line — underlines preceding character

        self.action_open_pdf = QAction(f"O{U}pen PDF", self)
        self.action_open_pdf.setShortcut(QKeySequence("Ctrl+O"))
        self.action_open_pdf.setToolTip("Open PDF (Ctrl+O)")
        self.action_open_pdf.triggered.connect(self.on_open_pdf)
        toolbar.addAction(self.action_open_pdf)

        self.action_open_images = QAction(f"Open I{U}mages", self)
        self.action_open_images.setShortcut(QKeySequence("Ctrl+I"))
        self.action_open_images.setToolTip("Open Images (Ctrl+I)")
        self.action_open_images.triggered.connect(self.on_open_images)
        toolbar.addAction(self.action_open_images)

        toolbar.addSeparator()

        self.action_undo = self.undo_stack.createUndoAction(self, "Undo")
        self.action_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self.action_undo.setToolTip("Undo (Ctrl+Z)")
        toolbar.addAction(self.action_undo)

        self.action_redo = self.undo_stack.createRedoAction(self, "Redo")
        self.action_redo.setShortcut(QKeySequence.StandardKey.Redo)
        self.action_redo.setToolTip("Redo (Ctrl+Y)")
        toolbar.addAction(self.action_redo)

        toolbar.addSeparator()

        self.action_auto_detect = QAction(f"Auto Align A{U}ll", self)
        self.action_auto_detect.setShortcut(QKeySequence("Ctrl+D"))
        self.action_auto_detect.setToolTip("Match and align all pieces (Ctrl+D)")
        self.action_auto_detect.triggered.connect(self.on_auto_detect)
        toolbar.addAction(self.action_auto_detect)

        self.action_snap = QAction(f"S{U}nap", self)
        self.action_snap.setShortcut(QKeySequence("Ctrl+S"))
        self.action_snap.setToolTip("Snap (Ctrl+S)")
        self.action_snap.triggered.connect(self.on_snap)
        toolbar.addAction(self.action_snap)

        toolbar.addSeparator()

        self.action_cut = QAction(f"Crop (X{U})", self)
        self.action_cut.setShortcut(QKeySequence("Ctrl+X"))
        self.action_cut.setToolTip("Crop (Ctrl+X)")
        self.action_cut.triggered.connect(self.on_cut)
        toolbar.addAction(self.action_cut)

        self.action_lock = QAction(f"L{U}ock/Unlock", self)
        self.action_lock.setShortcut(QKeySequence("Ctrl+L"))
        self.action_lock.setToolTip("Lock/Unlock (Ctrl+L)")
        self.action_lock.triggered.connect(self.on_lock_toggle)
        toolbar.addAction(self.action_lock)

        toolbar.addSeparator()

        self.action_export = QAction(f"E{U}xport TIFF", self)
        self.action_export.setShortcut(QKeySequence("Ctrl+E"))
        self.action_export.setToolTip("Export TIFF (Ctrl+E)")
        self.action_export.triggered.connect(self.on_export)
        toolbar.addAction(self.action_export)

        toolbar.addSeparator()

        self.action_fit = QAction(f"F{U}it View", self)
        self.action_fit.setShortcut(QKeySequence("Ctrl+F"))
        self.action_fit.setToolTip("Fit View (Ctrl+F)")
        self.action_fit.triggered.connect(self.on_fit_view)
        toolbar.addAction(self.action_fit)

        # Thumbnail panel
        self.thumbnail_panel = ThumbnailPanel()
        self.thumbnail_panel.piece_selected.connect(self.on_thumbnail_select)
        main_layout.addWidget(self.thumbnail_panel)

        # Canvas
        self.scene = PuzzleScene(undo_stack=self.undo_stack)
        self.scene.selection_changed_signal.connect(self.on_scene_selection_changed)
        self.scene.piece_moved_signal.connect(self._update_snap_enabled)
        self.scene.piece_moved_signal.connect(self.scene.update_scene_rect)
        self.view = PuzzleView(self.scene)
        self.view.select_next.connect(lambda: self._select_adjacent_piece(True))
        self.view.select_prev.connect(lambda: self._select_adjacent_piece(False))
        main_layout.addWidget(self.view, 1)

        # Status bar with embedded progress
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self._progress_label = QLabel()
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setMaximumWidth(200)
        self._progress_label.hide()
        self._progress_bar.hide()
        self.status_bar.addPermanentWidget(self._progress_label)
        self.status_bar.addPermanentWidget(self._progress_bar)
        self.status_bar.showMessage("Ready. Open a PDF or images to begin.")

    def _run_with_progress(self, title: str, fn, *args, on_done=None, **kwargs):
        """Run fn(*args, **kwargs, progress=Signal) in background with inline progress bar."""
        self._set_actions_enabled(False)
        self._progress_label.setText(title)
        self._progress_label.show()
        self._progress_bar.setValue(0)
        self._progress_bar.show()
        self.status_bar.showMessage("")

        worker = Worker(fn, *args, **kwargs)

        def on_progress(pct, msg):
            self._progress_bar.setValue(pct)
            self._progress_label.setText(msg)

        def _finish():
            self._current_worker = None
            self._progress_bar.hide()
            self._progress_label.hide()
            self._set_actions_enabled(True)

        def on_finished(result):
            _finish()
            if on_done:
                on_done(result)

        def on_error(msg):
            _finish()
            QMessageBox.critical(self, "Error", msg)

        worker.signals.progress.connect(on_progress)
        worker.signals.finished.connect(on_finished)
        worker.signals.error.connect(on_error)

        self._current_worker = worker
        self._thread_pool.start(worker)

    def _set_actions_enabled(self, enabled: bool):
        """Enable/disable toolbar actions and canvas editing during background work."""
        for action in (
            self.action_open_pdf, self.action_open_images,
            self.action_auto_detect, self.action_snap,
            self.action_cut, self.action_lock,
            self.action_export,
        ):
            action.setEnabled(enabled)
        self.scene.editing_enabled = enabled

    # --- PDF / Image loading ---

    def on_open_pdf(self):
        pdf_path, _ = QFileDialog.getOpenFileName(
            self, "Open PDF", "", "PDF files (*.pdf)"
        )
        if not pdf_path:
            return

        page_count, page_sizes = pdf_page_info(pdf_path)
        dlg = PdfImportDialog(pdf_path, page_count, page_sizes, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        pages = dlg.get_pages()
        self._import_pdf(pdf_path, pages)

    def _import_pdf(self, pdf_path: str, pages):
        if self._work_dir is None:
            self._work_dir = tempfile.mkdtemp(prefix="puzzle_joiner_")

        def task(progress):
            progress.emit(2, "Computing cache key...")

            # Determine which pages are requested (1-indexed)
            if pages:
                requested = sorted(pages)
            else:
                page_count, _ = pdf_page_info(pdf_path)
                requested = list(range(1, page_count + 1))

            # Check cache for each requested page
            cached_pages = {}
            uncached_page_nums = []
            for page_num in requested:
                cp = _cached_page_path(cache_dir, page_num)
                if os.path.exists(cp) and os.path.getsize(cp) > 0:
                    cached_pages[page_num] = cp
                else:
                    uncached_page_nums.append(page_num)

            results = []
            total = len(requested)

            # Load cached pages in parallel (compute crop rect on load)
            if cached_pages:
                load_args = [(pn, cached_pages[pn], requested.index(pn))
                             for pn in requested if pn in cached_pages]
                workers = max(1, os.cpu_count() // 2)
                with concurrent.futures.ProcessPoolExecutor(max_workers=workers, initializer=_set_low_priority) as executor:
                    for i, res in enumerate(executor.map(_load_cached_page, load_args)):
                        if res is not None:
                            results.append(res)
                        progress.emit(5 + int(40 * (i + 1) / len(load_args)),
                                      f"Loading cached page {i + 1}/{len(load_args)}...")

            if uncached_page_nums:
                # Split PDF to get per-page PDFs, only for uncached pages
                progress.emit(45, "Splitting PDF...")
                page_pdfs = split_pdf_pages(pdf_path, self._work_dir, set(uncached_page_nums))
                workers = max(1, os.cpu_count() // 2)
                n_uncached = len(page_pdfs)

                # Build args for convert+process workers
                convert_args = []
                for pdf_page_num, pdf_page_path in page_pdfs:
                    idx = requested.index(pdf_page_num)
                    cp = _cached_page_path(cache_dir, pdf_page_num)
                    ac_path = _cached_autocrop_path(cache_dir, pdf_page_num)
                    convert_args.append((pdf_page_path, pdf_page_num, idx, cp, ac_path))

                completed = 0
                with concurrent.futures.ProcessPoolExecutor(max_workers=workers, initializer=_set_low_priority) as executor:
                    futures = {executor.submit(_convert_and_process_page, a): a[2] for a in convert_args}
                    for future in concurrent.futures.as_completed(futures):
                        completed += 1
                        pct = 50 + int(45 * completed / n_uncached)
                        progress.emit(pct, f"Processing page {completed}/{n_uncached}...")
                        res = future.result()
                        if res is not None:
                            results.append(res)

            # Sort by page index (idx is last element)
            results.sort(key=lambda r: r[-1])
            progress.emit(98, "Done.")
            return results

        def on_done(result):
            if result is None:
                return
            for (path, bgra, crop_rect, idx) in result:
                self._add_piece_from_array(bgra, path, idx, crop_rect=crop_rect)
            self._layout_dir = cache_dir
            if load_layout(self._layout_dir, self.pieces):
                for piece in self.pieces:
                    piece.rebuild_display_pixmap()
                    piece.rebuild_thumbnail()
                self.scene.sync_all_from_pieces()
                self.thumbnail_panel.update_all()
            n = len(result)
            name = os.path.basename(pdf_path)
            self.status_bar.showMessage(f"Loaded {n} pages from {name}")
            self.view.fit_all()

        cache_dir = _cache_dir_for_pdf(pdf_path)
        self._run_with_progress("Importing PDF", task, on_done=on_done)

    def on_open_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open Images", "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp)"
        )
        if not paths:
            return
        self._import_images(paths)

    def _import_images(self, paths: list):
        def task(progress):
            result = []
            for idx, path in enumerate(paths):
                pct = int(90 * idx / max(len(paths), 1))
                progress.emit(pct, f"Loading {os.path.basename(path)}...")
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
                elif img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                result.append((path, img, idx))
            progress.emit(95, "Done.")
            return result

        def on_done(result):
            if not result:
                return
            for (path, bgra, idx) in result:
                self._add_piece_from_array(bgra, path, idx)
            self._layout_dir = _images_cache_dir(paths)
            if load_layout(self._layout_dir, self.pieces):
                for piece in self.pieces:
                    piece.rebuild_display_pixmap()
                    piece.rebuild_thumbnail()
                self.scene.sync_all_from_pieces()
                self.thumbnail_panel.update_all()
            self.status_bar.showMessage(f"Loaded {len(result)} images")
            self.view.fit_all()

        self._run_with_progress("Loading images", task, on_done=on_done)

    def _add_piece_from_array(self, bgra: np.ndarray, source_path: str, idx: int,
                              crop_rect: tuple = None):
        piece = PuzzlePiece()
        piece.source_path = source_path
        piece.image = bgra
        piece.layer_index = idx
        piece.crop_rect = crop_rect
        img = piece.get_cropped_image()
        h, w = img.shape[:2]
        # Default position: stacked horizontally
        piece.x = w / 2.0 + idx * (w + 20)
        piece.y = h / 2.0
        piece.rebuild_display_pixmap()
        piece.rebuild_thumbnail()
        self.pieces.append(piece)
        self.scene.add_piece(piece)
        self.thumbnail_panel.add_piece(piece)

    # --- Auto-detect ---

    def on_auto_detect(self):
        if not self.pieces:
            self.status_bar.showMessage("No pieces loaded.")
            return

        def task(progress):
            progress.emit(10, "Computing ORB features...")
            auto_detect_placements(self.pieces)
            progress.emit(90, "Syncing positions...")
            return True

        def on_done(result):
            self.scene.sync_all_from_pieces()
            self.thumbnail_panel.update_all()
            self.view.fit_all()
            self._update_snap_enabled()
            self._auto_save_layout()
            self.status_bar.showMessage("Auto-alignment complete.")

        self._run_with_progress("Aligning all pieces", task, on_done=on_done)

    # --- Snap ---

    def on_snap(self):
        piece = self.scene.get_selected_piece()
        if piece is None:
            self._snap_all()
            return
        neighbors = self.scene.get_neighbors(piece)
        if not neighbors:
            self.status_bar.showMessage("No neighbors found for snapping.")
            return

        # Capture before state
        old_x, old_y = piece.x, piece.y
        old_rot, old_scale = piece.rotation_deg, piece.scale
        old_matched = piece.is_matched

        def task(progress):
            progress.emit(20, "Computing features...")
            ok = snap_piece_to_neighbors(piece, neighbors)
            progress.emit(90, "Done.")
            return ok

        def on_done(ok):
            if ok:
                cmd = SnapCommand(
                    piece, old_x, old_y, old_rot, old_scale, old_matched,
                    piece.x, piece.y, piece.rotation_deg, piece.scale, True,
                    sync_fn=self._sync_piece,
                )
                # Already applied by snap_piece_to_neighbors, so push without re-doing
                self.undo_stack.push(cmd)
                self.status_bar.showMessage("Snap successful.")
            else:
                self.status_bar.showMessage("Snap found no match.")
            self._update_snap_enabled()

        self._run_with_progress("Snapping piece", task, on_done=on_done)

    def _snap_all(self):
        # Capture before states for all pieces
        before = {p: (p.x, p.y, p.rotation_deg, p.scale, p.is_matched)
                  for p in self.pieces}

        def task(progress):
            total = len(self.pieces)
            snapped = []
            for i, piece in enumerate(self.pieces):
                progress.emit(int(90 * i / max(total, 1)), f"Snapping piece {i+1}/{total}...")
                neighbors = self.scene.get_neighbors(piece)
                if neighbors:
                    ok = snap_piece_to_neighbors(piece, neighbors)
                    if ok:
                        piece.is_matched = True
                        snapped.append(piece)
            return snapped

        def on_done(snapped):
            if snapped:
                cmds = []
                for p in snapped:
                    ox, oy, orot, oscale, omatched = before[p]
                    cmds.append(SnapCommand(
                        p, ox, oy, orot, oscale, omatched,
                        p.x, p.y, p.rotation_deg, p.scale, True,
                        sync_fn=self._sync_piece,
                    ))
                cmd = SnapAllCommand(cmds)
                self.undo_stack.push(cmd)
            self.scene.sync_all_from_pieces()
            self.thumbnail_panel.update_all()
            self._update_snap_enabled()
            self.status_bar.showMessage("Snap all complete.")

        self._run_with_progress("Snapping all pieces", task, on_done=on_done)

    # --- Cut tool ---

    def on_cut(self):
        piece = self.scene.get_selected_piece()
        if piece is None:
            self.status_bar.showMessage("Select a piece first.")
            return
        dlg = CutToolDialog(piece, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            item = self.scene.piece_items.get(piece)
            if item:
                item.rebuild_pixmap()
            piece.rebuild_thumbnail()
            self.thumbnail_panel.update_piece(piece)
            self._auto_save_layout()
            self.status_bar.showMessage("Crop applied.")

    # --- Lock/Unlock ---

    def on_lock_toggle(self):
        piece = self.scene.get_selected_piece()
        if piece is None:
            self.status_bar.showMessage("Select a piece first.")
            return
        cmd = LockCommand(piece, sync_fn=self._sync_piece)
        self.undo_stack.push(cmd)
        state = "locked" if piece.is_locked else "unlocked"
        self.status_bar.showMessage(f"Piece {state}.")

    def _sync_piece(self, piece):
        """Sync piece visuals after undo/redo."""
        item = self.scene.piece_items.get(piece)
        if item:
            item.sync_from_piece()
            if piece.is_locked:
                item.setSelected(False)
        self.thumbnail_panel.update_piece(piece)
        self._update_snap_enabled()

    # --- Export ---

    def on_export(self):
        if not self.pieces:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export TIFF", "puzzle.tif", "TIFF (*.tif *.tiff)"
        )
        if not path:
            return

        base, ext = os.path.splitext(path)
        layered_path = f"{base}_layered{ext}"
        flat_path = f"{base}_flat{ext}"

        def task(progress):
            progress.emit(5, "Exporting layered TIFF...")
            export_layered_tiff(self.pieces, layered_path)
            progress.emit(50, "Exporting flattened TIFF...")
            export_flattened_tiff(self.pieces, flat_path)
            progress.emit(95, "Done.")
            return (layered_path, flat_path)

        def on_done(paths):
            self.status_bar.showMessage(f"Exported: {paths[0]} and {paths[1]}")

        self._run_with_progress("Exporting", task, on_done=on_done)

    # --- Fit View ---

    def on_fit_view(self):
        self.view.fit_all()

    # --- Selection sync ---

    def _update_snap_enabled(self):
        piece = self.scene.get_selected_piece()
        if piece is None:
            self.action_snap.setEnabled(False)
        else:
            self.action_snap.setEnabled(bool(self.scene.get_neighbors(piece)))

    def on_scene_selection_changed(self, piece):
        self.thumbnail_panel.select_piece(piece)
        self._update_snap_enabled()

    def on_thumbnail_select(self, piece: PuzzlePiece):
        # Clear current selection
        self.scene.clearSelection()
        item = self.scene.piece_items.get(piece)
        if item:
            item.setSelected(True)
            bb = piece.get_bounding_box()
            viewport_rect = self.view.mapToScene(self.view.viewport().rect()).boundingRect()
            if not viewport_rect.intersects(bb):
                self.view.smoothCenterOn(bb.center().x(), bb.center().y())
        self.thumbnail_panel.select_piece(piece)
        self._update_snap_enabled()

    def _select_adjacent_piece(self, forward: bool):
        if not self.pieces:
            return
        current = self.scene.get_selected_piece()
        if current is None:
            idx = 0 if forward else len(self.pieces) - 1
        else:
            try:
                cur_idx = self.pieces.index(current)
            except ValueError:
                cur_idx = 0
            idx = (cur_idx + (1 if forward else -1)) % len(self.pieces)
        self.on_thumbnail_select(self.pieces[idx])

    # --- Auto-save layout ---

    def _auto_save_layout(self):
        if self._layout_dir and self.pieces:
            save_layout(self._layout_dir, self.pieces)
